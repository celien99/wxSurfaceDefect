"""逐图粗扫→路由→复核→融合编排：单张原图在单个设备上保持 GPU 驻留。

粗扫补丁与复核结果都在 GPU 上拼接/路由/融合，只有路由图与最终异常图各做
一次 D2H；解码与批量建批放在有界预取线程，与 GPU 前向重叠。
"""
from __future__ import annotations

import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from queue import Queue

import cv2
import numpy as np
import torch

from hiad.constants import (
    TASK_TYPE_DYNAMIC_PATCH,
    TASK_TYPE_THUMBNAIL,
)
from hiad.data import (
    HRSample,
    HRImageIndex,
    MultiResolutionIndex,
    build_multiresolution_region,
    split_multiresolution_regions,
)
from hiad.data.patch_builder import build_patch_batch, build_thumbnail_batch
from hiad.detectors.base import BaseDetector
from hiad.inferencer.refinement import (
    refinement_tile_statistics,
    select_refinement_regions,
)
from hiad.runtime import mapops
from hiad.runtime.contracts import (
    ImageSize,
    RefinementStatistics,
    TaskInputRecord,
)
from hiad.runtime.inference_config import InferenceConfig
from hiad.task.contracts import (
    RefinementPatchTask,
    TaskDefinition,
    ThumbnailTask,
)

# 当前骨干（vit_base_patch16_dinov3）的解析常量：16 patch、768 embed、8 层。
_TOKEN_STRIDE = 16
_EMBED_DIM = 768
_LAYERS = 8
_SAFETY_FACTOR = 4.0


@dataclass(frozen=True)
class ImagePipelineOutput:
    """单张原图的完整逐图推理结果。"""

    image_path: str
    image_size: ImageSize
    final_map: np.ndarray
    thumbnail_score: float
    refinement_statistics: RefinementStatistics
    coarse_seconds: float
    routing_seconds: float
    refinement_seconds: float


@dataclass
class _PrefetchItem:
    """预取线程产出的解码图像与粗扫/缩略批量。"""

    image: np.ndarray
    sample: HRSample
    coarse_batch: object
    coarse_records: list[TaskInputRecord]
    thumbnail_batch: object
    thumbnail_record: TaskInputRecord
    thumbnail_size: int


class DeviceImagePipeline:
    """在单个设备上按图执行完整粗到细链路。

    每个设备持有一份全部任务模型；输入样本按图顺序处理，解码与建批在预取
    线程与 GPU 前向重叠。``device`` 取第一个检测器的设备。
    """

    def __init__(
        self,
        detectors: Mapping[str, BaseDetector],
        coarse_tasks: Sequence[TaskDefinition],
        refinement_task: RefinementPatchTask,
        *,
        inference_config: InferenceConfig,
        global_routing_weight: float,
        score_top_k: int,
        refinement_bridge_gap_tiles: int,
        map_gaussian_sigma: float,
        batch_cap: int = 0,
    ) -> None:
        self.detectors = dict(detectors)
        self.coarse_tasks = list(coarse_tasks)
        self.refinement_task = refinement_task
        self.inference_config = inference_config
        self.global_routing_weight = float(global_routing_weight)
        self.score_top_k = int(score_top_k)
        self.refinement_bridge_gap_tiles = int(refinement_bridge_gap_tiles)
        self.map_gaussian_sigma = float(map_gaussian_sigma)
        self.batch_cap = int(batch_cap)
        # 质量门禁由上层 inference() 统一评估；batch_cap 是自适应批的硬上限
        # （上层 --batch-size 注入，0 = 无上限）。
        self.device = next(iter(self.detectors.values())).device

    def _coarse_task(self) -> TaskDefinition:
        return next(task for task in self.coarse_tasks
                    if task["type"] == TASK_TYPE_DYNAMIC_PATCH)

    def _thumbnail_task(self) -> ThumbnailTask:
        return next(task for task in self.coarse_tasks
                    if task["type"] == TASK_TYPE_THUMBNAIL)

    def _records_per_batch(self, patch_size: int, context_views: int) -> int:
        """按显存预算估算单前向批可容纳的记录数（分析模型，保守安全系数）。

        每记录显存 ≈ 编码特征（层数×token²×embed×4B）× (1+上下文视图) × 安全 4。
        ``batch_memory_budget_gb=0`` 时取当前设备空闲显存的一半；非 CUDA 或
        预算为零时整批全跑。``batch_cap`` 大于零时作为硬上限（由上层
        ``--batch-size`` 注入）。
        """
        if self.device.type != "cuda":
            records = int(2**31 - 1)
        else:
            if self.inference_config.batch_memory_budget_gb <= 0:
                free_memory, _ = torch.cuda.mem_get_info(self.device)
                budget_bytes = int(free_memory * 0.5)
            else:
                budget_bytes = int(
                    self.inference_config.batch_memory_budget_gb * 1024**3
                )
            tokens = (patch_size // _TOKEN_STRIDE) ** 2
            per_record = (
                _SAFETY_FACTOR
                * _LAYERS
                * _EMBED_DIM
                * tokens
                * (1 + context_views)
                * 4
            )
            records = max(1, budget_bytes // max(1, per_record))
        if self.batch_cap > 0:
            records = min(records, self.batch_cap)
        return records

    def _chunk_batch(self, batch: Mapping[str, object], chunk_size: int):
        """把批量字典切成自适应子批，返回 ``(chunk, start, stop)`` 序列。

        张量字段按行切片，``low_resolution_index_<n>`` 等 list 字段同步切片，
        保证每张图上下文与其索引成对。
        """
        count = batch["image"].shape[0]
        if chunk_size <= 0 or chunk_size >= count:
            yield batch, 0, count
            return
        for start in range(0, count, chunk_size):
            stop = min(start + chunk_size, count)
            chunk = {key: value[start:stop] for key, value in batch.items()}
            yield chunk, start, stop

    def _forward_and_collect(
        self,
        detector: BaseDetector,
        batch: Mapping[str, object],
        patch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """按自适应子批前向并拼接，返回 ``(pixel_maps, token_scores)``。

        ``pixel_maps`` 形状 ``(N, 1, P, P)``；调用方负责在拼接/融合时做
        ``[:, 0]`` 二维切片（见 ``_coarse_forward`` / ``_refine_and_merge``）。
        """
        chunk_size = self._records_per_batch(
            patch_size,
            sum(1 for key in batch if key.startswith("low_resolution_image")),
        )
        pixel_parts: list[torch.Tensor] = []
        token_parts: list[torch.Tensor] = []
        for chunk, _start, _stop in self._chunk_batch(batch, chunk_size):
            fused_pixel, fused_token = detector.inference_batch(chunk)
            pixel_parts.append(fused_pixel)
            token_parts.append(fused_token)
        return torch.cat(pixel_parts, dim=0), torch.cat(token_parts, dim=0)

    def process_images(
        self,
        samples: Sequence[HRSample],
    ) -> list[ImagePipelineOutput]:
        """按图顺序处理样本，解码与建批在预取线程与 GPU 前向重叠。"""
        if not samples:
            return []
        outputs: list[ImagePipelineOutput] = []
        with _PrefetchWorker(samples, self._build_item) as worker:
            item = worker.next()
            while item is not None:
                outputs.append(self._process_item(item))
                item = worker.next()
        return outputs

    def _build_item(self, sample: HRSample) -> _PrefetchItem:
        """解码一张图并预建粗扫与缩略批量（CPU 线程执行）。

        质量门禁由上层 ``inference()`` 统一评估，这里只解码一次建批。
        """
        sample.open()
        image = sample.image.image
        if image is None:
            raise RuntimeError("Sample image was not decoded")
        image_size = (int(image.shape[1]), int(image.shape[0]))
        coarse_task = self._coarse_task()
        thumbnail_task = self._thumbnail_task()
        base = {
            "task_name": coarse_task["name"],
            "task_type": coarse_task["type"],
            "image_path": sample.image.image_path,
            "image_size": image_size,
            "model_input_size": (coarse_task["patch_size"], coarse_task["patch_size"]),
        }
        indexes = split_multiresolution_regions(
            image_size=image_size,
            patch_size=coarse_task["patch_size"],
            ds_factors=coarse_task["ds_factors"],
            stride=coarse_task["stride"],
        )
        coarse_batch, coarse_records = build_patch_batch(
            image, indexes, coarse_task["patch_size"], base
        )
        thumb_base = {
            "task_name": thumbnail_task["name"],
            "task_type": thumbnail_task["type"],
            "image_path": sample.image.image_path,
            "image_size": image_size,
            "model_input_size": (
                thumbnail_task["thumbnail_size"], thumbnail_task["thumbnail_size"],
            ),
        }
        thumbnail_batch, thumbnail_record = build_thumbnail_batch(
            image, thumbnail_task["thumbnail_size"], thumb_base
        )
        return _PrefetchItem(
            image=image,
            sample=sample,
            coarse_batch=coarse_batch,
            coarse_records=coarse_records,
            thumbnail_batch=thumbnail_batch,
            thumbnail_record=thumbnail_record,
            thumbnail_size=thumbnail_task["thumbnail_size"],
        )

    def _process_item(self, item: _PrefetchItem) -> ImagePipelineOutput:
        """在主线程执行 GPU 前向、路由、复核、融合，并负责关闭样本。"""
        image_size = (int(item.image.shape[1]), int(item.image.shape[0]))
        try:
            coarse_started = time.perf_counter()
            coarse_map, thumbnail_score, global_context_map = self._coarse_forward(
                item, image_size
            )
            coarse_seconds = time.perf_counter() - coarse_started

            routing_started = time.perf_counter()
            regions, _routing_np, _coarse = self._route(
                coarse_map, global_context_map, image_size
            )
            routing_seconds = time.perf_counter() - routing_started

            refinement_started = time.perf_counter()
            final_map = self._refine_and_merge(
                item, regions, coarse_map, image_size
            )
            refinement_seconds = time.perf_counter() - refinement_started

            statistics = refinement_tile_statistics(
                image_size,
                self.refinement_task["patch_size"],
                regions,
            )
            return ImagePipelineOutput(
                image_path=item.sample.image.image_path,
                image_size=image_size,
                final_map=final_map,
                thumbnail_score=thumbnail_score,
                refinement_statistics=statistics,
                coarse_seconds=coarse_seconds,
                routing_seconds=routing_seconds,
                refinement_seconds=refinement_seconds,
            )
        finally:
            item.sample.close()

    def _coarse_forward(
        self,
        item: _PrefetchItem,
        image_size: ImageSize,
    ) -> tuple[torch.Tensor, float, torch.Tensor]:
        """执行粗扫任务并返回原图分辨率 GPU 异常图、缩略图分数与全局先验。

        粗扫补丁图为 ``(N, 1, P, P)``，拼接前做 ``[:, 0]`` 二维切片（与 legacy
        ``_gather_patch_predictions`` 的 2D 契约一致）。
        """
        coarse_task = self._coarse_task()
        coarse_detector = self.detectors[coarse_task["name"]]
        pixel_maps, _token_scores = self._forward_and_collect(
            coarse_detector, item.coarse_batch, coarse_task["patch_size"]
        )
        coarse_map = mapops.stitch_patch_maps_torch(
            pixel_maps[:, 0], item.coarse_records, image_size, self.device
        )
        if self.map_gaussian_sigma > 0:
            coarse_map = mapops.gaussian_blur_torch(
                coarse_map, self.map_gaussian_sigma
            )

        thumbnail_detector = self.detectors[self._thumbnail_task()["name"]]
        thumb_pixel, thumb_token = thumbnail_detector.inference_batch(
            item.thumbnail_batch
        )
        thumbnail_score = float(
            mapops.top_k_token_scores_torch(
                thumb_token, thumbnail_detector.score_top_k
            ).cpu().item()
        )
        # 缩略图异常图按原图分辨率线性放大作为路由全局先验。保持 cv2.INTER_LINEAR
        # 与 legacy 路径逐位一致（不使用 F.interpolate，避免亚像素差异）。
        thumbnail_map_np = thumb_pixel[0, 0].cpu().numpy()
        global_context_map = torch.from_numpy(
            cv2.resize(
                thumbnail_map_np, image_size, interpolation=cv2.INTER_LINEAR
            )
        ).to(self.device)
        return coarse_map, thumbnail_score, global_context_map

    def _route(
        self,
        coarse_map: torch.Tensor,
        global_context_map: torch.Tensor,
        image_size: ImageSize,
    ) -> tuple[list[HRImageIndex], np.ndarray, torch.Tensor]:
        """构建路由图、选择复核区域（CPU 连通域），返回 ``(regions, routing_np, coarse)``。"""
        routing_map = mapops.build_routing_map_torch(
            coarse_map, global_context_map, self.global_routing_weight
        )
        threshold = float(
            torch.quantile(
                routing_map,
                self.refinement_task["refinement_quantile"],
            )
        )
        routing_np = routing_map.cpu().numpy()
        regions = select_refinement_regions(
            routing_np,
            threshold=threshold,
            tile_size=self.refinement_task["patch_size"],
            min_area=self.refinement_task["refinement_min_area"],
            safety_fraction=self.refinement_task["refinement_safety_fraction"],
            max_bridge_gap_tiles=self.refinement_bridge_gap_tiles,
        )
        return regions, routing_np, coarse_map

    def _refine_and_merge(
        self,
        item: _PrefetchItem,
        regions: Sequence[HRImageIndex],
        coarse_map: torch.Tensor,
        image_size: ImageSize,
    ) -> np.ndarray:
        """对候选区域建批前向并融合回粗扫图，返回 CPU ``float32`` 最终图。"""
        refinement_detector = self.detectors[self.refinement_task["name"]]
        task = self.refinement_task
        base = {
            "task_name": task["name"],
            "task_type": task["type"],
            "image_path": item.sample.image.image_path,
            "image_size": image_size,
            "model_input_size": (task["patch_size"], task["patch_size"]),
        }
        indexes = [
            build_multiresolution_region(image_size, region, task["ds_factors"])
            for region in regions
        ]
        if not indexes:
            return coarse_map.cpu().numpy().astype(np.float32)
        batch, _records = build_patch_batch(
            item.image, indexes, task["patch_size"], base
        )
        pixel_maps, _token_scores = self._forward_and_collect(
            refinement_detector, batch, task["patch_size"]
        )
        refinements = [
            (region, pixel_maps[i, 0]) for i, region in enumerate(regions)
        ]
        merged = mapops.merge_refinement_maps_torch(
            coarse_map, refinements, image_size, self.device
        )
        return merged.cpu().numpy().astype(np.float32)


class _PrefetchWorker:
    """有界预取：后台线程解码并建批下一张图，与主线程 GPU 前向重叠。

    容量 2 张图；主线程消费 ``next()`` 时若队列已空（异常情况）则在主线程
    同步构建，保证正确性不依赖调度。
    """

    def __init__(
        self,
        samples: Sequence[HRSample],
        build_item,
    ) -> None:
        self._samples = iter(samples)
        self._build_item = build_item
        self._queue: Queue[_PrefetchItem | None] = Queue(maxsize=2)
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._produce, daemon=True)
        self._thread.start()

    def _produce(self) -> None:
        try:
            for sample in self._samples:
                self._queue.put(self._build_item(sample))
            self._queue.put(None)
        except BaseException as error:  # noqa: BLE001 - 线程异常回传主线程
            self._error = error
            self._queue.put(None)

    def next(self) -> _PrefetchItem | None:
        item = self._queue.get()
        if self._error is not None:
            raise self._error
        return item

    def close(self) -> None:
        self._thread.join(timeout=5.0)

    def __enter__(self) -> _PrefetchWorker:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
