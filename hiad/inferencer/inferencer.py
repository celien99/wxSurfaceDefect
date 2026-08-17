from __future__ import annotations

import copy
import os
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from types import TracebackType
from typing import cast

import cv2
import numpy as np
import torch
from easydict import EasyDict
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from hiad.constants import (
    TASK_TYPE_DYNAMIC_PATCH,
    TASK_TYPE_REFINEMENT_PATCH,
    TASK_TYPE_THUMBNAIL,
)
from hiad.data import HRImageIndex, HRSample
from hiad.datasets import StreamingTaskDataset
from hiad.detectors.base import BaseDetector
from hiad.detectors.config import DetectorConfig, validate_required_config
from hiad.inferencer.modelmanager import ModelManager
from hiad.inferencer.refinement import (
    build_routing_map,
    merge_refinement_maps,
    select_refinement_regions,
)
from hiad.runtime.contracts import (
    DeviceInferenceResults,
    FloatMap,
    ImageArray,
    ImageQualityResult,
    ImageSize,
    InferenceResult,
    PatchPrediction,
    ScoreCalibration,
    ScoreVector,
)
from hiad.runtime.devices import validate_gpu_ids
from hiad.runtime.decision import (
    apply_quality_gate,
    classify_score,
    component_statistics,
    image_score_from_statistics,
    top_k_map_score,
)
from hiad.runtime.partition import round_robin_partition
from hiad.runtime.prediction import threshold_anomaly_maps
from hiad.runtime.quality import assess_image_quality
from hiad.runtime.score_calibration import (
    component_thresholds_for_samples,
    load_score_calibration,
    pixel_thresholds_for_samples,
    thresholds_for_samples,
)
from hiad.task import load_tasks
from hiad.task.contracts import RefinementPatchTask, TaskDefinition


RegionsByPath = dict[str, list[HRImageIndex]]


def _gather_patch_predictions(
    patches: list[PatchPrediction],
    image_size: ImageSize,
) -> FloatMap:
    """将原图补丁按坐标和 Hann 权重拼回连续异常图。

    Args:
        patches (list[PatchPrediction]): 补丁输入记录及对应二维异常图。记录中的
            ``source_xywh`` 使用原图像素坐标，``valid_source_hw`` 使用
            ``(height, width)``。
        image_size (ImageSize): 原图 ``(width, height)``。

    Returns:
        FloatMap: 原图分辨率 ``(height, width)`` 的连续 ``float32`` 异常图。

    Raises:
        KeyError: 补丁记录缺少坐标或有效区域字段。
        ValueError: 补丁不能完整覆盖原图，或预测形状无法与目标区域广播。

    Notes:
        二维 Hann 权重用于抑制补丁边缘伪影，并增加 ``0.05`` 最低权重以保证
        单补丁边界和整图外沿仍然得到覆盖。
    """
    image_width, image_height = image_size
    accumulated = np.zeros((image_height, image_width), dtype=np.float64)
    weight_map = np.zeros((image_height, image_width), dtype=np.float64)

    for record, prediction in patches:
        x, y, width, height = record["source_xywh"]
        valid_height, valid_width = record["valid_source_hw"]
        prediction = np.asarray(prediction, dtype=np.float32)
        # 边缘保底权重避免整图边界或单补丁区域出现零覆盖。
        row_hann = (
            np.hanning(height) if height > 1 else np.ones(1, dtype=np.float64)
        )
        column_hann = (
            np.hanning(width) if width > 1 else np.ones(1, dtype=np.float64)
        )
        weights = 0.05 + 0.95 * np.outer(row_hann, column_hann)
        valid_weights = weights[:valid_height, :valid_width]
        accumulated[y:y + valid_height, x:x + valid_width] += (
            prediction[:valid_height, :valid_width] * valid_weights
        )
        weight_map[y:y + valid_height, x:x + valid_width] += valid_weights

    if np.any(weight_map <= 0):
        raise ValueError("Patch predictions do not cover the complete source image")
    return (accumulated / weight_map).astype(np.float32)


def inference_in_device(
    test_samples: list[HRSample],
    task_group: list[TaskDefinition],
    model_manager: ModelManager,
    batch_size: int,
    *,
    regions_by_path: RegionsByPath | None = None,
) -> DeviceInferenceResults:
    """在单设备上执行一组任务，并按原图路径汇总结果。

    Args:
        test_samples (list[HRSample]): 路径唯一且带类别的推理样本。
        task_group (list[TaskDefinition]): 分配给当前设备的粗扫、复核或缩略任务。
        model_manager (ModelManager): 已在当前设备加载任务检查点的模型管理器。
        batch_size (int): DataLoader 批量大小。
        regions_by_path (RegionsByPath | None): 复核任务使用的原图 ``xywh`` 区域；
            ``None`` 表示按任务规则生成整图滑窗。

    Returns:
        DeviceInferenceResults: 以原图路径为键，累积图像尺寸、补丁异常图和可选
        缩略图异常图/分数的结果。

    Raises:
        KeyError: 任务模型或数据记录字段缺失。
        RuntimeError: 检测器预测数量与数据集输入记录数量不一致。
        ValueError: 任务类型不受支持，或数据集/模型输入不符合契约。
    """
    paths = [sample.image.image_path for sample in test_samples]
    results: DeviceInferenceResults = {
        path: {
            "image_size": None,
            "patches": [],
            "thumbnail": None,
            "thumbnail_score": None,
        }
        for path in paths
    }

    for task in task_group:
        task_name = task["name"]
        detector = model_manager.get_detector(task_name)
        dataset = StreamingTaskDataset(
            copy.deepcopy(test_samples),
            task,
            training=False,
            regions_by_path=regions_by_path,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        predictions = detector.inference_step(dataloader)
        if len(predictions) != len(dataset.records):
            raise RuntimeError(
                f"Task {task_name} returned {len(predictions)} predictions for "
                f"{len(dataset.records)} inputs"
            )

        if task["type"] in {TASK_TYPE_DYNAMIC_PATCH, TASK_TYPE_REFINEMENT_PATCH}:
            for record, prediction in zip(dataset.records, predictions):
                path = record["image_path"]
                results[path]["image_size"] = record["image_size"]
                results[path]["patches"].append((record, prediction["anomaly_map"]))
        elif task["type"] == TASK_TYPE_THUMBNAIL:
            for record, prediction in zip(dataset.records, predictions):
                path = record["image_path"]
                results[path]["image_size"] = record["image_size"]
                results[path]["thumbnail"] = prediction["anomaly_map"]
                results[path]["thumbnail_score"] = prediction["score"]
        else:
            raise ValueError(f"Unsupported task type: {task}")

    return results


class HRInferencer:
    """执行粗扫、全局路由、局部复核、校准判定与质量门禁。

    Args:
        detector_class (type[BaseDetector]): 每个任务使用的检测器类型。
        config (object): 已包含全部生产字段的映射或属性配置对象。
        checkpoint_root (str | os.PathLike[str]): 包含 ``tasks.json``、各任务权重和
            可选 ``score_calibration.json`` 的检查点目录。
        gpu_ids (object): 非空、不重复的 CUDA 设备编号列表。
        batch_size (int | None): 每任务推理批量大小；``None`` 表示使用当前样本数。
        require_score_calibration (bool): 是否必须加载分数校准；训练阶段拟合阈值前
            可设为 ``False``。

    Attributes:
        coarse_tasks_in_devices (list[list[TaskDefinition]]): 按设备分配的粗扫和缩略
            图任务。
        refinement_tasks_in_devices (list[list[TaskDefinition]]): 按设备分配的复核
            任务。
        refinement_task (RefinementPatchTask): 唯一复核任务及候选选择参数。
        model_managers (list[ModelManager]): 各设备已加载的任务模型。
        score_calibration (ScoreCalibration | None): 可选图像、像素和组件阈值。
        quality_thresholds (dict[str, float]): 曝光、截断比例和清晰度门禁阈值。
    """

    def __init__(
        self,
        detector_class: type[BaseDetector],
        config: object,
        checkpoint_root: str | os.PathLike[str],
        gpu_ids: object,
        batch_size: int | None = None,
        require_score_calibration: bool = True,
    ) -> None:
        gpu_ids = validate_gpu_ids(gpu_ids)
        if batch_size is not None and (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer or None")
        if not isinstance(require_score_calibration, bool):
            raise TypeError("require_score_calibration must be a boolean")

        checkpoint_root = os.path.abspath(checkpoint_root)
        normalized_config = EasyDict(config) if isinstance(config, dict) else config
        validate_required_config(normalized_config)
        detector_config = cast(DetectorConfig, normalized_config)
        self.batch_size: int | None = batch_size
        tasks_path = os.path.join(checkpoint_root, "tasks.json")
        if not os.path.isfile(tasks_path):
            raise FileNotFoundError(f"Task configuration not found: {tasks_path}")
        tasks = load_tasks(tasks_path)

        task_groups = [group for group in round_robin_partition(tasks, len(gpu_ids)) if group]

        self.coarse_tasks_in_devices: list[list[TaskDefinition]] = [
            [task for task in task_group if task["type"] != TASK_TYPE_REFINEMENT_PATCH]
            for task_group in task_groups
        ]
        self.refinement_tasks_in_devices: list[list[TaskDefinition]] = [
            [task for task in task_group if task["type"] == TASK_TYPE_REFINEMENT_PATCH]
            for task_group in task_groups
        ]
        refinement_tasks = [
            cast(RefinementPatchTask, task)
            for task in tasks
            if task["type"] == TASK_TYPE_REFINEMENT_PATCH
        ]
        self.refinement_task: RefinementPatchTask = refinement_tasks[0]
        self.model_managers: list[ModelManager] = [
            ModelManager(
                tasks,
                detector_class,
                detector_config,
                checkpoint_root,
                gpu_ids[index],
            )
            for index, tasks in enumerate(tqdm(task_groups, desc="Loading checkpoints..."))
        ]
        self.score_calibration: ScoreCalibration | None = (
            load_score_calibration(checkpoint_root)
            if require_score_calibration
            else None
        )
        self.map_gaussian_sigma: float = float(detector_config.map_gaussian_sigma)
        self.decision_recheck_margin_ratio: float = float(
            detector_config.decision_recheck_margin_ratio
        )
        self.quality_thresholds: dict[str, float] = {
            key: float(detector_config[key])
            for key in (
                "min_mean_luminance",
                "max_mean_luminance",
                "max_clipped_fraction",
                "min_focus_variance",
            )
        }
        self.global_routing_weight: float = float(
            detector_config.global_routing_weight
        )
        self.score_top_k: int = int(detector_config.score_top_k)
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=len(task_groups)
        )
        self._inference_lock = Lock()
        self._closed: bool = False

    def inference(
        self,
        test_samples: list[HRSample],
        *,
        display_size: int | list[int] | tuple[int, int] | None = None,
    ) -> InferenceResult:
        """按固定样本顺序执行完整粗到细推理并返回稳定结果结构。

        Args:
            test_samples (list[HRSample]): 非空、路径唯一且 ``clsname`` 非空的样本。
            display_size (int | list[int] | tuple[int, int] | None): 可选显示图尺寸；
                整数表示正方形，二元序列按 ``(width, height)`` 解释。

        Returns:
            InferenceResult: 与输入样本同序的原图路径、原图分辨率 ``float32``
            异常图、图像分数和质量结果。启用校准时还包含分类别阈值、组件摘要、
            ``OK/RECHECK/NG`` 判定、``0/1`` 二值掩码；请求显示图时包含 HWC RGB
            ``uint8`` 图像映射。

        Raises:
            TypeError: 样本、显示尺寸或配置类型错误。
            ValueError: 样本、任务结果、图像尺寸、补丁覆盖或阈值数据不一致。
            FileNotFoundError: 推理图像或检查点资源不存在。
            RuntimeError: 推理器已关闭、模型前向失败或必要任务结果缺失。

        Notes:
            缩略图异常图仅参与复核候选路由；最终像素证据由原图粗扫补丁和复核
            补丁组成。未加载校准时返回原始融合分数，不生成业务三态判定。
        """
        with self._inference_lock:
            if self._closed:
                raise RuntimeError("HRInferencer is closed")
            self._validate_samples(test_samples)
            quality_results = self._assess_quality(test_samples)

            batch_size = self.batch_size or len(test_samples)
            worker_results = self._run_inference_groups(
                test_samples,
                self.coarse_tasks_in_devices,
                batch_size,
            )

            merged: DeviceInferenceResults = {
                sample.image.image_path: {
                    "image_size": None,
                    "patches": [],
                    "thumbnail": None,
                    "thumbnail_score": None,
                }
                for sample in test_samples
            }
            for worker_result in worker_results:
                for path, result in worker_result.items():
                    if result["image_size"] is not None:
                        current_size = merged[path]["image_size"]
                        if current_size is not None and current_size != result["image_size"]:
                            raise ValueError(f"Task image sizes disagree for {path}")
                        merged[path]["image_size"] = result["image_size"]
                    merged[path]["patches"].extend(result["patches"])
                    if result["thumbnail"] is not None:
                        if merged[path]["thumbnail"] is not None:
                            raise ValueError(f"Duplicate thumbnail prediction for {path}")
                        merged[path]["thumbnail"] = result["thumbnail"]
                        merged[path]["thumbnail_score"] = result["thumbnail_score"]

            anomaly_maps: list[FloatMap] = []
            global_scores: list[float] = []
            refinement_task = self.refinement_task
            regions_by_path: RegionsByPath = {}
            for sample in test_samples:
                path = sample.image.image_path
                result = merged[path]
                if result["image_size"] is None or not result["patches"]:
                    raise ValueError(f"Incomplete dynamic patch prediction for {path}")
                patch_map = _gather_patch_predictions(result["patches"], result["image_size"])
                final_map = patch_map
                if (
                    result["thumbnail"] is None
                    or result["thumbnail_score"] is None
                ):
                    raise ValueError(f"Incomplete global context prediction for {path}")
                image_width, image_height = result["image_size"]
                global_context_map = cv2.resize(
                    np.asarray(result["thumbnail"], dtype=np.float32),
                    (image_width, image_height),
                    interpolation=cv2.INTER_LINEAR,
                )
                if self.map_gaussian_sigma > 0:
                    final_map = gaussian_filter(
                        final_map,
                        sigma=self.map_gaussian_sigma,
                    )
                final_map = np.asarray(final_map, dtype=np.float32)
                # 缩略图只参与候选路由；最终异常证据仍来自原图补丁与复核结果。
                routing_map = build_routing_map(
                    final_map,
                    global_context_map,
                    self.global_routing_weight,
                )
                regions_by_path[path] = select_refinement_regions(
                    routing_map,
                    threshold=float(
                        np.quantile(
                            routing_map,
                            refinement_task["refinement_quantile"],
                        )
                    ),
                    tile_size=refinement_task["patch_size"],
                    min_area=refinement_task["refinement_min_area"],
                    safety_fraction=refinement_task["refinement_safety_fraction"],
                )
                anomaly_maps.append(final_map)
                global_scores.append(float(result["thumbnail_score"]))

            anomaly_maps = self._apply_refinement(
                test_samples,
                anomaly_maps,
                regions_by_path,
                batch_size,
            )
            image_scores = np.asarray(
                [
                    max(top_k_map_score(anomaly_map, self.score_top_k), global_score)
                    for anomaly_map, global_score in zip(anomaly_maps, global_scores)
                ],
                dtype=np.float32,
            )
            output: InferenceResult = {
                "image_paths": [sample.image.image_path for sample in test_samples],
                "image_scores": image_scores,
                "anomaly_maps": anomaly_maps,
                "display_images": self._build_display_images(test_samples, display_size),
                "quality_results": quality_results,
            }
            if self.score_calibration is not None:
                thresholds = thresholds_for_samples(self.score_calibration, test_samples)
                pixel_thresholds = pixel_thresholds_for_samples(
                    self.score_calibration, test_samples
                )
                component_summaries = [
                    component_statistics(anomaly_map, pixel_threshold)
                    for anomaly_map, pixel_threshold in zip(anomaly_maps, pixel_thresholds)
                ]
                component_scores = [
                    image_score_from_statistics(summary, image_score)
                    for summary, image_score in zip(component_summaries, image_scores)
                ]
                decision_thresholds = (
                    component_thresholds_for_samples(
                        self.score_calibration,
                        test_samples,
                    )
                    if "global_component_threshold" in self.score_calibration
                    else thresholds
                )
                decisions = [
                    classify_score(
                        score,
                        threshold,
                        threshold * self.decision_recheck_margin_ratio,
                    )
                    for score, threshold in zip(component_scores, decision_thresholds)
                ]
                output["decision_thresholds"] = decision_thresholds
                output["component_scores"] = component_scores
                output["raw_image_scores"] = image_scores
                output["image_scores"] = np.asarray(component_scores, dtype=np.float32)
                output["image_thresholds"] = decision_thresholds
                output["decisions"] = decisions
                output["decision_reasons"] = [
                    "score_at_or_below_threshold"
                    if decision == "OK"
                    else "score_within_recheck_margin"
                    if decision == "RECHECK"
                    else "score_above_recheck_margin"
                    for decision in decisions
                ]
                output["component_summaries"] = component_summaries
                for index, quality in enumerate(quality_results):
                    decisions[index], quality_reason = apply_quality_gate(
                        decisions[index],
                        quality["reasons"],
                    )
                    if quality_reason is not None:
                        output["decision_reasons"][index] = quality_reason
                output["is_defect"] = [decision == "NG" for decision in decisions]
                output["pixel_thresholds"] = pixel_thresholds
                output["binary_anomaly_maps"] = threshold_anomaly_maps(
                    anomaly_maps, pixel_thresholds
                )
            return output

    def score_samples(self, test_samples: list[HRSample]) -> ScoreVector:
        """通过完整粗到细链路计算图像分数，不提供绕过复核的捷径。

        Args:
            test_samples (list[HRSample]): 满足 :meth:`inference` 输入约束的样本。

        Returns:
            ScoreVector: 与样本同序的一维 ``float32`` 最终图像分数；加载组件校准
            时为组件与原始全局分数的保守组合，否则为复核图 Top-K 与缩略分数的
            较大值。
        """
        return self.inference(test_samples)["image_scores"]

    def _run_inference_groups(
        self,
        test_samples: list[HRSample],
        task_groups: list[list[TaskDefinition]],
        batch_size: int,
        *,
        regions_by_path: RegionsByPath | None = None,
    ) -> list[DeviceInferenceResults]:
        """把任务组并发提交到各设备线程并等待全部结果。

        Args:
            test_samples (list[HRSample]): 当前推理样本。
            task_groups (list[list[TaskDefinition]]): 与模型管理器同序的设备任务组。
            batch_size (int): 单任务 DataLoader 批量大小。
            regions_by_path (RegionsByPath | None): 可选复核区域映射。

        Returns:
            list[DeviceInferenceResults]: 保持任务组提交顺序的各设备结果。

        Raises:
            RuntimeError: 线程中的模型加载、数据读取或推理失败。
        """
        pending: list[Future[DeviceInferenceResults]] = [
            self._executor.submit(
                inference_in_device,
                test_samples,
                task_group,
                manager,
                batch_size,
                regions_by_path=regions_by_path,
            )
            for task_group, manager in zip(task_groups, self.model_managers)
            if task_group
        ]
        return [future.result() for future in pending]

    def _apply_refinement(
        self,
        test_samples: list[HRSample],
        base_maps: list[FloatMap],
        regions_by_path: RegionsByPath,
        batch_size: int,
    ) -> list[FloatMap]:
        """执行高分辨率复核任务并把结果融合回粗扫异常图。

        Args:
            test_samples (list[HRSample]): 与粗扫图同序的源图样本。
            base_maps (list[FloatMap]): 每张原图的二维粗扫异常图。
            regions_by_path (RegionsByPath): 每张图需要复核的原图 ``xywh`` 区域。
            batch_size (int): 复核任务 DataLoader 批量大小。

        Returns:
            list[FloatMap]: 与样本和粗扫图同序的原图分辨率融合异常图。
        """
        worker_results = self._run_inference_groups(
            test_samples,
            self.refinement_tasks_in_devices,
            batch_size,
            regions_by_path=regions_by_path,
        )
        refinements_by_path: dict[str, list[PatchPrediction]] = {
            sample.image.image_path: []
            for sample in test_samples
        }
        for worker_result in worker_results:
            for path, result in worker_result.items():
                refinements_by_path[path].extend(result["patches"])

        refined_maps: list[FloatMap] = []
        for sample, base_map in zip(test_samples, base_maps):
            path = sample.image.image_path
            refinements: list[tuple[HRImageIndex, FloatMap]] = []
            for record, anomaly_map in refinements_by_path[path]:
                x, y, width, height = record["source_xywh"]
                refinements.append((
                    HRImageIndex(
                        x=x,
                        y=y,
                        width=width,
                        height=height,
                    ),
                    anomaly_map,
                ))
            refined_maps.append(
                merge_refinement_maps(
                    base_map,
                    refinements,
                    image_size=(base_map.shape[1], base_map.shape[0]),
                )
            )
        return refined_maps

    @staticmethod
    def _validate_samples(test_samples: object) -> None:
        """校验推理样本非空、类型正确、路径唯一且类别非空。

        Args:
            test_samples (object): 待验证的调用方输入。

        Raises:
            TypeError: 任一元素不是 :class:`HRSample`。
            ValueError: 输入不是非空列表、解析后的路径重复或类别为空。
        """
        if not isinstance(test_samples, list) or not test_samples:
            raise ValueError("test_samples must be a non-empty list")
        if any(not isinstance(sample, HRSample) for sample in test_samples):
            raise TypeError("Every test sample must be an HRSample")
        paths = [os.path.abspath(sample.image.image_path) for sample in test_samples]
        if len(paths) != len(set(paths)):
            raise ValueError("Inference sample image paths must be unique")
        if any(
            not isinstance(sample.clsname, str) or not sample.clsname.strip()
            for sample in test_samples
        ):
            raise ValueError("Every inference sample must have a non-empty clsname")

    def _assess_quality(
        self,
        test_samples: list[HRSample],
    ) -> list[ImageQualityResult]:
        """逐图执行采集质量检查；质量问题不直接判为缺陷。

        Args:
            test_samples (list[HRSample]): 已通过基础输入校验的样本。

        Returns:
            list[ImageQualityResult]: 与样本同序的曝光、截断和清晰度结果。

        Raises:
            OSError: 原图或前景掩码无法读取。
            RuntimeError: 图像打开后没有获得像素数组。
            ValueError: RGB 图像、前景掩码或质量阈值不符合约定。

        Notes:
            质量原因仅能在最终阶段把 ``OK`` 提升为 ``RECHECK``，不会直接产生
            ``NG``，也不会覆盖模型已经给出的 ``NG``。
        """
        results: list[ImageQualityResult] = []
        for sample in test_samples:
            sample.open()
            try:
                image = sample.image.image
                if image is None:
                    raise RuntimeError("Sample image was not loaded")
                results.append(
                    assess_image_quality(
                        image,
                        self.quality_thresholds,
                        (
                            sample.foreground.image
                            if sample.foreground is not None
                            else None
                        ),
                    )
                )
            finally:
                sample.close()
        return results

    @staticmethod
    def _build_display_images(
        test_samples: list[HRSample],
        display_size: int | list[int] | tuple[int, int] | None,
    ) -> dict[str, ImageArray] | None:
        """按需生成评估可视化使用的固定尺寸 RGB 图像。

        Args:
            test_samples (list[HRSample]): 需要生成显示图的源图样本。
            display_size (int | list[int] | tuple[int, int] | None): 目标
                ``(width, height)``；整数表示正方形，``None`` 表示不生成。

        Returns:
            dict[str, ImageArray] | None: 原图路径到 HWC ``uint8`` RGB 显示图的
            映射；未请求显示图时为 ``None``。

        Raises:
            TypeError: 显示尺寸格式错误。
            ValueError: 显示尺寸不是两个正整数。
            OSError: 样本图像无法读取。
            RuntimeError: 图像打开后没有获得像素数组。
        """
        if display_size is None:
            return None
        if isinstance(display_size, int) and not isinstance(display_size, bool):
            normalized_size = (display_size, display_size)
        elif isinstance(display_size, (tuple, list)):
            normalized_size = tuple(display_size)
        else:
            raise TypeError("display_size must be an integer or width-height pair")
        if len(normalized_size) != 2 or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in normalized_size
        ):
            raise ValueError("display_size must contain two positive integers")
        output_size = (int(normalized_size[0]), int(normalized_size[1]))
        display_images: dict[str, ImageArray] = {}
        for sample in test_samples:
            sample.open()
            try:
                image = sample.image.image
                if image is None:
                    raise RuntimeError("Sample image was not loaded")
                display_images[sample.image.image_path] = cv2.resize(
                    image,
                    output_size,
                    interpolation=cv2.INTER_LINEAR,
                )
            finally:
                sample.close()
        return display_images

    def close(self) -> None:
        """幂等关闭推理线程池并释放所有任务模型引用。

        正在执行的调用会由推理锁保护；关闭后再次调用 :meth:`inference` 或进入
        上下文管理器会抛出 ``RuntimeError``。
        """
        with self._inference_lock:
            if self._closed:
                return
            self._executor.shutdown(wait=True, cancel_futures=True)
            for manager in self.model_managers:
                manager.close()
            self.model_managers.clear()
            self._closed = True

    def __enter__(self) -> HRInferencer:
        if self._closed:
            raise RuntimeError("HRInferencer is closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
