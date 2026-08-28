from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from types import TracebackType
from typing import cast

import cv2
import numpy as np
from easydict import EasyDict

from hiad.constants import TASK_TYPE_REFINEMENT_PATCH
from hiad.data import HRSample
from hiad.detectors.base import BaseDetector
from hiad.detectors.config import DetectorConfig, validate_required_config
from hiad.inferencer.modelmanager import ModelManager
from hiad.inferencer.pipeline import DeviceImagePipeline, ImagePipelineOutput
from hiad.runtime.contracts import (
    FloatMap,
    ImageArray,
    ImageQualityResult,
    InferenceResult,
    InferenceTiming,
    RefinementStatistics,
    ScoreCalibration,
    ScoreVector,
)
from hiad.runtime.decision import (
    classify_score,
    component_statistics,
    image_score_from_statistics,
    top_k_map_score,
)
from hiad.runtime.devices import validate_gpu_ids
from hiad.runtime.inference_config import InferenceConfig, load_inference_config
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
        coarse_task_definitions (list[TaskDefinition]): 粗扫与缩略图任务定义。
        refinement_task (RefinementPatchTask): 唯一复核任务及候选选择参数。
        model_managers (list[ModelManager]): 各设备已加载的任务模型。
        score_calibration (ScoreCalibration | None): 可选图像、像素和组件阈值。
        quality_thresholds (dict[str, float]): 曝光、截断比例和清晰度质量阈值。
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

        refinement_tasks = [
            cast(RefinementPatchTask, task)
            for task in tasks
            if task["type"] == TASK_TYPE_REFINEMENT_PATCH
        ]
        self.refinement_task: RefinementPatchTask = refinement_tasks[0]
        self.coarse_task_definitions: list[TaskDefinition] = [
            task for task in tasks if task["type"] != TASK_TYPE_REFINEMENT_PATCH
        ]
        self.inference_config: InferenceConfig = load_inference_config(normalized_config)
        self.model_managers: list[ModelManager] = [
            ModelManager(
                tasks,
                detector_class,
                detector_config,
                checkpoint_root,
                gpu_ids[index],
            )
            for index in range(len(gpu_ids))
        ]
        self.score_calibration: ScoreCalibration | None = (
            load_score_calibration(checkpoint_root)
            if require_score_calibration
            else None
        )
        self.map_gaussian_sigma: float = float(detector_config.map_gaussian_sigma)
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
        self.refinement_bridge_gap_tiles: int = int(
            detector_config.refinement_bridge_gap_tiles
        )
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=len(gpu_ids)
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
            ``OK/NG`` 判定、``0/1`` 二值掩码；请求显示图时包含 HWC RGB
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
            total_started = time.perf_counter()
            quality_started = time.perf_counter()
            quality_results = self._assess_quality(test_samples)
            quality_seconds = time.perf_counter() - quality_started

            coarse_seconds, routing_seconds, refinement_seconds, (
                anomaly_maps, global_scores, refinement_statistics,
            ) = self._run_inference(test_samples)

            postprocess_started = time.perf_counter()
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
                "refinement_statistics": refinement_statistics,
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
                    classify_score(score, threshold)
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
                    else "score_above_threshold"
                    for decision in decisions
                ]
                output["component_summaries"] = component_summaries
                output["is_defect"] = [decision == "NG" for decision in decisions]
                output["pixel_thresholds"] = pixel_thresholds
                output["binary_anomaly_maps"] = threshold_anomaly_maps(
                    anomaly_maps,
                    pixel_thresholds,
                )
            postprocess_seconds = time.perf_counter() - postprocess_started
            output["inference_timing"] = InferenceTiming(
                quality_seconds=quality_seconds,
                coarse_seconds=coarse_seconds,
                routing_seconds=routing_seconds,
                refinement_seconds=refinement_seconds,
                postprocess_seconds=postprocess_seconds,
                total_seconds=time.perf_counter() - total_started,
            )
            return output

    def _run_inference(
        self,
        test_samples: list[HRSample],
    ) -> tuple[
        float,
        float,
        float,
        tuple[list[FloatMap], list[float], list[RefinementStatistics]],
    ]:
        """逐图 GPU 驻留路径：按图把样本均分到各设备，每设备跑完整链路。"""
        batch_size = self.batch_size or len(test_samples)
        pipelines = [
            DeviceImagePipeline(
                manager.detectors,
                self.coarse_task_definitions,
                self.refinement_task,
                inference_config=self.inference_config,
                global_routing_weight=self.global_routing_weight,
                score_top_k=self.score_top_k,
                refinement_bridge_gap_tiles=self.refinement_bridge_gap_tiles,
                map_gaussian_sigma=self.map_gaussian_sigma,
                batch_cap=batch_size,
                async_pipeline=self.inference_config.async_pipeline,
            )
            for manager in self.model_managers
        ]

        device_samples: list[list[HRSample]] = [[] for _ in pipelines]
        for index, sample in enumerate(test_samples):
            device_samples[index % len(pipelines)].append(sample)

        pending = [
            self._executor.submit(pipeline.process_images, samples)
            for pipeline, samples in zip(pipelines, device_samples)
            if samples
        ]
        worker_outputs = [future.result() for future in pending]
        flat = [output for outputs in worker_outputs for output in outputs]
        by_path = {output.image_path: output for output in flat}
        ordered: list[ImagePipelineOutput] = [
            by_path[sample.image.image_path] for sample in test_samples
        ]
        anomaly_maps = [output.final_map for output in ordered]
        global_scores = [output.thumbnail_score for output in ordered]
        refinement_statistics = [output.refinement_statistics for output in ordered]
        coarse_seconds = sum(output.coarse_seconds for output in ordered)
        routing_seconds = sum(output.routing_seconds for output in ordered)
        refinement_seconds = sum(output.refinement_seconds for output in ordered)
        return (
            coarse_seconds,
            routing_seconds,
            refinement_seconds,
            (anomaly_maps, global_scores, refinement_statistics),
        )

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
        """逐图执行采集质量检查；质量问题不改变模型 OK/NG 判定。

        Args:
            test_samples (list[HRSample]): 已通过基础输入校验的样本。

        Returns:
            list[ImageQualityResult]: 与样本同序的曝光、截断和清晰度结果。

        Raises:
            OSError: 原图或前景掩码无法读取。
            RuntimeError: 图像打开后没有获得像素数组。
            ValueError: RGB 图像、前景掩码或质量阈值不符合约定。

        Notes:
            质量状态和原因独立落盘，供采集链路处理；它们不会覆盖模型判定。
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
