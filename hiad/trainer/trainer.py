from __future__ import annotations

import copy
import logging
import os
from collections.abc import Iterable
from multiprocessing.pool import ApplyResult
from typing import cast

import numpy as np
import torch.multiprocessing as mp
from easydict import EasyDict

from hiad.constants import TASK_TYPE_DYNAMIC_PATCH, TASK_TYPE_REFINEMENT_PATCH
from hiad.data import HRSample
from hiad.detectors.base import BaseDetector
from hiad.detectors.config import DetectorConfig, validate_required_config
from hiad.runtime.contracts import ScoreCalibration
from hiad.runtime.devices import validate_gpu_ids
from hiad.runtime.logging import create_logger
from hiad.runtime.partition import round_robin_partition
from hiad.runtime.quality import assess_image_quality
from hiad.runtime.score_calibration import (
    build_component_calibration,
    build_score_calibration,
    save_score_calibration,
    summarize_anomaly_map,
)
from hiad.task import save_tasks, validate_tasks
from hiad.task.contracts import (
    DynamicPatchTask,
    RefinementPatchTask,
    TaskDefinition,
    ThumbnailTask,
)
from hiad.trainer.sources import validate_unified_training_samples
from hiad.trainer.worker import train_tasks_in_device


class HRTrainer:
    """为每个任务训练独立检测器，并用全部正常样本完成两阶段校准。

    Attributes:
        detector_class (type[BaseDetector]): 每个任务实例化的检测器类型。
        config (DetectorConfig): 已完成必需字段校验的检测器配置。
        batch_size (int): 任务训练批量大小。
        checkpoint_root (str): 任务权重、任务 JSON 和校准 JSON 输出目录。
        log_root (str): 主进程与各设备训练日志目录。
        tasks (list[TaskDefinition]): 已验证的粗扫、复核和缩略图任务。
        seed (int): 工作进程与采样器随机种子。
    """

    def __init__(
        self,
        detector_class: type[BaseDetector],
        config: object,
        batch_size: int,
        checkpoint_root: str | os.PathLike[str],
        log_root: str | os.PathLike[str],
        tasks: object,
        seed: int = 0,
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if tasks is None:
            raise ValueError("tasks must not be None")

        normalized_config = EasyDict(config) if isinstance(config, dict) else config
        validate_required_config(normalized_config)
        self.detector_class: type[BaseDetector] = detector_class
        self.config: DetectorConfig = cast(DetectorConfig, normalized_config)
        self.batch_size: int = batch_size
        self.checkpoint_root: str = os.fspath(checkpoint_root)
        self.log_root: str = os.fspath(log_root)
        self.tasks: list[TaskDefinition] = validate_tasks(tasks)
        self.seed: int = seed
        os.makedirs(self.checkpoint_root, exist_ok=True)
        os.makedirs(self.log_root, exist_ok=True)
        mp.set_start_method("spawn", force=True)

    def train(
        self,
        train_samples: Iterable[HRSample],
        gpu_ids: object,
        main_logger: logging.Logger | None = None,
    ) -> None:
        """完成质量门禁、任务训练以及图像/像素/组件两阶段校准。

        Args:
            train_samples (Iterable[HRSample]): 路径唯一、带类别且无缺陷的正常
                训练样本；全部样本都用于训练和正常分布校准。
            gpu_ids (object): 非空、不重复的 CUDA 设备编号列表。
            main_logger (logging.Logger | None): 可选主日志器；缺省时写入
                ``log_root/main.log``。

        Raises:
            TypeError: 训练样本、任务或设备参数类型不合法。
            ValueError: 样本不满足正常训练/采集质量约束，或配置、任务、校准数据
                不合法。
            RuntimeError: CUDA 不可用、图像未解码，或任一工作进程训练失败。
            OSError: 图像、日志或检查点文件无法读取或写入。

        Notes:
            第一阶段用全部正常图像的最终异常图拟合图像和像素阈值；第二阶段
            重新运行完整粗到细链路，拟合最终连通组件阈值。
        """
        sources = validate_unified_training_samples(train_samples)
        gpu_ids = validate_gpu_ids(gpu_ids)

        quality_thresholds = {
            key: float(self.config[key])
            for key in (
                "min_mean_luminance",
                "max_mean_luminance",
                "max_clipped_fraction",
                "min_focus_variance",
            )
        }
        for sample in sources.samples:
            # 正常训练图像不满足采集质量时必须提前失败，避免污染记忆与阈值。
            sample.open()
            try:
                image = sample.image.image
                if image is None:
                    raise RuntimeError("Training image was not loaded")
                quality = assess_image_quality(
                    image,
                    quality_thresholds,
                    sample.foreground.image if sample.foreground is not None else None,
                )
            finally:
                sample.close()
            if quality["status"] != "PASS":
                raise ValueError(
                    f"Normal training image failed quality gate: "
                    f"{sample.image.image_path}: {','.join(quality['reasons'])}"
                )

        if main_logger is None:
            main_logger = create_logger(
                "main",
                os.path.join(self.log_root, "main.log"),
                print_console=True,
            )

        tasks_path = os.path.join(self.checkpoint_root, "tasks.json")
        save_tasks(self.tasks, tasks_path)
        main_logger.info("Start training, devices: %s", gpu_ids)
        main_logger.info("Tasks config is saved as: %s", tasks_path)
        main_logger.info(
            "Normal training samples: %d (100%% used; no validation holdout)",
            len(sources.samples),
        )
        for index, task in enumerate(self.tasks, start=1):
            if task["type"] in {TASK_TYPE_DYNAMIC_PATCH, TASK_TYPE_REFINEMENT_PATCH}:
                patch_task = cast(DynamicPatchTask | RefinementPatchTask, task)
                main_logger.info(
                    "[%d/%d] Task %s, patch_size=%s, stride=%s, ds_factors=%s",
                    index,
                    len(self.tasks),
                    patch_task["name"],
                    patch_task["patch_size"],
                    patch_task["stride"],
                    patch_task["ds_factors"],
                )
            else:
                thumbnail_task = cast(ThumbnailTask, task)
                main_logger.info(
                    "[%d/%d] Task %s, thumbnail_size=%s",
                    index,
                    len(self.tasks),
                    thumbnail_task["name"],
                    thumbnail_task["thumbnail_size"],
                )
        main_logger.info("The training progress can be monitored in: %s", self.log_root)

        tasks_in_device = [
            task_group
            for task_group in round_robin_partition(self.tasks, len(gpu_ids))
            if task_group
        ]
        results: list[ApplyResult[None]] = []
        process_pool = mp.Pool(processes=len(tasks_in_device))
        try:
            for gpu_id, task_group in zip(gpu_ids, tasks_in_device):
                results.append(process_pool.apply_async(
                    train_tasks_in_device,
                    args=(
                        gpu_id,
                        self.detector_class,
                        self.config,
                        copy.deepcopy(list(sources.samples)),
                        task_group,
                        self.batch_size,
                        self.checkpoint_root,
                        self.log_root,
                        self.seed,
                    ),
                ))
            process_pool.close()
            process_pool.join()
        except Exception:
            process_pool.terminate()
            process_pool.join()
            raise

        for result in results:
            message = result.get()
            if message:
                main_logger.info(message)

        from hiad.inferencer import HRInferencer

        # 第一阶段只用正常样本估计图像分数和像素证据分布。
        main_logger.info(
            "Calibrating image and pixel thresholds from all normal training images"
        )
        calibration_scores: list[float] = []
        calibration_pixel_statistics: list[float] = []
        calibration_batch_size = int(self.config.calibration_batch_size)
        pixel_percentile = float(self.config.normal_pixel_percentile)
        with HRInferencer(
            detector_class=self.detector_class,
            config=self.config,
            checkpoint_root=self.checkpoint_root,
            gpu_ids=gpu_ids,
            batch_size=calibration_batch_size,
            require_score_calibration=False,
        ) as inferencer:
            for start in range(0, len(sources.samples), calibration_batch_size):
                batch_samples = list(
                    sources.samples[start:start + calibration_batch_size]
                )
                calibration_result = inferencer.inference(batch_samples)
                calibration_scores.extend(calibration_result["image_scores"].tolist())
                calibration_pixel_statistics.extend(
                    summarize_anomaly_map(anomaly_map, pixel_percentile)
                    for anomaly_map in calibration_result["anomaly_maps"]
                )

        percentile = float(self.config.normal_score_percentile)
        pixel_image_percentile = float(self.config.normal_pixel_image_percentile)
        calibration: ScoreCalibration = build_score_calibration(
            sources.samples,
            np.asarray(calibration_scores, dtype=np.float64),
            calibration_pixel_statistics,
            percentile=percentile,
            pixel_percentile=pixel_percentile,
            pixel_image_percentile=pixel_image_percentile,
        )
        calibration_path = save_score_calibration(calibration, self.checkpoint_root)

        # 第二阶段使用完整粗到细结果校准最终连通组件判定阈值。
        component_scores: list[float] = []
        with HRInferencer(
            detector_class=self.detector_class,
            config=self.config,
            checkpoint_root=self.checkpoint_root,
            gpu_ids=gpu_ids,
            batch_size=calibration_batch_size,
            require_score_calibration=True,
        ) as inferencer:
            for start in range(0, len(sources.samples), calibration_batch_size):
                batch_samples = list(
                    sources.samples[start:start + calibration_batch_size]
                )
                result = inferencer.inference(batch_samples)
                component_scores.extend(result["component_scores"])

        component_percentile = float(
            self.config.normal_component_percentile
        )
        calibration = build_component_calibration(
            calibration,
            sources.samples,
            component_scores,
            percentile=component_percentile,
        )
        calibration_path = save_score_calibration(calibration, self.checkpoint_root)
        main_logger.info(
            "Score calibration saved as %s: image_percentile=%.4f, "
            "global_image_threshold=%.6f, pixel_percentile=%.6f, "
            "pixel_image_percentile=%.4f, global_pixel_threshold=%.6f",
            calibration_path,
            calibration["percentile"],
            calibration["global_threshold"],
            calibration["pixel_percentile"],
            calibration["pixel_image_percentile"],
            calibration["global_pixel_threshold"],
        )
        for category, payload in calibration["categories"].items():
            main_logger.info(
                "Category %s thresholds: image=%.6f, pixel=%.6f "
                "(normal_images=%d)",
                category,
                payload["threshold"],
                payload["pixel_threshold"],
                payload["normal_image_count"],
            )

        main_logger.info("End training")
        main_logger.info("Checkpoints are saved as: %s", self.checkpoint_root)
