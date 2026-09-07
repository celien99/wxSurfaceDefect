from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import cast

import torch

from hiad.detectors.base import BaseDetector
from hiad.detectors.config import DetectorConfig, detector_config_for_task
from hiad.task.contracts import TaskDefinition


class ModelManager:
    """管理单个 CUDA 设备上的任务模型及其检查点生命周期。

    Attributes:
        detectors (dict[str, BaseDetector]): 任务名称到已加载检测器的映射；每个
            检测器均驻留在构造时指定的 CUDA 设备上。
    """

    def __init__(
        self,
        tasks: Sequence[TaskDefinition],
        detector_class: type[BaseDetector],
        config: DetectorConfig,
        checkpoint_root: str | os.PathLike[str],
        gpu_id: int,
    ) -> None:
        gpu_device = torch.device(f"cuda:{gpu_id}")
        self.detectors: dict[str, BaseDetector] = {}

        for task in tasks:
            task_name = task["name"]
            detector_config = detector_config_for_task(config, task)

            detector = detector_class(
                **cast(Mapping[str, object], detector_config),
                device=gpu_device,
                logger=None,
                seed=0,
            )
            checkpoint_path = os.path.join(
                checkpoint_root,
                f"{task_name}_weight.pkl",
            )
            detector.load_checkpoint(checkpoint_path)
            self.detectors[task_name] = detector

    def get_detector(self, task_name: str) -> BaseDetector:
        """按任务名称获取已加载的检测器。

        Args:
            task_name (str): 任务定义中的稳定名称。

        Returns:
            BaseDetector: 对应任务的检测器实例。

        Raises:
            KeyError: 当前设备没有加载该任务。
        """
        return self.detectors[task_name]

    def close(self) -> None:
        """释放管理器持有的检测器引用，允许框架回收模型和显存。"""
        self.detectors.clear()
