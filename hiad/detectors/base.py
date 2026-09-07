from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import torch
from torch.utils.data import DataLoader


class BaseDetector(ABC):
    """Dinomaly 检测器在训练、推理和持久化阶段必须遵守的接口。

    Attributes:
        patch_size (list[int]): 模型输入 ``[width, height]``。
        seed (int): 检测器训练和采样随机种子。
        logger (logging.Logger | None): 可选训练/推理日志器。
        device (torch.device): 模型和输入所在设备。
    """

    def __init__(
        self,
        patch_size: int | Sequence[int],
        device: torch.device,
        logger: logging.Logger | None = None,
        seed: int = 0,
        **_: object,
    ) -> None:
        if isinstance(patch_size, int):
            self.patch_size: list[int] = [patch_size, patch_size]
        else:
            self.patch_size = [int(value) for value in patch_size]
        self.seed: int = seed
        self.logger: logging.Logger | None = logger
        self.device: torch.device = device

    @abstractmethod
    def to_device(self, device: torch.device) -> None:
        """把检测器全部模块和状态移动到目标设备。"""
        raise NotImplementedError

    @abstractmethod
    def train_step(self, train_dataloader: DataLoader[Any], task_name: str) -> None:
        """训练当前任务的重建模块。"""
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, checkpoint_path: str | os.PathLike[str]) -> None:
        """保存当前任务推理所需状态。"""
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(self, checkpoint_path: str | os.PathLike[str]) -> None:
        """恢复当前任务推理所需状态。"""
        raise NotImplementedError
