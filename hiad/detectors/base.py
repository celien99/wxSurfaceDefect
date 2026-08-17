from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from hiad.data import HRImageIndex
from hiad.runtime.contracts import DetectorPrediction


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
        self.patch_size: list[int]
        if isinstance(patch_size, int):
            self.patch_size = [patch_size, patch_size]
        else:
            self.patch_size = [int(value) for value in patch_size]
        self.seed: int = seed
        self.logger: logging.Logger | None = logger
        self.device: torch.device = device

    @abstractmethod
    def embedding(self, input_tensor: torch.Tensor) -> list[torch.Tensor]:
        """将 BCHW 图像张量编码为多层特征张量。

        Args:
            input_tensor (torch.Tensor): ImageNet 标准化的 BCHW RGB 模型输入。

        Returns:
            list[torch.Tensor]: 顺序稳定的多层 BCHW 特征。
        """
        raise NotImplementedError

    @abstractmethod
    def to_device(self, device: torch.device) -> None:
        """把检测器全部模块和状态移动到目标设备。

        Args:
            device (torch.device): 目标 CPU 或 CUDA 设备。
        """
        raise NotImplementedError

    @abstractmethod
    def train_step(self, train_dataloader: DataLoader[Any], task_name: str) -> None:
        """训练当前任务的重建模块。

        Args:
            train_dataloader (DataLoader[Any]): 产生标准补丁字典的训练加载器。
            task_name (str): 用于日志和异常消息的任务名称。
        """
        raise NotImplementedError

    @abstractmethod
    def fit_normal_evidence(self, train_dataloader: DataLoader[Any]) -> None:
        """用正常训练样本拟合记忆特征与证据标准化参数。

        Args:
            train_dataloader (DataLoader[Any]): 只包含正常样本补丁的加载器。
        """
        raise NotImplementedError

    @abstractmethod
    def inference_step(
        self,
        test_dataloader: DataLoader[Any],
    ) -> list[DetectorPrediction]:
        """为每个模型输入生成异常图和图像分数。

        Args:
            test_dataloader (DataLoader[Any]): 不打乱顺序的标准补丁加载器。

        Returns:
            list[DetectorPrediction]: 与加载器条目同序的二维异常图和标量分数。
        """
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, checkpoint_path: str | os.PathLike[str]) -> None:
        """保存当前任务推理所需状态。

        Args:
            checkpoint_path (str | os.PathLike[str]): 目标检查点文件路径。
        """
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(self, checkpoint_path: str | os.PathLike[str]) -> None:
        """恢复当前任务推理所需状态。

        Args:
            checkpoint_path (str | os.PathLike[str]): 已存在的检查点文件路径。
        """
        raise NotImplementedError

    @torch.no_grad()
    def get_multi_resolution_embeddings(
        self,
        data: Mapping[str, Any],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor] | None]:
        """一次编码主补丁与上下文，再按原图区域对齐上下文特征。

        Args:
            data (Mapping[str, Any]): DataLoader 批次。``image`` 为 BCHW 主补丁；
                可选 ``low_resolution_image_<n>`` 为同尺寸上下文，配套
                ``low_resolution_index_<n>`` 保存主补丁在该上下文中的 JSON
                ``xywh`` 坐标。

        Returns:
            tuple[list[torch.Tensor], list[torch.Tensor] | None]: 主补丁多层 BCHW
            特征，以及裁剪到主区域、缩放到主特征空间并跨上下文取均值的对应特征；
            没有上下文输入时第二项为 ``None``。

        Raises:
            KeyError: 批次缺少 ``image`` 或某个上下文对应的索引字段。
            ValueError: 上下文索引不是合法的区域 JSON。
            RuntimeError: 编码、拼接或特征插值失败。
        """
        image = data["image"].to(self.device, non_blocking=True)
        low_resolution_image_keys = [
            key for key in data if key.startswith("low_resolution_image")
        ]

        if len(low_resolution_image_keys) == 0:
            return self.embedding(image), None

        low_resolution_image_keys.sort(key=lambda item: int(item.split("_")[-1]))

        # 合并后只执行一次冻结编码器前向，避免每个上下文重复付出编码成本。
        image_batches = [image]
        for key in low_resolution_image_keys:
            image_batches.append(data[key].to(self.device, non_blocking=True))
        batched_images = torch.cat(image_batches, dim=0)
        all_embeddings = self.embedding(batched_images)

        batch_size = image.shape[0]
        main_embeddings = [feature[:batch_size] for feature in all_embeddings]
        context_embeddings: list[list[torch.Tensor]] = [[] for _ in main_embeddings]

        for rs_index, low_resolution_image_key in enumerate(low_resolution_image_keys):
            low_resolution_index = data[
                low_resolution_image_key.replace("image", "index")
            ]
            start = (rs_index + 1) * batch_size
            low_resolution_embeddings = [
                feature[start : start + batch_size] for feature in all_embeddings
            ]

            for i, low_resolution_embedding in enumerate(low_resolution_embeddings):
                cropped_embeddings: list[torch.Tensor] = []
                for feature, index in zip(low_resolution_embedding, low_resolution_index):
                    feature_stride_h = self.patch_size[1] / feature.shape[1]
                    feature_stride_w = self.patch_size[0] / feature.shape[2]
                    index = HRImageIndex.from_str(index)
                    x_start = index.x / feature_stride_w
                    y_start = index.y / feature_stride_h
                    x_end = x_start + index.width / feature_stride_w
                    y_end = y_start + index.height / feature_stride_h
                    cropped_embeddings.append(
                        feature[:, int(y_start) : int(y_end), int(x_start) : int(x_end)]
                    )
                try:
                    downsampled = torch.stack(cropped_embeddings)
                except RuntimeError:
                    first_embedding = cropped_embeddings[0]
                    resized_embeddings = [
                        F.interpolate(
                            feature.unsqueeze(0),
                            size=first_embedding.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(0)
                        for feature in cropped_embeddings[1:]
                    ]
                    downsampled = torch.stack([first_embedding, *resized_embeddings])

                context_embeddings[i].append(
                    F.interpolate(
                        downsampled,
                        size=main_embeddings[i].shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                )

        aggregated_context = [
            torch.stack(layer_contexts, dim=0).mean(dim=0)
            for layer_contexts in context_embeddings
        ]
        return main_embeddings, aggregated_context
