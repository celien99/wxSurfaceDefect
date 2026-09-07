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


class BaseDetector(ABC):
    """Dinomaly 检测器在训练、推理和持久化阶段必须遵守的接口。

    Attributes:
        patch_size (list[int]): 模型输入 ``[width, height]``。
        seed (int): 检测器训练和采样随机种子。
        logger (logging.Logger | None): 可选训练/推理日志器。
        device (torch.device): 模型和输入所在设备。
        fusion_weights (Sequence[float] | None): 多尺度编码特征融合权重；``None``
            表示各尺度等权。
    """

    def __init__(
        self,
        patch_size: int | Sequence[int],
        device: torch.device,
        logger: logging.Logger | None = None,
        seed: int = 0,
        fusion_weights: Sequence[float] | None = None,
        **_: object,
    ) -> None:
        if isinstance(patch_size, int):
            self.patch_size: list[int] = [patch_size, patch_size]
        else:
            self.patch_size = [int(value) for value in patch_size]
        self.seed: int = seed
        self.logger: logging.Logger | None = logger
        self.device: torch.device = device
        self.fusion_weights: Sequence[float] | None = fusion_weights

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

    @torch.no_grad()
    def get_multi_resolution_fusion_embeddings(
        self,
        data: Mapping[str, Any],
    ) -> list[torch.Tensor]:
        """把主补丁与多尺度上下文编码特征按 ``fusion_weights`` 加权融合。

        一次前向编码主补丁与全部低分辨率上下文，随后把上下文特征裁剪并插值
        对齐到主补丁特征空间，按权重求和得到融合后的多层特征。

        Args:
            data (Mapping[str, Any]): DataLoader 批次。``image`` 为 BCHW 主补丁；
                可选 ``low_resolution_image_<n>`` 为同尺寸上下文，配套
                ``low_resolution_index_<n>`` 保存主补丁在该上下文中的 JSON
                ``xywh`` 坐标。

        Returns:
            list[torch.Tensor]: 与 ``embedding`` 形状一致的多层融合特征。

        Raises:
            ValueError: ``fusion_weights`` 数量与尺度数不匹配。
        """
        image = data["image"].to(self.device, non_blocking=True)
        low_resolution_image_keys = [
            key for key in data if key.startswith("low_resolution_image")
        ]
        if len(low_resolution_image_keys) == 0:
            return self.embedding(image)

        if self.fusion_weights is not None:
            if len(self.fusion_weights) != len(low_resolution_image_keys) + 1:
                raise ValueError(
                    "fusion_weights must have one value for the main image and "
                    "each low-resolution image"
                )
            fusion_weights = [
                weight / sum(self.fusion_weights) for weight in self.fusion_weights
            ]
        else:
            fusion_weights = [
                1 / (len(low_resolution_image_keys) + 1)
            ] * (len(low_resolution_image_keys) + 1)

        low_resolution_image_keys.sort(key=lambda item: int(item.split("_")[-1]))

        all_images = [image]
        for key in low_resolution_image_keys:
            all_images.append(data[key].to(self.device, non_blocking=True))
        all_images = torch.cat(all_images, dim=0)
        all_embeddings = self.embedding(all_images)

        batch_size = image.shape[0]
        main_embeddings = [feature[:batch_size] for feature in all_embeddings]
        embeddings = [[embedding * fusion_weights[0]] for embedding in main_embeddings]

        for rs_index, low_resolution_image_key in enumerate(low_resolution_image_keys):
            low_resolution_index = data[
                low_resolution_image_key.replace("image", "index")
            ]
            start = (rs_index + 1) * batch_size
            low_resolution_embeddings = [
                feature[start : start + batch_size] for feature in all_embeddings
            ]

            for i, low_resolution_embedding in enumerate(low_resolution_embeddings):
                downsampling_embedding: list[torch.Tensor] = []
                for feature, index in zip(
                    low_resolution_embedding, low_resolution_index
                ):
                    feature_stride_h = self.patch_size[1] / feature.shape[1]
                    feature_stride_w = self.patch_size[0] / feature.shape[2]
                    index = HRImageIndex.from_str(index)
                    x_start = index.x / feature_stride_w
                    y_start = index.y / feature_stride_h
                    x_end = x_start + index.width / feature_stride_w
                    y_end = y_start + index.height / feature_stride_h
                    downsampling_embedding.append(
                        feature[
                            :, int(y_start) : int(y_end), int(x_start) : int(x_end)
                        ]
                    )
                try:
                    downsampled = torch.stack(downsampling_embedding)
                except RuntimeError:
                    first_embedding = downsampling_embedding[0]
                    downsampled = torch.stack(
                        [
                            first_embedding,
                            *[
                                F.interpolate(
                                    feature.unsqueeze(0),
                                    size=first_embedding.shape[-2:],
                                    mode="bilinear",
                                ).squeeze(0)
                                for feature in downsampling_embedding[1:]
                            ],
                        ]
                    )

                embeddings[i].append(
                    fusion_weights[rs_index + 1]
                    * F.interpolate(
                        downsampled,
                        size=embeddings[i][-1].shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                )

        return [
            torch.sum(torch.stack(embedding), dim=0) for embedding in embeddings
        ]
