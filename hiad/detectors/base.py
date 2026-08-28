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
        cell_cache: Mapping[str, list[torch.Tensor]] | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor] | None]:
        """一次编码主补丁与上下文，再按原图区域对齐上下文特征。

        ``cell_cache`` 非空时启用 context 复用（spec 2026-08-27 旗舰 B）：
        主补丁单独编码，context 特征从缓存 cell 特征按 ``cell_id``/``cell_index``
        切片；此时 ``data`` 需携带与 ``image`` 首维对应的 ``cell_id``
        （list[str]）与 ``cell_index``（主补丁在该 cell 中缩放到模型输入坐标系
        的 JSON ``xywh``）。

        Args:
            data (Mapping[str, Any]): DataLoader 批次。``image`` 为 BCHW 主补丁；
                可选 ``low_resolution_image_<n>`` 为同尺寸上下文，配套
                ``low_resolution_index_<n>`` 保存主补丁在该上下文中的 JSON
                ``xywh`` 坐标。
            cell_cache (Mapping[str, list[torch.Tensor]] | None): context 复用
                的 cell 特征缓存；``None`` 时走逐 tile 独立编码现状路径。

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
        if cell_cache is not None:
            return self._context_reuse_embeddings(image, data, cell_cache)
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

    @torch.no_grad()
    def encode_grid_cells(
        self,
        cell_batch: Mapping[str, Any],
        cell_ids: Sequence[str],
    ) -> dict[str, list[torch.Tensor]]:
        """一次编码去重网格 cell，返回 ``cell_id -> 多层 BCHW 特征`` 的缓存。

        context 复用的第一步：整图只编码一次每个 cell，主补丁后续从缓存切片
        其 context 特征，避免相邻 tile 重复编码重叠 context。

        Args:
            cell_batch (Mapping[str, Any]): ``image`` 为 ``(N, 3, P, P)`` 的
                ImageNet 标准化去重 cell 图堆。
            cell_ids (Sequence[str]): 与 batch 行一一对应的 cell 标识。

        Returns:
            dict[str, list[torch.Tensor]]: 每个 cell 的多层 BCHW 特征
            （每层 ``(C, H, W)``，设备驻留）。
        """
        image = cell_batch["image"].to(self.device, non_blocking=True)
        embeddings = self.embedding(image)
        cache: dict[str, list[torch.Tensor]] = {}
        for index, cell_id in enumerate(cell_ids):
            cache[cell_id] = [
                layer[index].detach() for layer in embeddings
            ]
        return cache

    def _context_reuse_embeddings(
        self,
        image: torch.Tensor,
        data: Mapping[str, Any],
        cell_cache: Mapping[str, list[torch.Tensor]],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """context 复用：主补丁单独编码，context 从缓存 cell 特征切片。

        每个主补丁取"包含其中心"的 cell（``cell_id``），按 ``cell_index`` 在
        该 cell 特征空间切片并插值对齐到主补丁特征尺寸；切片坐标语义与现状
        ``low_resolution_index`` 一致（模型输入坐标除以特征步长）。

        Args:
            image (torch.Tensor): BCHW 主补丁图像（已在设备上）。
            data (Mapping[str, Any]): 含 ``cell_id``（list[str]）与
                ``cell_index``（list[str] JSON ``xywh``），均与 ``image``
                首维对应。
            cell_cache (Mapping[str, list[torch.Tensor]]): ``encode_grid_cells``
                产生的 cell 特征缓存。

        Returns:
            tuple[list[torch.Tensor], list[torch.Tensor]]: 主补丁多层 BCHW 特征
            与对齐后的 context 特征。
        """
        main_embeddings = self.embedding(image)
        cell_ids = data["cell_id"]
        cell_indexes = [HRImageIndex.from_str(value) for value in data["cell_index"]]
        context_embeddings: list[list[torch.Tensor]] = [[] for _ in main_embeddings]
        for cell_id, index in zip(cell_ids, cell_indexes):
            if cell_id not in cell_cache:
                raise KeyError(f"No cached cell features for cell_id={cell_id}")
            cell_features = cell_cache[cell_id]
            for layer_index, feature in enumerate(cell_features):
                feature_stride_h = self.patch_size[1] / feature.shape[1]
                feature_stride_w = self.patch_size[0] / feature.shape[2]
                x_start = index.x / feature_stride_w
                y_start = index.y / feature_stride_h
                x_end = x_start + index.width / feature_stride_w
                y_end = y_start + index.height / feature_stride_h
                cropped = feature[
                    :, int(y_start) : int(y_end), int(x_start) : int(x_end)
                ]
                context_embeddings[layer_index].append(
                    F.interpolate(
                        cropped.unsqueeze(0),
                        size=main_embeddings[layer_index].shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                )
        aggregated_context = [
            torch.stack(layer_contexts, dim=0).mean(dim=0)
            for layer_contexts in context_embeddings
        ]
        return main_embeddings, aggregated_context
