from __future__ import annotations

from typing import TypeAlias

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch.utils.data import Dataset

from hiad.data import LRPatch

PatchItem: TypeAlias = dict[str, torch.Tensor | str | int]


class PatchDataset(Dataset[PatchItem]):
    """将内存补丁转换为 ImageNet 标准化的模型输入字典。

    Attributes:
        patches (list[LRPatch]): 待转换的 RGB 补丁及可选上下文。
        training (bool): 是否用于训练；保留该状态供任务管线识别。
        task_name (str): 产生这些补丁的任务名称。
        mean (torch.Tensor): ``(3, 1, 1)`` ImageNet RGB 均值。
        std (torch.Tensor): ``(3, 1, 1)`` ImageNet RGB 标准差。
    """

    def __init__(self, patches: list[LRPatch], training: bool, task_name: str) -> None:
        super().__init__()
        self.patches: list[LRPatch] = patches
        self.training: bool = training
        self.task_name: str = task_name
        self.mean: torch.Tensor = torch.tensor(
            [0.485, 0.456, 0.406], dtype=torch.float32
        ).view(3, 1, 1)
        self.std: torch.Tensor = torch.tensor(
            [0.229, 0.224, 0.225], dtype=torch.float32
        ).view(3, 1, 1)
        self._inference_zero_masks: dict[tuple[int, int], torch.Tensor] = {}

    def _image_to_tensor(self, image: ArrayLike, name: str) -> torch.Tensor:
        """校验 HWC RGB 数组并转换为标准化 CHW 浮点张量。

        Args:
            image (ArrayLike): ``(height, width, 3)`` RGB 数组；支持
                ``uint8`` ``[0, 255]`` 或有限 ``float32`` 输入。``float32``
                输入应由调用方预先缩放到模型约定的数值范围，本方法不额外校验
                其范围。
            name (str): 用于校验错误消息的字段名称。

        Returns:
            torch.Tensor: ``(3, height, width)`` ImageNet 标准化 ``float32`` 张量。

        Raises:
            ValueError: 维度、通道数、尺寸、数据类型或有限性不符合约定。
        """
        image = np.asarray(image)
        if (
            image.ndim != 3
            or image.shape[2] != 3
            or image.shape[0] <= 0
            or image.shape[1] <= 0
            or image.dtype not in (np.uint8, np.float32)
            or (image.dtype == np.float32 and not np.isfinite(image).all())
        ):
            raise ValueError(f"{name} must be a finite HWC RGB image")
        source = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
        tensor = source.float() if image.dtype == np.uint8 else source.clone()
        if image.dtype == np.uint8:
            tensor.div_(255.0)
        return tensor.sub_(self.mean).div_(self.std)

    def __getitem__(self, idx: int) -> PatchItem:
        return self.transform_patch(self.patches[idx])

    def transform_patch(self, patch: LRPatch) -> PatchItem:
        """Convert one in-memory patch without constructing another dataset.

        Streaming inference calls this method repeatedly through one reusable
        converter, avoiding per-tile dataset and normalization-tensor setup.
        """
        image = self._image_to_tensor(patch.image, "patch.image")

        if patch.mask is not None:
            mask_array = np.asarray(patch.mask)
            if mask_array.ndim != 2 or mask_array.shape != tuple(image.shape[1:]):
                raise ValueError("patch.mask must match the patch image height and width")
            mask = torch.from_numpy(mask_array).ne(0).to(torch.float32)
        else:
            shape = (int(image.shape[1]), int(image.shape[2]))
            if self.training:
                mask = torch.zeros(shape, dtype=torch.float32)
            else:
                mask = self._inference_zero_masks.get(shape)
                if mask is None:
                    mask = torch.zeros(shape, dtype=torch.float32)
                    self._inference_zero_masks[shape] = mask

        item: PatchItem = {"image": image, "mask": mask}
        if patch.clsname is not None:
            item["clsname"] = patch.clsname
        if patch.label is not None:
            item["label"] = patch.label
        if patch.low_resolution_images is not None:
            if patch.low_resolution_indexes is None:
                raise RuntimeError("Context images and indexes must be aligned")
            for i, (low_image, low_index) in enumerate(
                zip(patch.low_resolution_images, patch.low_resolution_indexes)
            ):
                item[f"low_resolution_image_{i}"] = self._image_to_tensor(
                    low_image,
                    f"patch.low_resolution_images[{i}]",
                )
                item[f"low_resolution_index_{i}"] = str(low_index)
        return item

    def __len__(self) -> int:
        return len(self.patches)
