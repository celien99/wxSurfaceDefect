"""向量化批量建批：把逐 tile 的裁剪/缩放/归一化合并为一次批量操作。

所有算子与逐 tile 路径（``create_dynamic_patch`` + ``transform_patch``）保持
逐位一致，作为 A 工程层的数值一致入场券。
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import cv2
import numpy as np
import torch

from hiad.data import HRImageIndex, MultiResolutionIndex
from hiad.datasets.patch_dataset import PatchItem
from hiad.runtime.contracts import TaskInputRecord

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def normalize_rgb_stack(stack: np.ndarray) -> torch.Tensor:
    """把 ``(N, H, W, 3)`` uint8 批量转换为 ImageNet 标准化 CHW 张量。

    算子顺序与 ``PatchDataset._image_to_tensor`` 逐项一致（from_numpy →
    permute → float → div 255 → sub mean → div std），保证批量与逐 tile 结果
    逐位相等。
    """
    source = torch.from_numpy(np.ascontiguousarray(stack)).permute(0, 3, 1, 2)
    tensor = source.float().div_(255.0)
    return tensor.sub_(_IMAGENET_MEAN).div_(_IMAGENET_STD)


def crop_tile(image: np.ndarray, index: HRImageIndex) -> np.ndarray:
    """复刻 ``HRImage.__getitem__`` 的裁剪 + ``np.pad(mode='edge')`` 语义。

    网格滑窗区域均贴合原图边界，正常情况下不触发填充；该函数为越界路径提供
    与逐 tile 完全一致的边缘重复填充（仅右下越界场景存在）。
    """
    image_height, image_width = image.shape[:2]
    if index.x >= image_width or index.y >= image_height:
        raise ValueError(f"Patch origin is outside image bounds: {index}")
    valid_height = min(index.height, image_height - index.y)
    valid_width = min(index.width, image_width - index.x)
    tile = np.empty((index.height, index.width, 3), dtype=np.uint8)
    tile[:valid_height, :valid_width] = image[
        index.y:index.y + valid_height, index.x:index.x + valid_width
    ]
    if valid_width < index.width:
        tile[:, valid_width:] = tile[:, valid_width - 1:valid_width]
    if valid_height < index.height:
        tile[valid_height:] = tile[valid_height - 1:valid_height]
    return tile


def build_patch_batch(
    image: np.ndarray,
    indexes: Sequence[MultiResolutionIndex],
    patch_size: int,
    base_record: Mapping[str, object],
) -> tuple[PatchItem, list[TaskInputRecord]]:
    """一次批量提取主补丁与上下文，语义与逐 tile 路径逐位一致。

    Args:
        image: 已解码的 ``(height, width, 3)`` RGB ``uint8`` 原图。
        indexes: 与任务记录顺序一致的多尺度区域索引。
        patch_size: 正方形模型输入边长。
        base_record: 任务追溯字段模板（``task_name``/``task_type``/
            ``image_path``/``image_size``/``model_input_size``）。

    Returns:
        ``(batch, records)``：可直接送入 ``inference_batch`` 的批量输入字典
        （``image``、``low_resolution_image_<level>``、``low_resolution_index_<level>``），
        以及与输入逐项对应的 ``TaskInputRecord`` 列表。
    """
    count = len(indexes)
    image_height, image_width = image.shape[:2]
    main_stack = np.empty((count, patch_size, patch_size, 3), dtype=np.uint8)
    max_levels = max(
        (
            len(index.low_resolution_indexes)
            if index.low_resolution_indexes is not None
            else 0
        )
        for index in indexes
    ) if indexes else 0
    context_stacks = {
        level: np.empty((count, patch_size, patch_size, 3), dtype=np.uint8)
        for level in range(max_levels)
    }
    context_index_strings = {level: [] for level in range(max_levels)}
    records: list[TaskInputRecord] = []
    for i, index in enumerate(indexes):
        main_index = index.main_index
        main_stack[i] = crop_tile(image, main_index)
        records.append({
            **base_record,
            "source_xywh": (
                main_index.x, main_index.y, main_index.width, main_index.height,
            ),
            "valid_source_hw": (
                min(main_index.height, image_height - main_index.y),
                min(main_index.width, image_width - main_index.x),
            ),
        })
        if index.low_resolution_indexes is None:
            continue
        for level, low_index in enumerate(index.low_resolution_indexes):
            low = crop_tile(image, low_index)
            low_height, low_width = low.shape[:2]
            low = cv2.resize(
                low, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR
            )
            context_stacks[level][i] = low
            context_index_strings[level].append(str(HRImageIndex(
                x=int((main_index.x - low_index.x) / low_width * patch_size),
                y=int((main_index.y - low_index.y) / low_height * patch_size),
                width=int(main_index.width / low_index.width * patch_size),
                height=int(main_index.height / low_index.height * patch_size),
            )))

    batch: PatchItem = {"image": normalize_rgb_stack(main_stack)}
    for level in range(max_levels):
        batch[f"low_resolution_image_{level}"] = normalize_rgb_stack(
            context_stacks[level]
        )
        batch[f"low_resolution_index_{level}"] = context_index_strings[level]
    return batch, records


def build_thumbnail_batch(
    image: np.ndarray,
    thumbnail_size: int,
    base_record: Mapping[str, object],
) -> tuple[PatchItem, TaskInputRecord]:
    """把整图缩到模型尺寸并标准化为单样本批量（对应 ``down_sampling_to_LR``）。"""
    thumbnail = cv2.resize(
        image, (thumbnail_size, thumbnail_size), interpolation=cv2.INTER_LINEAR
    )
    batch: PatchItem = {"image": normalize_rgb_stack(thumbnail[None])}
    return batch, dict(base_record)
