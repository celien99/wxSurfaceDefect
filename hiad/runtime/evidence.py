"""高分辨率检测器使用的确定性证据图。"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from torch.nn import functional as F


def denormalize_imagenet_batch(image_batch: torch.Tensor) -> torch.Tensor:
    """将 ImageNet 标准化的 BCHW RGB 张量恢复到原始数值范围。

    Args:
        image_batch (torch.Tensor): ``(batch, 3, height, width)`` 浮点 RGB 张量。

    Returns:
        torch.Tensor: 与输入形状、设备和浮点类型一致的反标准化 RGB 张量；
        结果不会自动裁剪到 ``[0, 1]``。

    Raises:
        ValueError: 输入不是四维三通道浮点张量。
    """
    if (
        not isinstance(image_batch, torch.Tensor)
        or image_batch.ndim != 4
        or image_batch.shape[1] != 3
        or not image_batch.is_floating_point()
    ):
        raise ValueError("image_batch must be a floating-point BCHW RGB tensor")
    mean = image_batch.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = image_batch.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    return image_batch * std + mean


def high_frequency_map(image_batch: torch.Tensor) -> torch.Tensor:
    """生成单通道 Sobel/Laplacian 高频响应图。

    固定卷积核直接创建在输入设备上，不引入可训练参数，也不会造成 CPU/GPU 往返。

    Args:
        image_batch (torch.Tensor): ``(batch, channels, height, width)`` 图像张量。

    Returns:
        torch.Tensor: ``(batch, 1, height, width)`` 的 Sobel 梯度幅值与
        Laplacian 绝对响应之和。

    Raises:
        ValueError: 输入不是四维张量，或通道、空间维度为空。
    """
    if not isinstance(image_batch, torch.Tensor) or image_batch.ndim != 4:
        raise ValueError("image_batch must be a 4D BCHW tensor")
    if any(dimension == 0 for dimension in image_batch.shape[1:]):
        raise ValueError("image_batch must have non-empty channel and spatial dimensions")

    image = image_batch if image_batch.is_floating_point() else image_batch.float()
    image = image.mean(dim=1, keepdim=True)
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=image.device,
        dtype=image.dtype,
    ).reshape(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-1, -2)
    laplacian = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=image.device,
        dtype=image.dtype,
    ).reshape(1, 1, 3, 3)
    padded = F.pad(image, (1, 1, 1, 1), mode="replicate")
    gradient = torch.sqrt(
        F.conv2d(padded, sobel_x).square() + F.conv2d(padded, sobel_y).square()
    )
    curvature = F.conv2d(padded, laplacian).abs()
    return gradient + curvature


def _validate_weights(
    weights: Sequence[float],
    branch_count: int,
) -> NDArray[np.float64]:
    """校验证据分支权重并转换为一维有限浮点向量。

    Args:
        weights (Sequence[float]): 与证据分支逐项对应的权重。
        branch_count (int): 实际证据分支数量。

    Returns:
        NDArray[np.float64]: 一维非负有限权重向量，至少一个元素为正。

    Raises:
        ValueError: 数量不匹配、向量非一维、包含非有限/负数或总权重为零。
    """
    if branch_count != len(weights):
        raise ValueError("weights must match the number of evidence branches")
    numeric_weights = np.asarray(weights, dtype=np.float64)
    if numeric_weights.ndim != 1 or not np.isfinite(numeric_weights).all():
        raise ValueError("weights must be a finite vector")
    if np.any(numeric_weights < 0) or not numeric_weights.any():
        raise ValueError("weights must be non-negative with a positive sum")
    return numeric_weights


def fuse_evidence_tensors(
    branch_maps: Sequence[torch.Tensor],
    weights: Sequence[float],
) -> torch.Tensor:
    """融合对齐证据，并保留启用分支最大响应作为召回保护。

    Args:
        branch_maps (Sequence[torch.Tensor]): 形状、设备一致的有限浮点 BCHW
            证据图；允许任意批量和通道数。
        weights (Sequence[float]): 与分支逐项对应的非负权重，至少一个为正。

    Returns:
        torch.Tensor: 与单个分支形状一致的融合图，由加权均值和逐点最大值按
        ``0.75/0.25`` 组合。

    Raises:
        ValueError: 分支为空，权重无效，或分支形状、设备、类型、有限性不一致。
    """
    if not branch_maps:
        raise ValueError("at least one evidence branch is required")
    numeric_weights = _validate_weights(weights, len(branch_maps))
    reference = branch_maps[0]
    if not isinstance(reference, torch.Tensor) or reference.ndim != 4:
        raise ValueError("evidence branches must be BCHW tensors")
    if any(
        not isinstance(branch, torch.Tensor)
        or branch.shape != reference.shape
        or branch.device != reference.device
        for branch in branch_maps
    ):
        raise ValueError("all evidence tensors must share shape and device")
    if any(not branch.is_floating_point() for branch in branch_maps):
        raise ValueError("all evidence tensors must use a floating-point dtype")
    if any(not torch.isfinite(branch).all() for branch in branch_maps):
        raise ValueError("all evidence tensors must be finite")
    enabled = numeric_weights > 0
    stack = torch.stack(
        tuple(branch for branch, active in zip(branch_maps, enabled) if active),
        dim=0,
    )
    tensor_weights = torch.as_tensor(
        numeric_weights[enabled],
        device=reference.device,
        dtype=reference.dtype,
    ).view(-1, 1, 1, 1, 1)
    weighted = (stack * tensor_weights).sum(dim=0) / tensor_weights.sum()
    maximum = stack.amax(dim=0)
    return 0.75 * weighted + 0.25 * maximum
