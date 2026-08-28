"""GPU 驻留的异常图拼接、路由与量化算子；与 NumPy 参照保持舍入级一致。"""
from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import TypeAlias

import torch
import torch.nn.functional as F

from hiad.data import HRImageIndex
from hiad.runtime.contracts import TaskInputRecord

RefinementList: TypeAlias = Sequence[tuple[HRImageIndex, torch.Tensor]]


def _hann_1d_torch(length: int) -> torch.Tensor:
    """复刻 ``np.hanning`` 的对称窗公式（非 periodic），保证与 NumPy 一致。"""
    if length <= 1:
        return torch.ones(1, dtype=torch.float32)
    indices = torch.arange(length, dtype=torch.float32)
    window = 0.5 - 0.5 * torch.cos(2.0 * torch.pi * indices / (length - 1))
    if window.max() > 0:
        window = window / window.max()
    return window


@lru_cache(maxsize=32)
def hann_weights_torch(height: int, width: int, device: torch.device) -> torch.Tensor:
    """返回带 ``0.05`` 保底权重的二维 Hann 窗（与 ``refinement_blend_weights``
    数值一致的只读张量，按 ``(height, width, device)`` 缓存）。"""
    row_hann = _hann_1d_torch(height)
    column_hann = _hann_1d_torch(width)
    weights = 0.05 + 0.95 * torch.outer(row_hann, column_hann)
    return weights.to(device=device, dtype=torch.float32)


def stitch_patch_maps_torch(
    patch_maps: torch.Tensor,
    records: Sequence[TaskInputRecord],
    image_size: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """按坐标与 Hann 权重把补丁图拼回原图分辨率异常图。补丁重叠区域按加权平均融合。"""
    image_width, image_height = image_size
    accumulated = torch.zeros(
        (image_height, image_width), dtype=torch.float32, device=device
    )
    weight_map = torch.zeros(
        (image_height, image_width), dtype=torch.float32, device=device
    )
    for patch_map, record in zip(patch_maps, records):
        x, y, width, height = record["source_xywh"]
        valid_height, valid_width = record["valid_source_hw"]
        weights = hann_weights_torch(height, width, device)
        valid_weights = weights[:valid_height, :valid_width]
        region = patch_map[:valid_height, :valid_width] * valid_weights
        accumulated[y:y + valid_height, x:x + valid_width] += region
        weight_map[y:y + valid_height, x:x + valid_width] += valid_weights
    if torch.any(weight_map <= 0):
        raise ValueError("Patch predictions do not cover the complete source image")
    return accumulated / weight_map


def robust_unit_map_torch(values: torch.Tensor) -> torch.Tensor:
    """中位数映射为 ``0``、``0.995`` 分位数映射为 ``1``（对应 NumPy
    ``_robust_unit_map``）；两者相等时返回全零图。"""
    lower = torch.quantile(values, 0.5)
    upper = torch.quantile(values, 0.995)
    if upper <= lower:
        return torch.zeros_like(values)
    return ((values - lower) / (upper - lower)).clamp(0.0, 1.0)


def build_routing_map_torch(
    local_map: torch.Tensor,
    global_context_map: torch.Tensor,
    global_weight: float,
) -> torch.Tensor:
    """融合局部与全局排序先验（对应 NumPy ``build_routing_map``）。"""
    if local_map.ndim != 2 or local_map.shape != global_context_map.shape:
        raise ValueError("local and global context maps must be aligned 2D tensors")
    weight = float(global_weight)
    if weight != weight or not 0.0 <= weight <= 1.0:
        raise ValueError("global_weight must be finite and in [0, 1]")
    return (1.0 - weight) * robust_unit_map_torch(local_map) + weight * robust_unit_map_torch(
        global_context_map
    )


def merge_refinement_maps_torch(
    base_map: torch.Tensor,
    refinements: RefinementList,
    image_size: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """按原图坐标回填复核结果，重叠区按 Hann 权重平均，再与粗扫图逐点取最大。

    复核异常图形状必须与区域宽高一致（当前配置恒成立）；不一致时抛错而非
    静默换插值——F.interpolate 与 cv2.resize 的亚像素差异会破坏数值契约。
    """
    image_width, image_height = image_size
    if base_map.ndim != 2 or base_map.shape != (image_height, image_width):
        raise ValueError("base_map shape must match image_size")
    accumulated = torch.zeros_like(base_map)
    weight_map = torch.zeros_like(base_map)
    for index, prediction in refinements:
        if not isinstance(index, HRImageIndex):
            raise TypeError("Each refinement index must be an HRImageIndex")
        if index.x < 0 or index.y < 0 or index.width <= 0 or index.height <= 0:
            raise ValueError("Refinement index has invalid geometry")
        if index.x >= image_width or index.y >= image_height:
            raise ValueError("Refinement index origin is outside image_size")
        if prediction.ndim != 2 or prediction.shape != (index.height, index.width):
            raise ValueError(
                "GPU refinement merge requires prediction.shape == (index.height, index.width); "
                "shape mismatch would silently change interpolation semantics"
            )
        valid_width = min(index.width, image_width - index.x)
        valid_height = min(index.height, image_height - index.y)
        weights = hann_weights_torch(index.height, index.width, device)
        target = (
            slice(index.y, index.y + valid_height),
            slice(index.x, index.x + valid_width),
        )
        valid_weights = weights[:valid_height, :valid_width]
        accumulated[target] += prediction[:valid_height, :valid_width] * valid_weights
        weight_map[target] += valid_weights
    covered = weight_map > 0
    if not torch.any(covered):
        return base_map.clone()
    refinement_map = torch.zeros_like(base_map)
    refinement_map[covered] = accumulated[covered] / weight_map[covered]
    alpha = torch.clamp(weight_map, 0.0, 1.0)
    blended = (1.0 - alpha) * base_map + alpha * refinement_map
    return torch.maximum(base_map, blended)


def top_k_token_scores_torch(token_maps: torch.Tensor, top_k: int) -> torch.Tensor:
    """按样本聚合最高异常 token 的均值（对应 ``_top_k_token_scores``）。"""
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if token_maps.ndim != 4 or token_maps.shape[1] != 1:
        raise ValueError("token_maps must have shape [batch, 1, height, width]")
    values = token_maps.flatten(start_dim=1)
    count = min(top_k, values.shape[1])
    return torch.topk(values, k=count, dim=1).values.mean(dim=1)


def gaussian_blur_torch(map_2d: torch.Tensor, sigma: float) -> torch.Tensor:
    """对单通道 2D 图做高斯平滑（对应 ``scipy.ndimage.gaussian_filter`` 语义，
    供 ``map_gaussian_sigma > 0`` 时使用；默认 0 时不会被调用）。"""
    if sigma <= 0:
        return map_2d
    if map_2d.ndim != 2:
        raise ValueError("gaussian_blur_torch expects a 2D tensor")
    kernel_size = max(3, int(4 * sigma) | 1)
    return F.gaussian_blur(
        map_2d.unsqueeze(0).unsqueeze(0),
        kernel_size=(kernel_size, kernel_size),
        sigma=sigma,
    ).squeeze(0).squeeze(0)
