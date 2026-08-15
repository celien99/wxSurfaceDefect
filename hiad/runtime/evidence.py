"""Deterministic evidence maps used by high-resolution detectors."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch.nn import functional as F


def denormalize_imagenet_batch(image_batch: torch.Tensor) -> torch.Tensor:
    """Restore ImageNet-normalized BCHW RGB tensors to their source value range."""
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
    """Return a single-channel Sobel/Laplacian high-frequency map.

    The image channels are averaged before applying fixed kernels. Kernels are
    created on the input device, so this function has no parameters or module
    state and stays on the detector device.
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


def _validate_weights(weights: Sequence[float], branch_count: int) -> np.ndarray:
    if branch_count != len(weights):
        raise ValueError("weights must match the number of evidence branches")
    numeric_weights = np.asarray(weights, dtype=np.float64)
    if numeric_weights.ndim != 1 or not np.isfinite(numeric_weights).all():
        raise ValueError("weights must be a finite vector")
    if np.any(numeric_weights < 0) or not numeric_weights.any():
        raise ValueError("weights must be non-negative with a positive sum")
    return numeric_weights


def fuse_evidence_maps(
    branch_maps: Sequence[np.ndarray], weights: Sequence[float], max_evidence: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse same-shaped branch maps and optionally apply a safety maximum."""
    if len(branch_maps) == 0:
        raise ValueError("at least one evidence branch is required")
    numeric_weights = _validate_weights(weights, len(branch_maps))
    converted = [np.asarray(branch, dtype=np.float64) for branch in branch_maps]
    shape = converted[0].shape
    if len(shape) == 0:
        raise ValueError("evidence branches must have at least one dimension")
    if any(branch.shape != shape for branch in converted[1:]):
        raise ValueError("all evidence branches must have the same shape")
    if any(not np.isfinite(branch).all() for branch in converted):
        raise ValueError("all evidence branches must be finite")

    enabled = numeric_weights > 0
    stack = np.stack(converted, axis=0)[enabled]
    active_weights = numeric_weights[enabled]
    weighted = np.average(stack, axis=0, weights=active_weights)
    maximum = np.max(stack, axis=0)
    fused = 0.75 * weighted + 0.25 * maximum if max_evidence else weighted
    return fused, maximum


def fuse_evidence_tensors(
    branch_maps: Sequence[torch.Tensor],
    weights: Sequence[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse aligned accelerator tensors while retaining a max safety channel."""
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
    return 0.75 * weighted + 0.25 * maximum, maximum
