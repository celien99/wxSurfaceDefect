from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from hiad.models import TimmDinoV3Encoder

from .images import validate_image_array


def build_frozen_dino_encoder(config: Mapping[str, Any]) -> TimmDinoV3Encoder:
    encoder = TimmDinoV3Encoder(
        model_name=config["dino_backbone_name"],
        intermediate_layers=[config["dino_feature_layer"]],
        use_fp16=False,
    )
    if not isinstance(encoder.patch_size, int) or encoder.patch_size <= 0:
        raise ValueError(f"Invalid DINO patch size: {encoder.patch_size}")
    encoder.requires_grad_(False)
    encoder.eval()
    return encoder.cpu()


def _working_size(
    image_height: int,
    image_width: int,
    longest_edge: int,
    patch_size: int,
) -> tuple[int, int]:
    if image_height <= 0 or image_width <= 0:
        raise ValueError("DINO input dimensions must be positive")
    scale = longest_edge / max(image_height, image_width)
    scaled_height = max(1, int(round(image_height * scale)))
    scaled_width = max(1, int(round(image_width * scale)))
    working_height = int(math.ceil(scaled_height / patch_size) * patch_size)
    working_width = int(math.ceil(scaled_width / patch_size) * patch_size)
    return working_height, working_width


def extract_dino_grid(
    encoder: TimmDinoV3Encoder,
    rgb: np.ndarray,
    config: Mapping[str, Any],
    device: torch.device,
    working_longest_edge: int,
) -> tuple[torch.Tensor, np.ndarray, tuple[int, int], tuple[int, int]]:
    source = validate_image_array(rgb, config["input_scale"], "RGB")
    image_height, image_width = source.shape[:2]
    working_height, working_width = _working_size(
        image_height,
        image_width,
        working_longest_edge,
        encoder.patch_size,
    )

    # Copy so from_numpy yields a writable tensor (OpenCV/read-only views warn otherwise).
    source_tensor = (
        torch.from_numpy(np.array(source, copy=True, order="C"))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
    )
    source_tensor.div_(config["input_scale"])
    resized = F.interpolate(
        source_tensor,
        size=(working_height, working_width),
        mode="bilinear",
        align_corners=False,
    )
    mean = torch.tensor(config["mean"], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(config["std"], dtype=torch.float32).view(1, 3, 1, 1)
    normalized = resized.sub(mean).div(std)

    features = None
    outputs = None
    feature_map = None
    try:
        encoder.requires_grad_(False)
        encoder.eval()
        encoder.to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            outputs = encoder(normalized.to(device=device, dtype=torch.float32))
        if len(outputs) != 1 or outputs[0].ndim != 4 or outputs[0].shape[0] != 1:
            raise RuntimeError("DINO feature extraction returned an invalid feature shape")
        feature_map = outputs[0]
        grid_height, grid_width = feature_map.shape[-2:]
        expected_grid = (
            working_height // encoder.patch_size,
            working_width // encoder.patch_size,
        )
        if (grid_height, grid_width) != expected_grid:
            raise RuntimeError(
                f"DINO feature grid {(grid_height, grid_width)} does not match "
                f"{expected_grid}"
            )
        features = feature_map.permute(0, 2, 3, 1).reshape(-1, feature_map.shape[1])
        features = F.normalize(features.float(), dim=1).cpu()
        if not torch.isfinite(features).all():
            raise ValueError("DINO features contain NaN or infinite values")
    finally:
        encoder.requires_grad_(False)
        encoder.eval()
        encoder.cpu()
        del outputs, feature_map
        del source_tensor, resized, normalized
        if device.type == "cuda":
            torch.cuda.empty_cache()

    x_coordinates = (np.arange(grid_width, dtype=np.float32) + 0.5) * (
        image_width / grid_width
    )
    y_coordinates = (np.arange(grid_height, dtype=np.float32) + 0.5) * (
        image_height / grid_height
    )
    x_grid, y_grid = np.meshgrid(x_coordinates, y_coordinates)
    centers_xy = np.stack((x_grid, y_grid), axis=-1).reshape(-1, 2)
    return (
        features,
        np.ascontiguousarray(centers_xy, dtype=np.float32),
        (grid_height, grid_width),
        (working_height, working_width),
    )
