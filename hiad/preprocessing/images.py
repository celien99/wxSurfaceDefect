from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np


def validate_image_array(
    image: np.ndarray,
    input_scale: float,
    color_space: str,
) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")
    if (
        image.ndim != 3
        or image.shape[2] != 3
        or image.shape[0] <= 0
        or image.shape[1] <= 0
    ):
        raise ValueError("image must have non-empty HWC shape with three channels")
    if image.dtype not in {
        np.dtype(np.uint8),
        np.dtype(np.float32),
        np.dtype(np.float64),
    }:
        raise TypeError("image dtype must be uint8, float32, or float64")

    if image.dtype.kind == "f" and not np.isfinite(image).all():
        raise ValueError("image contains NaN or infinite values")
    minimum = float(np.min(image))
    maximum = float(np.max(image))
    if minimum < 0 or maximum > input_scale:
        raise ValueError(f"image values must be in [0, {input_scale}]")

    if color_space == "RGB":
        rgb = image
    elif color_space == "BGR":
        rgb = image[..., ::-1]
    else:
        raise ValueError("color_space must be RGB or BGR")

    target_dtype = np.uint8 if image.dtype == np.uint8 else np.float32
    return np.asarray(rgb, dtype=target_dtype, order="C")


def normalize_with_foreground(
    rgb: np.ndarray,
    foreground_mask: np.ndarray,
    config: Mapping[str, Any],
) -> np.ndarray:
    normalized = np.empty(rgb.shape, dtype=np.float32, order="C")
    np.divide(
        rgb,
        config["input_scale"],
        out=normalized,
        casting="unsafe",
    )
    mean = np.asarray(config["mean"], dtype=np.float32)
    std = np.asarray(config["std"], dtype=np.float32)
    np.subtract(normalized, mean, out=normalized)
    np.divide(normalized, std, out=normalized)
    if not np.isfinite(normalized).all():
        raise ValueError("Normalized image contains NaN or infinite values")

    normalized[~foreground_mask] = 0.0
    return normalized


def inverse_normalize_image(
    image: np.ndarray,
    config: Mapping[str, Any],
    output_size: int | tuple[int, int] | list[int] | None = None,
) -> np.ndarray:
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.float32
        or image.ndim != 3
        or image.shape[0] <= 0
        or image.shape[1] <= 0
        or image.shape[2] != 3
        or not np.isfinite(image).all()
    ):
        raise ValueError("Normalized image must be finite non-empty HWC float32 RGB")

    display_image = image
    if output_size is not None:
        if isinstance(output_size, bool):
            raise TypeError("output_size must be a positive integer or width-height pair")
        if isinstance(output_size, int):
            output_width = output_height = output_size
        elif isinstance(output_size, (tuple, list)) and len(output_size) == 2:
            output_width, output_height = output_size
        else:
            raise TypeError("output_size must be a positive integer or width-height pair")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (output_width, output_height)
        ):
            raise ValueError("output_size dimensions must be positive integers")
        if (output_height, output_width) != image.shape[:2]:
            interpolation = (
                cv2.INTER_AREA
                if output_height <= image.shape[0] and output_width <= image.shape[1]
                else cv2.INTER_LINEAR
            )
            display_image = cv2.resize(
                image,
                (output_width, output_height),
                interpolation=interpolation,
            )

    rgb_values = np.empty(display_image.shape, dtype=np.float32, order="C")
    std = np.asarray(config["std"], dtype=np.float32)
    mean = np.asarray(config["mean"], dtype=np.float32)
    np.multiply(display_image, std, out=rgb_values)
    np.add(rgb_values, mean, out=rgb_values)
    np.multiply(rgb_values, config["input_scale"], out=rgb_values)
    np.multiply(
        rgb_values,
        255.0 / config["input_scale"],
        out=rgb_values,
    )
    np.clip(rgb_values, 0.0, 255.0, out=rgb_values)
    np.rint(rgb_values, out=rgb_values)
    return rgb_values.astype(np.uint8)
