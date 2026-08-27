from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config import FusionConfig

GrayImage = NDArray[np.float32]


@dataclass(frozen=True)
class PhaseComponents:
    """四步相移分解后的相位、调制度、背景和可信度图。"""

    phase: GrayImage
    modulation: GrayImage
    background: GrayImage
    phase_confidence: GrayImage
    saturation_validity: GrayImage


def validate_image_group(images: Sequence[ArrayLike], name: str) -> list[NDArray[np.generic]]:
    """校验一组恰好 4 张同尺寸二维有限灰度图。"""
    arrays = [np.asarray(image) for image in images]
    if len(arrays) != 4:
        raise ValueError(f"{name} must contain exactly 4 phase-shift images.")
    first = arrays[0]
    if first.ndim != 2:
        raise ValueError(f"{name} images must be single-channel grayscale.")
    for index, image in enumerate(arrays):
        if image.ndim != 2:
            raise ValueError(f"{name} images must be single-channel grayscale.")
        if image.shape != first.shape:
            raise ValueError(
                f"{name}[{index}] shape mismatch: {image.shape} != {first.shape}"
            )
        if not np.isfinite(np.asarray(image, dtype=np.float64)).all():
            raise ValueError(f"{name}[{index}] contains NaN or Inf.")
        if image.dtype != first.dtype:
            raise ValueError(
                f"{name}[{index}] dtype mismatch: {image.dtype} != {first.dtype}"
            )
    return arrays


def images_to_unit_range(
    images: Sequence[ArrayLike],
    input_max: float | None,
) -> list[GrayImage]:
    """把一组灰度图线性缩放到 ``float32`` ``[0, 1]``。"""
    arrays = [np.asarray(image) for image in images]
    if not arrays:
        raise ValueError("images must be a non-empty sequence")
    scale = _resolve_input_max(arrays[0].dtype, input_max)
    return [
        np.clip(np.asarray(image, dtype=np.float32) / scale, 0.0, 1.0).astype(
            np.float32, copy=False
        )
        for image in arrays
    ]


def preprocess(image: ArrayLike, config: FusionConfig) -> GrayImage:
    """转换为 ``float32``，并按配置做可选高斯平滑。"""
    values = np.asarray(image, dtype=np.float32)
    if config.gaussian_sigma <= 0:
        return values
    return np.asarray(
        cv2.GaussianBlur(
            values,
            (config.gaussian_kernel, config.gaussian_kernel),
            config.gaussian_sigma,
        ),
        dtype=np.float32,
    )


def decompose_four_step(
    images: Sequence[ArrayLike],
    config: FusionConfig,
) -> PhaseComponents:
    """四步相移分解，得到相位、调制度、背景和两类可信度。"""
    validated = validate_image_group(images, "phase_images")
    unit_images = images_to_unit_range(validated, config.input_max)
    saturation_validity = _saturation_validity(
        np.maximum.reduce(unit_images),
        config.saturation_onset,
    )
    intensity_0, intensity_1, intensity_2, intensity_3 = [
        preprocess(image, config) for image in unit_images
    ]
    sin_term = intensity_3 - intensity_1
    cos_term = intensity_0 - intensity_2
    phase = np.arctan2(sin_term, cos_term).astype(np.float32)
    modulation = (0.5 * np.hypot(sin_term, cos_term)).astype(np.float32)
    background = (
        (intensity_0 + intensity_1 + intensity_2 + intensity_3) * 0.25
    ).astype(np.float32)
    modulation_ratio = modulation / np.maximum(background, 1e-3)
    phase_confidence = np.clip(
        modulation_ratio / config.modulation_confidence_floor,
        0.0,
        1.0,
    ).astype(np.float32)
    return PhaseComponents(
        phase=phase,
        modulation=modulation,
        background=background,
        phase_confidence=phase_confidence,
        saturation_validity=saturation_validity,
    )


def unwrap_phase(
    phase: ArrayLike,
    axis: int,
    confidence: ArrayLike,
    config: FusionConfig,
) -> GrayImage:
    """沿指定轴展开相位，并用平滑值替换低可信区域。"""
    unwrapped = np.unwrap(np.asarray(phase, dtype=np.float32), axis=axis).astype(np.float32)
    confidence_map = np.asarray(confidence, dtype=np.float32)
    kernel = (config.phase_smooth_kernel, config.phase_smooth_kernel)
    smooth = cv2.GaussianBlur(unwrapped, kernel, 0)
    low_confidence = confidence_map < config.modulation_confidence_floor
    if np.any(low_confidence):
        unwrapped = unwrapped.copy()
        unwrapped[low_confidence] = smooth[low_confidence]
    return np.asarray(cv2.GaussianBlur(unwrapped, kernel, 0), dtype=np.float32)


def _resolve_input_max(dtype: np.dtype[np.generic], input_max: float | None) -> float:
    if input_max is not None:
        return float(input_max)
    if dtype == np.uint8:
        return 255.0
    if dtype == np.uint16:
        return 65535.0
    return 1.0


def _saturation_validity(peak: GrayImage, onset: float) -> GrayImage:
    if onset >= 1.0:
        return np.ones_like(peak, dtype=np.float32)
    validity = np.ones_like(peak, dtype=np.float32)
    saturated = peak >= 1.0
    transitioning = (peak >= onset) & ~saturated
    span = 1.0 - onset
    validity[transitioning] = (1.0 - peak[transitioning]) / span
    validity[saturated] = 0.0
    return np.clip(validity, 0.0, 1.0).astype(np.float32)
