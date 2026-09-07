from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config import FusionConfig
from .phase import decompose_four_step, validate_image_group
from .responses import (
    detect_dark_spots,
    detect_scratch,
    detect_shape,
    detect_texture,
    normalized_weights,
    robust_normalize,
)

GrayImage = NDArray[np.float32]
UInt8Image = NDArray[np.uint8]


@dataclass(frozen=True)
class FusionResult:
    """8→1 融合结果及可检查的中间响应。"""

    fused: UInt8Image
    shape: GrayImage
    scratch: GrayImage
    dark: GrayImage
    texture: GrayImage
    phase_confidence: GrayImage
    saturation_validity: GrayImage
    x_phase: GrayImage
    y_phase: GrayImage
    x_modulation: GrayImage
    y_modulation: GrayImage
    x_background: GrayImage
    y_background: GrayImage


class CarbonFiberStripeFusion:
    """把 X/Y 各 4 张程控相移图融合为 1 张缺陷增强图。"""

    def __init__(self, config: FusionConfig | None = None) -> None:
        self.config = config or FusionConfig()

    def fuse(
        self,
        x_images: Sequence[ArrayLike],
        y_images: Sequence[ArrayLike],
    ) -> UInt8Image:
        """返回 ``(H, W)`` ``uint8`` 融合图。"""
        return self.analyze(x_images, y_images).fused

    def analyze(
        self,
        x_images: Sequence[ArrayLike],
        y_images: Sequence[ArrayLike],
    ) -> FusionResult:
        """计算融合图及四类响应、可信度和相移中间量。"""
        x_arrays = validate_image_group(x_images, "x_images")
        y_arrays = validate_image_group(y_images, "y_images")
        if x_arrays[0].shape != y_arrays[0].shape:
            raise ValueError("x_images and y_images must share the same spatial shape")
        if x_arrays[0].dtype != y_arrays[0].dtype:
            raise ValueError("x_images and y_images must share the same dtype")

        x_components = decompose_four_step(x_arrays, self.config)
        y_components = decompose_four_step(y_arrays, self.config)
        shape = detect_shape(x_components, y_components, self.config)
        scratch = detect_scratch(x_components, y_components, self.config)
        dark = detect_dark_spots(x_components, y_components, self.config)
        texture = detect_texture(x_components, y_components, self.config)
        phase_confidence = (
            0.5 * (x_components.phase_confidence + y_components.phase_confidence)
        ).astype(np.float32)
        saturation_validity = (
            0.5 * (x_components.saturation_validity + y_components.saturation_validity)
        ).astype(np.float32)
        fused = self._fuse_maps(shape, scratch, dark, texture)
        return FusionResult(
            fused=fused,
            shape=shape,
            scratch=scratch,
            dark=dark,
            texture=texture,
            phase_confidence=phase_confidence,
            saturation_validity=saturation_validity,
            x_phase=x_components.phase,
            y_phase=y_components.phase,
            x_modulation=x_components.modulation,
            y_modulation=y_components.modulation,
            x_background=x_components.background,
            y_background=y_components.background,
        )

    def _fuse_maps(
        self,
        shape: GrayImage,
        scratch: GrayImage,
        dark: GrayImage,
        texture: GrayImage,
    ) -> UInt8Image:
        weights = normalized_weights(self.config)
        fused = (
            weights[0] * shape
            + weights[1] * scratch
            + weights[2] * dark
            + weights[3] * texture
        )
        fused = np.maximum(fused - self.config.soft_threshold, 0.0)
        fused = robust_normalize(fused, self.config)
        kernel = self.config.output_blur_kernel
        if kernel > 1:
            fused = cv2.GaussianBlur(fused, (kernel, kernel), 0)
        return np.clip(fused * 255.0, 0, 255).astype(np.uint8)
