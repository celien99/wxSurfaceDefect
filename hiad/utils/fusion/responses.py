from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import median_filter

from .config import FusionConfig
from .phase import PhaseComponents, unwrap_phase

GrayImage = NDArray[np.float32]


def robust_normalize(image: ArrayLike, config: FusionConfig) -> GrayImage:
    """用分位数把非负响应映射到 ``[0, 1]``；动态范围不足时归零。"""
    values = np.nan_to_num(
        np.asarray(image, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    low = float(np.percentile(values, config.percentile_low))
    high = float(np.percentile(values, config.percentile_high))
    if high - low < config.min_dynamic_range:
        return np.zeros_like(values)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def morphological_tophat_residual(image: ArrayLike, config: FusionConfig) -> GrayImage:
    """多尺度开运算最小基线后的顶帽残差。"""
    values = np.asarray(image, dtype=np.float32)
    opened_maps: list[GrayImage] = []
    for scale in config.texture_period_scales:
        ksize = max(3, int(config.texture_period * scale))
        if ksize % 2 == 0:
            ksize += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        opened_maps.append(
            np.asarray(
                cv2.morphologyEx(values, cv2.MORPH_OPEN, kernel),
                dtype=np.float32,
            )
        )
    baseline = np.min(np.stack(opened_maps, axis=0), axis=0)
    return np.maximum(values - baseline, 0.0).astype(np.float32)


def normalized_weights(config: FusionConfig) -> NDArray[np.float32]:
    """把四通道固定权重归一化为和为 1。"""
    weights = np.array(
        [
            config.weight_phase,
            config.weight_scratch,
            config.weight_dark,
            config.weight_texture,
        ],
        dtype=np.float32,
    )
    return weights / float(weights.sum())


def detect_shape(
    x_components: PhaseComponents,
    y_components: PhaseComponents,
    config: FusionConfig,
) -> GrayImage:
    """由 X/Y 相位梯度合成形貌响应。"""
    x_phase = unwrap_phase(
        x_components.phase,
        axis=config.x_phase_axis,
        confidence=x_components.phase_confidence,
        config=config,
    )
    y_phase = unwrap_phase(
        y_components.phase,
        axis=config.y_phase_axis,
        confidence=y_components.phase_confidence,
        config=config,
    )
    gradient_x = _axis_gradient(x_phase, config.x_phase_axis, config.phase_gradient_kernel)
    gradient_y = _axis_gradient(y_phase, config.y_phase_axis, config.phase_gradient_kernel)
    magnitude = np.hypot(gradient_x, gradient_y)
    confidence = np.sqrt(
        np.maximum(x_components.phase_confidence * y_components.phase_confidence, 0.0)
    )
    return robust_normalize(magnitude * confidence, config)


def detect_scratch(
    x_components: PhaseComponents,
    y_components: PhaseComponents,
    config: FusionConfig,
) -> GrayImage:
    """由调制度局部高频残差检测划痕。"""
    modulation = 0.5 * (x_components.modulation + y_components.modulation)
    local_mean = cv2.GaussianBlur(
        modulation,
        (config.modulation_local_kernel, config.modulation_local_kernel),
        0,
    )
    normalized = modulation / np.maximum(local_mean, 1e-3)
    local_std = _local_std(normalized, config.modulation_std_kernel)
    low_frequency = cv2.GaussianBlur(local_std, (0, 0), config.highpass_sigma)
    high_frequency = np.abs(local_std - low_frequency)
    confidence = 0.5 * (
        x_components.phase_confidence * x_components.saturation_validity
        + y_components.phase_confidence * y_components.saturation_validity
    )
    residual = morphological_tophat_residual(high_frequency * confidence, config)
    return robust_normalize(residual, config)


def detect_dark_spots(
    x_components: PhaseComponents,
    y_components: PhaseComponents,
    config: FusionConfig,
) -> GrayImage:
    """由浮点中值背景残差检测暗点。"""
    background = 0.5 * (x_components.background + y_components.background)
    local_background = median_filter(
        background,
        size=int(config.dark_background_kernel),
        mode="nearest",
    ).astype(np.float32)
    dark = np.maximum(local_background - background, 0.0)
    validity = 0.5 * (
        x_components.saturation_validity + y_components.saturation_validity
    )
    return robust_normalize(dark * validity, config)


def detect_texture(
    x_components: PhaseComponents,
    y_components: PhaseComponents,
    config: FusionConfig,
) -> GrayImage:
    """由带通纹理能量和顶帽残差检测纹理异常。"""
    background = 0.5 * (x_components.background + y_components.background)
    low = cv2.GaussianBlur(background, (0, 0), config.texture_sigma_large)
    high = cv2.GaussianBlur(background, (0, 0), config.texture_sigma_small)
    band = np.asarray(high - low, dtype=np.float32)
    energy = _local_std(band, config.texture_local_kernel)
    validity = 0.5 * (
        x_components.saturation_validity + y_components.saturation_validity
    )
    residual = morphological_tophat_residual(energy * validity, config)
    return robust_normalize(residual, config)


def _axis_gradient(image: GrayImage, axis: int, ksize: int) -> GrayImage:
    if axis == 1:
        gradient = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=ksize)
    else:
        gradient = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=ksize)
    return np.asarray(gradient, dtype=np.float32)


def _local_std(image: GrayImage, kernel: int) -> GrayImage:
    mean = cv2.blur(image, (kernel, kernel))
    mean_sq = cv2.blur(image * image, (kernel, kernel))
    variance = np.maximum(mean_sq - mean * mean, 0.0)
    return np.sqrt(variance).astype(np.float32)
