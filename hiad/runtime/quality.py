from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import cv2
import numpy as np
from numpy.typing import NDArray

from .contracts import ImageQualityResult


_QUALITY_KEYS: Final = (
    "min_mean_luminance",
    "max_mean_luminance",
    "max_clipped_fraction",
    "min_focus_variance",
)


def assess_image_quality(
    image: NDArray[np.generic],
    thresholds: Mapping[str, float],
    valid_mask: NDArray[np.generic] | None = None,
) -> ImageQualityResult:
    """在异常推理前检查 RGB 图像的曝光、截断比例和清晰度。

    Args:
        image (NDArray[np.generic]): ``(height, width, 3)`` HWC ``uint8`` RGB 图像。
        thresholds (Mapping[str, float]): 包含最小/最大平均亮度、最大截断比例和
            最小清晰度方差的有限阈值映射。
        valid_mask (NDArray[np.generic] | None): 可选二维有效区域；尺寸不一致时
            使用最近邻插值对齐到图像高宽，非零像素视为有效。

    Returns:
        ImageQualityResult: ``PASS`` 或 ``RECHECK`` 状态、原因码及三项质量指标。

    Raises:
        ValueError: 图像格式、阈值范围或有效掩码不符合约定。
    """
    values = np.asarray(image)
    if values.dtype != np.uint8 or values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("image must be an HWC uint8 RGB array")
    if not isinstance(thresholds, Mapping) or any(
        key not in thresholds for key in _QUALITY_KEYS
    ):
        raise ValueError("quality thresholds are incomplete")
    limits = {key: float(thresholds[key]) for key in _QUALITY_KEYS}
    if not all(np.isfinite(value) for value in limits.values()):
        raise ValueError("quality thresholds must be finite")
    if not 0 <= limits["min_mean_luminance"] < limits["max_mean_luminance"] <= 1:
        raise ValueError("luminance thresholds must satisfy 0 <= min < max <= 1")
    if not 0 <= limits["max_clipped_fraction"] <= 1:
        raise ValueError("max_clipped_fraction must be in [0, 1]")
    if limits["min_focus_variance"] < 0:
        raise ValueError("min_focus_variance must be non-negative")

    gray = cv2.cvtColor(values, cv2.COLOR_RGB2GRAY)
    normalized = gray.astype(np.float32) / 255.0
    if valid_mask is None:
        valid = np.ones(gray.shape, dtype=bool)
    else:
        valid = np.asarray(valid_mask)
        if valid.ndim != 2:
            raise ValueError("valid_mask must be two-dimensional")
        if valid.shape != gray.shape:
            valid = cv2.resize(
                valid.astype(np.uint8),
                (gray.shape[1], gray.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        valid = valid.astype(bool)
        if not np.any(valid):
            raise ValueError("valid_mask must select at least one pixel")
    valid_luminance = normalized[valid]
    mean_luminance = float(valid_luminance.mean())
    clipped_fraction = float(
        np.mean((valid_luminance <= 0.01) | (valid_luminance >= 0.99))
    )
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    focus_variance = float(laplacian[valid].var())
    reasons: list[str] = []
    if mean_luminance < limits["min_mean_luminance"]:
        reasons.append("mean_luminance_below_minimum")
    if mean_luminance > limits["max_mean_luminance"]:
        reasons.append("mean_luminance_above_maximum")
    if clipped_fraction > limits["max_clipped_fraction"]:
        reasons.append("clipped_fraction_above_maximum")
    if focus_variance < limits["min_focus_variance"]:
        reasons.append("focus_variance_below_minimum")
    return {
        "status": "RECHECK" if reasons else "PASS",
        "reasons": reasons,
        "mean_luminance": mean_luminance,
        "clipped_fraction": clipped_fraction,
        "focus_variance": focus_variance,
    }
