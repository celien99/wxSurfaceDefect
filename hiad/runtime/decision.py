from __future__ import annotations

import math

import cv2
import numpy as np
from numpy.typing import ArrayLike

from .contracts import (
    ComponentStatistics,
    DecisionState,
    StrongestComponent,
)


def _empty_statistics() -> ComponentStatistics:
    """创建没有阈值内异常区域时的稳定摘要结构。"""
    return {
        "component_count": 0,
        "anomalous_pixel_count": 0,
        "largest_component_area": 0,
        "strongest_component": None,
    }


def component_statistics(
    anomaly_map: ArrayLike,
    pixel_threshold: float,
) -> ComponentStatistics:
    """在校准像素阈值上汇总八连通异常区域。

    Args:
        anomaly_map (ArrayLike): 原图分辨率的非空二维异常分数图。非有限像素
            不参与连通区域统计。
        pixel_threshold (float): 像素异常阈值，分数大于等于该值视为异常。

    Returns:
        ComponentStatistics: 连通区域数量、异常像素总数、最大面积及最强区域；
        外接框使用原图像素 ``xywh`` 坐标。

    Raises:
        ValueError: 异常图不是非空二维数组，或阈值不是有限数。
    """
    threshold = float(pixel_threshold)
    values = np.asarray(anomaly_map, dtype=np.float32)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("anomaly_map must be a non-empty two-dimensional array")
    if not np.isfinite(threshold):
        raise ValueError("pixel_threshold must be finite")

    binary = np.asarray(np.isfinite(values) & (values >= threshold), dtype=np.uint8)
    component_count, labels, statistics, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    if component_count == 1:
        return _empty_statistics()

    active = binary.astype(bool)
    active_labels = labels[active]
    active_values = values[active].astype(np.float64, copy=False)
    sums = np.bincount(
        active_labels,
        weights=active_values,
        minlength=component_count,
    )
    maxima = np.full(component_count, -np.inf, dtype=np.float64)
    np.maximum.at(maxima, active_labels, active_values)

    strongest_component: StrongestComponent | None = None
    strongest_score = -math.inf
    total_pixels = float(values.size)
    for label in range(1, component_count):
        area = int(statistics[label, cv2.CC_STAT_AREA])
        area_fraction = area / total_pixels
        mean_score = float(sums[label] / area)
        # 面积比例的加性奖励让连续弱异常优先于孤立尖峰，同时保持分辨率缩放不变性。
        component_score = mean_score + math.sqrt(area_fraction)
        if component_score <= strongest_score:
            continue
        strongest_score = component_score
        strongest_component = {
            "area": area,
            "area_fraction": area_fraction,
            "mean_score": mean_score,
            "max_score": float(maxima[label]),
            "score": component_score,
            "bbox_xywh": [
                int(statistics[label, cv2.CC_STAT_LEFT]),
                int(statistics[label, cv2.CC_STAT_TOP]),
                int(statistics[label, cv2.CC_STAT_WIDTH]),
                int(statistics[label, cv2.CC_STAT_HEIGHT]),
            ],
        }

    areas = statistics[1:, cv2.CC_STAT_AREA]
    return {
        "component_count": component_count - 1,
        "anomalous_pixel_count": int(binary.sum()),
        "largest_component_area": int(areas.max()),
        "strongest_component": strongest_component,
    }


def image_score_from_statistics(
    statistics: ComponentStatistics,
    fallback_score: float,
) -> float:
    """组合连通区域摘要与模型全局分数，保留更保守的结果。

    Args:
        statistics (ComponentStatistics): 最终异常图的连通区域摘要。
        fallback_score (float): 模型或 Top-K 产生的备用图像级分数；非有限值按
            ``0.0`` 处理。

    Returns:
        float: 最强组件分数与备用分数的较大值。
    """
    fallback = float(fallback_score)
    if not math.isfinite(fallback):
        fallback = 0.0
    strongest = statistics["strongest_component"]
    if strongest is None:
        return fallback
    return float(max(fallback, strongest["score"]))


def top_k_map_score(anomaly_map: ArrayLike, top_k: int) -> float:
    """从最终融合图计算不依赖整图均值的局部 Top-K 分数。

    Args:
        anomaly_map (ArrayLike): 任意形状的有限异常分数数组。
        top_k (int): 参与均值的最高分像素数量；超过像素总数时使用全部像素。

    Returns:
        float: 最高 ``top_k`` 个异常分数的均值。

    Raises:
        ValueError: 异常图为空、包含非有限值，或 ``top_k`` 不是正整数。
    """
    values = np.asarray(anomaly_map, dtype=np.float32).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("anomaly_map must contain finite values")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    count = min(top_k, values.size)
    return float(np.partition(values, values.size - count)[-count:].mean())


def classify_score(
    score: float,
    threshold: float,
) -> DecisionState:
    """根据校准阈值生成二分类判定。

    Args:
        score (float): 待判定的图像或组件异常分数。
        threshold (float): 正常样本校准阈值。
    Returns:
        DecisionState: 不超过阈值为 ``OK``，超过阈值为 ``NG``；分数或阈值
        非有限时保守返回 ``NG``。
    """
    score = float(score)
    threshold = float(threshold)
    if not (math.isfinite(score) and math.isfinite(threshold)):
        return "NG"
    if score <= threshold:
        return "OK"
    return "NG"
