from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage

from .common import validate_mask_pairs


def _counts_at_thresholds(
    values: ArrayLike,
    ascending_thresholds: NDArray[np.float64],
) -> NDArray[np.int64]:
    """批量统计每个升序阈值下大于等于阈值的元素数量。

    Args:
        values (ArrayLike): 待统计的任意形状数值数组。
        ascending_thresholds (NDArray[np.float64]): 严格按升序排列的阈值向量。

    Returns:
        NDArray[np.int64]: 与阈值等长、按调用方原阈值方向排列的累计数量。
    """
    bucket_indexes = np.searchsorted(
        ascending_thresholds,
        np.asarray(values).reshape(-1),
        side="right",
    )
    bucket_counts = np.bincount(
        bucket_indexes,
        minlength=len(ascending_thresholds) + 1,
    )
    counts_at_ascending_thresholds = np.cumsum(bucket_counts[::-1])[::-1][1:]
    return counts_at_ascending_thresholds[::-1]


def compute_pro(
    prediction_masks: Sequence[ArrayLike],
    gt_masks: Sequence[ArrayLike],
    *,
    fpr_limit: float = 0.3,
    num_thresholds: int = 200,
    **_: object,
) -> dict[str, float]:
    """在可变原生分辨率掩码上计算有界 FPR 区间的 AUPRO。

    Args:
        prediction_masks (Sequence[ArrayLike]): 每张图的二维有限异常分数图。
        gt_masks (Sequence[ArrayLike]): 与预测逐项同形状的二维二值真值掩码。
        fpr_limit (float): 积分使用的最大假阳性率，范围 ``(0, 1]``。
        num_thresholds (int): 从最大分数到最小分数采样的阈值数量，至少为 ``2``。

    Returns:
        dict[str, float]: 包含归一化到 ``[0, 1]`` 的 ``pixel_pro``。

    Raises:
        TypeError: ``num_thresholds`` 不是整数或是布尔值。
        ValueError: 参数、掩码不合法，或数据没有背景像素/异常连通区域。
    """
    if not 0 < fpr_limit <= 1:
        raise ValueError("fpr_limit must be in (0, 1]")
    if isinstance(num_thresholds, bool) or not isinstance(num_thresholds, int):
        raise TypeError("num_thresholds must be an integer")
    if num_thresholds < 2:
        raise ValueError("num_thresholds must be at least 2")

    pairs = validate_mask_pairs(prediction_masks, gt_masks)
    minimum = min(float(prediction.min()) for prediction, _ in pairs)
    maximum = max(float(prediction.max()) for prediction, _ in pairs)
    margin = max(
        np.finfo(np.float64).eps,
        abs(maximum) * 1e-12,
        abs(minimum) * 1e-12,
    )
    thresholds = np.linspace(maximum + margin, minimum - margin, num_thresholds)
    ascending_thresholds = thresholds[::-1]
    false_positives = np.zeros(num_thresholds, dtype=np.int64)
    overlap_sums = np.zeros(num_thresholds, dtype=np.float64)
    background_count = 0
    region_count = 0

    for prediction, target in pairs:
        boolean_target = target.astype(bool, copy=False)
        background = ~boolean_target
        background_count += int(background.sum())
        false_positives += _counts_at_thresholds(
            prediction[background],
            ascending_thresholds,
        )

        components, _ = ndimage.label(boolean_target)
        for component_id, region_slice in enumerate(
            ndimage.find_objects(components),
            start=1,
        ):
            if region_slice is None:
                continue
            local_components = components[region_slice]
            region = local_components == component_id
            region_size = int(region.sum())
            overlap_sums += _counts_at_thresholds(
                prediction[region_slice][region],
                ascending_thresholds,
            ) / region_size
            region_count += 1

    if background_count == 0:
        raise ValueError("PRO requires at least one background pixel")
    if region_count == 0:
        raise ValueError("PRO requires at least one anomalous region")

    fprs = false_positives / background_count
    pros = overlap_sums / region_count

    order = np.argsort(fprs, kind="stable")
    sorted_fprs = np.asarray(fprs)[order]
    sorted_pros = np.asarray(pros)[order]
    unique_fprs = np.unique(sorted_fprs)
    envelope = np.asarray(
        [sorted_pros[sorted_fprs == fpr].max() for fpr in unique_fprs]
    )

    within = unique_fprs < fpr_limit
    curve_fprs = unique_fprs[within].tolist()
    curve_pros = envelope[within].tolist()
    curve_fprs.append(float(fpr_limit))
    curve_pros.append(float(np.interp(fpr_limit, unique_fprs, envelope)))
    if curve_fprs[0] > 0:
        curve_fprs.insert(0, 0.0)
        curve_pros.insert(0, 0.0)
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:
        trapezoid = np.trapz
    area = trapezoid(np.asarray(curve_pros), np.asarray(curve_fprs))
    return {"pixel_pro": float(np.clip(area / fpr_limit, 0.0, 1.0))}
