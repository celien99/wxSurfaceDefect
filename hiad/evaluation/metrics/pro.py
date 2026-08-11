import numpy as np
from scipy import ndimage

from .common import validate_mask_pairs


def _counts_at_thresholds(values, ascending_thresholds):
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
    prediction_masks,
    gt_masks,
    *,
    fpr_limit: float = 0.3,
    num_thresholds: int = 200,
    **kwargs,
):
    """Compute native-resolution AUPRO for variable-size mask pairs."""
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
