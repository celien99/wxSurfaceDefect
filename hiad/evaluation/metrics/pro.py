import numpy as np
from scipy import ndimage

from .common import validate_mask_pairs


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
    boolean_pairs = [
        (prediction, target.astype(bool, copy=False))
        for prediction, target in pairs
    ]
    background_count = sum(int((~target).sum()) for _, target in boolean_pairs)
    if background_count == 0:
        raise ValueError("PRO requires at least one background pixel")

    regions = []
    for pair_index, (_, target) in enumerate(boolean_pairs):
        components, component_count = ndimage.label(target)
        for component_id in range(1, component_count + 1):
            region = components == component_id
            regions.append((pair_index, region, int(region.sum())))
    if not regions:
        raise ValueError("PRO requires at least one anomalous region")

    all_values = np.concatenate(
        [prediction.reshape(-1) for prediction, _ in boolean_pairs]
    )
    minimum = float(all_values.min())
    maximum = float(all_values.max())
    margin = max(
        np.finfo(np.float64).eps,
        abs(maximum) * 1e-12,
        abs(minimum) * 1e-12,
    )
    thresholds = np.linspace(maximum + margin, minimum - margin, num_thresholds)

    fprs = []
    pros = []
    for threshold in thresholds:
        binary_predictions = [
            prediction >= threshold for prediction, _ in boolean_pairs
        ]
        false_positives = sum(
            int(np.logical_and(binary, ~target).sum())
            for binary, (_, target) in zip(binary_predictions, boolean_pairs)
        )
        overlaps = [
            float(binary_predictions[pair_index][region].sum()) / region_size
            for pair_index, region, region_size in regions
        ]
        fprs.append(false_positives / background_count)
        pros.append(float(np.mean(overlaps)))

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
    area = np.trapz(np.asarray(curve_pros), np.asarray(curve_fprs))
    return {"pixel_pro": float(np.clip(area / fpr_limit, 0.0, 1.0))}
