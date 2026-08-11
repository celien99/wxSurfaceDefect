import numpy as np
import pytest

from hiad.evaluation.metrics.numpy_backend import (
    compute_pixelwise_metrics as compute_exact_pixelwise_metrics,
)
from hiad.evaluation.metrics.pro import compute_pro
from hiad.evaluation.metrics.torch_backend import compute_pixelwise_metrics


def test_streaming_pixel_metrics_match_exact_metrics_for_separated_scores():
    prediction_masks = [
        np.array([[0.1, 0.4], [0.8, 0.9]], dtype=np.float32),
        np.array([[0.2, 0.3], [0.6, 0.7]], dtype=np.float32),
    ]
    gt_masks = [
        np.array([[0, 0], [1, 1]], dtype=np.uint8),
        np.array([[0, 0], [1, 1]], dtype=np.uint8),
    ]

    expected = compute_exact_pixelwise_metrics(prediction_masks, gt_masks)
    actual = compute_pixelwise_metrics(
        prediction_masks,
        gt_masks,
        device="cpu",
        num_thresholds=1024,
    )

    for metric in ("pixel_auroc", "pixel_ap", "pixel_f1"):
        assert actual[metric] == pytest.approx(expected[metric])
    assert actual["seg_threshold"] == pytest.approx(
        expected["seg_threshold"],
        abs=0.001,
    )


def test_streaming_pixel_metrics_handle_constant_scores():
    scores = [np.ones((2, 2), dtype=np.float32)]
    labels = [np.array([[0, 0], [1, 1]], dtype=np.uint8)]

    actual = compute_pixelwise_metrics(
        scores,
        labels,
        device="cpu",
        num_thresholds=8,
    )

    assert actual == pytest.approx(
        {
            "pixel_auroc": 0.5,
            "pixel_ap": 0.5,
            "pixel_f1": 2 / 3,
            "seg_threshold": 1.0,
        }
    )


def _brute_force_pro(prediction_masks, gt_masks, fpr_limit, num_thresholds):
    minimum = min(float(mask.min()) for mask in prediction_masks)
    maximum = max(float(mask.max()) for mask in prediction_masks)
    margin = max(
        np.finfo(np.float64).eps,
        abs(maximum) * 1e-12,
        abs(minimum) * 1e-12,
    )
    thresholds = np.linspace(maximum + margin, minimum - margin, num_thresholds)
    background_count = sum(int((target == 0).sum()) for target in gt_masks)
    regions = []
    from scipy import ndimage

    for pair_index, target in enumerate(gt_masks):
        components, component_count = ndimage.label(target)
        for component_id in range(1, component_count + 1):
            region = components == component_id
            regions.append((pair_index, region, int(region.sum())))

    fprs = []
    pros = []
    for threshold in thresholds:
        binaries = [prediction >= threshold for prediction in prediction_masks]
        fprs.append(
            sum(
                int(np.logical_and(binary, target == 0).sum())
                for binary, target in zip(binaries, gt_masks)
            ) / background_count
        )
        pros.append(np.mean([
            binaries[pair_index][region].sum() / region_size
            for pair_index, region, region_size in regions
        ]))

    order = np.argsort(fprs, kind="stable")
    sorted_fprs = np.asarray(fprs)[order]
    sorted_pros = np.asarray(pros)[order]
    unique_fprs = np.unique(sorted_fprs)
    envelope = np.asarray([
        sorted_pros[sorted_fprs == fpr].max() for fpr in unique_fprs
    ])
    within = unique_fprs < fpr_limit
    curve_fprs = unique_fprs[within].tolist() + [fpr_limit]
    curve_pros = envelope[within].tolist() + [
        float(np.interp(fpr_limit, unique_fprs, envelope))
    ]
    if curve_fprs[0] > 0:
        curve_fprs.insert(0, 0.0)
        curve_pros.insert(0, 0.0)
    return np.trapezoid(curve_pros, curve_fprs) / fpr_limit


def test_streaming_pro_matches_original_threshold_evaluation():
    prediction_masks = [
        np.array(
            [[0.0, 0.2, 0.4], [0.1, 0.8, 0.7], [0.3, 0.9, 0.6]],
            dtype=np.float32,
        ),
        np.array(
            [[0.9, 0.1, 0.2], [0.8, 0.3, 0.0], [0.7, 0.4, 0.5]],
            dtype=np.float32,
        ),
    ]
    gt_masks = [
        np.array([[0, 0, 0], [0, 1, 1], [0, 1, 0]], dtype=np.uint8),
        np.array([[1, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=np.uint8),
    ]
    fpr_limit = 0.4
    num_thresholds = 11

    expected = _brute_force_pro(
        prediction_masks,
        gt_masks,
        fpr_limit,
        num_thresholds,
    )
    actual = compute_pro(
        prediction_masks,
        gt_masks,
        fpr_limit=fpr_limit,
        num_thresholds=num_thresholds,
    )["pixel_pro"]

    assert actual == pytest.approx(expected)
