from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np
from scipy import ndimage


class MaskRejected(RuntimeError):
    def __init__(self, reason: str, metrics: Mapping[str, float] | None = None):
        super().__init__(reason)
        self.metrics = dict(metrics or {})


def clean_warped_mask(
    warped_mask: np.ndarray,
    boundary_expand_ratio: float,
) -> np.ndarray:
    if (
        not isinstance(warped_mask, np.ndarray)
        or warped_mask.dtype != np.bool_
        or warped_mask.ndim != 2
        or not warped_mask.any()
    ):
        raise MaskRejected("invalid_mask_cleanup_input")

    occupied_rows = np.flatnonzero(warped_mask.any(axis=1))
    occupied_columns = np.flatnonzero(warped_mask.any(axis=0))
    if occupied_rows.size == 0 or occupied_columns.size == 0:
        raise MaskRejected("invalid_cleanup_prior_box")
    prior_height = int(occupied_rows[-1] - occupied_rows[0] + 1)
    prior_width = int(occupied_columns[-1] - occupied_columns[0] + 1)
    expand_radius = int(
        math.ceil(boundary_expand_ratio * max(prior_width, prior_height))
    )

    cleaned = np.ascontiguousarray(warped_mask, dtype=np.bool_)
    kernel = None
    if expand_radius > 0:
        kernel_size = 2 * expand_radius + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        closed = cv2.morphologyEx(
            cleaned.view(np.uint8),
            cv2.MORPH_CLOSE,
            kernel,
        ).view(np.bool_)
        cleaned = np.logical_or(cleaned, closed)

    cleaned = np.ascontiguousarray(ndimage.binary_fill_holes(cleaned), dtype=np.bool_)
    if expand_radius > 0:
        expanded = cv2.dilate(
            cleaned.view(np.uint8),
            kernel,
            iterations=1,
        ).view(np.bool_)
        cleaned = np.logical_or(cleaned, expanded)
    cleaned = np.ascontiguousarray(ndimage.binary_fill_holes(cleaned), dtype=np.bool_)

    hole_filled = ndimage.binary_fill_holes(cleaned)
    if not np.array_equal(hole_filled, cleaned):
        raise MaskRejected("mask_cleanup_left_internal_holes")
    if not cleaned.any():
        raise MaskRejected("empty_cleaned_mask")
    return cleaned


def validate_warped_mask(
    warped_mask: np.ndarray,
    reference_mask: np.ndarray,
    config: Mapping[str, Any],
    *,
    affine_abs_det: float,
) -> tuple[np.ndarray, dict[str, float]]:
    metrics: dict[str, float] = {}

    def reject(reason: str) -> None:
        raise MaskRejected(reason, metrics)

    if (
        not isinstance(warped_mask, np.ndarray)
        or not isinstance(reference_mask, np.ndarray)
        or warped_mask.dtype != np.bool_
        or reference_mask.dtype != np.bool_
        or warped_mask.ndim != 2
        or reference_mask.ndim != 2
    ):
        reject("invalid_gate_mask_geometry")
    if (
        isinstance(affine_abs_det, bool)
        or not isinstance(affine_abs_det, (int, float))
        or not math.isfinite(float(affine_abs_det))
        or float(affine_abs_det) <= 0
    ):
        reject("invalid_affine_abs_det")

    image_height, image_width = warped_mask.shape
    warped_area = int(np.count_nonzero(warped_mask))
    reference_area = int(np.count_nonzero(reference_mask))
    if warped_area == 0 or reference_area == 0:
        reject("zero_area_gate_mask")

    # Scale-normalize: expected input area ≈ reference_area * |det(A[:2,:2])|.
    abs_det = float(affine_abs_det)
    expected_input_area = reference_area * abs_det
    metrics["affine_abs_det"] = abs_det
    metrics["expected_input_area"] = float(expected_input_area)

    reference_coverage = warped_area / expected_input_area
    if not math.isfinite(reference_coverage):
        reject("nonfinite_reference_coverage")
    metrics["reference_coverage"] = float(reference_coverage)
    if reference_coverage < config["min_reference_coverage"]:
        reject("reference_coverage_below_threshold")

    area_ratio_deviation = abs(reference_coverage - 1.0)
    if not math.isfinite(area_ratio_deviation):
        reject("nonfinite_area_ratio_deviation")
    metrics["area_ratio_deviation"] = float(area_ratio_deviation)
    if area_ratio_deviation > config["max_area_ratio_deviation"]:
        reject("area_ratio_deviation_above_threshold")

    try:
        cleaned_mask = clean_warped_mask(
            warped_mask,
            config["boundary_expand_ratio"],
        )
    except MaskRejected as error:
        raise MaskRejected(str(error), metrics) from error
    if (
        cleaned_mask.dtype != np.bool_
        or cleaned_mask.shape != (image_height, image_width)
        or not cleaned_mask.any()
    ):
        reject("invalid_cleaned_mask")
    if not np.array_equal(ndimage.binary_fill_holes(cleaned_mask), cleaned_mask):
        reject("cleaned_mask_contains_holes")

    occupied_rows = np.flatnonzero(cleaned_mask.any(axis=1))
    occupied_columns = np.flatnonzero(cleaned_mask.any(axis=0))
    if occupied_rows.size == 0 or occupied_columns.size == 0:
        reject("empty_cleaned_mask_bounds")
    cleaned_box_xyxy = np.array(
        [
            occupied_columns[0],
            occupied_rows[0],
            occupied_columns[-1],
            occupied_rows[-1],
        ],
        dtype=np.float64,
    )
    if (
        not np.isfinite(cleaned_box_xyxy).all()
        or cleaned_box_xyxy[0] < 0
        or cleaned_box_xyxy[1] < 0
        or cleaned_box_xyxy[2] >= image_width
        or cleaned_box_xyxy[3] >= image_height
        or cleaned_box_xyxy[2] <= cleaned_box_xyxy[0]
        or cleaned_box_xyxy[3] <= cleaned_box_xyxy[1]
    ):
        reject("invalid_cleaned_mask_bounds")
    return cleaned_mask, metrics
