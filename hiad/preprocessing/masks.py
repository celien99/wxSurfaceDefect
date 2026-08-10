from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np
from scipy import ndimage

from hiad.foreground import compute_reference_coverage


class MaskRejected(RuntimeError):
    def __init__(self, reason: str, metrics: Mapping[str, float] | None = None):
        super().__init__(reason)
        self.metrics = dict(metrics or {})


def clean_foreground_mask(
    sam_mask: np.ndarray,
    warped_prior: np.ndarray,
    boundary_expand_ratio: float,
) -> np.ndarray:
    if (
        not isinstance(sam_mask, np.ndarray)
        or not isinstance(warped_prior, np.ndarray)
        or sam_mask.dtype != np.bool_
        or warped_prior.dtype != np.bool_
        or sam_mask.ndim != 2
        or warped_prior.shape != sam_mask.shape
        or not sam_mask.any()
        or not warped_prior.any()
    ):
        raise MaskRejected("invalid_mask_cleanup_input")

    occupied_rows = np.flatnonzero(warped_prior.any(axis=1))
    occupied_columns = np.flatnonzero(warped_prior.any(axis=0))
    if occupied_rows.size == 0 or occupied_columns.size == 0:
        raise MaskRejected("invalid_cleanup_prior_box")
    prior_height = int(occupied_rows[-1] - occupied_rows[0] + 1)
    prior_width = int(occupied_columns[-1] - occupied_columns[0] + 1)
    expand_radius = int(
        math.ceil(boundary_expand_ratio * max(prior_width, prior_height))
    )

    cleaned = np.logical_or(sam_mask, warped_prior)
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
        np.logical_or(cleaned, closed, out=cleaned)

    cleaned = np.ascontiguousarray(ndimage.binary_fill_holes(cleaned), dtype=np.bool_)
    if expand_radius > 0:
        expanded = cv2.dilate(
            cleaned.view(np.uint8),
            kernel,
            iterations=1,
        ).view(np.bool_)
        np.logical_or(cleaned, expanded, out=cleaned)
    cleaned = np.ascontiguousarray(ndimage.binary_fill_holes(cleaned), dtype=np.bool_)

    hole_filled = ndimage.binary_fill_holes(cleaned)
    if not np.array_equal(hole_filled, cleaned):
        raise MaskRejected("mask_cleanup_left_internal_holes")
    if not cleaned.any():
        raise MaskRejected("empty_cleaned_mask")
    return cleaned


def validate_and_clean_mask(
    sam_mask: np.ndarray,
    warped_prior: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, float]]:
    metrics: dict[str, float] = {}

    def reject(reason: str) -> None:
        raise MaskRejected(reason, metrics)

    if (
        sam_mask.dtype != np.bool_
        or warped_prior.dtype != np.bool_
        or sam_mask.ndim != 2
        or warped_prior.shape != sam_mask.shape
    ):
        reject("invalid_gate_mask_geometry")
    image_height, image_width = sam_mask.shape
    sam_area = int(np.count_nonzero(sam_mask))
    prior_area = int(np.count_nonzero(warped_prior))
    if sam_area == 0 or prior_area == 0:
        reject("zero_area_gate_mask")

    intersection = int(np.count_nonzero(sam_mask & warped_prior))
    reference_coverage = compute_reference_coverage(sam_mask, warped_prior)
    if not math.isfinite(reference_coverage):
        reject("nonfinite_reference_coverage")
    metrics["reference_coverage"] = float(reference_coverage)
    if reference_coverage < config["min_reference_coverage"]:
        reject("reference_coverage_below_threshold")

    union = sam_area + prior_area - intersection
    if union <= 0:
        reject("zero_union_gate_mask")
    sam_prior_iou = intersection / union
    area_ratio_deviation = abs(sam_area / prior_area - 1.0)
    if not math.isfinite(sam_prior_iou) or not math.isfinite(area_ratio_deviation):
        reject("nonfinite_sam_gate_metrics")
    metrics["sam_prior_iou"] = float(sam_prior_iou)
    metrics["area_ratio_deviation"] = float(area_ratio_deviation)
    if sam_prior_iou < config["min_sam_prior_iou"]:
        reject("sam_prior_iou_below_threshold")
    if area_ratio_deviation > config["max_area_ratio_deviation"]:
        reject("sam_area_ratio_deviation_above_threshold")

    try:
        cleaned_mask = clean_foreground_mask(
            sam_mask,
            warped_prior,
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
