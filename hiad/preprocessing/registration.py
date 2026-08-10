from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from hiad.models import TimmDinoV3Encoder

from .dino import extract_dino_grid
from .masks import MaskRejected


def register_and_warp_mask(
    rgb: np.ndarray,
    *,
    encoder: TimmDinoV3Encoder,
    prototypes: Mapping[str, Any],
    template: Mapping[str, Any],
    reference_mask: np.ndarray,
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    metrics: dict[str, Any] = {}

    def reject(reason: str) -> None:
        metrics["reason"] = reason
        raise MaskRejected(reason, metrics)

    if isinstance(rgb, np.ndarray) and rgb.ndim >= 2:
        image_height, image_width = rgb.shape[:2]
        image_diagonal = math.hypot(image_width, image_height)
        if not math.isfinite(image_diagonal) or image_diagonal <= 0:
            reject("invalid_image_diagonal")

    input_features, input_centers_xy, _, _ = extract_dino_grid(
        encoder,
        rgb,
        config,
        device,
        template["working_longest_edge"],
    )
    image_height, image_width = rgb.shape[:2]
    image_diagonal = math.hypot(image_width, image_height)
    if not math.isfinite(image_diagonal) or image_diagonal <= 0:
        reject("invalid_image_diagonal")

    foreground_cells = template["foreground_cells"]
    reference_features = F.normalize(
        template["features"][foreground_cells].float(),
        dim=1,
    )
    input_features = F.normalize(input_features.float(), dim=1)
    similarities = reference_features @ input_features.T

    reference_to_input = similarities.argmax(dim=1)
    input_to_reference = similarities.argmax(dim=0)
    reference_indices = torch.arange(reference_features.shape[0], dtype=torch.long)
    mutual = input_to_reference[reference_to_input] == reference_indices

    foreground_prototype = F.normalize(prototypes["foreground"], dim=0)
    background_prototype = F.normalize(prototypes["background"], dim=0)
    prototype_foreground = (
        input_features @ foreground_prototype > input_features @ background_prototype
    )
    mutual &= prototype_foreground[reference_to_input]

    matched_reference_indices = reference_indices[mutual]
    matched_input_indices = reference_to_input[mutual]
    match_count = int(matched_reference_indices.numel())
    metrics["match_count"] = match_count
    del similarities, reference_features, input_features

    if match_count < config["min_dino_matches"]:
        reject("insufficient_dino_matches")

    # torch has no flatnonzero; match numpy.flatnonzero semantics.
    reference_foreground_indices = torch.nonzero(
        foreground_cells.reshape(-1),
        as_tuple=True,
    )[0]
    reference_points = template["centers_xy"][
        reference_foreground_indices[matched_reference_indices]
    ].numpy()
    input_points = input_centers_xy[matched_input_indices.numpy()]

    try:
        affine, inlier_mask = cv2.estimateAffinePartial2D(
            reference_points,
            input_points,
            method=cv2.RANSAC,
        )
    except cv2.error:
        reject("affine_estimation_failed")
    if affine is None or inlier_mask is None or affine.shape != (2, 3):
        reject("affine_estimation_failed")
    if not np.isfinite(affine).all():
        reject("nonfinite_affine")
    linear_determinant = float(np.linalg.det(affine[:, :2]))
    if not math.isfinite(linear_determinant) or abs(linear_determinant) <= np.finfo(
        np.float64
    ).eps:
        reject("singular_affine")
    metrics["affine_abs_det"] = float(abs(linear_determinant))

    inliers = np.asarray(inlier_mask).reshape(-1).astype(bool, copy=False)
    if inliers.shape != (match_count,):
        reject("invalid_affine_inliers")
    inlier_count = int(np.count_nonzero(inliers))
    inlier_ratio = inlier_count / match_count
    metrics["inlier_ratio"] = float(inlier_ratio)
    if inlier_count == 0:
        reject("empty_affine_inliers")

    projected_points = reference_points @ affine[:, :2].T + affine[:, 2]
    reprojection_errors = np.linalg.norm(
        projected_points[inliers] - input_points[inliers],
        axis=1,
    )
    mean_reprojection_error = float(np.mean(reprojection_errors))
    reprojection_ratio = mean_reprojection_error / image_diagonal
    if not math.isfinite(reprojection_ratio):
        reject("nonfinite_reprojection")
    metrics["reprojection_ratio"] = float(reprojection_ratio)

    try:
        warped_prior = cv2.warpAffine(
            reference_mask.view(np.uint8),
            affine,
            (image_width, image_height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    except cv2.error:
        reject("prior_warp_failed")
    if warped_prior.shape != (image_height, image_width):
        reject("invalid_warped_prior_shape")
    warped_prior = warped_prior.view(np.bool_)
    occupied_rows = np.flatnonzero(warped_prior.any(axis=1))
    occupied_columns = np.flatnonzero(warped_prior.any(axis=0))
    if occupied_rows.size == 0 or occupied_columns.size == 0:
        reject("empty_warped_prior")

    metrics["reason"] = None
    return warped_prior.astype(bool, copy=False), metrics
