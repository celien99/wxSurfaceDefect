from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from hiad.models import TimmDinoV3Encoder

from .constants import MAX_SAM_POSITIVE_POINTS
from .dino import extract_dino_grid


def generate_registration_prompts(
    rgb: np.ndarray,
    *,
    encoder: TimmDinoV3Encoder,
    prototypes: Mapping[str, Any],
    template: Mapping[str, Any],
    reference_mask: np.ndarray,
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    dict[str, Any],
]:
    metrics: dict[str, Any] = {}

    def reject(reason: str):
        metrics["reason"] = reason
        return None, None, None, metrics

    if isinstance(rgb, np.ndarray) and rgb.ndim >= 2:
        image_height, image_width = rgb.shape[:2]
        image_diagonal = math.hypot(image_width, image_height)
        if not math.isfinite(image_diagonal) or image_diagonal <= 0:
            return reject("invalid_image_diagonal")

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
        return reject("invalid_image_diagonal")

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
    matched_similarities = similarities[
        matched_reference_indices,
        matched_input_indices,
    ]
    match_count = int(matched_reference_indices.numel())
    metrics["match_count"] = match_count
    del similarities, reference_features, input_features

    if match_count < config["min_dino_matches"]:
        return reject("insufficient_dino_matches")

    reference_foreground_indices = torch.flatnonzero(foreground_cells)
    reference_points = template["centers_xy"][
        reference_foreground_indices[matched_reference_indices]
    ].numpy()
    input_indices = matched_input_indices.numpy()
    input_points = input_centers_xy[input_indices]
    match_scores = matched_similarities.numpy()

    try:
        affine, inlier_mask = cv2.estimateAffinePartial2D(
            reference_points,
            input_points,
            method=cv2.RANSAC,
        )
    except cv2.error:
        return reject("affine_estimation_failed")
    if affine is None or inlier_mask is None or affine.shape != (2, 3):
        return reject("affine_estimation_failed")
    if not np.isfinite(affine).all():
        return reject("nonfinite_affine")
    linear_determinant = float(np.linalg.det(affine[:, :2]))
    if not math.isfinite(linear_determinant) or abs(linear_determinant) <= np.finfo(
        np.float64
    ).eps:
        return reject("singular_affine")

    inliers = np.asarray(inlier_mask).reshape(-1).astype(bool, copy=False)
    if inliers.shape != (match_count,):
        return reject("invalid_affine_inliers")
    inlier_count = int(np.count_nonzero(inliers))
    inlier_ratio = inlier_count / match_count
    metrics["inlier_ratio"] = float(inlier_ratio)
    if inlier_count == 0:
        return reject("empty_affine_inliers")

    projected_points = reference_points @ affine[:, :2].T + affine[:, 2]
    reprojection_errors = np.linalg.norm(
        projected_points[inliers] - input_points[inliers],
        axis=1,
    )
    mean_reprojection_error = float(np.mean(reprojection_errors))
    reprojection_ratio = mean_reprojection_error / image_diagonal
    if not math.isfinite(reprojection_ratio):
        return reject("nonfinite_reprojection")
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
        return reject("prior_warp_failed")
    if warped_prior.shape != (image_height, image_width):
        return reject("invalid_warped_prior_shape")
    warped_prior = warped_prior.view(np.bool_)
    occupied_rows = np.flatnonzero(warped_prior.any(axis=1))
    occupied_columns = np.flatnonzero(warped_prior.any(axis=0))
    if occupied_rows.size == 0 or occupied_columns.size == 0:
        return reject("empty_warped_prior")

    box_xyxy = np.array(
        [
            occupied_columns[0],
            occupied_rows[0],
            occupied_columns[-1],
            occupied_rows[-1],
        ],
        dtype=np.float32,
    )
    if (
        not np.isfinite(box_xyxy).all()
        or box_xyxy[2] <= box_xyxy[0]
        or box_xyxy[3] <= box_xyxy[1]
    ):
        return reject("invalid_prior_box")

    positive_points = []
    used_input_cells: set[int] = set()
    inlier_order = np.flatnonzero(inliers)
    inlier_order = inlier_order[np.argsort(-match_scores[inlier_order], kind="stable")]
    for match_index in inlier_order:
        input_cell = int(input_indices[match_index])
        if input_cell in used_input_cells:
            continue
        point_x, point_y = input_points[match_index]
        if (
            not np.isfinite((point_x, point_y)).all()
            or point_x < 0
            or point_x >= image_width
            or point_y < 0
            or point_y >= image_height
        ):
            continue
        pixel_x = int(np.rint(point_x))
        pixel_y = int(np.rint(point_y))
        if (
            pixel_x < 0
            or pixel_x >= image_width
            or pixel_y < 0
            or pixel_y >= image_height
            or not warped_prior[pixel_y, pixel_x]
        ):
            continue
        positive_points.append((float(point_x), float(point_y)))
        used_input_cells.add(input_cell)
        if len(positive_points) == MAX_SAM_POSITIVE_POINTS:
            break
    if not positive_points:
        return reject("empty_positive_prompts")

    positive_points_xy = np.ascontiguousarray(positive_points, dtype=np.float32)
    metrics["reason"] = None
    return box_xyxy, positive_points_xy, warped_prior, metrics
