from __future__ import annotations

import math

import cv2
import numpy as np

from hiad.data import HRImageIndex


def _robust_unit_map(values: np.ndarray) -> np.ndarray:
    lower = float(np.quantile(values, 0.5))
    upper = float(np.quantile(values, 0.995))
    if upper <= lower:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lower) / (upper - lower), 0.0, 1.0).astype(np.float32)


def build_routing_map(local_map, global_context_map, global_weight: float) -> np.ndarray:
    """Build a scale-independent routing prior without changing local evidence."""
    local = np.asarray(local_map, dtype=np.float32)
    global_context = np.asarray(global_context_map, dtype=np.float32)
    if (
        local.ndim != 2
        or local.shape != global_context.shape
        or not np.isfinite(local).all()
        or not np.isfinite(global_context).all()
    ):
        raise ValueError("local and global context maps must be aligned finite 2D arrays")
    weight = float(global_weight)
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("global_weight must be finite and in [0, 1]")
    return (
        (1.0 - weight) * _robust_unit_map(local)
        + weight * _robust_unit_map(global_context)
    ).astype(np.float32)


def _validate_selection_arguments(
    anomaly_map, threshold, tile_size, min_area, safety_fraction
) -> np.ndarray:
    values = np.asarray(anomaly_map, dtype=np.float32)
    if values.ndim != 2 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("anomaly_map must be a non-empty finite two-dimensional array")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if isinstance(tile_size, bool) or not isinstance(tile_size, int) or tile_size <= 0:
        raise ValueError("tile_size must be a positive integer")
    if isinstance(min_area, bool) or not isinstance(min_area, int) or min_area <= 0:
        raise ValueError("min_area must be a positive integer")
    if (
        isinstance(safety_fraction, bool)
        or not isinstance(safety_fraction, (int, float))
        or not np.isfinite(safety_fraction)
        or safety_fraction <= 0
        or safety_fraction > 1
    ):
        raise ValueError("safety_fraction must be finite and in the range (0, 1]")
    return values


def _tile_for_center(center_x, center_y, tile_size, image_width, image_height):
    x = min(max(math.floor(center_x - tile_size / 2), 0), max(image_width - tile_size, 0))
    y = min(max(math.floor(center_y - tile_size / 2), 0), max(image_height - tile_size, 0))
    return HRImageIndex(x=x, y=y, width=tile_size, height=tile_size)


def _tile_axis_starts(length: int, tile_size: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length, tile_size))
    starts[-1] = length - tile_size
    return list(dict.fromkeys(starts))


def select_refinement_regions(
    anomaly_map, threshold, tile_size, min_area, safety_fraction
) -> list[HRImageIndex]:
    """Select suspicious tiles plus deterministic native-coordinate safety tiles."""
    values = _validate_selection_arguments(
        anomaly_map, threshold, tile_size, min_area, safety_fraction
    )
    image_height, image_width = values.shape
    selected = []
    binary = np.zeros(values.shape, dtype=np.uint8)
    if float(values.max()) > float(values.min()):
        binary = np.asarray(values >= threshold, dtype=np.uint8)
    component_count, _, statistics, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    for component_index in range(1, component_count):
        if int(statistics[component_index, cv2.CC_STAT_AREA]) < min_area:
            continue
        left = int(statistics[component_index, cv2.CC_STAT_LEFT])
        top = int(statistics[component_index, cv2.CC_STAT_TOP])
        width = int(statistics[component_index, cv2.CC_STAT_WIDTH])
        height = int(statistics[component_index, cv2.CC_STAT_HEIGHT])
        for y in range(top, top + height, tile_size):
            for x in range(left, left + width, tile_size):
                selected.append(
                    _tile_for_center(
                        min(x + tile_size / 2, left + width - 0.5),
                        min(y + tile_size / 2, top + height - 0.5),
                        tile_size,
                        image_width,
                        image_height,
                    )
                )

    selected = list(dict.fromkeys(selected))
    safety_tiles = [
        HRImageIndex(x=x, y=y, width=tile_size, height=tile_size)
        for y in _tile_axis_starts(image_height, tile_size)
        for x in _tile_axis_starts(image_width, tile_size)
    ]
    safety_count = max(1, math.ceil(len(safety_tiles) * float(safety_fraction)))
    safety_indexes = np.linspace(
        0, len(safety_tiles) - 1, num=safety_count, dtype=np.intp
    )
    for safety_index in safety_indexes:
        tile = safety_tiles[int(safety_index)]
        if tile not in selected:
            selected.append(tile)
    return selected


def merge_refinement_maps(base_map, refinements, image_size) -> np.ndarray:
    """Merge refinement predictions by their source-image coordinates."""
    base = np.asarray(base_map, dtype=np.float32)
    if base.ndim != 2 or base.size == 0 or not np.isfinite(base).all():
        raise ValueError("base_map must be a non-empty finite two-dimensional array")
    if (
        not isinstance(image_size, (tuple, list))
        or len(image_size) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in image_size)
    ):
        raise ValueError("image_size must contain two positive integers")
    image_width, image_height = image_size
    if base.shape != (image_height, image_width):
        raise ValueError("base_map shape must match image_size")

    merged = np.array(base, copy=True)
    for refinement in refinements:
        if not isinstance(refinement, tuple) or len(refinement) != 2:
            raise TypeError("Each refinement must be an (HRImageIndex, anomaly_map) tuple")
        index, prediction = refinement
        if not isinstance(index, HRImageIndex):
            raise TypeError("Each refinement index must be an HRImageIndex")
        if index.x < 0 or index.y < 0 or index.width <= 0 or index.height <= 0:
            raise ValueError("Refinement index has invalid geometry")
        if index.x >= image_width or index.y >= image_height:
            raise ValueError("Refinement index origin is outside image_size")
        prediction = np.asarray(prediction, dtype=np.float32)
        if prediction.ndim != 2 or prediction.size == 0 or not np.isfinite(prediction).all():
            raise ValueError("Refinement anomaly maps must be non-empty finite arrays")
        if prediction.shape != (index.height, index.width):
            prediction = cv2.resize(
                prediction,
                (index.width, index.height),
                interpolation=cv2.INTER_LINEAR,
            )
        valid_width = min(index.width, image_width - index.x)
        valid_height = min(index.height, image_height - index.y)
        target = merged[index.y:index.y + valid_height, index.x:index.x + valid_width]
        merged[index.y:index.y + valid_height, index.x:index.x + valid_width] = np.maximum(
            target,
            prediction[:valid_height, :valid_width],
        )
    return merged
