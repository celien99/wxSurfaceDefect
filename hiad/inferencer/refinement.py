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


def _tile_axis_starts(length: int, tile_size: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length, tile_size))
    starts[-1] = length - tile_size
    return list(dict.fromkeys(starts))


def _radical_inverse(index: int, base: int) -> float:
    result = 0.0
    fraction = 1.0 / base
    while index:
        result += (index % base) * fraction
        index //= base
        fraction /= base
    return result


def _spatial_safety_indexes(row_count: int, column_count: int, count: int) -> list[int]:
    total = row_count * column_count
    if count >= total:
        return list(range(total))
    selected = []
    seen = set()
    sequence_index = 1
    while len(selected) < count:
        row = min(int(_radical_inverse(sequence_index, 2) * row_count), row_count - 1)
        column = min(
            int(_radical_inverse(sequence_index, 3) * column_count),
            column_count - 1,
        )
        flattened = row * column_count + column
        if flattened not in seen:
            seen.add(flattened)
            selected.append(flattened)
        sequence_index += 1
    return selected


def select_refinement_regions(
    anomaly_map, threshold, tile_size, min_area, safety_fraction
) -> list[HRImageIndex]:
    """Select suspicious tiles plus deterministic native-coordinate safety tiles."""
    values = _validate_selection_arguments(
        anomaly_map, threshold, tile_size, min_area, safety_fraction
    )
    image_height, image_width = values.shape
    x_starts = _tile_axis_starts(image_width, tile_size)
    y_starts = _tile_axis_starts(image_height, tile_size)
    binary = np.zeros(values.shape, dtype=np.uint8)
    if float(values.max()) > float(values.min()):
        binary = np.asarray(values >= threshold, dtype=np.uint8)
    component_count, labels, statistics, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    valid_components = np.zeros(component_count, dtype=bool)
    valid_components[1:] = statistics[1:, cv2.CC_STAT_AREA] >= min_area
    candidate_y, candidate_x = np.nonzero(valid_components[labels])
    occupied_tiles = set()
    if candidate_x.size:
        x_indexes = np.searchsorted(x_starts, candidate_x, side="right") - 1
        y_indexes = np.searchsorted(y_starts, candidate_y, side="right") - 1
        occupied_tiles.update(zip(y_indexes.tolist(), x_indexes.tolist()))
    selected = [
        HRImageIndex(
            x=x_starts[x_index],
            y=y_starts[y_index],
            width=tile_size,
            height=tile_size,
        )
        for y_index, x_index in sorted(occupied_tiles)
    ]
    safety_tiles = [
        HRImageIndex(x=x, y=y, width=tile_size, height=tile_size)
        for y in y_starts
        for x in x_starts
    ]
    safety_count = max(1, math.ceil(len(safety_tiles) * float(safety_fraction)))
    safety_indexes = _spatial_safety_indexes(
        len(y_starts),
        len(x_starts),
        safety_count,
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

    accumulated = np.zeros_like(base, dtype=np.float64)
    weight_map = np.zeros_like(base, dtype=np.float64)
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
        row_hann = np.hanning(index.height) if index.height > 1 else np.ones(1)
        column_hann = np.hanning(index.width) if index.width > 1 else np.ones(1)
        if row_hann.max() > 0:
            row_hann /= row_hann.max()
        if column_hann.max() > 0:
            column_hann /= column_hann.max()
        weights = 0.05 + 0.95 * np.outer(row_hann, column_hann)
        target_slice = (
            slice(index.y, index.y + valid_height),
            slice(index.x, index.x + valid_width),
        )
        valid_weights = weights[:valid_height, :valid_width]
        accumulated[target_slice] += (
            prediction[:valid_height, :valid_width] * valid_weights
        )
        weight_map[target_slice] += valid_weights

    covered = weight_map > 0
    if not np.any(covered):
        return np.array(base, copy=True)
    refinement_map = np.zeros_like(base, dtype=np.float64)
    refinement_map[covered] = accumulated[covered] / weight_map[covered]
    alpha = np.clip(weight_map, 0.0, 1.0)
    return ((1.0 - alpha) * base + alpha * refinement_map).astype(np.float32)
