from __future__ import annotations

import math
from collections.abc import Sequence
from functools import lru_cache

import cv2
import numpy as np
from numpy.typing import ArrayLike

from hiad.data import HRImageIndex
from hiad.runtime.contracts import FloatMap, ImageSize, RefinementStatistics


@lru_cache(maxsize=32)
def refinement_blend_weights(height: int, width: int) -> FloatMap:
    """Return cached Hann weights with a non-zero edge floor.

    Refinement and coarse-map stitching repeatedly use the same patch shapes.
    Keeping the small weight kernels cached avoids rebuilding them for every
    patch while returning a read-only array so callers cannot corrupt the cache.
    """
    if height <= 0 or width <= 0:
        raise ValueError("Patch dimensions must be positive")
    row_hann = np.hanning(height) if height > 1 else np.ones(1)
    column_hann = np.hanning(width) if width > 1 else np.ones(1)
    if row_hann.max() > 0:
        row_hann /= row_hann.max()
    if column_hann.max() > 0:
        column_hann /= column_hann.max()
    weights = (0.05 + 0.95 * np.outer(row_hann, column_hann)).astype(
        np.float32,
        copy=False,
    )
    weights.setflags(write=False)
    return weights


def refinement_tile_statistics(
    image_size: ImageSize,
    tile_size: int,
    regions: Sequence[HRImageIndex],
) -> RefinementStatistics:
    """Summarize refinement-grid coverage for runtime observability.

    The selected count is deduplicated by native tile origin, so overlap or
    repeated task records cannot inflate the reported compute coverage.
    """
    image_width, image_height = image_size
    x_starts = _tile_axis_starts(image_width, tile_size)
    y_starts = _tile_axis_starts(image_height, tile_size)
    valid_origins = {
        (int(region.y), int(region.x))
        for region in regions
    }
    selected_tiles = sum(
        1
        for y in y_starts
        for x in x_starts
        if (y, x) in valid_origins
    )
    total_tiles = len(x_starts) * len(y_starts)
    return {
        "total_tiles": total_tiles,
        "selected_tiles": selected_tiles,
        "coverage_ratio": float(selected_tiles / total_tiles),
    }


def _robust_unit_map(values: ArrayLike) -> FloatMap:
    """用中位数和高分位数把排序证据稳健缩放到 ``[0, 1]``。

    Args:
        values (ArrayLike): 有限二维异常排序图。

    Returns:
        FloatMap: 中位数映射为 ``0``、``0.995`` 分位数映射为 ``1`` 的
        ``float32`` 图；两者相等时返回全零图。
    """
    lower = float(np.quantile(values, 0.5))
    upper = float(np.quantile(values, 0.995))
    if upper <= lower:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lower) / (upper - lower), 0.0, 1.0).astype(np.float32)


def build_routing_map(
    local_map: ArrayLike,
    global_context_map: ArrayLike,
    global_weight: float,
) -> FloatMap:
    """融合局部与全局排序先验，只用于复核路由而不替换局部证据。

    Args:
        local_map (ArrayLike): 原图分辨率的局部补丁异常图。
        global_context_map (ArrayLike): 与局部图同形状的缩略图全局先验。
        global_weight (float): 全局先验权重，范围 ``[0, 1]``。

    Returns:
        FloatMap: 与输入同形状的二维 ``float32`` 路由图，范围 ``[0, 1]``。

    Raises:
        ValueError: 两张图不是形状一致的有限二维数组，或权重超出范围。

    Notes:
        两个输入分别用中位数和 ``0.995`` 分位数做稳健归一化，因此路由图只表达
        图内相对排序；最终异常证据仍由局部粗扫和高分辨率复核结果组成。
    """
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
    anomaly_map: ArrayLike,
    threshold: float,
    tile_size: int,
    min_area: int,
    safety_fraction: float,
    max_bridge_gap_tiles: int,
) -> FloatMap:
    """校验复核候选选择所需的异常图和网格参数。

    Args:
        anomaly_map (ArrayLike): 非空有限二维路由图。
        threshold (float): 有限候选阈值。
        tile_size (int): 正方形复核块边长。
        min_area (int): 最小连通区域像素数。
        safety_fraction (float): 确定性安全采样比例，范围 ``(0, 1]``。
        max_bridge_gap_tiles (int): 候选补丁之间可桥接的最大缺口数，允许为零。

    Returns:
        FloatMap: 转换为 ``float32`` 的路由图。

    Raises:
        ValueError: 任一输入不满足形状、有限性或数值范围约束。
    """
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
    if (
        isinstance(max_bridge_gap_tiles, bool)
        or not isinstance(max_bridge_gap_tiles, int)
        or max_bridge_gap_tiles < 0
    ):
        raise ValueError("max_bridge_gap_tiles must be a non-negative integer")
    return values


def _tile_axis_starts(length: int, tile_size: int) -> list[int]:
    """计算单轴不重复且覆盖末端边界的复核块起点。

    Args:
        length (int): 原图当前轴像素长度。
        tile_size (int): 复核块在当前轴的像素长度。

    Returns:
        list[int]: 升序且去重的起点；块大于图像时仅返回 ``0``。
    """
    if length <= tile_size:
        return [0]
    starts = list(range(0, length, tile_size))
    starts[-1] = length - tile_size
    return list(dict.fromkeys(starts))


def _radical_inverse(index: int, base: int) -> float:
    """计算低差异空间采样使用的指定进制反根值。

    Args:
        index (int): 非负序列编号。
        base (int): 大于一的进制基数。

    Returns:
        float: 范围 ``[0, 1)`` 的反根值。
    """
    result = 0.0
    fraction = 1.0 / base
    while index:
        result += (index % base) * fraction
        index //= base
        fraction /= base
    return result


def _spatial_safety_indexes(row_count: int, column_count: int, count: int) -> list[int]:
    """用基数 ``2/3`` 的低差异序列选择确定性空间安全网格。

    Args:
        row_count (int): 复核网格行数。
        column_count (int): 复核网格列数。
        count (int): 需要选择的唯一网格数量。

    Returns:
        list[int]: 按生成顺序排列的扁平网格索引。
    """
    total = row_count * column_count
    if count >= total:
        return list(range(total))
    selected: list[int] = []
    seen: set[int] = set()
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
    anomaly_map: ArrayLike,
    threshold: float,
    tile_size: int,
    min_area: int,
    safety_fraction: float,
    max_bridge_gap_tiles: int = 0,
) -> list[HRImageIndex]:
    """选择可疑区域，并加入确定性的原图坐标安全采样块以保护召回。

    Args:
        anomaly_map (ArrayLike): 原图分辨率的有限二维路由异常图。
        threshold (float): 候选像素阈值，通常由当前图的配置分位数产生。
        tile_size (int): 正方形复核块边长，单位为原图像素。
        min_area (int): 八连通候选区域的最小像素数。
        safety_fraction (float): 除异常候选外必须覆盖的确定性网格比例，范围
            ``(0, 1]``。
        max_bridge_gap_tiles (int): 同一行或列中被候选补丁夹住时，额外桥接的
            最大微补丁缺口数。该扩展不扩大确定性安全采样。

    Returns:
        list[HRImageIndex]: 原图 ``xywh`` 复核区域。先按行列顺序返回异常候选块
        及其被夹住的微补丁缺口，再追加不重复的低差异安全采样块；至少包含一个
        安全块。

    Raises:
        ValueError: 异常图为空、非二维或含非有限值，阈值非有限，或网格参数
            不符合范围。
    """
    values = _validate_selection_arguments(
        anomaly_map,
        threshold,
        tile_size,
        min_area,
        safety_fraction,
        max_bridge_gap_tiles,
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
    component_maxima = np.full(component_count, -np.inf, dtype=np.float32)
    np.maximum.at(component_maxima, labels.reshape(-1), values.reshape(-1))
    # Keep small but sharp peaks: a tiny defect can occupy fewer pixels than
    # min_area after coarse routing, while still being strong enough to warrant
    # a high-resolution look.
    peak_threshold = float(threshold) + 0.5 * max(
        0.0,
        float(values.max()) - float(threshold),
    )
    valid_components = np.zeros(component_count, dtype=bool)
    valid_components[1:] = (
        (statistics[1:, cv2.CC_STAT_AREA] >= min_area)
        | (component_maxima[1:] >= peak_threshold)
    )
    candidate_y, candidate_x = np.nonzero(valid_components[labels])
    occupied_tiles: set[tuple[int, int]] = set()
    if candidate_x.size:
        x_indexes = np.searchsorted(x_starts, candidate_x, side="right") - 1
        y_indexes = np.searchsorted(y_starts, candidate_y, side="right") - 1
        occupied_tiles.update(zip(y_indexes.tolist(), x_indexes.tolist()))
    expanded_tiles = set(occupied_tiles)
    tiles_by_row: dict[int, list[int]] = {}
    tiles_by_column: dict[int, list[int]] = {}
    for y_index, x_index in occupied_tiles:
        tiles_by_row.setdefault(y_index, []).append(x_index)
        tiles_by_column.setdefault(x_index, []).append(y_index)
    for fixed_index, coordinates in tiles_by_row.items():
        coordinates.sort()
        for left, right in zip(coordinates, coordinates[1:]):
            if 0 < right - left - 1 <= max_bridge_gap_tiles:
                expanded_tiles.update(
                    (fixed_index, x_index)
                    for x_index in range(left + 1, right)
                )
    for fixed_index, coordinates in tiles_by_column.items():
        coordinates.sort()
        for top, bottom in zip(coordinates, coordinates[1:]):
            if 0 < bottom - top - 1 <= max_bridge_gap_tiles:
                expanded_tiles.update(
                    (y_index, fixed_index)
                    for y_index in range(top + 1, bottom)
                )

    # When no component survived routing, inspect the strongest one or two
    # tiles. This protects broad, moderate defects whose pixels all sit below
    # the high quantile route threshold, without forcing full-image refinement.
    if float(values.max()) > float(values.min()):
        tile_scores = [
            (
                float(
                    values[
                        y : min(y + tile_size, image_height),
                        x : min(x + tile_size, image_width),
                    ].max()
                ),
                y_index,
                x_index,
            )
            for y_index, y in enumerate(y_starts)
            for x_index, x in enumerate(x_starts)
        ]
        guard_count = 2 if not occupied_tiles and len(tile_scores) > 1 else 1
        for _, y_index, x_index in sorted(
            tile_scores,
            key=lambda item: (-item[0], item[1], item[2]),
        )[:guard_count]:
            expanded_tiles.add((y_index, x_index))

    selected_indexes = sorted(expanded_tiles)
    selected = [
        HRImageIndex(
            x=x_starts[x_index],
            y=y_starts[y_index],
            width=tile_size,
            height=tile_size,
        )
        for y_index, x_index in selected_indexes
    ]
    column_count = len(x_starts)
    tile_count = len(y_starts) * column_count
    safety_count = max(1, math.ceil(tile_count * float(safety_fraction)))
    safety_indexes = _spatial_safety_indexes(
        len(y_starts),
        column_count,
        safety_count,
    )
    selected_set = set(selected_indexes)
    for safety_index in safety_indexes:
        flattened = int(safety_index)
        tile_index = (flattened // column_count, flattened % column_count)
        if tile_index not in selected_set:
            y_index, x_index = tile_index
            tile = HRImageIndex(
                x=x_starts[x_index],
                y=y_starts[y_index],
                width=tile_size,
                height=tile_size,
            )
            selected.append(tile)
            selected_set.add(tile_index)
    return selected


def merge_refinement_maps(
    base_map: ArrayLike,
    refinements: Sequence[tuple[HRImageIndex, ArrayLike]],
    image_size: ImageSize | list[int],
) -> FloatMap:
    """按原图坐标回填复核结果，并在重叠区域平滑融合。

    Args:
        base_map (ArrayLike): 原图分辨率的非空有限二维粗扫异常图。
        refinements (Sequence[tuple[HRImageIndex, ArrayLike]]): 原图 ``xywh`` 区域及
            对应二维复核异常图；图像尺寸不匹配时线性缩放到区域宽高。
        image_size (ImageSize | list[int]): 原图 ``(width, height)``，必须与
            ``base_map`` 形状一致。

    Returns:
        FloatMap: 原图分辨率 ``float32`` 融合图。没有有效覆盖时返回粗扫图副本；
        重叠区域按带最低边缘权重的二维 Hann 窗平均，再与粗扫图逐点取最大值，
        从而确保精修不会压低已有异常证据。

    Raises:
        TypeError: 复核条目不是 ``(HRImageIndex, anomaly_map)`` 二元组。
        ValueError: 原图尺寸、区域几何或复核异常图不符合约定。
    """
    base = np.asarray(base_map, dtype=np.float32)
    if base.ndim != 2 or base.size == 0 or not np.isfinite(base).all():
        raise ValueError("base_map must be a non-empty finite two-dimensional array")
    if (
        not isinstance(image_size, (tuple, list))
        or len(image_size) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in image_size
        )
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
        # Hann 权重削弱补丁边缘伪影，保底权重确保边界像素仍能被复核结果覆盖。
        weights = refinement_blend_weights(index.height, index.width)
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
    blended = (1.0 - alpha) * base + alpha * refinement_map
    return np.maximum(base, blended).astype(np.float32)
