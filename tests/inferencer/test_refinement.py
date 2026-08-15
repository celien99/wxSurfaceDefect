import numpy as np
import pytest

from hiad.data import HRImageIndex, build_multiresolution_region
from hiad.inferencer.refinement import (
    build_routing_map,
    merge_refinement_maps,
    select_refinement_regions,
)


def test_select_refinement_regions_tiles_connected_candidates():
    anomaly_map = np.zeros((8, 12), dtype=np.float32)
    anomaly_map[2:5, 7:10] = 0.9

    regions = select_refinement_regions(
        anomaly_map,
        threshold=0.5,
        tile_size=4,
        min_area=4,
        safety_fraction=0.25,
    )

    assert any(
        region.x <= 8 < region.x + region.width
        and region.y <= 3 < region.y + region.height
        for region in regions
    )


def test_select_refinement_regions_clips_bottom_right_tile_to_native_bounds():
    anomaly_map = np.zeros((6, 10), dtype=np.float32)
    anomaly_map[4:, 8:] = 1.0

    regions = select_refinement_regions(
        anomaly_map,
        threshold=0.5,
        tile_size=4,
        min_area=1,
        safety_fraction=0.5,
    )

    assert HRImageIndex(x=6, y=2, width=4, height=4) in regions


def test_select_refinement_regions_covers_large_component_with_multiple_tiles():
    anomaly_map = np.zeros((16, 16), dtype=np.float32)
    anomaly_map[2:14, 2:14] = 1.0

    regions = select_refinement_regions(
        anomaly_map,
        threshold=0.5,
        tile_size=4,
        min_area=1,
        safety_fraction=0.01,
    )

    component_tiles = [
        region
        for region in regions
        if region.x < 14 and region.y < 14 and region.x + 4 > 2 and region.y + 4 > 2
    ]
    assert len(component_tiles) >= 9


def test_select_refinement_regions_adds_deterministic_safety_coverage():
    anomaly_map = np.zeros((6, 10), dtype=np.float32)

    first = select_refinement_regions(
        anomaly_map,
        threshold=0.5,
        tile_size=4,
        min_area=1,
        safety_fraction=0.25,
    )
    second = select_refinement_regions(
        anomaly_map,
        threshold=0.5,
        tile_size=4,
        min_area=1,
        safety_fraction=0.25,
    )

    assert first == second
    assert first == [
        HRImageIndex(x=0, y=0, width=4, height=4),
        HRImageIndex(x=6, y=2, width=4, height=4),
    ]


def test_select_refinement_regions_keeps_a_top_quantile_plateau():
    anomaly_map = np.zeros((20, 20), dtype=np.float32)
    anomaly_map[18:, 18:] = 1.0
    threshold = float(np.quantile(anomaly_map, 0.995))

    regions = select_refinement_regions(
        anomaly_map,
        threshold=threshold,
        tile_size=4,
        min_area=1,
        safety_fraction=0.01,
    )

    assert threshold == 1.0
    assert any(region.x == 16 and region.y == 16 for region in regions)


def test_select_refinement_regions_rejects_zero_safety_coverage():
    with pytest.raises(ValueError, match="safety_fraction"):
        select_refinement_regions(
            np.zeros((4, 4), dtype=np.float32),
            threshold=0.5,
            tile_size=4,
            min_area=1,
            safety_fraction=0.0,
        )


def test_merge_refinement_maps_uses_native_coordinates_and_valid_edge_extent():
    base_map = np.full((4, 6), 0.2, dtype=np.float32)
    refinement_map = np.full((4, 4), 0.8, dtype=np.float32)

    merged = merge_refinement_maps(
        base_map,
        [(HRImageIndex(x=4, y=2, width=4, height=4), refinement_map)],
        image_size=(6, 4),
    )

    assert merged.shape == (4, 6)
    assert np.all(merged[:2, :] == 0.2)
    assert np.all(merged[2:, :4] == 0.2)
    assert np.all(merged[2:, 4:] == 0.8)


def test_refinement_region_keeps_nested_multiscale_context_at_image_edge():
    region = build_multiresolution_region(
        image_size=(100, 80),
        main_index=HRImageIndex(x=84, y=64, width=16, height=16),
        ds_factors=[0, 1],
    )

    assert region.main_index == HRImageIndex(x=84, y=64, width=16, height=16)
    assert region.low_resolution_indexes == [
        HRImageIndex(x=68, y=48, width=32, height=32)
    ]


def test_routing_map_uses_global_context_without_replacing_local_evidence():
    local = np.zeros((4, 4), dtype=np.float32)
    local[0, 0] = 1.0
    global_context = np.zeros((4, 4), dtype=np.float32)
    global_context[3, 3] = 1.0

    routing = build_routing_map(local, global_context, global_weight=0.25)

    assert routing[0, 0] > routing[1, 1]
    assert routing[3, 3] > routing[1, 1]
    assert routing[0, 0] > routing[3, 3]
