import numpy as np
import torch
import pytest

from hiad.data import HRImageIndex
from hiad.inferencer.refinement import (
    build_routing_map as build_routing_map_np,
    merge_refinement_maps as merge_refinement_maps_np,
)
from hiad.runtime.mapops import (
    build_routing_map_torch,
    gaussian_blur_torch,
    hann_weights_torch,
    merge_refinement_maps_torch,
    robust_unit_map_torch,
    stitch_patch_maps_torch,
    top_k_token_scores_torch,
)


def test_hann_matches_numpy_reference():
    device = torch.device("cpu")
    weights_t = hann_weights_torch(16, 16, device).numpy()
    from hiad.inferencer.refinement import refinement_blend_weights
    weights_np = refinement_blend_weights(16, 16)
    np.testing.assert_allclose(weights_t, weights_np, atol=1e-6)


def test_stitch_matches_numpy_gather():
    device = torch.device("cpu")
    maps = torch.tensor(
        [[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]],
        dtype=torch.float32,
    )
    records = [
        {
            "task_name": "dynamic_patch", "task_type": "dynamic_patch",
            "image_path": "a", "image_size": (4, 2),
            "model_input_size": (2, 2), "source_xywh": (0, 0, 2, 2),
            "valid_source_hw": (2, 2),
        },
        {
            "task_name": "dynamic_patch", "task_type": "dynamic_patch",
            "image_path": "a", "image_size": (4, 2),
            "model_input_size": (2, 2), "source_xywh": (2, 0, 2, 2),
            "valid_source_hw": (2, 2),
        },
    ]
    result = stitch_patch_maps_torch(maps, records, (4, 2), device).numpy()
    assert result.shape == (2, 4)
    np.testing.assert_allclose(
        result[:, :2], maps[0].numpy(), atol=1e-6
    )
    np.testing.assert_allclose(
        result[:, 2:], maps[1].numpy(), atol=1e-6
    )


def test_stitch_raises_when_uncovered():
    device = torch.device("cpu")
    records = [
        {
            "task_name": "dynamic_patch", "task_type": "dynamic_patch",
            "image_path": "a", "image_size": (6, 2),
            "model_input_size": (2, 2), "source_xywh": (0, 0, 2, 2),
            "valid_source_hw": (2, 2),
        },
    ]
    with pytest.raises(ValueError, match="cover"):
        stitch_patch_maps_torch(
            torch.zeros((1, 2, 2)), records, (6, 2), device
        )


def test_robust_unit_map_matches_numpy():
    values = torch.tensor(np.random.default_rng(0).random((8, 8)), dtype=torch.float32)
    expected = np.zeros_like(values.numpy(), dtype=np.float32)
    lower = float(np.quantile(values.numpy(), 0.5))
    upper = float(np.quantile(values.numpy(), 0.995))
    if upper > lower:
        expected = np.clip((values.numpy() - lower) / (upper - lower), 0.0, 1.0).astype(np.float32)
    np.testing.assert_allclose(
        robust_unit_map_torch(values).numpy(), expected, atol=1e-6
    )


def test_routing_matches_numpy_reference():
    local = torch.tensor(np.random.default_rng(1).random((8, 8)), dtype=torch.float32)
    global_map = torch.tensor(np.random.default_rng(2).random((8, 8)), dtype=torch.float32)
    expected = build_routing_map_np(local.numpy(), global_map.numpy(), 0.25)
    np.testing.assert_allclose(
        build_routing_map_torch(local, global_map, 0.25).numpy(),
        expected,
        atol=1e-6,
    )


def test_merge_matches_numpy_reference():
    base = torch.zeros((8, 8), dtype=torch.float32)
    refinements = [
        (HRImageIndex(x=2, y=2, width=4, height=4),
         torch.ones((4, 4), dtype=torch.float32) * 0.5),
    ]
    expected = merge_refinement_maps_np(base.numpy(), [(ref[0], ref[1].numpy())], (8, 8))
    np.testing.assert_allclose(
        merge_refinement_maps_torch(base, refinements, (8, 8), torch.device("cpu")).numpy(),
        expected,
        atol=1e-5,
    )


def test_merge_rejects_unsupported_resize():
    base = torch.zeros((8, 8), dtype=torch.float32)
    refinements = [
        (HRImageIndex(x=0, y=0, width=4, height=4), torch.zeros((2, 2), dtype=torch.float32)),
    ]
    with pytest.raises(ValueError, match="shape"):
        merge_refinement_maps_torch(base, refinements, (8, 8), torch.device("cpu"))


def test_top_k_token_scores_matches_reference():
    token_maps = torch.tensor(
        [[[[0.1, 0.9, 0.3], [0.2, 0.8, 0.4], [0.5, 0.6, 0.7]]]],
        dtype=torch.float32,
    )
    result = top_k_token_scores_torch(token_maps, 2)
    assert result.shape == (1,)
    assert abs(float(result[0]) - (0.9 + 0.8) / 2) < 1e-6


def test_gaussian_blur_is_shape_preserving():
    values = torch.zeros((8, 8), dtype=torch.float32)
    result = gaussian_blur_torch(values, 1.0)
    assert result.shape == (8, 8)
