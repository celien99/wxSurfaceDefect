import numpy as np
import pytest
import torch

from hiad.runtime.evidence import (
    fuse_evidence_maps,
    fuse_evidence_tensors,
    high_frequency_map,
)


def test_constant_image_has_zero_high_frequency_response():
    image = torch.ones((2, 3, 8, 8), dtype=torch.float32)

    evidence = high_frequency_map(image)

    assert evidence.shape == (2, 1, 8, 8)
    torch.testing.assert_close(evidence, torch.zeros_like(evidence))


def test_step_edge_has_high_frequency_response():
    image = torch.zeros((1, 1, 9, 9), dtype=torch.float32)
    image[:, :, :, 5:] = 1

    evidence = high_frequency_map(image)

    assert float(evidence.max()) > 0
    assert float(evidence[:, :, :, 4:6].max()) > float(evidence[:, :, :, :3].max())


@pytest.mark.parametrize(
    "branch_maps, weights, match",
    [
        ([], [], "at least one"),
        ([np.zeros((2, 2))], [], "weights"),
        ([np.zeros((2, 2)), np.zeros((3, 2))], [1, 1], "shape"),
        ([np.zeros((2, 2))], [-1], "non-negative"),
        ([np.asarray([[np.nan]])], [1], "finite"),
    ],
)
def test_fusion_validates_inputs(branch_maps, weights, match):
    with pytest.raises(ValueError, match=match):
        fuse_evidence_maps(branch_maps, weights)


def test_fusion_safety_max_preserves_strongest_evidence():
    first = np.asarray([[0.1, 0.8], [0.2, 0.3]])
    second = np.asarray([[0.4, 0.2], [0.9, 0.1]])

    fused, maximum = fuse_evidence_maps([first, second], [1.0, 3.0])

    np.testing.assert_allclose(maximum, np.maximum(first, second))
    weighted = (first + 3.0 * second) / 4.0
    assert np.all(fused >= weighted)
    assert np.all(fused <= maximum)


def test_tensor_fusion_preserves_device_shape_and_safety_channel():
    first = torch.tensor([[[[0.1, 0.8]]]])
    second = torch.tensor([[[[0.9, 0.2]]]])

    fused, maximum = fuse_evidence_tensors([first, second], [1.0, 1.0])

    assert fused.shape == first.shape
    torch.testing.assert_close(maximum, torch.maximum(first, second))
    assert torch.all(fused >= (first + second) / 2.0)


def test_tensor_fusion_rejects_nonfinite_evidence():
    with pytest.raises(ValueError, match="finite"):
        fuse_evidence_tensors([torch.tensor([[[[float("inf")]]]])], [1.0])
