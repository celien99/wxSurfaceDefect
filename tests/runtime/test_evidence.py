import numpy as np
import pytest
import torch

from hiad.runtime.evidence import (
    denormalize_imagenet_batch,
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


def test_high_frequency_uses_denormalized_production_input():
    raw = torch.zeros((1, 3, 9, 9), dtype=torch.float32)
    raw[:, :, :, 5:] = 1.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    normalized = (raw - mean) / std

    restored = denormalize_imagenet_batch(normalized)

    torch.testing.assert_close(restored, raw)
    torch.testing.assert_close(high_frequency_map(restored), high_frequency_map(raw))


def test_tensor_fusion_preserves_device_shape_and_safety_contribution():
    first = torch.tensor([[[[0.1, 0.8]]]])
    second = torch.tensor([[[[0.9, 0.2]]]])

    fused = fuse_evidence_tensors([first, second], [1.0, 1.0])

    assert fused.shape == first.shape
    assert torch.all(fused >= (first + second) / 2.0)


def test_zero_weight_branch_is_excluded_from_safety_maximum():
    enabled = torch.tensor([[[[0.2]]]])
    disabled = torch.tensor([[[[100.0]]]])

    fused = fuse_evidence_tensors([enabled, disabled], [1.0, 0.0])

    torch.testing.assert_close(fused, enabled)


def test_tensor_fusion_rejects_nonfinite_evidence():
    with pytest.raises(ValueError, match="finite"):
        fuse_evidence_tensors([torch.tensor([[[[float("inf")]]]])], [1.0])
