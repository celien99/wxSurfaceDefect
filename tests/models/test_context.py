import pytest
import torch

from hiad.models.context import ConditionalFeatureFusion


def _features(batch=2, channels=8):
    return [
        torch.randn(batch, channels, 4, 5, requires_grad=True),
        torch.randn(batch, channels, 2, 3, requires_grad=True),
    ]


def test_context_fusion_is_identity_without_context():
    module = ConditionalFeatureFusion(embed_dim=8, layers=2)
    main = _features()

    output = module(main, None)

    assert output[0] is main[0]
    assert output[1] is main[1]


def test_context_fusion_preserves_shape_and_produces_gradients():
    module = ConditionalFeatureFusion(embed_dim=8, layers=2)
    main = _features()
    context = [torch.randn(2, 8, 3, 4), torch.randn(2, 8, 1, 2)]

    output = module(main, context)
    loss = sum(value.square().mean() for value in output)
    loss.backward()

    assert [value.shape for value in output] == [value.shape for value in main]
    assert all(parameter.grad is not None for parameter in module.parameters())
    assert all(torch.isfinite(value).all() for value in output)


def test_context_fusion_rejects_mismatched_feature_lists():
    module = ConditionalFeatureFusion(embed_dim=8, layers=2)
    with pytest.raises(ValueError, match="same number of layers"):
        module(_features(), [torch.randn(2, 8, 3, 4)])
