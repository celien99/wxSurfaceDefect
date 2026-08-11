import torch

from hiad.detectors.dinomaly.utils import global_cosine_hm_percent


def test_zero_hard_mining_matches_plain_global_cosine_loss():
    target = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    prediction = torch.tensor([[[[0.5, 0.5]], [[0.5, 0.5]]]], requires_grad=True)

    loss = global_cosine_hm_percent([target], [prediction], p=0.0, factor=0.1)
    expected = torch.mean(
        1 - torch.nn.functional.cosine_similarity(
            target.reshape(1, -1), prediction.reshape(1, -1)
        )
    )

    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
