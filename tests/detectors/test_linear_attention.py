import pytest
import torch

from hiad.detectors.dinomaly.models.vision_transformer import LinearAttention2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP test")
def test_linear_attention2_keeps_token_reductions_finite_under_amp():
    attention = LinearAttention2(dim=4, num_heads=1, qkv_bias=False).cuda()
    with torch.no_grad():
        attention.qkv.weight.fill_(100.0)
        attention.proj.weight.copy_(torch.eye(4, device="cuda"))
        attention.proj.bias.zero_()

    tokens = torch.ones((1, 1024, 4), device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output, _ = attention(tokens)
        loss = output.float().mean() * 1e-4

    assert torch.isfinite(output).all()
    loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in attention.parameters()
    )
