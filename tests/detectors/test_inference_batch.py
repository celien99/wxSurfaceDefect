import torch

import hiad.detectors.hr_dinomaly as hr_module
from hiad.detectors.hr_dinomaly import HRDinomaly
from hiad.runtime import mapops


class _StubEncoder:
    """最小可用的编码器替身：固定 384 维、8 层特征。"""

    def __init__(self, model_name="stub", intermediate_layers=None,
                 use_fp16=False, weights_path=None):
        self.embed_dim = 384
        self.intermediate_layers = tuple(intermediate_layers or (2, 3, 4, 5, 6, 7, 8, 9))

    def forward(self, inputs):
        torch.manual_seed(0)
        batch, _, height, width = inputs.shape
        token_h, token_w = height // 16, width // 16
        return [
            torch.randn(batch, self.embed_dim, token_h, token_w)
            for _ in self.intermediate_layers
        ]


def _make_detector():
    import torch
    original = hr_module.TimmDinoV3Encoder
    hr_module.TimmDinoV3Encoder = _StubEncoder
    try:
        detector = HRDinomaly(
            backbone_name="stub",
            total_iters=10,
            log_per_steps=1,
            patch_size=32,
            logger=None,
            device=torch.device("cpu"),
            use_context_conditioning=True,
            semantic_weight=0.6,
            memory_weight=0.3,
            high_frequency_weight=0.1,
        )
    finally:
        hr_module.TimmDinoV3Encoder = original
    detector.semantic_center = 0.0
    detector.semantic_scale = 1.0
    detector.memory_center = 0.0
    detector.memory_scale = 1.0
    detector.high_frequency_center = 0.0
    detector.high_frequency_scale = 1.0
    return detector


def _make_batch(batch_size=2):
    torch.manual_seed(1)
    image = torch.randn(batch_size, 3, 32, 32)
    context = torch.randn(batch_size, 3, 32, 32)
    indexes = ['{"x": 8, "y": 8, "width": 16, "height": 16}'] * batch_size
    return {"image": image, "low_resolution_image_0": context,
            "low_resolution_index_0": indexes}


def test_inference_batch_returns_gpu_shaped_tensors():
    detector = _make_detector()
    detector.feature_memory.update(
        detector.get_multi_resolution_embeddings(_make_batch())[0]
    )
    fused_pixel, fused_token = detector.inference_batch(_make_batch())
    assert fused_pixel.shape == (2, 1, 32, 32)
    assert fused_token.shape == (2, 1, 2, 2)
    assert fused_pixel.is_floating_point()
    assert torch.isfinite(fused_pixel).all()


def test_inference_step_parity_with_inference_batch():
    from torch.utils.data import DataLoader

    detector = _make_detector()
    batch = _make_batch(batch_size=3)
    detector.feature_memory.update(
        detector.get_multi_resolution_embeddings(batch)[0]
    )
    fused_pixel, fused_token = detector.inference_batch(batch)

    class _SingleBatchDataset:
        def __len__(self):
            return 1

        def __getitem__(self, index):
            return batch

    predictions = detector.inference_step(DataLoader(_SingleBatchDataset(), batch_size=1))
    expected_scores = mapops.top_k_token_scores_torch(
        fused_token, detector.score_top_k
    )
    for i, prediction in enumerate(predictions):
        pixel = prediction["anomaly_map"]
        assert pixel.shape == (32, 32)
        assert abs(prediction["score"] - float(expected_scores[i])) < 1e-6
        torch.testing.assert_close(
            torch.as_tensor(pixel), fused_pixel[i, 0], atol=0.0, rtol=0.0
        )
