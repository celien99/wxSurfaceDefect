import torch

from hiad.detectors.hr_dinomaly import HRDinomaly
from hiad.models.memory import NormalFeatureMemory


def test_top_k_token_score_does_not_average_the_full_map():
    token_maps = torch.tensor([[[[0.01, 0.02], [0.03, 0.95]]]])

    score = HRDinomaly._top_k_token_scores(token_maps, top_k=2)

    torch.testing.assert_close(score, torch.tensor([0.49]))


def test_pixel_map_maximizes_layers_after_upsampling():
    detector = object.__new__(HRDinomaly)
    encoder = torch.tensor([[[[1.0, 1.0]], [[0.0, 0.0]]]])
    decoder_left = torch.tensor([[[[0.0, 1.0]], [[1.0, 0.0]]]])
    decoder_right = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])

    anomaly_map, token_map = detector.cal_anomaly_maps(
        [encoder, encoder],
        [decoder_left, decoder_right],
        output_size=(3, 1),
    )

    torch.testing.assert_close(token_map, torch.ones((1, 1, 1, 2)))
    torch.testing.assert_close(
        anomaly_map,
        torch.tensor([[[[1.0, 0.5, 1.0]]]]),
    )


def test_memory_evidence_uses_conditioned_layers_before_dinomaly_aggregation():
    detector = object.__new__(HRDinomaly)
    detector.patch_size = [2, 2]
    detector.device = torch.device("cpu")
    detector.feature_memory = NormalFeatureMemory(embed_dim=2, layers=8)
    detector.high_frequency_center = 0.0
    detector.high_frequency_scale = 1.0
    detector.semantic_center = 0.0
    detector.semantic_scale = 1.0
    detector.memory_center = 0.0
    detector.memory_scale = 1.0
    detector.evidence_weights = (0.6, 0.3, 0.1)
    conditioned = [torch.zeros((1, 2, 1, 2)) for _ in range(8)]
    detector.feature_memory.update(conditioned)
    detector.feature_memory.update([feature + 0.01 for feature in conditioned])
    semantic_encoder = [torch.ones((1, 2, 1, 2)) for _ in range(2)]
    semantic_decoder = [torch.ones((1, 2, 1, 2)) for _ in range(2)]

    pixel_map, token_map = detector._fused_evidence(
        {"image": torch.zeros((1, 3, 2, 2))},
        conditioned,
        semantic_encoder,
        semantic_decoder,
    )

    assert pixel_map.shape == (1, 1, 2, 2)
    assert token_map.shape == (1, 1, 1, 2)
