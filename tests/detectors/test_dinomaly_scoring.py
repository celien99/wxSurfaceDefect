import numpy as np
import torch

from hiad.detectors.hr_dinomaly import HRDinomaly


def test_image_score_keeps_the_strongest_local_evidence():
    scores = HRDinomaly.get_image_score([
        [0.02, 0.03, 0.91, 0.04],
        [0.12, 0.08],
    ])

    np.testing.assert_allclose(scores, np.asarray([0.91, 0.12], dtype=np.float32))


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
