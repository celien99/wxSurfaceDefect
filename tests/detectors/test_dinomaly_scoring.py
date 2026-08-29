import torch

from hiad.detectors.hr_dinomaly import HRDinomaly


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
