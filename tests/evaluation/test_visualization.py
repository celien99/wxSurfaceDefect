import numpy as np

from hiad.evaluation.visualization import scale_anomaly_map_for_display


def test_heatmap_uses_segmentation_threshold_as_a_fixed_scale():
    prediction = np.array([[0.0, 0.25, 0.5, 1.0]], dtype=np.float32)

    scaled = scale_anomaly_map_for_display(prediction, threshold=0.5)

    np.testing.assert_allclose(scaled, prediction)


def test_heatmap_without_threshold_handles_constant_maps():
    scaled = scale_anomaly_map_for_display(
        np.ones((2, 2), dtype=np.float32),
    )

    np.testing.assert_array_equal(scaled, np.zeros((2, 2), dtype=np.float32))
