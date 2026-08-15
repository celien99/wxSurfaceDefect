import numpy as np

from hiad.runtime.quality import assess_image_quality


def _thresholds():
    return {
        "min_mean_luminance": 0.1,
        "max_mean_luminance": 0.9,
        "max_clipped_fraction": 0.2,
        "min_focus_variance": 5.0,
    }


def test_quality_gate_rechecks_dark_flat_image():
    image = np.zeros((32, 32, 3), dtype=np.uint8)

    result = assess_image_quality(image, _thresholds())

    assert result["status"] == "RECHECK"
    assert "mean_luminance_below_minimum" in result["reasons"]
    assert "focus_variance_below_minimum" in result["reasons"]


def test_quality_gate_accepts_well_exposed_sharp_image():
    checker = (np.indices((32, 32)).sum(axis=0) % 2 * 128 + 64).astype(np.uint8)
    image = np.repeat(checker[:, :, None], 3, axis=2)

    result = assess_image_quality(image, _thresholds())

    assert result["status"] == "PASS"
    assert result["reasons"] == []


def test_quality_gate_ignores_masked_black_background():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    checker = (np.indices((16, 16)).sum(axis=0) % 2 * 128 + 64).astype(np.uint8)
    image[8:24, 8:24] = checker[:, :, None]
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 255

    result = assess_image_quality(image, _thresholds(), mask)

    assert result["status"] == "PASS"
