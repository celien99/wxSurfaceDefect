from types import SimpleNamespace

import numpy as np
import pytest

from hiad.runtime.score_calibration import (
    build_component_calibration,
    build_score_calibration,
    component_thresholds_for_samples,
    load_score_calibration,
    summarize_anomaly_map,
)


def _sample(clsname):
    return SimpleNamespace(clsname=clsname)


def test_load_score_calibration_requires_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Score calibration not found"):
        load_score_calibration(tmp_path)


def test_pixel_calibration_uses_scalar_map_summaries():
    samples = [_sample("part"), _sample("part")]
    maps = [
        np.asarray([[0.1, 0.2]], dtype=np.float32),
        np.asarray([[0.3, 0.4]], dtype=np.float32),
    ]
    statistics = [summarize_anomaly_map(anomaly_map, 0.5) for anomaly_map in maps]

    calibration = build_score_calibration(
        samples,
        [0.1, 0.2],
        statistics,
        percentile=0.5,
        pixel_percentile=0.5,
        pixel_image_percentile=0.5,
    )

    assert calibration["global_pixel_threshold"] == np.quantile(statistics, 0.5)


def test_component_calibration_adds_category_decision_thresholds():
    samples = [_sample("a"), _sample("a"), _sample("b")]
    calibration = build_score_calibration(
        samples,
        [0.1, 0.2, 0.4],
        [0.3, 0.5, 0.7],
        percentile=0.5,
        pixel_percentile=0.9,
        pixel_image_percentile=0.5,
    )

    completed = build_component_calibration(
        calibration,
        samples,
        [1.0, 2.0, 4.0],
        percentile=0.5,
    )

    np.testing.assert_allclose(
        component_thresholds_for_samples(completed, samples),
        [1.5, 1.5, 4.0],
    )
