import json
from types import SimpleNamespace

import numpy as np

from hiad.runtime.score_calibration import (
    build_score_calibration,
    load_score_calibration,
    pixel_thresholds_for_samples,
    summarize_anomaly_map,
)


def _sample(clsname):
    return SimpleNamespace(clsname=clsname)


def test_load_score_calibration_accepts_existing_files_without_version_checks(tmp_path):
    calibration = {
        "schema_version": 1,
        "score_top_k": 4,
        "global_threshold": 0.3,
        "categories": {"part": {"threshold": 0.2}},
    }
    (tmp_path / "score_calibration.json").write_text(json.dumps(calibration))

    assert load_score_calibration(tmp_path) == calibration
    assert pixel_thresholds_for_samples(calibration, [_sample("part")]) is None


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
        score_top_k=4,
    )

    assert calibration["global_pixel_threshold"] == np.quantile(statistics, 0.5)
    assert "schema_version" not in calibration
