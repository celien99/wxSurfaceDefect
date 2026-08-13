from __future__ import annotations

import json
import os

import numpy as np


SCORE_CALIBRATION_FILE = "score_calibration.json"


def _validate_percentile(value) -> float:
    value = float(value)
    if not np.isfinite(value) or not 0 < value < 1:
        raise ValueError("normal_score_percentile must be in the open interval (0, 1)")
    return value


def _score_threshold(scores, percentile: float) -> float:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Calibration scores must be a non-empty finite vector")
    return float(np.quantile(values, percentile))


def summarize_anomaly_map(anomaly_map, percentile: float) -> float:
    """Reduce one full-resolution map to a scalar for bounded-memory calibration."""
    values = np.asarray(anomaly_map)
    return float(np.quantile(values, _validate_percentile(percentile)))


def build_score_calibration(
    samples,
    scores,
    pixel_statistics,
    *,
    percentile,
    pixel_percentile,
    pixel_image_percentile,
    score_top_k: int,
) -> dict:
    samples = tuple(samples)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(samples),):
        raise ValueError("Calibration score count must match the normal sample count")
    pixel_statistics = np.asarray(pixel_statistics, dtype=np.float64)
    if pixel_statistics.shape != (len(samples),):
        raise ValueError("Pixel statistic count must match the normal sample count")
    if isinstance(score_top_k, bool) or not isinstance(score_top_k, int) or score_top_k <= 0:
        raise ValueError("score_top_k must be a positive integer")
    percentile = _validate_percentile(percentile)
    pixel_percentile = _validate_percentile(pixel_percentile)
    pixel_image_percentile = _validate_percentile(pixel_image_percentile)

    grouped_scores: dict[str, list[float]] = {}
    grouped_pixel_statistics: dict[str, list[float]] = {}
    for sample, score, pixel_statistic in zip(samples, scores, pixel_statistics):
        category = sample.clsname
        if not isinstance(category, str) or not category.strip():
            raise ValueError("Every calibration sample must have a non-empty clsname")
        grouped_scores.setdefault(category, []).append(float(score))
        grouped_pixel_statistics.setdefault(category, []).append(float(pixel_statistic))

    return {
        "score_top_k": score_top_k,
        "percentile": percentile,
        "pixel_percentile": pixel_percentile,
        "pixel_image_percentile": pixel_image_percentile,
        "normal_image_count": len(samples),
        "global_threshold": _score_threshold(scores, percentile),
        "global_pixel_threshold": _score_threshold(
            pixel_statistics, pixel_image_percentile
        ),
        "categories": {
            category: {
                "normal_image_count": len(category_scores),
                "threshold": _score_threshold(category_scores, percentile),
                "pixel_threshold": _score_threshold(
                    grouped_pixel_statistics[category], pixel_image_percentile
                ),
            }
            for category, category_scores in sorted(grouped_scores.items())
        },
    }


def save_score_calibration(calibration: dict, checkpoint_root: str) -> str:
    path = os.path.join(checkpoint_root, SCORE_CALIBRATION_FILE)
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as stream:
        json.dump(calibration, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_path, path)
    return path


def load_score_calibration(checkpoint_root: str, *, required: bool = True) -> dict | None:
    path = os.path.join(checkpoint_root, SCORE_CALIBRATION_FILE)
    if not os.path.isfile(path):
        if required:
            raise FileNotFoundError(f"Score calibration not found: {path}")
        return None
    with open(path, "r", encoding="utf-8") as stream:
        calibration = json.load(stream)
    return calibration


def thresholds_for_samples(calibration: dict, samples) -> np.ndarray:
    global_threshold = float(calibration["global_threshold"])
    categories = calibration["categories"]
    return np.asarray(
        [
            float(categories.get(sample.clsname, {}).get("threshold", global_threshold))
            for sample in samples
        ],
        dtype=np.float32,
    )


def pixel_thresholds_for_samples(calibration: dict, samples) -> np.ndarray | None:
    global_value = calibration.get("global_pixel_threshold")
    if global_value is None:
        return None
    global_threshold = float(global_value)
    categories = calibration["categories"]
    return np.asarray(
        [
            float(
                categories.get(sample.clsname, {}).get(
                    "pixel_threshold", global_threshold
                )
            )
            for sample in samples
        ],
        dtype=np.float32,
    )
