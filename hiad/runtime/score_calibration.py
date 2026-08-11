from __future__ import annotations

import json
import os

import numpy as np


SCORE_CALIBRATION_FILE = "score_calibration.json"
SCORE_CALIBRATION_DOMAIN = "dinomaly_max_layer_topk_token"
SCORE_CALIBRATION_SCHEMA_VERSION = 1


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


def build_score_calibration(samples, scores, *, percentile, score_top_k: int) -> dict:
    samples = tuple(samples)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(samples),):
        raise ValueError("Calibration score count must match the normal sample count")
    if isinstance(score_top_k, bool) or not isinstance(score_top_k, int) or score_top_k <= 0:
        raise ValueError("score_top_k must be a positive integer")
    percentile = _validate_percentile(percentile)

    grouped_scores: dict[str, list[float]] = {}
    for sample, score in zip(samples, scores):
        category = sample.clsname
        if not isinstance(category, str) or not category.strip():
            raise ValueError("Every calibration sample must have a non-empty clsname")
        grouped_scores.setdefault(category, []).append(float(score))

    return {
        "schema_version": SCORE_CALIBRATION_SCHEMA_VERSION,
        "domain": SCORE_CALIBRATION_DOMAIN,
        "score_top_k": score_top_k,
        "percentile": percentile,
        "normal_image_count": len(samples),
        "global_threshold": _score_threshold(scores, percentile),
        "categories": {
            category: {
                "normal_image_count": len(category_scores),
                "threshold": _score_threshold(category_scores, percentile),
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
    if not isinstance(calibration, dict):
        raise ValueError("Score calibration must be a mapping")
    if calibration.get("schema_version") != SCORE_CALIBRATION_SCHEMA_VERSION:
        raise ValueError("Unsupported score calibration schema version")
    if calibration.get("domain") != SCORE_CALIBRATION_DOMAIN:
        raise ValueError("Score calibration domain does not match Dinomaly scoring")
    _validate_percentile(calibration.get("percentile"))
    global_threshold = float(calibration.get("global_threshold"))
    if not np.isfinite(global_threshold):
        raise ValueError("Score calibration global threshold must be finite")
    categories = calibration.get("categories")
    if not isinstance(categories, dict):
        raise ValueError("Score calibration categories must be a mapping")
    for category, payload in categories.items():
        if not isinstance(category, str) or not category or not isinstance(payload, dict):
            raise ValueError("Score calibration contains an invalid category entry")
        threshold = float(payload.get("threshold"))
        if not np.isfinite(threshold):
            raise ValueError(f"Score threshold for category {category} must be finite")
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
