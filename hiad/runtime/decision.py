from __future__ import annotations

import math

import cv2
import numpy as np


def _empty_statistics() -> dict:
    return {
        "component_count": 0,
        "anomalous_pixel_count": 0,
        "largest_component_area": 0,
        "strongest_component": None,
    }


def component_statistics(anomaly_map, pixel_threshold) -> dict:
    """Summarize eight-connected regions at a calibrated pixel threshold."""
    threshold = float(pixel_threshold)
    values = np.asarray(anomaly_map, dtype=np.float32)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("anomaly_map must be a non-empty two-dimensional array")
    if not np.isfinite(threshold):
        raise ValueError("pixel_threshold must be finite")

    binary = np.asarray(np.isfinite(values) & (values >= threshold), dtype=np.uint8)
    component_count, labels, statistics, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    if component_count == 1:
        return _empty_statistics()

    active = binary.astype(bool)
    active_labels = labels[active]
    active_values = values[active].astype(np.float64, copy=False)
    sums = np.bincount(
        active_labels,
        weights=active_values,
        minlength=component_count,
    )
    maxima = np.full(component_count, -np.inf, dtype=np.float64)
    np.maximum.at(maxima, active_labels, active_values)

    strongest_component = None
    strongest_score = -math.inf
    total_pixels = float(values.size)
    for label in range(1, component_count):
        area = int(statistics[label, cv2.CC_STAT_AREA])
        area_fraction = area / total_pixels
        mean_score = float(sums[label] / area)
        component_score = mean_score * (1.0 + math.sqrt(area_fraction))
        if component_score <= strongest_score:
            continue
        strongest_score = component_score
        strongest_component = {
            "area": area,
            "area_fraction": area_fraction,
            "mean_score": mean_score,
            "max_score": float(maxima[label]),
            "score": component_score,
            "bbox_xywh": [
                int(statistics[label, cv2.CC_STAT_LEFT]),
                int(statistics[label, cv2.CC_STAT_TOP]),
                int(statistics[label, cv2.CC_STAT_WIDTH]),
                int(statistics[label, cv2.CC_STAT_HEIGHT]),
            ],
        }

    areas = statistics[1:, cv2.CC_STAT_AREA]
    return {
        "component_count": component_count - 1,
        "anomalous_pixel_count": int(binary.sum()),
        "largest_component_area": int(areas.max()),
        "strongest_component": strongest_component,
    }


def image_score_from_statistics(statistics, fallback_score) -> float:
    """Combine a compact component summary with a finite model-score fallback."""
    fallback = float(fallback_score)
    if not math.isfinite(fallback):
        fallback = 0.0
    strongest = statistics["strongest_component"]
    if strongest is None:
        return fallback
    return float(max(fallback, strongest["score"]))


def image_score_from_components(anomaly_map, pixel_threshold, fallback_score) -> float:
    """Compute a resolution-independent score from connected anomaly regions."""
    return image_score_from_statistics(
        component_statistics(anomaly_map, pixel_threshold),
        fallback_score,
    )


def top_k_map_score(anomaly_map, top_k: int) -> float:
    """Return a resolution-stable local score from the final fused map."""
    values = np.asarray(anomaly_map, dtype=np.float32).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("anomaly_map must contain finite values")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    count = min(top_k, values.size)
    return float(np.partition(values, values.size - count)[-count:].mean())


def classify_score(score, threshold, recheck_margin) -> str:
    """Return the conservative three-state image decision for a calibrated score."""
    score = float(score)
    threshold = float(threshold)
    recheck_margin = float(recheck_margin)
    if not (math.isfinite(score) and math.isfinite(threshold)):
        return "RECHECK"
    if not math.isfinite(recheck_margin) or recheck_margin < 0:
        raise ValueError("recheck_margin must be a finite non-negative number")
    if score <= threshold:
        return "OK"
    if score <= threshold + recheck_margin:
        return "RECHECK"
    return "NG"


def apply_quality_gate(decision: str, reasons) -> tuple[str, str | None]:
    """Promote an otherwise acceptable image without hiding an NG decision."""
    if decision == "OK" and reasons:
        return "RECHECK", "quality_gate:" + ",".join(str(reason) for reason in reasons)
    return decision, None
