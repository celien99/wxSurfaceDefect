import math
from collections.abc import Mapping
from numbers import Real

import torch

from hiad.constants import SUPPORTED_ANOMALY_DISTANCES


DETECTOR_CHECKPOINT_KEYS = {
    "schema_version",
    "bottleneck",
    "decoder",
    "anomaly_distance",
    "fusion_weights",
    "use_fp16",
}


def _validate_model_state(state, name: str) -> None:
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"{name} state must be a non-empty mapping")
    for key, value in state.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} state keys must be non-empty strings")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} state value {key!r} must be a tensor")


def validate_detector_checkpoint(
    payload,
    *,
    expected_version: int,
) -> dict:
    if not isinstance(payload, Mapping):
        raise TypeError("Detector checkpoint must contain a mapping")
    state = dict(payload)
    if set(state) != DETECTOR_CHECKPOINT_KEYS:
        raise ValueError("Detector checkpoint has unexpected or missing keys")
    if state["schema_version"] != expected_version:
        raise ValueError("Unsupported detector checkpoint schema version")

    _validate_model_state(state["bottleneck"], "bottleneck")
    _validate_model_state(state["decoder"], "decoder")

    if state["anomaly_distance"] not in SUPPORTED_ANOMALY_DISTANCES:
        raise ValueError("Detector checkpoint anomaly_distance is invalid")
    if not isinstance(state["use_fp16"], bool):
        raise TypeError("Detector checkpoint use_fp16 must be a boolean")
    fusion_weights = state["fusion_weights"]
    if fusion_weights is not None:
        if not isinstance(fusion_weights, (list, tuple)) or not fusion_weights:
            raise ValueError("fusion_weights must be absent or a non-empty sequence")
        if any(
            isinstance(weight, bool)
            or not isinstance(weight, Real)
            or not math.isfinite(float(weight))
            or float(weight) < 0
            for weight in fusion_weights
        ):
            raise ValueError("fusion_weights must contain finite non-negative values")
        if sum(float(weight) for weight in fusion_weights) <= 0:
            raise ValueError("fusion_weights must have a positive sum")
    return state
