import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from hiad.checkpoints import atomic_write_json

from .config import MultiRiskConfig
from .contracts import CalibratedImageResult, RawImageScores, RawSubscore


MULTIRISK_CALIBRATION_FILE = "multirisk_calibration.json"
MULTIRISK_CALIBRATION_SCHEMA_VERSION = 2
MULTIRISK_CALIBRATION_DOMAIN = "full_image_multirisk_token_v2"

_REQUIRED_ARTIFACT_KEYS = {
    "schema_version",
    "domain",
    "normal_image_count",
    "decision_percentile",
    "scoring_config",
    "scoring_config_sha256",
    "branch_sorted_scores",
    "joint_sorted_scores",
    "raw_diagnostics",
    "warnings",
}


def _config_fingerprint(config_payload: Mapping) -> str:
    encoded = json.dumps(
        dict(config_payload),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sorted_finite(values, name: str, *, expected_count: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a non-empty finite one-dimensional sequence")
    if expected_count is not None and array.size != expected_count:
        raise ValueError(f"{name} must contain {expected_count} values")
    if np.any(array[1:] < array[:-1]):
        raise ValueError(f"{name} must be sorted in ascending order")
    return array


def _midrank_percentile(sorted_scores: np.ndarray, value: float) -> float:
    if not np.isfinite(value):
        raise ValueError("ECDF values must be finite")
    left = int(np.searchsorted(sorted_scores, value, side="left"))
    right = int(np.searchsorted(sorted_scores, value, side="right"))
    return float((left + 0.5 * (right - left)) / sorted_scores.size)


def _tail_sample_warnings(
    normal_count: int,
    decision_percentile: float,
) -> list[str]:
    minimum_count = math.ceil(1.0 / (1.0 - decision_percentile))
    warnings = []
    if normal_count < minimum_count:
        warnings.append(
            f"Decision percentile {decision_percentile:.6g} is under-resolved: "
            f"{normal_count} independent normal images provide empirical tail "
            f"resolution {1.0 / normal_count:.6g}; {minimum_count} would resolve "
            "the requested tail"
        )
    if decision_percentile >= 0.995 and normal_count < 1000:
        warnings.append(
            "P99.5 tail estimate uses fewer than 1000 independent normal images"
        )
    return warnings


def _validate_raw_scores(
    raw_scores: RawImageScores,
    subscore_order: tuple[str, ...],
) -> dict[str, RawSubscore]:
    if not isinstance(raw_scores, RawImageScores):
        raise TypeError("Calibration inputs must be RawImageScores")
    by_key = raw_scores.by_key()
    unexpected = set(by_key).difference(subscore_order)
    if unexpected:
        raise ValueError(f"Raw scores contain unconfigured keys: {sorted(unexpected)}")
    for key, subscore in by_key.items():
        expected_branch = "peak" if key == "peak" else key.split(".", 1)[0]
        if subscore.branch != expected_branch:
            raise ValueError(f"Raw subscore {key} has an inconsistent branch")
    for branch in ("peak", "region", "line", "context"):
        if not any(subscore.branch == branch for subscore in by_key.values()):
            raise ValueError(f"Raw scores are missing the {branch} branch")
    return by_key


def _branch_risks(
    raw_scores: RawImageScores,
    sorted_scores: Mapping[str, np.ndarray],
    subscore_order: tuple[str, ...],
    branch_order: tuple[str, ...],
) -> dict[str, tuple[float, RawSubscore]]:
    by_key = _validate_raw_scores(raw_scores, subscore_order)
    risks = {}
    for branch in branch_order:
        winner = None
        for key in subscore_order:
            subscore = by_key.get(key)
            if subscore is None or subscore.branch != branch:
                continue
            percentile = _midrank_percentile(sorted_scores[key], subscore.value)
            if winner is None or percentile > winner[0]:
                winner = (percentile, subscore)
        if winner is None:
            raise ValueError(f"No applicable subscore exists for branch {branch}")
        risks[branch] = winner
    return risks


def build_calibration(
    raw_scores: Sequence[RawImageScores],
    scoring_config: MultiRiskConfig,
) -> dict:
    if not isinstance(scoring_config, MultiRiskConfig):
        raise TypeError("scoring_config must be MultiRiskConfig")
    samples = tuple(raw_scores)
    if not samples:
        raise ValueError("Calibration requires normal full-image scores")
    normal_count = len(samples)

    subscore_order = scoring_config.subscore_order
    values_by_key = {key: [] for key in subscore_order}
    for sample in samples:
        by_key = _validate_raw_scores(sample, subscore_order)
        for key, subscore in by_key.items():
            values_by_key[key].append(float(subscore.value))

    sorted_scores = {}
    diagnostics = {}
    for key in subscore_order:
        values = np.asarray(values_by_key[key], dtype=np.float64)
        if values.size == 0:
            raise ValueError(f"Calibration has no applicable values for {key}")
        if not np.isfinite(values).all():
            raise ValueError(f"Calibration values for {key} must be finite")
        sorted_values = np.sort(values)
        sorted_scores[key] = sorted_values
        diagnostics[key] = {
            "count": int(values.size),
            "min": float(values.min()),
            "max": float(values.max()),
            "p99_9": float(np.percentile(values, 99.9)),
        }

    joint_risks = []
    for sample in samples:
        branch_risks = _branch_risks(
            sample,
            sorted_scores,
            subscore_order,
            scoring_config.branch_order,
        )
        joint_risks.append(max(risk for risk, _ in branch_risks.values()))

    warnings = _tail_sample_warnings(
        normal_count,
        scoring_config.decision_percentile,
    )
    payload = {
        "schema_version": MULTIRISK_CALIBRATION_SCHEMA_VERSION,
        "domain": MULTIRISK_CALIBRATION_DOMAIN,
        "normal_image_count": normal_count,
        "decision_percentile": scoring_config.decision_percentile,
        "scoring_config": scoring_config.to_dict(),
        "scoring_config_sha256": scoring_config.fingerprint,
        "branch_sorted_scores": {
            key: sorted_scores[key].astype(float).tolist()
            for key in subscore_order
        },
        "joint_sorted_scores": np.sort(np.asarray(joint_risks, dtype=np.float64)).tolist(),
        "raw_diagnostics": diagnostics,
        "warnings": warnings,
    }
    return _validate_calibration(payload, expected_config=scoring_config)


def _validate_calibration(
    payload,
    *,
    expected_config: MultiRiskConfig | None = None,
) -> dict:
    if not isinstance(payload, Mapping):
        raise TypeError("Multi-risk calibration must contain a mapping")
    calibration = dict(payload)
    if set(calibration) != _REQUIRED_ARTIFACT_KEYS:
        raise ValueError("Multi-risk calibration has an invalid schema")
    if calibration["schema_version"] != MULTIRISK_CALIBRATION_SCHEMA_VERSION:
        raise ValueError("Unsupported multi-risk calibration schema version")
    if calibration["domain"] != MULTIRISK_CALIBRATION_DOMAIN:
        raise ValueError("Unsupported multi-risk calibration domain")

    normal_count = calibration["normal_image_count"]
    if isinstance(normal_count, bool) or not isinstance(normal_count, int) or normal_count <= 0:
        raise ValueError("normal_image_count must be a positive integer")
    decision_percentile = calibration["decision_percentile"]
    if (
        isinstance(decision_percentile, bool)
        or not isinstance(decision_percentile, (int, float))
        or not np.isfinite(decision_percentile)
        or not 0 < decision_percentile < 1
    ):
        raise ValueError("decision_percentile must be finite and in (0, 1)")
    config_payload = calibration["scoring_config"]
    if not isinstance(config_payload, Mapping):
        raise TypeError("scoring_config must be a mapping")
    fingerprint = calibration["scoring_config_sha256"]
    if not isinstance(fingerprint, str) or fingerprint != _config_fingerprint(config_payload):
        raise ValueError("Multi-risk scoring configuration fingerprint is invalid")
    if expected_config is not None:
        if fingerprint != expected_config.fingerprint or dict(config_payload) != expected_config.to_dict():
            raise ValueError("Runtime scoring configuration differs from calibration")

    config_decision_percentile = config_payload.get("decision_percentile")
    if (
        isinstance(config_decision_percentile, bool)
        or not isinstance(config_decision_percentile, (int, float))
        or float(config_decision_percentile) != float(decision_percentile)
    ):
        raise ValueError("Artifact and scoring configuration decision percentiles differ")
    subscore_order = config_payload.get("subscore_order")
    branch_order = config_payload.get("branch_order")
    if (
        not isinstance(subscore_order, list)
        or not subscore_order
        or any(not isinstance(key, str) or not key for key in subscore_order)
        or len(subscore_order) != len(set(subscore_order))
    ):
        raise ValueError("Calibration subscore_order is invalid")
    if (
        not isinstance(branch_order, list)
        or any(not isinstance(branch, str) for branch in branch_order)
        or len(branch_order) != len(set(branch_order))
    ):
        raise ValueError("Calibration branch_order is invalid")
    if set(branch_order) != {"peak", "region", "line", "context"}:
        raise ValueError("Calibration branch_order is incomplete")

    score_payload = calibration["branch_sorted_scores"]
    if not isinstance(score_payload, Mapping) or set(score_payload) != set(subscore_order):
        raise ValueError("Calibration subscore distributions are incomplete")
    sorted_scores = {}
    for key in subscore_order:
        values = _sorted_finite(score_payload[key], f"branch_sorted_scores.{key}")
        if values.size > normal_count:
            raise ValueError(f"Calibration distribution {key} exceeds normal_image_count")
        sorted_scores[key] = values

    diagnostics = calibration["raw_diagnostics"]
    if not isinstance(diagnostics, Mapping) or set(diagnostics) != set(subscore_order):
        raise ValueError("Calibration raw diagnostics are incomplete")
    for key, values in sorted_scores.items():
        diagnostic = diagnostics[key]
        if not isinstance(diagnostic, Mapping) or set(diagnostic) != {"count", "min", "max", "p99_9"}:
            raise ValueError(f"Calibration diagnostics for {key} have an invalid schema")
        expected = np.asarray(
            [values.min(), values.max(), np.percentile(values, 99.9)],
            dtype=np.float64,
        )
        actual = np.asarray(
            [diagnostic["min"], diagnostic["max"], diagnostic["p99_9"]],
            dtype=np.float64,
        )
        diagnostic_count = diagnostic["count"]
        if (
            isinstance(diagnostic_count, bool)
            or not isinstance(diagnostic_count, int)
            or diagnostic_count != values.size
            or not np.isfinite(actual).all()
        ):
            raise ValueError(f"Calibration diagnostics for {key} are invalid")
        if not np.allclose(actual, expected, rtol=0.0, atol=1e-12):
            raise ValueError(f"Calibration diagnostics for {key} do not match scores")

    joint_scores = _sorted_finite(
        calibration["joint_sorted_scores"],
        "joint_sorted_scores",
        expected_count=normal_count,
    )
    if np.any(joint_scores < 0) or np.any(joint_scores > 1):
        raise ValueError("Joint calibration risks must lie in [0, 1]")
    warnings = calibration["warnings"]
    if not isinstance(warnings, list) or any(not isinstance(item, str) for item in warnings):
        raise ValueError("Calibration warnings must be a list of strings")
    expected_warnings = _tail_sample_warnings(
        normal_count,
        float(decision_percentile),
    )
    if warnings != expected_warnings:
        raise ValueError("Calibration warnings do not match its tail sample state")
    return calibration


def save_calibration(calibration, generation_root) -> None:
    payload = _validate_calibration(calibration)
    atomic_write_json(payload, Path(generation_root) / MULTIRISK_CALIBRATION_FILE)


def load_calibration(
    generation_root,
    scoring_config: MultiRiskConfig,
) -> dict:
    path = Path(generation_root) / MULTIRISK_CALIBRATION_FILE
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return _validate_calibration(payload, expected_config=scoring_config)


def calibrate_image(
    raw_scores: RawImageScores,
    calibration,
) -> CalibratedImageResult:
    calibration = _validate_calibration(calibration)
    return _calibrate_image_validated(raw_scores, calibration)


def _calibrate_image_validated(
    raw_scores: RawImageScores,
    calibration: dict,
) -> CalibratedImageResult:
    config_payload = calibration["scoring_config"]
    subscore_order = tuple(config_payload["subscore_order"])
    branch_order = tuple(config_payload["branch_order"])
    sorted_scores = {
        key: np.asarray(values, dtype=np.float64)
        for key, values in calibration["branch_sorted_scores"].items()
    }
    branch_risks = _branch_risks(
        raw_scores,
        sorted_scores,
        subscore_order,
        branch_order,
    )
    dominant_branch = branch_order[0]
    for branch in branch_order[1:]:
        if branch_risks[branch][0] > branch_risks[dominant_branch][0]:
            dominant_branch = branch
    raw_joint_risk = branch_risks[dominant_branch][0]
    joint_percentile = _midrank_percentile(
        np.asarray(calibration["joint_sorted_scores"], dtype=np.float64),
        raw_joint_risk,
    )
    decision_percentile = float(calibration["decision_percentile"])
    defect_margin = max(0.0, joint_percentile - decision_percentile) / (
        1.0 - decision_percentile
    )
    raw_by_branch = {branch: {} for branch in branch_order}
    for subscore in raw_scores.subscores:
        raw_by_branch[subscore.branch][subscore.key] = float(subscore.value)
    return CalibratedImageResult(
        is_defect=joint_percentile >= decision_percentile,
        decision_percentile=decision_percentile,
        joint_percentile=joint_percentile,
        defect_margin=defect_margin,
        dominant_branch=dominant_branch,
        branch_percentiles={
            branch: branch_risks[branch][0]
            for branch in branch_order
        },
        raw_branch_scores=raw_by_branch,
        candidate_location=branch_risks[dominant_branch][1].candidate,
    )


def calibrate_batch(
    raw_scores: Sequence[RawImageScores],
    calibration,
) -> list[CalibratedImageResult]:
    calibration = _validate_calibration(calibration)
    return [_calibrate_image_validated(scores, calibration) for scores in raw_scores]
