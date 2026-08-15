import copy
import math
from collections.abc import Mapping

from hiad.constants import (
    TASK_TYPE_DYNAMIC_PATCH,
    TASK_TYPE_REFINEMENT_PATCH,
    TASK_TYPE_THUMBNAIL,
)


REQUIRED_CONFIG_KEYS = (
    "backbone_name",
    "total_iters",
    "thumbnail_total_iters",
    "log_per_steps",
    "bottleneck_dropout",
    "grad_clip_norm",
    "hard_mining_final",
    "hard_mining_warmup_iters",
    "easy_grad_factor",
    "encoder_amp",
    "decoder_amp",
    "allow_tf32",
    "semantic_weight",
    "memory_weight",
    "high_frequency_weight",
    "global_routing_weight",
    "patches_per_source",
    "score_top_k",
    "normal_score_percentile",
    "normal_component_percentile",
    "normal_pixel_percentile",
    "normal_pixel_image_percentile",
    "calibration_batch_size",
    "map_gaussian_sigma",
    "decision_recheck_margin_ratio",
    "min_mean_luminance",
    "max_mean_luminance",
    "max_clipped_fraction",
    "min_focus_variance",
)


def _config_value(config, key):
    if isinstance(config, Mapping):
        if key not in config:
            raise ValueError(f"Missing required config setting: {key}")
        return config[key]
    if not hasattr(config, key):
        raise ValueError(f"Missing required config setting: {key}")
    return getattr(config, key)


def _positive_int(value, key):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")


def _finite_number(value, key):
    if isinstance(value, bool):
        raise ValueError(f"{key} must be a finite number")
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{key} must be a finite number") from error
    if not math.isfinite(value):
        raise ValueError(f"{key} must be a finite number")
    return value


def validate_required_config(config) -> None:
    """Reject incomplete or incompatible production detector configuration."""
    values = {key: _config_value(config, key) for key in REQUIRED_CONFIG_KEYS}
    if not isinstance(values["backbone_name"], str) or not values["backbone_name"].strip():
        raise ValueError("backbone_name must be a non-empty string")
    for key in (
        "total_iters",
        "thumbnail_total_iters",
        "log_per_steps",
        "patches_per_source",
        "score_top_k",
        "calibration_batch_size",
    ):
        _positive_int(values[key], key)
    for key in ("encoder_amp", "decoder_amp", "allow_tf32"):
        if not isinstance(values[key], bool):
            raise ValueError(f"{key} must be a boolean")

    dropout = _finite_number(values["bottleneck_dropout"], "bottleneck_dropout")
    hard_mining = _finite_number(values["hard_mining_final"], "hard_mining_final")
    grad_clip = _finite_number(values["grad_clip_norm"], "grad_clip_norm")
    easy_factor = _finite_number(values["easy_grad_factor"], "easy_grad_factor")
    if not 0 <= dropout < 1:
        raise ValueError("bottleneck_dropout must be in [0, 1)")
    if grad_clip < 0:
        raise ValueError("grad_clip_norm must be non-negative")
    if not 0 <= hard_mining <= 1 or not 0 <= easy_factor <= 1:
        raise ValueError("hard-mining settings must be in [0, 1]")
    if (
        isinstance(values["hard_mining_warmup_iters"], bool)
        or not isinstance(values["hard_mining_warmup_iters"], int)
        or values["hard_mining_warmup_iters"] < 0
    ):
        raise ValueError("hard_mining_warmup_iters must be a non-negative integer")

    evidence_weights = tuple(
        _finite_number(values[key], key)
        for key in ("semantic_weight", "memory_weight", "high_frequency_weight")
    )
    if any(weight < 0 for weight in evidence_weights) or sum(evidence_weights) <= 0:
        raise ValueError("evidence weights must be non-negative with a positive sum")
    if not 0 <= _finite_number(
        values["global_routing_weight"], "global_routing_weight"
    ) <= 1:
        raise ValueError("global_routing_weight must be in [0, 1]")
    for key in (
        "normal_score_percentile",
        "normal_component_percentile",
        "normal_pixel_percentile",
        "normal_pixel_image_percentile",
    ):
        if not 0 < _finite_number(values[key], key) < 1:
            raise ValueError(f"{key} must be in the open interval (0, 1)")
    if _finite_number(values["map_gaussian_sigma"], "map_gaussian_sigma") < 0:
        raise ValueError("map_gaussian_sigma must be non-negative")
    if not 0 <= _finite_number(
        values["decision_recheck_margin_ratio"], "decision_recheck_margin_ratio"
    ) <= 1:
        raise ValueError("decision_recheck_margin_ratio must be in [0, 1]")

    min_luminance = _finite_number(values["min_mean_luminance"], "min_mean_luminance")
    max_luminance = _finite_number(values["max_mean_luminance"], "max_mean_luminance")
    clipped_fraction = _finite_number(values["max_clipped_fraction"], "max_clipped_fraction")
    focus_variance = _finite_number(values["min_focus_variance"], "min_focus_variance")
    if not 0 <= min_luminance < max_luminance <= 1:
        raise ValueError("luminance thresholds must satisfy 0 <= min < max <= 1")
    if not 0 <= clipped_fraction <= 1:
        raise ValueError("max_clipped_fraction must be in [0, 1]")
    if focus_variance < 0:
        raise ValueError("min_focus_variance must be non-negative")


def detector_config_for_task(config, task):
    task_type = task.get("type")
    if task_type == TASK_TYPE_DYNAMIC_PATCH:
        detector_config = copy.deepcopy(config.patch)
        detector_config.patch_size = task["patch_size"]
        return detector_config
    if task_type == TASK_TYPE_REFINEMENT_PATCH:
        detector_config = copy.deepcopy(config.refinement)
        detector_config.patch_size = task["patch_size"]
        return detector_config
    if task_type == TASK_TYPE_THUMBNAIL:
        detector_config = copy.deepcopy(config.thumbnail)
        detector_config.pop("thumbnail_size", None)
        detector_config.patch_size = task["thumbnail_size"]
        detector_config.total_iters = detector_config.thumbnail_total_iters
        return detector_config
    raise ValueError(f"Unsupported task type: {task}")
