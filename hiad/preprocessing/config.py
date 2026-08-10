from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from hiad.constants import DINO_PATCH_SIZE

from .constants import CONFIG_KEYS, PREPROCESSING_SCHEMA_VERSION


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(f"{name} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _finite_triplet(value: Any, name: str) -> list[float]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must contain exactly three finite values")
    try:
        items = list(value)
    except TypeError as error:
        raise TypeError(f"{name} must contain exactly three finite values") from error
    if len(items) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    return [_finite_float(item, f"{name}[{index}]") for index, item in enumerate(items)]


def require_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-empty string")
    return value.strip()


def canonicalize_preprocessing_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise TypeError("preprocessing config must be a mapping")
    if any(not isinstance(key, str) for key in config):
        raise TypeError("preprocessing config keys must be strings")

    missing = sorted(set(CONFIG_KEYS) - set(config))
    unknown = sorted(set(config) - set(CONFIG_KEYS))
    if missing:
        raise ValueError(f"Missing preprocessing config keys: {missing}")
    if unknown:
        raise ValueError(f"Unknown preprocessing config keys: {unknown}")

    schema_version = config["schema_version"]
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise TypeError("schema_version must be an integer")
    if schema_version != PREPROCESSING_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported preprocessing schema_version: {schema_version}; "
            f"expected {PREPROCESSING_SCHEMA_VERSION}"
        )

    color_space = require_nonempty_string(
        config["array_color_space"],
        "array_color_space",
    ).upper()
    if color_space not in {"RGB", "BGR"}:
        raise ValueError("array_color_space must be RGB or BGR")

    input_scale = _finite_float(config["input_scale"], "input_scale")
    if input_scale <= 0:
        raise ValueError("input_scale must be positive")

    mean = _finite_triplet(config["mean"], "mean")
    std = _finite_triplet(config["std"], "std")
    if any(value <= 0 for value in std):
        raise ValueError("std values must be positive")

    feature_layer = config["dino_feature_layer"]
    if isinstance(feature_layer, bool) or not isinstance(feature_layer, int):
        raise TypeError("dino_feature_layer must be an integer")

    working_longest_edge = config["working_longest_edge"]
    if isinstance(working_longest_edge, bool) or not isinstance(
        working_longest_edge, int
    ):
        raise TypeError("working_longest_edge must be an integer")
    if working_longest_edge <= 0:
        raise ValueError("working_longest_edge must be positive")
    if working_longest_edge % DINO_PATCH_SIZE != 0:
        raise ValueError(
            f"working_longest_edge must be a positive multiple of {DINO_PATCH_SIZE}"
        )

    min_dino_matches = config["min_dino_matches"]
    if isinstance(min_dino_matches, bool) or not isinstance(min_dino_matches, int):
        raise TypeError("min_dino_matches must be an integer")
    if min_dino_matches <= 0:
        raise ValueError("min_dino_matches must be positive")

    ratios = {
        "boundary_expand_ratio": _finite_float(
            config["boundary_expand_ratio"],
            "boundary_expand_ratio",
        ),
        "min_dino_inlier_ratio": _finite_float(
            config["min_dino_inlier_ratio"],
            "min_dino_inlier_ratio",
        ),
        "max_dino_reprojection_ratio": _finite_float(
            config["max_dino_reprojection_ratio"],
            "max_dino_reprojection_ratio",
        ),
        "min_reference_coverage": _finite_float(
            config["min_reference_coverage"],
            "min_reference_coverage",
        ),
    }
    for name, value in ratios.items():
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1]")

    max_area_ratio_deviation = _finite_float(
        config["max_area_ratio_deviation"],
        "max_area_ratio_deviation",
    )
    if max_area_ratio_deviation < 0:
        raise ValueError("max_area_ratio_deviation must be non-negative")

    return {
        "schema_version": schema_version,
        "array_color_space": color_space,
        "input_scale": input_scale,
        "mean": mean,
        "std": std,
        "reference_manifest": require_nonempty_string(
            config["reference_manifest"],
            "reference_manifest",
        ),
        "dino_backbone_name": require_nonempty_string(
            config["dino_backbone_name"],
            "dino_backbone_name",
        ),
        "dino_feature_layer": feature_layer,
        "working_longest_edge": working_longest_edge,
        "boundary_expand_ratio": ratios["boundary_expand_ratio"],
        "min_dino_matches": min_dino_matches,
        "min_dino_inlier_ratio": ratios["min_dino_inlier_ratio"],
        "max_dino_reprojection_ratio": ratios["max_dino_reprojection_ratio"],
        "max_area_ratio_deviation": max_area_ratio_deviation,
        "min_reference_coverage": ratios["min_reference_coverage"],
    }
