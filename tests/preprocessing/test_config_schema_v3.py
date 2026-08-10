from __future__ import annotations

import copy

import pytest

from hiad.preprocessing.config import canonicalize_preprocessing_config
from hiad.preprocessing.constants import PREPROCESSING_SCHEMA_VERSION


def _valid_v3_config() -> dict:
    return {
        "schema_version": 3,
        "array_color_space": "RGB",
        "input_scale": 255.0,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "reference_manifest": "configs/foreground_references.yaml",
        "dino_backbone_name": "vit_base_patch16_dinov3.lvd1689m",
        "dino_feature_layer": -1,
        "working_longest_edge": 1024,
        "boundary_expand_ratio": 0.01,
        "min_dino_matches": 8,
        "min_dino_inlier_ratio": 0.50,
        "max_dino_reprojection_ratio": 0.03,
        "max_area_ratio_deviation": 0.35,
        "min_reference_coverage": 1.0,
    }


def test_canonicalize_accepts_minimal_v3_config():
    canonical = canonicalize_preprocessing_config(_valid_v3_config())
    assert canonical["schema_version"] == 3
    assert canonical["schema_version"] == PREPROCESSING_SCHEMA_VERSION
    assert canonical["working_longest_edge"] == 1024


def test_canonicalize_rejects_sam_keys():
    raw = _valid_v3_config()
    raw["sam2_model_id"] = "x"
    with pytest.raises(ValueError):
        canonicalize_preprocessing_config(raw)


def test_canonicalize_requires_working_longest_edge():
    raw = _valid_v3_config()
    del raw["working_longest_edge"]
    with pytest.raises((KeyError, ValueError)):
        canonicalize_preprocessing_config(raw)


def test_canonicalize_rejects_non_patch_multiple_longest_edge():
    raw = _valid_v3_config()
    raw["working_longest_edge"] = 1023
    with pytest.raises(ValueError):
        canonicalize_preprocessing_config(raw)


def test_canonicalize_rejects_other_sam_keys():
    for key, value in (
        ("sam2_dtype", "float16"),
        ("sam2_batch_size", 1),
        ("min_sam_prior_iou", 0.5),
    ):
        raw = _valid_v3_config()
        raw[key] = value
        with pytest.raises(ValueError):
            canonicalize_preprocessing_config(copy.deepcopy(raw))
