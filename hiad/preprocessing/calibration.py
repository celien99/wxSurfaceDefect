from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from hiad.checkpoints import atomic_torch_save, atomic_write_json

from .artifacts import (
    atomic_write_bytes,
    encode_binary_rle,
    load_reference_assets,
    load_reference_entry,
    preprocessing_bundle_root,
    sha256_file,
    sha256_state_dict,
    validate_registry_category,
)
from .config import canonicalize_preprocessing_config
from .constants import (
    PREPROCESSING_CONFIG_FILE,
    PREPROCESSING_MANIFEST_FILE,
    PREPROCESSING_REGISTRY_FILE,
    PREPROCESSING_REGISTRY_SCHEMA_VERSION,
    PREPROCESSING_SCHEMA_VERSION,
    PROTOTYPES_FILE,
    REFERENCE_MASK_FILE,
    REFERENCE_TEMPLATE_FILE,
)
from .dino import build_frozen_dino_encoder, extract_dino_grid
from .images import validate_image_array
from .sam import load_sam2_components, sam2_longest_edge


def _validate_categories(categories: Sequence[str]) -> tuple[str, ...]:
    if isinstance(categories, (str, bytes)):
        raise TypeError("categories must be a sequence of category names")
    normalized = tuple(validate_registry_category(category) for category in categories)
    if not normalized:
        raise ValueError("categories must not be empty")
    if tuple(sorted(set(normalized))) != normalized:
        raise ValueError("categories must be sorted and unique")
    return normalized


def _write_preprocessing_bundle(
    canonical_config: Mapping[str, Any],
    category: str,
    checkpoint_root: str,
    device: torch.device,
    *,
    encoder,
    working_longest_edge: int,
    dino_weights_hash: str,
    sam2_weights_hash: str,
    sam2_revision: str | None,
    logger=None,
) -> None:
    reference_rgb = None
    reference_mask = None
    try:
        reference_image_path, reference_mask_path = load_reference_entry(
            canonical_config["reference_manifest"],
            category,
        )
        reference_hashes = {
            "image": sha256_file(reference_image_path),
            "mask": sha256_file(reference_mask_path),
        }
        reference_rgb, reference_mask = load_reference_assets(
            reference_image_path,
            reference_mask_path,
        )
        if reference_hashes != {
            "image": sha256_file(reference_image_path),
            "mask": sha256_file(reference_mask_path),
        }:
            raise RuntimeError("Reference assets changed during calibration")
        reference_rgb = validate_image_array(
            reference_rgb,
            canonical_config["input_scale"],
            "RGB",
        )
        features, centers_xy, grid_size, working_size = extract_dino_grid(
            encoder,
            reference_rgb,
            canonical_config,
            device,
            working_longest_edge,
        )

        grid_height, grid_width = grid_size
        foreground_cells = cv2.resize(
            reference_mask.astype(np.uint8),
            (grid_width, grid_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool).reshape(-1)
        if not foreground_cells.any() or foreground_cells.all():
            raise ValueError(
                "Reference DINO grid must contain foreground and background cells"
            )

        foreground_features = features[torch.from_numpy(foreground_cells)]
        background_features = features[torch.from_numpy(~foreground_cells)]
        foreground_mean = foreground_features.mean(dim=0)
        background_mean = background_features.mean(dim=0)
        if foreground_mean.norm() <= torch.finfo(torch.float32).eps:
            raise ValueError("Reference foreground prototype has zero norm")
        if background_mean.norm() <= torch.finfo(torch.float32).eps:
            raise ValueError("Reference background prototype has zero norm")
        foreground_prototype = F.normalize(foreground_mean, dim=0)
        background_prototype = F.normalize(background_mean, dim=0)
        if not torch.isfinite(foreground_prototype).all() or not torch.isfinite(
            background_prototype
        ).all():
            raise ValueError("Reference DINO prototypes contain NaN or infinite values")

        os.makedirs(checkpoint_root, exist_ok=True)
        config_path = os.path.join(checkpoint_root, PREPROCESSING_CONFIG_FILE)
        prototypes_path = os.path.join(checkpoint_root, PROTOTYPES_FILE)
        template_path = os.path.join(checkpoint_root, REFERENCE_TEMPLATE_FILE)
        reference_mask_output_path = os.path.join(checkpoint_root, REFERENCE_MASK_FILE)
        manifest_path = os.path.join(checkpoint_root, PREPROCESSING_MANIFEST_FILE)

        config_bytes = yaml.safe_dump(
            canonical_config,
            sort_keys=True,
            allow_unicode=False,
        ).encode("utf-8")
        rle_bytes = json.dumps(
            encode_binary_rle(reference_mask),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        atomic_write_bytes(config_path, config_bytes)
        atomic_torch_save(
            {
                "foreground": foreground_prototype.cpu().float(),
                "background": background_prototype.cpu().float(),
                "foreground_count": int(foreground_features.shape[0]),
                "background_count": int(background_features.shape[0]),
            },
            prototypes_path,
        )
        atomic_torch_save(
            {
                "features": features.half(),
                "centers_xy": torch.from_numpy(centers_xy),
                "foreground_cells": torch.from_numpy(foreground_cells.copy()),
                "original_size": tuple(reference_mask.shape),
                "working_size": working_size,
                "grid_size": grid_size,
                "patch_size": encoder.patch_size,
                "feature_layer": canonical_config["dino_feature_layer"],
                "working_longest_edge": working_longest_edge,
            },
            template_path,
        )
        atomic_write_bytes(reference_mask_output_path, rle_bytes)

        asset_hashes = {
            PREPROCESSING_CONFIG_FILE: sha256_file(config_path),
            PROTOTYPES_FILE: sha256_file(prototypes_path),
            REFERENCE_TEMPLATE_FILE: sha256_file(template_path),
            REFERENCE_MASK_FILE: sha256_file(reference_mask_output_path),
        }
        manifest = {
            "schema_version": PREPROCESSING_SCHEMA_VERSION,
            "category": category,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dino": {
                "backbone_name": canonical_config["dino_backbone_name"],
                "feature_layer": canonical_config["dino_feature_layer"],
                "patch_size": encoder.patch_size,
                "weights_sha256": dino_weights_hash,
            },
            "sam2": {
                "model_id": canonical_config["sam2_model_id"],
                "revision": sam2_revision,
                "weights_sha256": sam2_weights_hash,
            },
            "normalization": {
                "input_scale": canonical_config["input_scale"],
                "mean": canonical_config["mean"],
                "std": canonical_config["std"],
            },
            "reference_sha256": reference_hashes,
            "asset_sha256": asset_hashes,
        }
        atomic_write_bytes(
            manifest_path,
            json.dumps(manifest, sort_keys=True, indent=2).encode("utf-8") + b"\n",
        )
        if logger is not None:
            logger.info(
                "Preprocessing bundle calibrated for clsname %r at %s",
                category,
                checkpoint_root,
            )
    finally:
        del reference_rgb, reference_mask


def calibrate_preprocessing_registry(
    config: Mapping[str, Any],
    categories: Sequence[str],
    generation_root: str,
    device: torch.device,
    logger=None,
) -> tuple[str, ...]:
    canonical_config = canonicalize_preprocessing_config(config)
    categories = _validate_categories(categories)
    generation_root = os.path.abspath(generation_root)
    os.makedirs(generation_root, exist_ok=True)

    encoder = None
    processor = None
    sam2_model = None
    try:
        encoder = build_frozen_dino_encoder(canonical_config)
        sam2_model, processor = load_sam2_components(canonical_config["sam2_model_id"])
        working_longest_edge = sam2_longest_edge(processor)
        dino_weights_hash = sha256_state_dict(encoder.state_dict())
        sam2_weights_hash = sha256_state_dict(sam2_model.state_dict())
        sam2_revision = getattr(sam2_model.config, "_commit_hash", None)
        if sam2_revision is not None and (
            not isinstance(sam2_revision, str) or not sam2_revision.strip()
        ):
            raise ValueError("Loaded SAM2 revision metadata is invalid")

        for category in categories:
            _write_preprocessing_bundle(
                canonical_config,
                category,
                str(preprocessing_bundle_root(generation_root, category)),
                device,
                encoder=encoder,
                working_longest_edge=working_longest_edge,
                dino_weights_hash=dino_weights_hash,
                sam2_weights_hash=sam2_weights_hash,
                sam2_revision=sam2_revision,
                logger=logger,
            )
        atomic_write_json(
            {
                "schema_version": PREPROCESSING_REGISTRY_SCHEMA_VERSION,
                "categories": list(categories),
            },
            os.path.join(generation_root, PREPROCESSING_REGISTRY_FILE),
        )
        return categories
    finally:
        if encoder is not None:
            encoder.requires_grad_(False)
            encoder.eval()
            encoder.cpu()
        if sam2_model is not None:
            sam2_model.cpu()
        del encoder, processor, sam2_model
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
