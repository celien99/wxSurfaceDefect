from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

from hiad.checkpoints import safe_torch_load

from .config import canonicalize_preprocessing_config, require_nonempty_string
from .constants import (
    CONFIG_KEYS,
    PREPROCESSING_DIRECTORY,
    PREPROCESSING_CONFIG_FILE,
    PREPROCESSING_MANIFEST_FILE,
    PREPROCESSING_REGISTRY_FILE,
    PREPROCESSING_REGISTRY_SCHEMA_VERSION,
    PREPROCESSING_SCHEMA_VERSION,
    PROTOTYPES_FILE,
    REFERENCE_MASK_FILE,
    REFERENCE_TEMPLATE_FILE,
)


def validate_registry_category(category: str) -> str:
    category = require_nonempty_string(category, "category")
    if "/" in category or "\\" in category:
        raise ValueError("category must not contain a path separator")
    category_path = Path(category)
    if category_path.is_absolute() or category_path.parts != (category,):
        raise ValueError("category must be a single relative path component")
    if category in {".", ".."}:
        raise ValueError("category must not be a traversal component")
    return category


def preprocessing_bundle_root(generation_root, category: str) -> Path:
    category = validate_registry_category(category)
    generation_root = Path(generation_root).resolve()
    preprocessing_root = (generation_root / PREPROCESSING_DIRECTORY).resolve()
    bundle_root = (preprocessing_root / category).resolve()
    if bundle_root.parent != preprocessing_root:
        raise ValueError("category preprocessing bundle escapes the generation root")
    return bundle_root


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_state_dict(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Model state entry {name!r} is not a tensor")
        cpu_tensor = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(cpu_tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        shape_bytes = json.dumps(list(cpu_tensor.shape), separators=(",", ":")).encode("ascii")
        digest.update(shape_bytes)
        digest.update(b"\0")
        digest.update(cpu_tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
        digest.update(b"\0")
    return digest.hexdigest()


def load_reference_entry(reference_manifest: str, category: str) -> tuple[str, str]:
    category = require_nonempty_string(category, "category")
    manifest_path = os.path.abspath(reference_manifest)
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Reference manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as file:
        manifest = yaml.safe_load(file)
    if not isinstance(manifest, Mapping):
        raise ValueError("Reference manifest must be a mapping")
    categories = manifest.get("categories")
    if not isinstance(categories, Mapping):
        raise ValueError("Reference manifest must contain a categories mapping")
    if category not in categories:
        raise KeyError(f"Reference manifest has no entry for category {category!r}")

    entry = categories[category]
    if not isinstance(entry, Mapping) or set(entry) != {"image", "mask"}:
        raise ValueError(
            f"Reference entry for category {category!r} must contain only image and mask"
        )
    image_path = os.path.abspath(require_nonempty_string(entry["image"], "reference image"))
    mask_path = os.path.abspath(require_nonempty_string(entry["mask"], "reference mask"))
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Reference image not found: {image_path}")
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(f"Reference mask not found: {mask_path}")
    return image_path, mask_path


def load_reference_assets(image_path: str, mask_path: str) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(image_path) as image_file:
        rgb = np.array(image_file.convert("RGB"), dtype=np.uint8, copy=True)
    with Image.open(mask_path) as mask_file:
        mask_values = np.array(mask_file.convert("L"), copy=True)

    if rgb.shape[:2] != mask_values.shape:
        raise ValueError(f"Reference image/mask dimensions differ: {rgb.shape[:2]} vs {mask_values.shape}")
    unique_values = set(np.unique(mask_values).tolist())
    if unique_values not in ({0, 1}, {0, 255}):
        raise ValueError(
            f"Reference mask must be binary with values {{0, 1}} or {{0, 255}}, got "
            f"{sorted(unique_values)}"
        )
    mask = mask_values != 0
    if not mask.any() or mask.all():
        raise ValueError("Reference mask must contain both foreground and background")

    component_count, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if component_count != 2:
        raise ValueError("Reference mask must contain exactly one connected foreground object")
    return np.ascontiguousarray(rgb), np.ascontiguousarray(mask)


def encode_binary_rle(mask: np.ndarray) -> dict[str, Any]:
    array = np.asarray(mask)
    if array.ndim != 2 or array.dtype != np.bool_:
        raise TypeError("RLE input must be a two-dimensional boolean array")
    height, width = array.shape
    if height <= 0 or width <= 0:
        raise ValueError("RLE input must be non-empty")

    counts: list[int] = []
    current = 0
    count = 0
    for value in array.reshape(-1, order="C"):
        pixel = int(value)
        if pixel == current:
            count += 1
        else:
            counts.append(count)
            count = 1
            current = pixel
    counts.append(count)
    return {"size": [height, width], "counts": counts}


def decode_binary_rle(payload: Mapping[str, Any]) -> np.ndarray:
    if not isinstance(payload, Mapping) or set(payload) != {"size", "counts"}:
        raise ValueError("RLE payload must contain only size and counts")
    size = payload["size"]
    if not isinstance(size, list) or len(size) != 2:
        raise ValueError("RLE size must be [height, width]")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in size):
        raise ValueError("RLE height and width must be positive integers")

    counts = payload["counts"]
    if not isinstance(counts, list) or not counts:
        raise ValueError("RLE counts must be a non-empty list")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
        raise ValueError("RLE counts must be non-negative integers")

    expected_pixels = size[0] * size[1]
    if sum(counts) != expected_pixels:
        raise ValueError("RLE counts do not match the declared mask size")
    flat = np.empty(expected_pixels, dtype=np.bool_)
    offset = 0
    value = False
    for count in counts:
        flat[offset:offset + count] = value
        offset += count
        value = not value
    return flat.reshape((size[0], size[1]), order="C")


def atomic_write_bytes(path: str, payload: bytes) -> None:
    temporary_path = f"{path}.tmp-{os.getpid()}"
    try:
        with open(temporary_path, "wb") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def load_json_mapping(path: str, name: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return payload


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 hex digest")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a SHA-256 hex digest") from error
    return value.lower()


def validate_preprocessing_bundle(
    checkpoint_root: str,
    runtime_config: Mapping[str, Any] | None = None,
    expected_category: str | None = None,
) -> dict[str, Any]:
    checkpoint_root = os.path.abspath(checkpoint_root)
    asset_names = (
        PREPROCESSING_CONFIG_FILE,
        PROTOTYPES_FILE,
        REFERENCE_TEMPLATE_FILE,
        REFERENCE_MASK_FILE,
    )
    paths = {name: os.path.join(checkpoint_root, name) for name in (*asset_names, PREPROCESSING_MANIFEST_FILE)}
    for name, path in paths.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Preprocessing bundle file {name!r} not found: {path}")

    with open(paths[PREPROCESSING_CONFIG_FILE], "r", encoding="utf-8") as file:
        saved_config = canonicalize_preprocessing_config(yaml.safe_load(file))
    if runtime_config is not None:
        canonical_runtime = canonicalize_preprocessing_config(runtime_config)
        if canonical_runtime != saved_config:
            differing_keys = [key for key in CONFIG_KEYS if canonical_runtime[key] != saved_config[key]]
            raise ValueError(
                f"Runtime preprocessing config conflicts with checkpoint keys: "
                f"{differing_keys}"
            )

    manifest = load_json_mapping(paths[PREPROCESSING_MANIFEST_FILE], PREPROCESSING_MANIFEST_FILE)
    required_manifest_keys = {
        "schema_version",
        "category",
        "created_at_utc",
        "dino",
        "normalization",
        "reference_sha256",
        "asset_sha256",
    }
    if "sam2" in manifest:
        raise ValueError("Preprocessing manifest must not contain sam2 metadata")
    if set(manifest) != required_manifest_keys:
        raise ValueError("Preprocessing manifest has unexpected or missing keys")
    if manifest["schema_version"] != PREPROCESSING_SCHEMA_VERSION:
        raise ValueError("Preprocessing manifest schema does not match this runtime")
    category = require_nonempty_string(manifest["category"], "manifest category")
    if expected_category is not None and category != require_nonempty_string(
        expected_category, "expected_category"
    ):
        raise ValueError(
            f"Checkpoint category {category!r} does not match expected category "
            f"{expected_category!r}"
        )
    require_nonempty_string(manifest["created_at_utc"], "created_at_utc")

    dino = manifest["dino"]
    if not isinstance(dino, Mapping) or set(dino) != {
        "backbone_name",
        "feature_layer",
        "patch_size",
        "weights_sha256",
    }:
        raise ValueError("Preprocessing manifest DINO metadata is invalid")
    if dino["backbone_name"] != saved_config["dino_backbone_name"]:
        raise ValueError("Checkpoint DINO backbone does not match preprocessing config")
    if dino["feature_layer"] != saved_config["dino_feature_layer"]:
        raise ValueError("Checkpoint DINO feature layer does not match preprocessing config")
    if isinstance(dino["patch_size"], bool) or not isinstance(dino["patch_size"], int):
        raise ValueError("Checkpoint DINO patch size must be an integer")
    if dino["patch_size"] <= 0:
        raise ValueError("Checkpoint DINO patch size must be positive")
    _require_sha256(dino["weights_sha256"], "DINO weights hash")

    expected_normalization = {
        "input_scale": saved_config["input_scale"],
        "mean": saved_config["mean"],
        "std": saved_config["std"],
    }
    if manifest["normalization"] != expected_normalization:
        raise ValueError("Checkpoint normalization metadata does not match preprocessing config")

    reference_hashes = manifest["reference_sha256"]
    if not isinstance(reference_hashes, Mapping) or set(reference_hashes) != {"image", "mask"}:
        raise ValueError("Checkpoint reference hashes are invalid")
    _require_sha256(reference_hashes["image"], "reference image hash")
    _require_sha256(reference_hashes["mask"], "reference mask hash")

    asset_hashes = manifest["asset_sha256"]
    if not isinstance(asset_hashes, Mapping) or set(asset_hashes) != set(asset_names):
        raise ValueError("Checkpoint derived-asset hashes are invalid")
    for name in asset_names:
        recorded_hash = _require_sha256(asset_hashes[name], f"{name} hash")
        actual_hash = sha256_file(paths[name])
        if actual_hash != recorded_hash:
            raise ValueError(f"Preprocessing asset hash mismatch: {name}")

    return {
        "checkpoint_root": checkpoint_root,
        "config": saved_config,
        "manifest": manifest,
        "paths": paths,
    }


def validate_preprocessing_registry(
    generation_root,
    runtime_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    generation_root = Path(generation_root).resolve()
    registry_path = generation_root / PREPROCESSING_REGISTRY_FILE
    if not registry_path.is_file():
        raise FileNotFoundError(
            f"Preprocessing registry not found: {registry_path}"
        )
    registry = load_json_mapping(registry_path, PREPROCESSING_REGISTRY_FILE)
    if set(registry) != {"schema_version", "categories"}:
        raise ValueError("Preprocessing registry has an invalid schema")
    if registry["schema_version"] != PREPROCESSING_REGISTRY_SCHEMA_VERSION:
        raise ValueError("Unsupported preprocessing registry schema version")
    categories = registry["categories"]
    if not isinstance(categories, list) or not categories:
        raise ValueError("Preprocessing registry categories must be a non-empty list")
    validated_categories = tuple(validate_registry_category(category) for category in categories)
    if tuple(sorted(set(validated_categories))) != validated_categories:
        raise ValueError("Preprocessing registry categories must be sorted and unique")

    bundles = {}
    identity = None
    config = None
    for category in validated_categories:
        bundle = validate_preprocessing_bundle(
            str(preprocessing_bundle_root(generation_root, category)),
            runtime_config=runtime_config,
            expected_category=category,
        )
        bundle_identity = {
            "dino": bundle["manifest"]["dino"],
        }
        if identity is None:
            identity = bundle_identity
            config = bundle["config"]
        elif bundle_identity != identity or bundle["config"] != config:
            raise ValueError(
                "Preprocessing registry bundles must share runtime model identity and configuration"
            )
        bundles[category] = bundle

    return {
        "generation_root": str(generation_root),
        "categories": validated_categories,
        "bundles": bundles,
        "config": config,
    }


def _positive_int_pair(value: Any, name: str) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must contain two integers")
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value):
        raise ValueError(f"{name} must contain two positive integers")
    return int(value[0]), int(value[1])


def _validate_artifacts(
    bundle: Mapping[str, Any],
    prototypes: Any,
    template: Any,
    reference_mask: np.ndarray,
) -> None:
    if not isinstance(prototypes, Mapping) or set(prototypes) != {
        "foreground",
        "background",
        "foreground_count",
        "background_count",
    }:
        raise ValueError("Foreground prototype artifact has invalid keys")
    foreground = prototypes["foreground"]
    background = prototypes["background"]
    if (
        not isinstance(foreground, torch.Tensor)
        or not isinstance(background, torch.Tensor)
        or foreground.dtype != torch.float32
        or background.dtype != torch.float32
        or foreground.ndim != 1
        or background.shape != foreground.shape
        or foreground.numel() == 0
        or not torch.isfinite(foreground).all()
        or not torch.isfinite(background).all()
    ):
        raise ValueError("Foreground prototype tensors are invalid")
    if foreground.norm() <= torch.finfo(torch.float32).eps:
        raise ValueError("Foreground prototype has zero norm")
    if background.norm() <= torch.finfo(torch.float32).eps:
        raise ValueError("Background prototype has zero norm")
    for count_name in ("foreground_count", "background_count"):
        count = prototypes[count_name]
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"{count_name} must be a positive integer")

    template_keys = {
        "features", "centers_xy", "foreground_cells", "original_size", "working_size",
        "grid_size", "patch_size", "feature_layer", "working_longest_edge",
    }
    if not isinstance(template, Mapping) or set(template) != template_keys:
        raise ValueError("Reference feature template has invalid keys")
    features = template["features"]
    centers_xy = template["centers_xy"]
    foreground_cells = template["foreground_cells"]
    if (
        not isinstance(features, torch.Tensor)
        or features.dtype != torch.float16
        or features.ndim != 2
        or features.shape[1] != foreground.numel()
        or features.shape[0] <= 0
        or not torch.isfinite(features).all()
    ):
        raise ValueError("Reference feature tensor is invalid")
    if torch.any(features.float().norm(dim=1) <= torch.finfo(torch.float32).eps):
        raise ValueError("Reference feature tensor contains a zero descriptor")
    if (
        not isinstance(centers_xy, torch.Tensor)
        or centers_xy.dtype != torch.float32
        or centers_xy.shape != (features.shape[0], 2)
        or not torch.isfinite(centers_xy).all()
    ):
        raise ValueError("Reference feature centers are invalid")
    if (
        not isinstance(foreground_cells, torch.Tensor)
        or foreground_cells.dtype != torch.bool
        or foreground_cells.shape != (features.shape[0],)
        or not foreground_cells.any()
        or foreground_cells.all()
    ):
        raise ValueError("Reference foreground-cell selector is invalid")

    original_size = _positive_int_pair(template["original_size"], "original_size")
    working_size = _positive_int_pair(template["working_size"], "working_size")
    grid_size = _positive_int_pair(template["grid_size"], "grid_size")
    patch_size = template["patch_size"]
    if isinstance(patch_size, bool) or not isinstance(patch_size, int) or patch_size <= 0:
        raise ValueError("Reference template patch_size must be positive")
    if working_size != (grid_size[0] * patch_size, grid_size[1] * patch_size):
        raise ValueError("Reference working size, grid size, and patch size are inconsistent")
    if grid_size[0] * grid_size[1] != features.shape[0]:
        raise ValueError("Reference grid size does not match feature count")
    if prototypes["foreground_count"] != int(foreground_cells.sum().item()):
        raise ValueError("Foreground prototype count does not match feature selector")
    if prototypes["background_count"] != int((~foreground_cells).sum().item()):
        raise ValueError("Background prototype count does not match feature selector")
    image_height, image_width = original_size
    if (
        torch.any(centers_xy[:, 0] < 0)
        or torch.any(centers_xy[:, 0] >= image_width)
        or torch.any(centers_xy[:, 1] < 0)
        or torch.any(centers_xy[:, 1] >= image_height)
    ):
        raise ValueError("Reference feature centers fall outside the original image")
    if template["feature_layer"] != bundle["config"]["dino_feature_layer"]:
        raise ValueError("Reference feature layer does not match preprocessing config")
    longest_edge = template["working_longest_edge"]
    if isinstance(longest_edge, bool) or not isinstance(longest_edge, int) or longest_edge <= 0:
        raise ValueError("Reference working_longest_edge must be positive")
    if reference_mask.dtype != np.bool_ or reference_mask.shape != original_size:
        raise ValueError("Reference RLE mask does not match template geometry")
    if not reference_mask.any() or reference_mask.all():
        raise ValueError("Reference RLE mask must contain foreground and background")
    if patch_size != bundle["manifest"]["dino"]["patch_size"]:
        raise ValueError("Reference patch size does not match preprocessing manifest")


def load_preprocessing_artifacts(
    bundle: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    paths = bundle["paths"]
    prototypes = safe_torch_load(
        paths[PROTOTYPES_FILE],
        required_keys={"foreground", "background", "foreground_count", "background_count"},
    )
    template = safe_torch_load(
        paths[REFERENCE_TEMPLATE_FILE],
        required_keys={
            "features", "centers_xy", "foreground_cells", "original_size", "working_size",
            "grid_size", "patch_size", "feature_layer", "working_longest_edge",
        },
    )
    reference_mask = decode_binary_rle(
        load_json_mapping(paths[REFERENCE_MASK_FILE], REFERENCE_MASK_FILE)
    )
    _validate_artifacts(bundle, prototypes, template, reference_mask)
    return dict(prototypes), dict(template), reference_mask
