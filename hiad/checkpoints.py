import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Iterable

import torch


CHECKPOINT_SCHEMA_VERSION = 1
CURRENT_FILE = "current.json"
GENERATION_MANIFEST_FILE = "generation_manifest.json"
_GENERATION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    return Path(temporary)


def atomic_write_json(payload: Mapping, destination) -> None:
    if not isinstance(payload, Mapping):
        raise TypeError("JSON payload must be a mapping")
    destination = Path(destination)
    temporary = _atomic_path(destination)
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(dict(payload), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch_save(payload: Mapping, destination) -> None:
    if not isinstance(payload, Mapping):
        raise TypeError("PyTorch checkpoint payload must be a mapping")
    destination = Path(destination)
    temporary = _atomic_path(destination)
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def safe_torch_load(path, *, required_keys: Iterable[str], map_location="cpu") -> dict:
    required_keys = set(required_keys)
    if any(not isinstance(key, str) or not key for key in required_keys):
        raise ValueError("required_keys must contain non-empty strings")
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("PyTorch checkpoint must contain a mapping")
    missing = sorted(required_keys.difference(payload))
    if missing:
        raise ValueError(f"Checkpoint is missing required keys: {missing}")
    return dict(payload)


def _validate_generation_id(generation_id: str) -> str:
    if not isinstance(generation_id, str) or not _GENERATION_ID_PATTERN.fullmatch(
        generation_id
    ):
        raise ValueError("generation_id contains unsupported characters")
    return generation_id


def begin_generation(checkpoint_root, *, generation_id: str) -> Path:
    generation_id = _validate_generation_id(generation_id)
    generation = Path(checkpoint_root) / "generations" / generation_id
    generation.mkdir(parents=True, exist_ok=False)
    return generation


def _validate_artifact_name(name: str) -> str:
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError(f"Invalid generation artifact name: {name!r}")
    path = Path(name)
    if ".." in path.parts or str(path) in {".", GENERATION_MANIFEST_FILE}:
        raise ValueError(f"Invalid generation artifact name: {name!r}")
    return path.as_posix()


def publish_generation(checkpoint_root, generation, *, required_files: Iterable[str]) -> None:
    checkpoint_root = Path(checkpoint_root).resolve()
    generation = Path(generation).resolve()
    generations_root = (checkpoint_root / "generations").resolve()
    if generation.parent != generations_root:
        raise ValueError("Generation must be a direct child of the generations directory")
    generation_id = _validate_generation_id(generation.name)

    names = sorted({_validate_artifact_name(name) for name in required_files})
    if not names:
        raise ValueError("At least one generation artifact is required")
    files = {}
    for name in names:
        artifact = generation / name
        if not artifact.is_file():
            raise FileNotFoundError(f"Generation artifact not found: {artifact}")
        files[name] = _sha256_file(artifact)

    manifest = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "generation_id": generation_id,
        "files": files,
    }
    manifest_path = generation / GENERATION_MANIFEST_FILE
    atomic_write_json(manifest, manifest_path)
    current = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "generation_id": generation_id,
        "manifest_sha256": _sha256_file(manifest_path),
    }
    atomic_write_json(current, checkpoint_root / CURRENT_FILE)


def _read_json_mapping(path: Path, description: str) -> dict:
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid {description}: {path}") from error
    if not isinstance(payload, Mapping):
        raise TypeError(f"{description} must contain a mapping")
    return dict(payload)


def resolve_current_generation(checkpoint_root) -> Path:
    checkpoint_root = Path(checkpoint_root).resolve()
    current_path = checkpoint_root / CURRENT_FILE
    if not current_path.is_file():
        raise FileNotFoundError(f"Checkpoint publication file not found: {current_path}")
    current = _read_json_mapping(current_path, "checkpoint publication")
    required_current = {"schema_version", "generation_id", "manifest_sha256"}
    if set(current) != required_current:
        raise ValueError("Checkpoint publication has an invalid schema")
    if current["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Unsupported checkpoint publication schema version")

    generation_id = _validate_generation_id(current["generation_id"])
    generation = checkpoint_root / "generations" / generation_id
    manifest_path = generation / GENERATION_MANIFEST_FILE
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Generation manifest not found: {manifest_path}")
    if _sha256_file(manifest_path) != current["manifest_sha256"]:
        raise ValueError("Generation manifest hash mismatch")

    manifest = _read_json_mapping(manifest_path, "generation manifest")
    required_manifest = {"schema_version", "generation_id", "files"}
    if set(manifest) != required_manifest:
        raise ValueError("Generation manifest has an invalid schema")
    if manifest["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Unsupported generation manifest schema version")
    if manifest["generation_id"] != generation_id:
        raise ValueError("Generation manifest id mismatch")
    if not isinstance(manifest["files"], Mapping) or not manifest["files"]:
        raise ValueError("Generation manifest files must be a non-empty mapping")

    for raw_name, expected_hash in manifest["files"].items():
        name = _validate_artifact_name(raw_name)
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError(f"Invalid artifact hash for {name}")
        artifact = generation / name
        if not artifact.is_file():
            raise FileNotFoundError(f"Generation artifact not found: {artifact}")
        if _sha256_file(artifact) != expected_hash:
            raise ValueError(f"Generation artifact hash mismatch: {name}")
    return generation
