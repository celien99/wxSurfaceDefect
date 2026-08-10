import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from hiad.data import HRSample


@dataclass(frozen=True)
class TrainingSources:
    samples: tuple[HRSample, ...]
    categories: tuple[str, ...]


def _resolve_training_path(data_root: Path, filename: str) -> Path:
    if not isinstance(filename, str) or not filename:
        raise ValueError("Training record filename must be a non-empty string")
    relative_path = Path(filename)
    if relative_path.is_absolute():
        raise ValueError("Training record filename must be relative to data_root")
    return (data_root / relative_path).resolve()


def validate_unified_training_samples(samples) -> TrainingSources:
    training_samples = tuple(samples)
    if not training_samples:
        raise ValueError("Training samples must not be empty")

    resolved_paths = []
    categories = set()
    for index, sample in enumerate(training_samples):
        if not isinstance(sample, HRSample):
            raise TypeError("Training samples must contain HRSample objects")
        if sample.mask is not None or (
            sample.label is not None
            and (isinstance(sample.label, bool) or sample.label != 0)
        ):
            raise ValueError(
                "Training samples must contain only mask-free normal images; "
                f"invalid sample index: {index}"
            )
        category = sample.clsname
        if not isinstance(category, str) or not category.strip():
            raise ValueError(
                f"Training sample index {index} must have a non-empty clsname"
            )
        image_path = sample.image.image_path
        if not isinstance(image_path, (str, os.PathLike)) or not os.fspath(image_path):
            raise ValueError("Training sample image paths must be non-empty")
        resolved_paths.append(Path(image_path).expanduser().resolve())
        categories.add(category)

    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("Duplicate resolved path in training samples")
    return TrainingSources(
        samples=training_samples,
        categories=tuple(sorted(categories)),
    )


def load_unified_training_samples(data_root) -> tuple[list[HRSample], tuple[str, ...]]:
    root = Path(data_root).expanduser().resolve()
    metadata_path = root / "train_uni.jsonl"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Unified training metadata not found: {metadata_path}")

    samples = []
    with metadata_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(
                    f"Blank metadata record at {metadata_path}:{line_number}"
                )
            record = json.loads(line)
            if not isinstance(record, Mapping):
                raise TypeError(
                    f"Metadata record at {metadata_path}:{line_number} must be a mapping"
                )
            label = record.get("label")
            category = record.get("clsname")
            if (
                isinstance(label, bool)
                or label != 0
                or record.get("mask") is not None
                or not isinstance(category, str)
                or not category.strip()
            ):
                raise ValueError(
                    f"{metadata_path} must contain only normal records with clsname; "
                    f"invalid record at line {line_number}"
                )
            samples.append(
                HRSample(
                    image=os.fspath(_resolve_training_path(root, record.get("filename"))),
                    clsname=category,
                    label=0,
                    label_name=record.get("label_name"),
                )
            )

    sources = validate_unified_training_samples(samples)
    return list(sources.samples), sources.categories
