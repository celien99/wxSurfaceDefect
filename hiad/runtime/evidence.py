from collections.abc import Callable, Mapping, Sequence

import torch
from PIL import Image

from hiad.constants import TASK_TYPE_DYNAMIC_PATCH, TASK_TYPE_THUMBNAIL
from hiad.datasets import StreamingTaskDataset
from hiad.scoring.pipeline import (
    assemble_context_evidence,
    assemble_patch_evidence,
)


def _scoring_identity(detector) -> dict:
    fusion_weights = detector.fusion_weights
    return {
        "anomaly_distance": detector.anomaly_distance,
        "use_fp16": detector.use_fp16,
        "fusion_weights": (
            None if fusion_weights is None else list(fusion_weights)
        ),
    }


def _image_size_metadata(samples) -> dict:
    metadata = {}
    for sample in samples:
        path = sample.image.image_path
        with Image.open(path) as image_file:
            width, height = image_file.size
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image size for {path}: {(width, height)}")
        metadata[path] = {"image_size": (int(width), int(height))}
    return metadata


def collect_task_evidence(
    samples: Sequence,
    tasks: Sequence[dict],
    preprocessor_registry,
    detector_provider: Callable[[dict], object],
    batch_size: int,
    *,
    task_names: Sequence[str] | None = None,
    detector_releaser: Callable[[object], None] | None = None,
) -> tuple[dict, dict | None]:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not callable(detector_provider):
        raise TypeError("detector_provider must be callable")
    if detector_releaser is not None and not callable(detector_releaser):
        raise TypeError("detector_releaser must be callable or None")
    if preprocessor_registry is None:
        raise ValueError("preprocessor_registry must not be None")
    if not isinstance(samples, (list, tuple)) or not samples:
        raise ValueError("samples must be a non-empty sequence")

    task_by_name = {task["name"]: task for task in tasks}
    if len(task_by_name) != len(tasks):
        raise ValueError("Task names must be unique")
    ordered_names = list(task_by_name) if task_names is None else list(task_names)
    if len(ordered_names) != len(set(ordered_names)):
        raise ValueError("task_names must not contain duplicates")
    if set(ordered_names) != set(task_by_name):
        raise ValueError("task_names must contain every configured task exactly once")

    image_metadata = _image_size_metadata(samples)
    results = {
        path: {
            "image_size": metadata["image_size"],
            "patches": [],
            "context": None,
        }
        for path, metadata in image_metadata.items()
    }
    dynamic_scoring_identity = None

    for task_name in ordered_names:
        task = task_by_name[task_name]
        dataset = StreamingTaskDataset(
            list(samples),
            task,
            preprocessor_registry,
            training=False,
        )
        detector = detector_provider(task)
        try:
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )
            outputs = detector.predict_evidence(dataloader, task_name)

            if task["type"] == TASK_TYPE_DYNAMIC_PATCH:
                if dynamic_scoring_identity is not None:
                    raise ValueError("A worker cannot run multiple dynamic patch detectors")
                dynamic_scoring_identity = _scoring_identity(detector)
                assembled = assemble_patch_evidence(dataset.records, outputs)
                for path, patches in assembled.items():
                    results[path]["patches"].extend(patches)
            elif task["type"] == TASK_TYPE_THUMBNAIL:
                assembled = assemble_context_evidence(dataset.records, outputs)
                for path, context in assembled.items():
                    if results[path]["context"] is not None:
                        raise ValueError(f"Duplicate Context evidence for {path}")
                    results[path]["context"] = context
            else:
                raise ValueError(f"Unsupported task type: {task}")
        finally:
            if detector_releaser is not None:
                detector_releaser(detector)

    return results, dynamic_scoring_identity
