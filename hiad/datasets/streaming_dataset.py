from __future__ import annotations

from typing import Any

import numpy
from PIL import Image
from torch.utils.data import Dataset

from hiad.constants import (
    SUPPORTED_TASK_TYPES,
    TASK_TYPE_DYNAMIC_PATCH,
    TASK_TYPE_REFINEMENT_PATCH,
    TASK_TYPE_THUMBNAIL,
)
from hiad.data import (
    HRSample,
    HRImageIndex,
    build_multiresolution_region,
    create_dynamic_patch,
    split_multiresolution_regions,
)
from hiad.datasets.patch_dataset import PatchDataset


class StreamingTaskDataset(Dataset):
    """Build task patches on demand without full-image residency."""

    def __init__(
        self,
        samples: list[HRSample],
        task: dict,
        *,
        training: bool,
        regions_by_path: dict[str, list[HRImageIndex]] | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(samples, list) or not samples:
            raise ValueError("StreamingTaskDataset requires a non-empty sample list")
        if any(not isinstance(sample, HRSample) for sample in samples):
            raise TypeError("Every streaming sample must be an HRSample")
        if not isinstance(task, dict) or task.get("type") not in SUPPORTED_TASK_TYPES:
            raise ValueError(f"Unsupported task: {task}")
        paths = [sample.image.image_path for sample in samples]
        duplicates = sorted({path for path in paths if paths.count(path) > 1})
        if duplicates:
            raise ValueError(f"Duplicate source image path: {duplicates[0]}")

        self.samples = samples
        self.task = task
        self.training = training
        self.regions_by_path = self._validate_regions_by_path(regions_by_path, paths)
        self._entries: list[tuple[int, Any]] = []
        self.records = []
        # At most one decoded full image is retained between consecutive
        # patches from the same path to bound worker memory.
        self._cached_path: str | None = None
        self._cached_image: numpy.ndarray | None = None
        self._build_index()

    @staticmethod
    def _validate_regions_by_path(regions_by_path, paths):
        if regions_by_path is None:
            return None
        if not isinstance(regions_by_path, dict):
            raise TypeError("regions_by_path must be a mapping or None")
        unknown_paths = set(regions_by_path) - set(paths)
        if unknown_paths:
            raise ValueError("regions_by_path contains a source image outside this dataset")
        validated = {}
        for path, regions in regions_by_path.items():
            if not isinstance(regions, list) or not regions:
                raise ValueError("Each refinement source must have at least one region")
            if any(not isinstance(region, HRImageIndex) for region in regions):
                raise TypeError("Every refinement region must be an HRImageIndex")
            validated[path] = list(regions)
        return validated

    @staticmethod
    def _read_image_size(image_path: str) -> tuple[int, int]:
        with Image.open(image_path) as image_file:
            width, height = image_file.size
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image size for {image_path}: {(width, height)}")
        return int(width), int(height)

    @staticmethod
    def _thumbnail_model_input_size(thumbnail_size) -> tuple[int, int]:
        if isinstance(thumbnail_size, int) and not isinstance(thumbnail_size, bool):
            if thumbnail_size <= 0:
                raise ValueError("thumbnail_size must be positive")
            return thumbnail_size, thumbnail_size
        if (
            isinstance(thumbnail_size, (tuple, list))
            and len(thumbnail_size) == 2
            and all(
                not isinstance(value, bool) and isinstance(value, int) and value > 0
                for value in thumbnail_size
            )
        ):
            return int(thumbnail_size[0]), int(thumbnail_size[1])
        raise ValueError("thumbnail_size must be a positive integer or width-height pair")

    def _build_index(self) -> None:
        task = self.task
        task_name = task["name"]
        task_type = task["type"]

        if task_type in {TASK_TYPE_DYNAMIC_PATCH, TASK_TYPE_REFINEMENT_PATCH}:
            for sample_index, sample in enumerate(self.samples):
                path = sample.image.image_path
                image_size = self._read_image_size(path)
                image_width, image_height = image_size
                if self.regions_by_path is None:
                    indexes = split_multiresolution_regions(
                        image_size=image_size,
                        patch_size=task["patch_size"],
                        ds_factors=task["ds_factors"],
                        stride=task["stride"],
                    )
                else:
                    indexes = [
                        build_multiresolution_region(
                            image_size,
                            region,
                            task["ds_factors"],
                        )
                        for region in self.regions_by_path.get(path, [])
                    ]
                for region in indexes:
                    source = region.main_index
                    valid_height = min(source.height, image_height - source.y)
                    valid_width = min(source.width, image_width - source.x)
                    self._entries.append((sample_index, region))
                    self.records.append({
                        "task_name": task_name,
                        "task_type": task_type,
                        "image_path": path,
                        "image_size": image_size,
                        "model_input_size": (source.width, source.height),
                        "source_xywh": (source.x, source.y, source.width, source.height),
                        "valid_source_hw": (valid_height, valid_width),
                    })
            return

        model_input_size = self._thumbnail_model_input_size(task["thumbnail_size"])
        for sample_index, sample in enumerate(self.samples):
            path = sample.image.image_path
            image_size = self._read_image_size(path)
            self._entries.append((sample_index, None))
            self.records.append({
                "task_name": task_name,
                "task_type": task_type,
                "image_path": path,
                "image_size": image_size,
                "model_input_size": model_input_size,
            })

    def __len__(self) -> int:
        return len(self._entries)

    def source_sample_index(self, index: int) -> int:
        return self._entries[index][0]

    def _load_image(self, sample: HRSample) -> numpy.ndarray:
        path = sample.image.image_path
        if self._cached_path == path and self._cached_image is not None:
            return self._cached_image
        with Image.open(path) as image_file:
            image_array = numpy.array(image_file.convert("RGB"), copy=True)
        if (
            not isinstance(image_array, numpy.ndarray)
            or image_array.dtype != numpy.uint8
            or image_array.ndim != 3
            or image_array.shape[2] != 3
            or not numpy.isfinite(image_array).all()
        ):
            raise ValueError("Source image must be a finite HWC uint8 RGB image")
        self._cached_path = path
        self._cached_image = image_array
        return image_array

    def __getitem__(self, index: int) -> dict:
        sample_index, region = self._entries[index]
        sample = self.samples[sample_index]
        image_array = self._load_image(sample)

        sample.image.image = image_array
        try:
            if sample.mask is not None:
                sample.mask.open()
            if region is None:
                patch = sample.down_sampling_to_LR(self.task["thumbnail_size"])
            else:
                patch = create_dynamic_patch(sample, region)
            return PatchDataset(
                patches=[patch],
                training=self.training,
                task_name=self.task["name"],
            )[0]
        finally:
            sample.image.image = None
            if sample.mask is not None:
                sample.mask.close()
