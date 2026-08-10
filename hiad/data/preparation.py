from dataclasses import dataclass

from hiad.constants import (
    SUPPORTED_TASK_TYPES,
    TASK_TYPE_DYNAMIC_PATCH,
    TASK_TYPE_THUMBNAIL,
)
from hiad.data import (
    HRSample,
    create_dynamic_patch,
    split_multiresolution_regions,
)


@dataclass(frozen=True)
class PreparedInputRecord:
    task_name: str
    task_type: str
    image_path: str
    image_size: tuple[int, int]
    model_input_size: tuple[int, int]
    source_xywh: tuple[int, int, int, int] | None = None
    valid_source_hw: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if not self.task_name or self.task_type not in SUPPORTED_TASK_TYPES:
            raise ValueError("Prepared input task identity is invalid")
        if not self.image_path:
            raise ValueError("Prepared input image_path must be non-empty")
        for name, size in (
            ("image_size", self.image_size),
            ("model_input_size", self.model_input_size),
        ):
            if (
                not isinstance(size, tuple)
                or len(size) != 2
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value <= 0
                    for value in size
                )
            ):
                raise ValueError(f"{name} must be a positive integer pair")
        if self.task_type == TASK_TYPE_THUMBNAIL:
            if self.source_xywh is not None or self.valid_source_hw is not None:
                raise ValueError("Thumbnail records must not contain local source geometry")
            return
        if (
            not isinstance(self.source_xywh, tuple)
            or len(self.source_xywh) != 4
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in self.source_xywh
            )
        ):
            raise ValueError("Dynamic records require integer source_xywh")
        if (
            not isinstance(self.valid_source_hw, tuple)
            or len(self.valid_source_hw) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in self.valid_source_hw
            )
        ):
            raise ValueError("Dynamic records require positive valid_source_hw")


def build_task_inputs_from_open_samples(samples, tasks, *, logger=None):
    """Transitional helper for Task 4/5 callers that still expect open samples.

    Prefer ``StreamingTaskDataset`` for new training/inference paths.
    """
    if not isinstance(samples, list) or not samples:
        raise ValueError("Task input creation requires a non-empty sample list")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Task input creation requires configured tasks")

    task_inputs = {}
    for task in tasks:
        if task["type"] not in SUPPORTED_TASK_TYPES:
            raise ValueError(f"Unsupported task type: {task}")
        if task["name"] in task_inputs:
            raise ValueError(f"Duplicate task name: {task['name']}")
        task_inputs[task["name"]] = {"patches": [], "records": []}

    image_metadata = {}
    for sample_index, sample in enumerate(samples):
        if not isinstance(sample, HRSample):
            raise TypeError("Every task input sample must be an HRSample")
        if sample.image.image is None or not sample.image.is_processed:
            raise RuntimeError("Task inputs require open preprocessed source samples")
        if (
            logger is not None
            and len(samples) > 10
            and (sample_index + 1) % max(1, int(len(samples) * 0.1)) == 0
        ):
            logger.info(f"{int((sample_index + 1) / len(samples) * 100)}% of data loaded")

        path = sample.image.image_path
        if path in image_metadata:
            raise ValueError(f"Duplicate source image path: {path}")
        image_size = sample.image.size()
        image_width, image_height = image_size
        image_metadata[path] = {"image_size": image_size}

        for task in tasks:
            prepared = task_inputs[task["name"]]
            if task["type"] == TASK_TYPE_DYNAMIC_PATCH:
                indexes = split_multiresolution_regions(
                    image_size=image_size,
                    patch_size=task["patch_size"],
                    ds_factors=task["ds_factors"],
                    stride=task["stride"],
                )
                for index in indexes:
                    source = index.main_index
                    valid_height = min(source.height, image_height - source.y)
                    valid_width = min(source.width, image_width - source.x)
                    patch = create_dynamic_patch(sample, index)
                    patch.valid_source_hw = (valid_height, valid_width)
                    model_input_size = patch.image.shape[:2][::-1]
                    prepared["patches"].append(patch)
                    prepared["records"].append(
                        PreparedInputRecord(
                            task_name=task["name"],
                            task_type=task["type"],
                            image_path=path,
                            image_size=image_size,
                            model_input_size=model_input_size,
                            source_xywh=(
                                source.x,
                                source.y,
                                source.width,
                                source.height,
                            ),
                            valid_source_hw=(valid_height, valid_width),
                        )
                    )
            else:
                patch = sample.down_sampling_to_LR(task["thumbnail_size"])
                model_input_size = patch.image.shape[:2][::-1]
                prepared["patches"].append(patch)
                prepared["records"].append(
                    PreparedInputRecord(
                        task_name=task["name"],
                        task_type=task["type"],
                        image_path=path,
                        image_size=image_size,
                        model_input_size=model_input_size,
                    )
                )
    return task_inputs, image_metadata
