import copy
from typing import Dict, List, Optional

from hiad.constants import (
    DINO_PATCH_SIZE,
    TASK_TYPE_DYNAMIC_PATCH,
    TASK_TYPE_THUMBNAIL,
)


def _validate_model_size(value, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value % DINO_PATCH_SIZE != 0
    ):
        raise ValueError(
            f"{name} must be a positive multiple of {DINO_PATCH_SIZE}"
        )
    return value


class DynamicTaskGenerator:
    """Create one patch task whose indexes are generated from each image's native size."""

    def __init__(self, patch_size: int, stride: int = None, ds_factors: List[int] = None):
        patch_size = _validate_model_size(patch_size, "patch_size")
        if stride is not None and (
            isinstance(stride, bool)
            or not isinstance(stride, int)
            or stride <= 0
            or stride > patch_size
        ):
            raise ValueError("stride must be in the range [1, patch_size]")
        ds_factors = [0] if ds_factors is None else list(ds_factors)
        if (
            not ds_factors
            or any(
                isinstance(factor, bool)
                or not isinstance(factor, int)
                or factor < 0
                for factor in ds_factors
            )
            or ds_factors[0] != 0
            or ds_factors != sorted(set(ds_factors))
        ):
            raise ValueError("ds_factors must be unique, sorted, non-negative, and start with 0")
        self.patch_size = patch_size
        self.stride = stride
        self.ds_factors = ds_factors

    def create_tasks(self, thumbnail_size: Optional[int] = None) -> List[Dict]:
        tasks = [{
            "name": TASK_TYPE_DYNAMIC_PATCH,
            "type": TASK_TYPE_DYNAMIC_PATCH,
            "patch_size": self.patch_size,
            "stride": self.stride,
            "ds_factors": self.ds_factors,
        }]
        if thumbnail_size is not None:
            thumbnail_size = _validate_model_size(thumbnail_size, "thumbnail_size")
            tasks.append({
                "name": TASK_TYPE_THUMBNAIL,
                "type": TASK_TYPE_THUMBNAIL,
                "thumbnail_size": thumbnail_size,
            })
        return tasks


def validate_tasks(tasks: List[Dict]) -> List[Dict]:
    """Validate the single supported dynamic-patch/thumbnail task schema."""
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("At least one task is required")

    validated = copy.deepcopy(tasks)
    dynamic_tasks = [
        task
        for task in validated
        if task.get("type") == TASK_TYPE_DYNAMIC_PATCH
    ]
    thumbnail_tasks = [
        task
        for task in validated
        if task.get("type") == TASK_TYPE_THUMBNAIL
    ]
    if len(dynamic_tasks) != 1:
        raise ValueError("Exactly one dynamic_patch task is required")
    if len(thumbnail_tasks) > 1:
        raise ValueError("At most one thumbnail task is allowed")
    if len(dynamic_tasks) + len(thumbnail_tasks) != len(validated):
        raise ValueError("Only dynamic_patch and thumbnail tasks are supported")

    dynamic = dynamic_tasks[0]
    if dynamic.get("name") != TASK_TYPE_DYNAMIC_PATCH:
        raise ValueError("The dynamic patch task name must be 'dynamic_patch'")
    DynamicTaskGenerator(
        patch_size=dynamic["patch_size"],
        stride=dynamic["stride"],
        ds_factors=dynamic["ds_factors"],
    )

    if thumbnail_tasks:
        thumbnail = thumbnail_tasks[0]
        if thumbnail.get("name") != TASK_TYPE_THUMBNAIL:
            raise ValueError("The thumbnail task name must be 'thumbnail'")
        _validate_model_size(thumbnail["thumbnail_size"], "thumbnail_size")
    return validated
