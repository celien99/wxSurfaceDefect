import copy
import math
from typing import Dict, List, Optional

from hiad.constants import (
    DINO_PATCH_SIZE,
    TASK_TYPE_DYNAMIC_PATCH,
    TASK_TYPE_REFINEMENT_PATCH,
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

    def create_tasks(
        self,
        thumbnail_size: Optional[int] = None,
        *,
        micro_patch_size: Optional[int] = None,
        refinement_quantile: Optional[float] = None,
        refinement_min_area: Optional[int] = None,
        refinement_safety_fraction: Optional[float] = None,
    ) -> List[Dict]:
        tasks = [{
            "name": TASK_TYPE_DYNAMIC_PATCH,
            "type": TASK_TYPE_DYNAMIC_PATCH,
            "patch_size": self.patch_size,
            "stride": self.stride,
            "ds_factors": self.ds_factors,
        }]
        refinement_values = (
            refinement_quantile,
            refinement_min_area,
            refinement_safety_fraction,
        )
        if micro_patch_size is None:
            raise ValueError("micro_patch_size is required by the new architecture")
        else:
            micro_patch_size = _validate_model_size(micro_patch_size, "micro_patch_size")
            if any(value is None for value in refinement_values):
                raise ValueError("micro_patch_size requires all refinement options")
            refinement_task = {
                "name": TASK_TYPE_REFINEMENT_PATCH,
                "type": TASK_TYPE_REFINEMENT_PATCH,
                "patch_size": micro_patch_size,
                "stride": micro_patch_size,
                "ds_factors": self.ds_factors,
                "refinement_quantile": refinement_quantile,
                "refinement_min_area": refinement_min_area,
                "refinement_safety_fraction": refinement_safety_fraction,
            }
            _validate_refinement_task(refinement_task)
            tasks.append(refinement_task)
        if thumbnail_size is None:
            raise ValueError("thumbnail_size is required by the new architecture")
        thumbnail_size = _validate_model_size(thumbnail_size, "thumbnail_size")
        tasks.append({
            "name": TASK_TYPE_THUMBNAIL,
            "type": TASK_TYPE_THUMBNAIL,
            "thumbnail_size": thumbnail_size,
        })
        return tasks


def _validate_refinement_task(task: Dict) -> None:
    if task.get("name") != TASK_TYPE_REFINEMENT_PATCH:
        raise ValueError("The refinement patch task name must be 'refinement_patch'")
    DynamicTaskGenerator(
        patch_size=task["patch_size"],
        stride=task["stride"],
        ds_factors=task["ds_factors"],
    )
    threshold = task.get("refinement_quantile")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or not 0 < threshold < 1
    ):
        raise ValueError("refinement_quantile must be finite and in the range (0, 1)")
    min_area = task.get("refinement_min_area")
    if isinstance(min_area, bool) or not isinstance(min_area, int) or min_area <= 0:
        raise ValueError("refinement_min_area must be a positive integer")
    safety_fraction = task.get("refinement_safety_fraction")
    if (
        isinstance(safety_fraction, bool)
        or not isinstance(safety_fraction, (int, float))
        or not math.isfinite(safety_fraction)
        or safety_fraction <= 0
        or safety_fraction > 1
    ):
        raise ValueError(
            "refinement_safety_fraction must be finite and in the range (0, 1]"
        )


def validate_tasks(tasks: List[Dict]) -> List[Dict]:
    """Validate the mandatory base, refinement, and thumbnail task metadata."""
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
    refinement_tasks = [
        task
        for task in validated
        if task.get("type") == TASK_TYPE_REFINEMENT_PATCH
    ]
    if len(dynamic_tasks) != 1:
        raise ValueError("Exactly one dynamic_patch task is required")
    if len(thumbnail_tasks) != 1:
        raise ValueError("Exactly one thumbnail task is required")
    if len(refinement_tasks) != 1:
        raise ValueError("Exactly one refinement_patch task is required")
    if len(dynamic_tasks) + len(thumbnail_tasks) + len(refinement_tasks) != len(validated):
        raise ValueError("Only dynamic_patch, refinement_patch, and thumbnail tasks are supported")

    dynamic = dynamic_tasks[0]
    if dynamic.get("name") != TASK_TYPE_DYNAMIC_PATCH:
        raise ValueError("The dynamic patch task name must be 'dynamic_patch'")
    DynamicTaskGenerator(
        patch_size=dynamic["patch_size"],
        stride=dynamic["stride"],
        ds_factors=dynamic["ds_factors"],
    )

    _validate_refinement_task(refinement_tasks[0])

    thumbnail = thumbnail_tasks[0]
    if thumbnail.get("name") != TASK_TYPE_THUMBNAIL:
        raise ValueError("The thumbnail task name must be 'thumbnail'")
    _validate_model_size(thumbnail["thumbnail_size"], "thumbnail_size")
    return validated
