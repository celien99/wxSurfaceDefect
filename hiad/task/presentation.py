from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from hiad.constants import (
    TASK_TYPE_DYNAMIC_PATCH,
    TASK_TYPE_REFINEMENT_PATCH,
    TASK_TYPE_THUMBNAIL,
)

from .contracts import (
    DynamicPatchTask,
    RefinementPatchTask,
    TaskDefinition,
    ThumbnailTask,
)
from .task import validate_tasks


def print_task_summary(tasks: Sequence[TaskDefinition]) -> None:
    """校验并打印与落盘任务一致的简明配置摘要。

    Args:
        tasks (Sequence[TaskDefinition]): 粗扫、复核和缩略图任务序列。

    Raises:
        TypeError: 任一任务不是字典。
        ValueError: 任务集合或字段不符合生产约束。
    """
    validated = validate_tasks(tasks)
    dynamic_task = cast(
        DynamicPatchTask,
        next(
            task
            for task in validated
            if task["type"] == TASK_TYPE_DYNAMIC_PATCH
        ),
    )
    print(
        "Dynamic patch task: "
        f"patch_size={dynamic_task['patch_size']}, "
        f"stride={dynamic_task['stride']}, "
        f"ds_factors={dynamic_task['ds_factors']}"
    )
    refinement_task = cast(
        RefinementPatchTask,
        next(
            task
            for task in validated
            if task["type"] == TASK_TYPE_REFINEMENT_PATCH
        ),
    )
    print(
        "Refinement patch task: "
        f"patch_size={refinement_task['patch_size']}, "
        f"quantile={refinement_task['refinement_quantile']}, "
        f"min_area={refinement_task['refinement_min_area']}, "
        f"safety_fraction={refinement_task['refinement_safety_fraction']}"
    )
    thumbnail_task = cast(
        ThumbnailTask,
        next(
            task
            for task in validated
            if task["type"] == TASK_TYPE_THUMBNAIL
        ),
    )
    print(
        "Thumbnail task: "
        f"thumbnail_size={thumbnail_task['thumbnail_size']}"
    )
