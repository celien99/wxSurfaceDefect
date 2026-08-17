from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import cast

from hiad.constants import (
    DINO_PATCH_SIZE,
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


def _validate_model_size(value: object, name: str) -> int:
    """校验模型边长是 DINO patch 大小的正整数倍。

    Args:
        value (object): 待校验模型输入边长。
        name (str): 用于错误消息的字段名。

    Returns:
        int: 通过校验的正整数边长。

    Raises:
        ValueError: 值不是正整数、是布尔值或不能被 DINO patch 大小整除。
    """
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
    """生成粗扫、复核和缩略图组成的生产任务集合。

    Attributes:
        patch_size (int): 粗扫正方形补丁边长，必须能被 DINO patch 大小整除。
        stride (int | None): 粗扫滑窗步长；``None`` 表示无重叠切分。
        ds_factors (list[int]): 唯一、升序、从 ``0`` 开始的上下文尺度指数。
    """

    def __init__(
        self,
        patch_size: int,
        stride: int | None = None,
        ds_factors: list[int] | None = None,
    ) -> None:
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
        self.patch_size: int = patch_size
        self.stride: int | None = stride
        self.ds_factors: list[int] = ds_factors

    def create_tasks(
        self,
        thumbnail_size: int | None = None,
        *,
        micro_patch_size: int | None = None,
        refinement_quantile: float | None = None,
        refinement_min_area: int | None = None,
        refinement_safety_fraction: float | None = None,
    ) -> list[TaskDefinition]:
        """创建唯一的一组生产任务，并在写盘前完成参数校验。

        Args:
            thumbnail_size (int | None): 整图缩略任务边长，必须为 DINO patch
                大小的正整数倍。
            micro_patch_size (int | None): 高分辨率复核补丁边长。
            refinement_quantile (float | None): 路由异常图候选分位数，范围
                ``(0, 1)``。
            refinement_min_area (int | None): 候选连通区域最小像素数。
            refinement_safety_fraction (float | None): 安全采样网格比例，范围
                ``(0, 1]``。

        Returns:
            list[TaskDefinition]: 固定按粗扫、复核、缩略图顺序排列的三个任务。

        Raises:
            ValueError: 任一必需参数缺失、尺寸不能整除或复核参数超出范围。
        """
        dynamic_task: DynamicPatchTask = {
            "name": TASK_TYPE_DYNAMIC_PATCH,
            "type": TASK_TYPE_DYNAMIC_PATCH,
            "patch_size": self.patch_size,
            "stride": self.stride,
            "ds_factors": self.ds_factors,
        }
        tasks: list[TaskDefinition] = [dynamic_task]
        refinement_values = (
            refinement_quantile,
            refinement_min_area,
            refinement_safety_fraction,
        )
        if micro_patch_size is None:
            raise ValueError("micro_patch_size is required by the new architecture")
        micro_patch_size = _validate_model_size(micro_patch_size, "micro_patch_size")
        if any(value is None for value in refinement_values):
            raise ValueError("micro_patch_size requires all refinement options")
        refinement_task: RefinementPatchTask = {
            "name": TASK_TYPE_REFINEMENT_PATCH,
            "type": TASK_TYPE_REFINEMENT_PATCH,
            "patch_size": micro_patch_size,
            "stride": micro_patch_size,
            "ds_factors": self.ds_factors,
            "refinement_quantile": cast(float, refinement_quantile),
            "refinement_min_area": cast(int, refinement_min_area),
            "refinement_safety_fraction": cast(float, refinement_safety_fraction),
        }
        _validate_refinement_task(refinement_task)
        tasks.append(refinement_task)
        if thumbnail_size is None:
            raise ValueError("thumbnail_size is required by the new architecture")
        thumbnail_size = _validate_model_size(thumbnail_size, "thumbnail_size")
        thumbnail_task: ThumbnailTask = {
            "name": TASK_TYPE_THUMBNAIL,
            "type": TASK_TYPE_THUMBNAIL,
            "thumbnail_size": thumbnail_size,
        }
        tasks.append(thumbnail_task)
        return tasks


def _validate_refinement_task(task: Mapping[str, object]) -> None:
    """校验复核任务名称、尺寸、尺度以及候选区域选择参数。

    Args:
        task (Mapping[str, object]): 未验证的复核任务对象。

    Raises:
        KeyError: 缺少补丁尺寸、步长或尺度字段。
        ValueError: 名称、尺寸、尺度、分位数、最小面积或安全比例不合法。
    """
    if task.get("name") != TASK_TYPE_REFINEMENT_PATCH:
        raise ValueError("The refinement patch task name must be 'refinement_patch'")
    DynamicTaskGenerator(
        patch_size=cast(int, task["patch_size"]),
        stride=cast(int, task["stride"]),
        ds_factors=cast(list[int], task["ds_factors"]),
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


def validate_tasks(tasks: object) -> list[TaskDefinition]:
    """校验生产链路必须且只能包含粗扫、复核和缩略图三个任务。

    Args:
        tasks (object): 从 JSON 或调用方接收的未验证任务对象。

    Returns:
        list[TaskDefinition]: 保留输入顺序的深拷贝任务列表。

    Raises:
        TypeError: 任一任务条目不是字典。
        ValueError: 任务为空、数量/类型不唯一，或任一字段违反生产约束。
        KeyError: 必需任务字段缺失。
    """
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("At least one task is required")
    if any(not isinstance(task, dict) for task in tasks):
        raise TypeError("Every task must be a dictionary")

    validated = copy.deepcopy(cast(list[dict[str, object]], tasks))
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
        patch_size=cast(int, dynamic["patch_size"]),
        stride=cast(int | None, dynamic["stride"]),
        ds_factors=cast(list[int], dynamic["ds_factors"]),
    )

    _validate_refinement_task(refinement_tasks[0])

    thumbnail = thumbnail_tasks[0]
    if thumbnail.get("name") != TASK_TYPE_THUMBNAIL:
        raise ValueError("The thumbnail task name must be 'thumbnail'")
    _validate_model_size(thumbnail["thumbnail_size"], "thumbnail_size")
    # 上述分支已经验证了判别字段和各任务的必需参数，运行形态仍保持为普通字典。
    return cast(list[TaskDefinition], validated)
