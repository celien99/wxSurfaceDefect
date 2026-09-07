from typing import Final, Literal, TypeAlias

DINO_PATCH_SIZE: Final = 16

# 整图缩略前向画布边长（正方形），用于提取每源图全局 recenter 锚；必须能被
# DINO patch 大小整除。训练与推理共用同一画布，保证 recenter 语义一致。
ANCHOR_CANVAS: Final = 512

TaskType: TypeAlias = Literal["dynamic_patch", "refinement_patch", "thumbnail"]
TASK_TYPE_DYNAMIC_PATCH: Final[Literal["dynamic_patch"]] = "dynamic_patch"
TASK_TYPE_REFINEMENT_PATCH: Final[Literal["refinement_patch"]] = "refinement_patch"
TASK_TYPE_THUMBNAIL: Final[Literal["thumbnail"]] = "thumbnail"
SUPPORTED_TASK_TYPES: Final[frozenset[TaskType]] = frozenset({
    TASK_TYPE_DYNAMIC_PATCH,
    TASK_TYPE_REFINEMENT_PATCH,
    TASK_TYPE_THUMBNAIL,
})
