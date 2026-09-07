from typing import Final, Literal, TypeAlias

DINO_PATCH_SIZE: Final = 16

TaskType: TypeAlias = Literal["dynamic_patch", "refinement_patch", "thumbnail"]
TASK_TYPE_DYNAMIC_PATCH: Final[Literal["dynamic_patch"]] = "dynamic_patch"
TASK_TYPE_REFINEMENT_PATCH: Final[Literal["refinement_patch"]] = "refinement_patch"
TASK_TYPE_THUMBNAIL: Final[Literal["thumbnail"]] = "thumbnail"
SUPPORTED_TASK_TYPES: Final[frozenset[TaskType]] = frozenset({
    TASK_TYPE_DYNAMIC_PATCH,
    TASK_TYPE_REFINEMENT_PATCH,
    TASK_TYPE_THUMBNAIL,
})
