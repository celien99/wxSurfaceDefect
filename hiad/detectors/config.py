import copy

from hiad.constants import TASK_TYPE_DYNAMIC_PATCH, TASK_TYPE_THUMBNAIL


def detector_config_for_task(config, task):
    task_type = task.get("type")
    if task_type == TASK_TYPE_DYNAMIC_PATCH:
        detector_config = copy.deepcopy(config.patch)
        detector_config.patch_size = task["patch_size"]
        return detector_config
    if task_type == TASK_TYPE_THUMBNAIL:
        detector_config = copy.deepcopy(config.thumbnail)
        detector_config.pop("thumbnail_size", None)
        detector_config.patch_size = task["thumbnail_size"]
        return detector_config
    raise ValueError(f"Unsupported task type: {task}")
