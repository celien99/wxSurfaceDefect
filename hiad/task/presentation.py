from hiad.constants import TASK_TYPE_DYNAMIC_PATCH

from .task import validate_tasks


def print_task_summary(tasks) -> None:
    validated = validate_tasks(tasks)
    dynamic_task = next(
        task
        for task in validated
        if task["type"] == TASK_TYPE_DYNAMIC_PATCH
    )
    print(
        "Dynamic patch task: "
        f"patch_size={dynamic_task['patch_size']}, "
        f"stride={dynamic_task['stride']}, "
        f"ds_factors={dynamic_task['ds_factors']}"
    )
