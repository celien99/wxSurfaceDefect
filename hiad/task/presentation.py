from hiad.constants import TASK_TYPE_DYNAMIC_PATCH, TASK_TYPE_REFINEMENT_PATCH

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
    refinement_task = next(
        task
        for task in validated
        if task["type"] == TASK_TYPE_REFINEMENT_PATCH
    )
    print(
        "Refinement patch task: "
        f"patch_size={refinement_task['patch_size']}, "
        f"quantile={refinement_task['refinement_quantile']}, "
        f"min_area={refinement_task['refinement_min_area']}, "
        f"safety_fraction={refinement_task['refinement_safety_fraction']}"
    )
