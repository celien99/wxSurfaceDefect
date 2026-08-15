import pytest

from hiad.constants import TASK_TYPE_DYNAMIC_PATCH, TASK_TYPE_REFINEMENT_PATCH
from hiad.task import DynamicTaskGenerator, validate_tasks


def test_dynamic_task_generator_persists_refinement_metadata():
    tasks = DynamicTaskGenerator(
        patch_size=64,
        stride=64,
        ds_factors=[0],
    ).create_tasks(
        thumbnail_size=64,
        micro_patch_size=32,
        refinement_quantile=0.9,
        refinement_min_area=12,
        refinement_safety_fraction=0.25,
    )

    assert tasks == [
        {
            "name": TASK_TYPE_DYNAMIC_PATCH,
            "type": TASK_TYPE_DYNAMIC_PATCH,
            "patch_size": 64,
            "stride": 64,
            "ds_factors": [0],
        },
        {
            "name": TASK_TYPE_REFINEMENT_PATCH,
            "type": TASK_TYPE_REFINEMENT_PATCH,
            "patch_size": 32,
            "stride": 32,
            "ds_factors": [0],
            "refinement_quantile": 0.9,
            "refinement_min_area": 12,
            "refinement_safety_fraction": 0.25,
        },
        {
            "name": "thumbnail",
            "type": "thumbnail",
            "thumbnail_size": 64,
        },
    ]


def test_refinement_task_requires_complete_positive_configuration():
    base_task = {
        "name": TASK_TYPE_DYNAMIC_PATCH,
        "type": TASK_TYPE_DYNAMIC_PATCH,
        "patch_size": 64,
        "stride": 64,
        "ds_factors": [0],
    }
    incomplete_refinement = {
        "name": TASK_TYPE_REFINEMENT_PATCH,
        "type": TASK_TYPE_REFINEMENT_PATCH,
        "patch_size": 32,
        "stride": 32,
        "ds_factors": [0],
        "refinement_quantile": 0.9,
        "refinement_min_area": 12,
        "refinement_safety_fraction": 0.0,
    }
    thumbnail_task = {
        "name": "thumbnail",
        "type": "thumbnail",
        "thumbnail_size": 64,
    }

    with pytest.raises(ValueError, match="refinement_safety_fraction"):
        validate_tasks([base_task, incomplete_refinement, thumbnail_task])


def test_refinement_task_is_required():
    generator = DynamicTaskGenerator(patch_size=64, stride=64, ds_factors=[0])

    with pytest.raises(ValueError, match="micro_patch_size"):
        generator.create_tasks(thumbnail_size=64)
