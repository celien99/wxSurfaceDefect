from __future__ import annotations

import numpy as np

from hiad.runtime.contracts import (
    DetectorPrediction,
    FloatMap,
    ImageQualityResult,
    InferenceResult,
    ScoreCalibration,
)
from hiad.task.contracts import DynamicPatchTask, TaskDefinition, ThumbnailTask


def task_name(task: TaskDefinition) -> str:
    return task["name"]


dynamic_task: DynamicPatchTask = {
    "name": "dynamic_patch",
    "type": "dynamic_patch",
    "patch_size": 512,
    "stride": 256,
    "ds_factors": [0, 1],
}
thumbnail_task: ThumbnailTask = {
    "name": "thumbnail",
    "type": "thumbnail",
    "thumbnail_size": 512,
}
anomaly_map: FloatMap = np.zeros((8, 8), dtype=np.float32)
prediction: DetectorPrediction = {"anomaly_map": anomaly_map, "score": 0.25}
quality: ImageQualityResult = {
    "status": "PASS",
    "reasons": [],
    "mean_luminance": 0.5,
    "clipped_fraction": 0.0,
    "focus_variance": 1.0,
}
calibration: ScoreCalibration = {
    "percentile": 0.99,
    "pixel_percentile": 0.99,
    "pixel_image_percentile": 0.99,
    "normal_image_count": 1,
    "global_threshold": 0.5,
    "global_pixel_threshold": 0.4,
    "categories": {
        "seat": {
            "normal_image_count": 1,
            "threshold": 0.5,
            "pixel_threshold": 0.4,
        }
    },
}
result: InferenceResult = {
    "image_paths": ["sample.png"],
    "anomaly_maps": [anomaly_map],
    "image_scores": np.asarray([0.25], dtype=np.float32),
    "quality_results": [quality],
}

assert task_name(dynamic_task) == "dynamic_patch"
assert task_name(thumbnail_task) == "thumbnail"
assert prediction["score"] == 0.25
assert calibration["categories"]["seat"]["threshold"] == 0.5
assert result["quality_results"][0]["status"] == "PASS"
