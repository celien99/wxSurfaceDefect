import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from hiad.inferencer.inferencer import HRInferencer
from hiad.inferencer.pipeline import DeviceImagePipeline
from hiad.runtime.inference_config import InferenceConfig
from hiad.data import HRSample


class _StubDetector:
    def __init__(self, patch_size):
        self.device = torch.device("cpu")
        self.score_top_k = 4
        self.patch_size = [patch_size, patch_size]

    def inference_batch(self, data):
        batch, _, height, width = data["image"].shape
        return (
            torch.full((batch, 1, height, width), 0.2),
            torch.full((batch, 1, height // 16, width // 16), 0.2),
        )

    def load_checkpoint(self, path):
        pass


def test_hr_inferencer_pipeline_and_legacy_paths_agree_on_shape_contract(tmp_path):
    """两条路径都产出同结构的 InferenceResult（不要求数值一致，由训练机 parity 判定）。"""
    image_path = tmp_path / "a.png"
    from PIL import Image
    Image.fromarray(np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(image_path)
    sample = HRSample(image=str(image_path), clsname="part")

    inferencer = HRInferencer.__new__(HRInferencer)
    inferencer.batch_size = None
    inferencer.score_calibration = None
    inferencer.map_gaussian_sigma = 0.0
    inferencer.global_routing_weight = 0.25
    inferencer.score_top_k = 4
    inferencer.refinement_bridge_gap_tiles = 1
    inferencer.quality_thresholds = {
        "min_mean_luminance": 0.05, "max_mean_luminance": 0.95,
        "max_clipped_fraction": 0.5, "min_focus_variance": 10.0,
    }
    inferencer.inference_config = InferenceConfig()
    # 用 stub 覆盖模型管理器，避免加载真实检查点
    from unittest.mock import MagicMock
    manager = MagicMock()
    manager.detectors = {
        "dynamic_patch": _StubDetector(16),
        "thumbnail": _StubDetector(16),
        "refinement_patch": _StubDetector(16),
    }
    inferencer.model_managers = [manager]
    # 以下属性是 __init__ 的产物；__new__ 绕过了 __init__，需显式补齐
    inferencer.coarse_tasks_in_devices = [["dynamic_patch", "thumbnail"]]
    inferencer.refinement_tasks_in_devices = [["refinement_patch"]]
    inferencer.coarse_task_definitions = [
        {"name": "dynamic_patch", "type": "dynamic_patch",
         "patch_size": 16, "stride": 16, "ds_factors": [0, 1]},
        {"name": "thumbnail", "type": "thumbnail", "thumbnail_size": 16},
    ]
    inferencer.refinement_task = {
        "name": "refinement_patch", "type": "refinement_patch",
        "patch_size": 16, "stride": 16, "ds_factors": [0, 1],
        "refinement_quantile": 0.5, "refinement_min_area": 1,
        "refinement_safety_fraction": 0.25,
    }
    inferencer._closed = False
    inferencer._inference_lock = Lock()
    inferencer._executor = ThreadPoolExecutor(max_workers=1)

    try:
        result = inferencer.inference([sample], use_pipeline=True)
        assert result["image_paths"] == [str(image_path)]
        assert result["anomaly_maps"][0].shape == (32, 32)
        assert "inference_timing" in result
    finally:
        inferencer.close()
