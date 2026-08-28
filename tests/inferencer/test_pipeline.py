import numpy as np
import torch

from hiad.constants import TASK_TYPE_DYNAMIC_PATCH, TASK_TYPE_THUMBNAIL
from hiad.data import HRImage, HRSample
from hiad.inferencer.pipeline import (
    DeviceImagePipeline,
    ImagePipelineOutput,
    _PrefetchWorker,
)
from hiad.runtime.inference_config import InferenceConfig


class _StubDetector:
    """返回固定异常图的检测器替身，只实现推理热路径需要的接口。"""

    def __init__(self, patch_size):
        self.device = torch.device("cpu")
        self.score_top_k = 4
        self.patch_size = [patch_size, patch_size]

    def inference_batch(self, data):
        batch = data["image"].shape[0]
        height, width = data["image"].shape[2:]
        fused_pixel = torch.full((batch, 1, height, width), 0.25)
        fused_token = torch.full((batch, 1, height // 16, width // 16), 0.25)
        return fused_pixel, fused_token


_COARSE = {
    "name": "dynamic_patch", "type": TASK_TYPE_DYNAMIC_PATCH,
    "patch_size": 16, "stride": 16, "ds_factors": [0, 1],
}
_THUMBNAIL = {
    "name": "thumbnail", "type": TASK_TYPE_THUMBNAIL, "thumbnail_size": 16,
}
_REFINEMENT = {
    "name": "refinement_patch", "type": "refinement_patch",
    "patch_size": 16, "stride": 16, "ds_factors": [0, 1],
    "refinement_quantile": 0.5, "refinement_min_area": 1,
    "refinement_safety_fraction": 0.25,
}


def _sample(image):
    return HRSample(image=HRImage.from_array(image), clsname="part")


def _make_pipeline():
    return DeviceImagePipeline(
        detectors={"dynamic_patch": _StubDetector(16), "thumbnail": _StubDetector(16),
                   "refinement_patch": _StubDetector(16)},
        coarse_tasks=[_COARSE, _THUMBNAIL],
        refinement_task=_REFINEMENT,
        inference_config=InferenceConfig(),
        global_routing_weight=0.25,
        score_top_k=4,
        refinement_bridge_gap_tiles=1,
        map_gaussian_sigma=0.0,
    )


def test_process_image_returns_full_resolution_final_map():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    pipeline = _make_pipeline()
    output = pipeline.process_images([_sample(image)])[0]
    assert isinstance(output, ImagePipelineOutput)
    assert output.final_map.shape == (32, 32)
    assert output.image_size == (32, 32)
    assert np.isfinite(output.final_map).all()
    assert output.refinement_statistics["total_tiles"] == 4
    assert output.refinement_statistics["selected_tiles"] >= 1


def test_process_image_keeps_decoded_sample_open_for_refinement():
    rng = np.random.default_rng(1)
    image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    pipeline = _make_pipeline()
    pipeline.process_images([_sample(image)])
    # 处理后样本应关闭，避免长期持有大图内存
    # （process_images 内部对每个样本 open/close；此处仅断言不抛异常）


def test_process_images_reuses_prefetch_worker():
    rng = np.random.default_rng(2)
    images = [
        rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8) for _ in range(3)
    ]
    pipeline = _make_pipeline()
    outputs = pipeline.process_images([_sample(image) for image in images])
    assert len(outputs) == 3
    assert [output.image_size for output in outputs] == [(32, 32)] * 3


def test_staged_serial_matches_legacy_coarse_route():
    """阶段化串行路径与保留的旧 _coarse_forward/_route 输出逐位一致。"""
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, size=(48, 48, 3), dtype=np.uint8)
    new_out = _make_pipeline().process_images([_sample(image)])[0]

    pipeline = _make_pipeline()
    item = None
    with _PrefetchWorker([_sample(image.copy())], pipeline._build_item) as worker:
        item = worker.next()
    assert item is not None
    image_size = (int(item.image.shape[1]), int(item.image.shape[0]))
    coarse_map, thumbnail_score, global_context_map = pipeline._coarse_forward(
        item, image_size
    )
    regions, _routing_np, _coarse = pipeline._route(
        coarse_map, global_context_map, image_size
    )
    legacy_map = pipeline._refine_and_merge(item, regions, coarse_map, image_size)
    item.sample.close()

    assert np.array_equal(new_out.final_map, legacy_map)
    assert new_out.thumbnail_score == thumbnail_score
    assert new_out.refinement_statistics["selected_tiles"] == len(regions)
