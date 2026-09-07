import numpy as np
import torch
from numpy.typing import NDArray

from hiad.data import (
    HRImage,
    HRImageIndex,
    HRSample,
    create_dynamic_patch,
    split_image_regions,
)
from hiad.data.patch_builder import (
    build_patch_batch,
    build_thumbnail_batch,
    crop_tile,
)
from hiad.datasets import PatchDataset


def _reference_item(image: NDArray[np.uint8], index: HRImageIndex):
    """逐 tile 参照路径：create_dynamic_patch + transform_patch。"""
    sample = HRSample(image=HRImage.from_array(image), clsname="part")
    sample.open()
    patch = create_dynamic_patch(sample, index)
    converter = PatchDataset(patches=[patch], training=False, task_name="dynamic_patch")
    return converter.transform_patch(patch)


def test_build_patch_batch_is_bit_identical_to_per_tile_path():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(64, 96, 3), dtype=np.uint8)
    indexes = split_image_regions((96, 64), patch_size=32, stride=32)

    base_record = {
        "task_name": "dynamic_patch",
        "task_type": "dynamic_patch",
        "image_path": "<array>",
        "image_size": (96, 64),
        "model_input_size": (32, 32),
    }
    batch, records = build_patch_batch(image, indexes, 32, base_record)

    assert len(records) == len(indexes)
    for i, index in enumerate(indexes):
        reference = _reference_item(image, index)
        assert torch.equal(batch["image"][i], reference["image"])
        assert records[i]["source_xywh"] == (index.x, index.y, 32, 32)


def test_build_patch_batch_matches_records_for_refinement():
    rng = np.random.default_rng(1)
    image = rng.integers(0, 256, size=(48, 48, 3), dtype=np.uint8)
    index = HRImageIndex(x=8, y=8, width=16, height=16)
    base_record = {
        "task_name": "refinement_patch",
        "task_type": "refinement_patch",
        "image_path": "<array>",
        "image_size": (48, 48),
        "model_input_size": (16, 16),
    }
    batch, records = build_patch_batch(image, [index], 16, base_record)
    assert records[0]["source_xywh"] == (8, 8, 16, 16)
    assert batch["image"].shape == (1, 3, 16, 16)


def test_thumbnail_batch_matches_downsampling_reference():
    rng = np.random.default_rng(2)
    image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    sample = HRSample(image=HRImage.from_array(image), clsname="part")
    sample.open()
    reference_patch = sample.down_sampling_to_LR(16)
    converter = PatchDataset(patches=[reference_patch], training=False, task_name="thumbnail")
    reference = converter.transform_patch(reference_patch)
    batch, _ = build_thumbnail_batch(
        image, 16, {"task_name": "thumbnail", "task_type": "thumbnail",
                    "image_path": "<array>", "image_size": (64, 64),
                    "model_input_size": (16, 16)}
    )
    assert torch.equal(batch["image"][0], reference["image"])
