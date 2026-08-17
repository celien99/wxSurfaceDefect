import numpy as np
import pytest
import torch

from hiad.data import LRPatch
from hiad.datasets.patch_dataset import PatchDataset


def test_reusable_patch_converter_matches_indexed_conversion():
    patch = LRPatch(
        image=np.full((8, 8, 3), 128, dtype=np.uint8),
        clsname="part",
        label=0,
    )
    indexed = PatchDataset([patch], training=False, task_name="dynamic_patch")
    reusable = PatchDataset([], training=False, task_name="dynamic_patch")

    expected = indexed[0]
    first = reusable.transform_patch(patch)
    second = reusable.transform_patch(patch)

    torch.testing.assert_close(first["image"], expected["image"])
    torch.testing.assert_close(second["image"], expected["image"])
    assert first["mask"].data_ptr() == second["mask"].data_ptr()
    assert first["clsname"] == second["clsname"] == "part"
    assert first["label"] == second["label"] == 0


def test_reusable_patch_converter_still_rejects_nonfinite_float_input():
    image = np.zeros((4, 4, 3), dtype=np.float32)
    image[0, 0, 0] = np.nan
    converter = PatchDataset([], training=False, task_name="dynamic_patch")

    with pytest.raises(ValueError, match="finite HWC RGB image"):
        converter.transform_patch(LRPatch(image=image))


def test_float_patch_normalization_does_not_mutate_source_array():
    image = np.full((4, 4, 3), 0.5, dtype=np.float32)
    original = image.copy()
    converter = PatchDataset([], training=False, task_name="dynamic_patch")

    converter.transform_patch(LRPatch(image=image))

    np.testing.assert_array_equal(image, original)
