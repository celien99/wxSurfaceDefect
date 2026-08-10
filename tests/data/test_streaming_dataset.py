from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from hiad.constants import TASK_TYPE_DYNAMIC_PATCH, TASK_TYPE_THUMBNAIL
from hiad.data import HRSample, split_multiresolution_regions
from hiad.data.preparation import PreparedInputRecord
from hiad.datasets.streaming_dataset import StreamingTaskDataset


class FakeRegistry:
    def __init__(self) -> None:
        self.process_calls = 0

    def get(self, clsname):
        return self

    def process_file(self, path, category=None):
        self.process_calls += 1
        with Image.open(path) as image_file:
            rgb = np.asarray(image_file.convert("RGB"), dtype=np.float32)
        return np.ascontiguousarray(rgb / 255.0, dtype=np.float32)


def _write_png(path, width: int, height: int, value: int = 128) -> None:
    array = np.full((height, width, 3), value, dtype=np.uint8)
    Image.fromarray(array, mode="RGB").save(path)


def _dynamic_task(**overrides):
    task = {
        "name": TASK_TYPE_DYNAMIC_PATCH,
        "type": TASK_TYPE_DYNAMIC_PATCH,
        "patch_size": 16,
        "stride": 16,
        "ds_factors": [0],
    }
    task.update(overrides)
    return task


def test_streaming_dataset_len_matches_regions(tmp_path):
    paths = [tmp_path / "a.png", tmp_path / "b.png"]
    for path in paths:
        _write_png(path, 32, 32)

    samples = [
        HRSample(str(paths[0]), clsname="cat", label=0),
        HRSample(str(paths[1]), clsname="cat", label=0),
    ]
    task = _dynamic_task()
    expected = 0
    for path in paths:
        with Image.open(path) as image_file:
            image_size = image_file.size
        expected += len(
            split_multiresolution_regions(
                image_size=image_size,
                patch_size=task["patch_size"],
                ds_factors=task["ds_factors"],
                stride=task["stride"],
            )
        )

    registry = FakeRegistry()
    dataset = StreamingTaskDataset(
        samples,
        task,
        registry,
        training=True,
    )

    assert len(dataset) == expected
    assert expected == 8
    assert len(dataset.records) == expected
    assert all(isinstance(record, PreparedInputRecord) for record in dataset.records)

    item = dataset[0]
    assert item["image"].shape[0] == 3
    assert item["image"].shape[1:] == (16, 16)
    assert item["mask"].shape == (16, 16)
    assert "clsname" in item
    assert item["clsname"] == "cat"


def test_streaming_dataset_releases_sample_handles_and_bounds_cache(tmp_path):
    path = tmp_path / "only.png"
    _write_png(path, 32, 32)
    samples = [HRSample(str(path), clsname="cat", label=0)]
    registry = FakeRegistry()
    dataset = StreamingTaskDataset(
        samples,
        _dynamic_task(),
        registry,
        training=False,
    )

    for index in range(len(dataset)):
        item = dataset[index]
        assert item["image"].shape[0] == 3
        assert samples[0].image.image is None
        assert samples[0].image.is_processed is False

    # One-image cache: preprocess once per path even when many patches.
    assert registry.process_calls == 1
    assert dataset._cached_image is not None
    assert dataset._cached_image.ndim == 3


def test_streaming_dataset_thumbnail_item_shape(tmp_path):
    path = tmp_path / "thumb.png"
    _write_png(path, 40, 24)
    samples = [HRSample(str(path), clsname="cat")]
    task = {
        "name": TASK_TYPE_THUMBNAIL,
        "type": TASK_TYPE_THUMBNAIL,
        "thumbnail_size": 8,
    }
    dataset = StreamingTaskDataset(
        samples,
        task,
        FakeRegistry(),
        training=True,
    )
    assert len(dataset) == 1
    assert len(dataset.records) == 1
    assert dataset.records[0].task_type == TASK_TYPE_THUMBNAIL
    assert dataset.records[0].model_input_size == (8, 8)
    item = dataset[0]
    assert tuple(item["image"].shape) == (3, 8, 8)
