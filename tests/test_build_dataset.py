import json

import numpy as np
import pytest
from PIL import Image

from runs.build_dataset import build_dataset


def _image(path, size=(32, 24), value=64):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, color=value).save(path)


def _manifest(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_build_dataset_uses_explicit_splits_labels_and_categories(tmp_path):
    source = tmp_path / "source"
    _image(source / "line_a" / "001.bmp")
    _image(source / "line_a" / "002.bmp")
    _image(source / "line_b" / "001.bmp")
    manifest = tmp_path / "manifest.jsonl"
    _manifest(
        manifest,
        [
            {"filename": "line_a/001.bmp", "split": "train", "clsname": "panel", "label": 0},
            {"filename": "line_a/002.bmp", "split": "test", "clsname": "panel", "label": 1, "label_name": "scratch"},
            {"filename": "line_b/001.bmp", "split": "test", "clsname": "fabric"},
        ],
    )

    output = tmp_path / "dataset"
    summary = build_dataset(source, manifest, output)

    train = [json.loads(line) for line in (output / "train_uni.jsonl").read_text().splitlines()]
    test = [json.loads(line) for line in (output / "test_uni.jsonl").read_text().splitlines()]
    assert summary["categories"] == ["fabric", "panel"]
    assert train[0]["filename"] == "images/train/panel/line_a/001.bmp"
    assert [record["label"] for record in test] == [1, None]
    assert all((output / record["filename"]).is_file() for record in train + test)


def test_build_dataset_applies_per_record_roi_to_image_and_mask(tmp_path):
    source = tmp_path / "source"
    _image(source / "part.png", size=(40, 30))
    _image(source / "part_mask.png", size=(40, 30), value=1)
    manifest = tmp_path / "manifest.jsonl"
    _manifest(
        manifest,
        [{
            "filename": "part.png",
            "mask": "part_mask.png",
            "split": "test",
            "clsname": "part",
            "label": 1,
            "roi": [5, 4, 20, 16],
        }],
    )

    output = tmp_path / "dataset"
    build_dataset(source, manifest, output)
    record = json.loads((output / "test_uni.jsonl").read_text())
    with Image.open(output / record["filename"]) as image:
        assert image.size == (20, 16)
    with Image.open(output / record["mask"]) as mask:
        assert mask.size == (20, 16)
        assert np.asarray(mask).min() == 255


def test_build_dataset_rejects_nonempty_output(tmp_path):
    source = tmp_path / "source"
    _image(source / "image.bmp")
    manifest = tmp_path / "manifest.jsonl"
    _manifest(
        manifest,
        [{"filename": "image.bmp", "split": "train", "clsname": "part", "label": 0}],
    )
    output = tmp_path / "dataset"
    output.mkdir()
    (output / "keep.txt").write_text("user data")

    with pytest.raises(FileExistsError, match="not empty"):
        build_dataset(source, manifest, output)

    assert (output / "keep.txt").read_text() == "user data"


def test_build_dataset_rejects_anomalous_training_records(tmp_path):
    source = tmp_path / "source"
    _image(source / "image.bmp")
    manifest = tmp_path / "manifest.jsonl"
    _manifest(
        manifest,
        [{"filename": "image.bmp", "split": "train", "clsname": "part", "label": 1}],
    )

    with pytest.raises(ValueError, match="normal and mask-free"):
        build_dataset(source, manifest, tmp_path / "dataset")
