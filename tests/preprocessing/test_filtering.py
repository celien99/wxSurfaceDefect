from __future__ import annotations

import numpy as np
import pytest

from hiad.data import HRSample
from hiad.preprocessing.filtering import filter_registerable_samples
from hiad.preprocessing.masks import MaskRejected


class _FakePreprocessor:
    def __init__(self, reject_paths: set[str]) -> None:
        self.reject_paths = reject_paths

    def process_file(self, path, category=None):
        if path in self.reject_paths:
            raise MaskRejected("dino_inlier_ratio_below_threshold")
        return np.zeros((4, 4, 3), dtype=np.float32)


class _FakeRegistry:
    def __init__(self, reject_paths: set[str]) -> None:
        self._prep = _FakePreprocessor(reject_paths)

    def get(self, clsname):
        return self._prep


def test_filter_registerable_samples_keeps_passing_paths(tmp_path):
    good = str(tmp_path / "good.png")
    bad = str(tmp_path / "bad.png")
    samples = [
        HRSample(good, clsname="bottle", label=0),
        HRSample(bad, clsname="bottle", label=0),
        HRSample(str(tmp_path / "also_good.png"), clsname="bottle", label=0),
    ]
    # Paths that do not exist are fine: fake preprocessor never opens files.
    kept = filter_registerable_samples(
        samples,
        _FakeRegistry({bad}),
    )
    assert [sample.image.image_path for sample in kept] == [
        good,
        str(tmp_path / "also_good.png"),
    ]


def test_filter_registerable_samples_raises_when_all_rejected(tmp_path):
    path = str(tmp_path / "only.png")
    samples = [HRSample(path, clsname="bottle", label=0)]
    with pytest.raises(ValueError, match="No training samples passed"):
        filter_registerable_samples(samples, _FakeRegistry({path}))
