from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from easydict import EasyDict

from hiad.constants import TASK_TYPE_DYNAMIC_PATCH, TASK_TYPE_THUMBNAIL
from hiad.detectors.config import (
    REQUIRED_CONFIG_KEYS,
    detector_config_for_task,
    validate_required_config,
)
from hiad.detectors.hr_dinomaly import HRDinomaly


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "dinomaly.yaml"


def _config():
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


class _LoadableModule:
    def __init__(self):
        self.loaded_state = None

    def load_state_dict(self, state):
        self.loaded_state = state


def test_default_config_declares_every_required_production_setting():
    config = _config()

    assert set(REQUIRED_CONFIG_KEYS).issubset(config)
    validate_required_config(config)


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("semantic_weight", -0.1, "evidence weights"),
        ("global_routing_weight", 1.1, "global_routing_weight"),
        ("normal_component_percentile", 1.0, "normal_component_percentile"),
        ("decision_recheck_margin_ratio", 1.1, "decision_recheck_margin_ratio"),
        ("min_mean_luminance", 0.99, "luminance thresholds"),
        ("min_focus_variance", float("nan"), "min_focus_variance"),
    ],
)
def test_required_config_rejects_malformed_architecture_settings(key, value, match):
    config = deepcopy(_config())
    config[key] = value

    with pytest.raises(ValueError, match=match):
        validate_required_config(config)


def test_required_config_rejects_missing_architecture_setting():
    config = _config()
    del config["memory_weight"]

    with pytest.raises(ValueError, match="Missing required config setting: memory_weight"):
        validate_required_config(config)


def test_detector_config_derives_task_fields_from_single_config():
    config = EasyDict(_config())
    patch_config = detector_config_for_task(
        config,
        {
            "name": TASK_TYPE_DYNAMIC_PATCH,
            "type": TASK_TYPE_DYNAMIC_PATCH,
            "patch_size": 512,
            "stride": 512,
            "ds_factors": [0, 1],
        },
    )
    thumbnail_config = detector_config_for_task(
        config,
        {
            "name": TASK_TYPE_THUMBNAIL,
            "type": TASK_TYPE_THUMBNAIL,
            "thumbnail_size": 512,
        },
    )

    assert patch_config.use_context_conditioning is True
    assert thumbnail_config.use_context_conditioning is False
    assert thumbnail_config.total_iters == config.thumbnail_total_iters
    assert "patch_size" not in config


def test_checkpoint_save_persists_inference_state(monkeypatch):
    class Module:
        def state_dict(self):
            return {"state": "present"}

    detector = object.__new__(HRDinomaly)
    detector.context_conditioner = Module()
    detector.feature_memory = Module()
    detector.bottleneck = Module()
    detector.decoder = Module()
    detector.high_frequency_center = 0.2
    detector.high_frequency_scale = 0.7
    detector.semantic_center = 0.1
    detector.semantic_scale = 0.4
    detector.memory_center = 0.3
    detector.memory_scale = 0.8
    captured = {}

    monkeypatch.setattr(
        "hiad.detectors.hr_dinomaly.torch.save",
        lambda payload, _: captured.update(payload),
    )
    detector.save_checkpoint("unused.pkl")

    assert captured == {
        "context_conditioner": {"state": "present"},
        "feature_memory": {"state": "present"},
        "bottleneck": {"state": "present"},
        "decoder": {"state": "present"},
        "high_frequency_center": 0.2,
        "high_frequency_scale": 0.7,
        "semantic_center": 0.1,
        "semantic_scale": 0.4,
        "memory_center": 0.3,
        "memory_scale": 0.8,
    }


def test_checkpoint_omits_context_state_for_single_resolution_task(monkeypatch):
    class Module:
        def state_dict(self):
            return {"state": "present"}

    detector = object.__new__(HRDinomaly)
    detector.context_conditioner = None
    detector.feature_memory = Module()
    detector.bottleneck = Module()
    detector.decoder = Module()
    detector.high_frequency_center = 0.2
    detector.high_frequency_scale = 0.7
    detector.semantic_center = 0.1
    detector.semantic_scale = 0.4
    detector.memory_center = 0.3
    detector.memory_scale = 0.8
    captured = {}

    monkeypatch.setattr(
        "hiad.detectors.hr_dinomaly.torch.save",
        lambda payload, _: captured.update(payload),
    )
    detector.save_checkpoint("unused.pkl")

    assert "context_conditioner" not in captured


def test_checkpoint_load_restores_inference_state(monkeypatch):
    detector = object.__new__(HRDinomaly)
    detector.device = "cuda:0"
    detector.context_conditioner = _LoadableModule()
    detector.feature_memory = _LoadableModule()
    detector.bottleneck = _LoadableModule()
    detector.decoder = _LoadableModule()
    payload = {
        "context_conditioner": {"context": 1},
        "feature_memory": {"memory": 2},
        "bottleneck": {"bottleneck": 3},
        "decoder": {"decoder": 4},
        "high_frequency_center": 0.2,
        "high_frequency_scale": 0.7,
        "semantic_center": 0.1,
        "semantic_scale": 0.4,
        "memory_center": 0.3,
        "memory_scale": 0.8,
    }
    monkeypatch.setattr(
        "hiad.detectors.hr_dinomaly.torch.load",
        lambda *_args, **_kwargs: payload,
    )

    detector.load_checkpoint("checkpoint.pkl")

    assert detector.context_conditioner.loaded_state == {"context": 1}
    assert detector.feature_memory.loaded_state == {"memory": 2}
    assert detector.bottleneck.loaded_state == {"bottleneck": 3}
    assert detector.decoder.loaded_state == {"decoder": 4}
    assert detector.high_frequency_center == 0.2
    assert detector.high_frequency_scale == 0.7
    assert detector.semantic_center == 0.1
    assert detector.semantic_scale == 0.4
    assert detector.memory_center == 0.3
    assert detector.memory_scale == 0.8
