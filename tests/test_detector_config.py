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
        ("backbone_weights_path", 123, "backbone_weights_path must be a string"),
        ("global_routing_weight", 1.1, "global_routing_weight"),
        ("refinement_bridge_gap_tiles", -1, "refinement_bridge_gap_tiles"),
        ("normal_component_percentile", 1.0, "normal_component_percentile"),
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
    del config["global_routing_weight"]

    with pytest.raises(ValueError, match="Missing required config setting: global_routing_weight"):
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

    assert patch_config.patch_size == 512
    assert thumbnail_config.total_iters == config.thumbnail_total_iters
    assert "patch_size" not in config


def test_checkpoint_save_persists_inference_state(monkeypatch):
    class Module:
        def state_dict(self):
            return {"state": "present"}

        def named_parameters(self):
            return []

    detector = object.__new__(HRDinomaly)
    detector.bottleneck = Module()
    detector.decoder = Module()
    detector.fusion_weights = None
    detector.score_top_k = 4
    detector.encoder_amp = True
    detector.decoder_amp = True
    detector.allow_tf32 = True
    captured = {}

    monkeypatch.setattr(
        "hiad.detectors.hr_dinomaly.torch.save",
        lambda payload, _: captured.update(payload),
    )
    detector.save_checkpoint("unused.pkl")

    assert captured == {
        "bottleneck": {"state": "present"},
        "decoder": {"state": "present"},
        "fusion_weights": None,
        "score_top_k": 4,
        "layer_aggregation": "max",
        "encoder_amp": True,
        "decoder_amp": True,
        "allow_tf32": True,
    }


def test_checkpoint_load_restores_inference_state(monkeypatch):
    detector = object.__new__(HRDinomaly)
    detector.device = "cpu"
    detector.bottleneck = _LoadableModule()
    detector.decoder = _LoadableModule()
    detector.encoder_amp = True
    detector.score_top_k = 4
    payload = {
        "bottleneck": {"bottleneck": 3},
        "decoder": {"decoder": 4},
        "fusion_weights": [0.5, 0.5],
        "score_top_k": 8,
        "layer_aggregation": "max",
        "encoder_amp": True,
    }
    monkeypatch.setattr(
        "hiad.detectors.hr_dinomaly.torch.load",
        lambda *_args, **_kwargs: payload,
    )

    detector.load_checkpoint("checkpoint.pkl")

    assert detector.bottleneck.loaded_state == {"bottleneck": 3}
    assert detector.decoder.loaded_state == {"decoder": 4}
    assert detector.fusion_weights == [0.5, 0.5]
    assert detector.score_top_k == 8
