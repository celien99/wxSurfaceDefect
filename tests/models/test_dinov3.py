"""``TimmDinoV3Encoder`` 本地主干权重加载的单元测试。"""

from __future__ import annotations

import pytest

from hiad.constants import DINO_PATCH_SIZE
from hiad.models.dinov3 import TimmDinoV3Encoder


class _PatchEmbed:
    """模拟主干 patch 嵌入，patch 边长必须与项目常量一致。"""

    def __init__(self) -> None:
        self.patch_size = DINO_PATCH_SIZE


class _FakeBackbone:
    """模拟 ``timm`` 主干：记录 ``create_model`` 参数并捕获装载的状态字典。"""

    def __init__(self) -> None:
        self.patch_embed = _PatchEmbed()
        self.num_features = 768
        self.loaded_state = None

    def parameters(self):
        return []

    def eval(self):
        return self

    def load_state_dict(self, state, strict=True):
        self.loaded_state = state


def _patch_create_model(monkeypatch, fake_backbone, captured):
    monkeypatch.setattr(
        "hiad.models.dinov3.timm.create_model",
        lambda model_name, pretrained, num_classes: captured.update(
            {
                "model_name": model_name,
                "pretrained": pretrained,
                "num_classes": num_classes,
            }
        )
        or fake_backbone,
    )


def test_encoder_loads_local_weights_without_network(monkeypatch, tmp_path):
    weights_path = tmp_path / "backbone.pth"
    weights_path.write_bytes(b"placeholder")
    state_dict = {"some.weight": "tensor"}
    fake_backbone = _FakeBackbone()
    captured = {}
    _patch_create_model(monkeypatch, fake_backbone, captured)
    monkeypatch.setattr(
        "hiad.models.dinov3.torch.load",
        lambda path, map_location, weights_only: state_dict,
    )

    encoder = TimmDinoV3Encoder(
        model_name="vit_base_patch16_dinov3.lvd1689m",
        intermediate_layers=[2, 3],
        weights_path=str(weights_path),
    )

    assert captured["pretrained"] is False
    assert captured["num_classes"] == 0
    assert fake_backbone.loaded_state == state_dict
    assert encoder.patch_size == DINO_PATCH_SIZE


def test_encoder_without_weights_path_falls_back_to_pretrained(monkeypatch):
    fake_backbone = _FakeBackbone()
    captured = {}
    _patch_create_model(monkeypatch, fake_backbone, captured)

    TimmDinoV3Encoder(
        model_name="vit_base_patch16_dinov3.lvd1689m",
        intermediate_layers=[2, 3],
        weights_path=None,
    )

    assert captured["pretrained"] is True


def test_encoder_treats_empty_weights_path_as_dev_fallback(monkeypatch):
    fake_backbone = _FakeBackbone()
    captured = {}
    _patch_create_model(monkeypatch, fake_backbone, captured)

    TimmDinoV3Encoder(
        model_name="vit_base_patch16_dinov3.lvd1689m",
        intermediate_layers=[2, 3],
        weights_path="",
    )

    assert captured["pretrained"] is True


def test_encoder_rejects_missing_weights_file(monkeypatch):
    fake_backbone = _FakeBackbone()
    captured = {}
    _patch_create_model(monkeypatch, fake_backbone, captured)

    with pytest.raises(FileNotFoundError, match="Backbone weights file not found"):
        TimmDinoV3Encoder(
            model_name="vit_base_patch16_dinov3.lvd1689m",
            intermediate_layers=[2, 3],
            weights_path="/nonexistent/backbone.pth",
        )
