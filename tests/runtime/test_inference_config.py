import pytest

from hiad.runtime.inference_config import InferenceConfig, load_inference_config


def test_defaults_when_section_absent():
    config = load_inference_config({})
    assert config == InferenceConfig()


def test_parses_optional_inference_section():
    config = load_inference_config({"inference": {"batch_memory_budget_gb": 4.0}})
    assert config.batch_memory_budget_gb == 4.0
    assert config.preprocess_backend == "vectorized_cpu"


def test_accepts_attribute_object_without_section():
    from types import SimpleNamespace
    config = load_inference_config(SimpleNamespace(backbone_name="x"))
    assert config == InferenceConfig()


def test_rejects_non_mapping_section():
    with pytest.raises(ValueError):
        load_inference_config({"inference": [1, 2, 3]})


def test_rejects_negative_budget():
    with pytest.raises(ValueError):
        load_inference_config({"inference": {"batch_memory_budget_gb": -1}})


def test_rejects_unknown_preprocess_backend():
    with pytest.raises(ValueError):
        load_inference_config({"inference": {"preprocess_backend": "cuda"}})


def test_async_pipeline_parses_bool():
    config = load_inference_config({"inference": {"async_pipeline": True}})
    assert config.async_pipeline is True


def test_async_pipeline_defaults_false():
    config = load_inference_config({})
    assert config.async_pipeline is False


def test_async_pipeline_rejects_non_bool():
    with pytest.raises(ValueError):
        load_inference_config({"inference": {"async_pipeline": "yes"}})
