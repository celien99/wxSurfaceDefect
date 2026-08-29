from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class InferenceConfig:
    """推理侧可调参数；只在推理读取，绝不进入训练必需配置。

    Attributes:
        batch_memory_budget_gb (float): 单个前向批的显存预算（GB）；``0`` 表示
            由空闲显存自动决定。
        preprocess_backend (str): 预处理后端；当前仅支持 ``vectorized_cpu``。
        async_pipeline (bool): 阶段级异步流水（P0 双缓冲）开关；默认关闭，
            训练机 parity + 性能门槛通过后翻转默认。
        decoder_amp (bool): 推理时是否用 FP16 autocast 跑重建解码器（P1）；
            线性层 FP16、einsum 核心保持 FP32。2026-08-28 训练机 parity +
            安全门通过后翻为默认开启。
    """

    batch_memory_budget_gb: float = 0.0
    preprocess_backend: str = "vectorized_cpu"
    async_pipeline: bool = False
    decoder_amp: bool = True


def _inference_section(config: object) -> object:
    """读取可选 ``inference`` 小节；缺失时返回空映射。"""
    if isinstance(config, Mapping):
        value = config.get("inference")
        return value if value is not None else {}
    return getattr(config, "inference", {}) or {}


def _finite_budget(value: object) -> float:
    """把预算转换为有限非负浮点数。"""
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("inference.batch_memory_budget_gb must be a number") from error
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError("inference.batch_memory_budget_gb must be finite")
    if number < 0:
        raise ValueError("inference.batch_memory_budget_gb must be non-negative")
    return number


def load_inference_config(config: object) -> InferenceConfig:
    """从可选 ``inference:`` YAML 小节读取推理参数，缺省时返回默认值。

    Args:
        config (object): 已验证的配置映射或属性对象。

    Returns:
        InferenceConfig: 解析并校验后的推理参数。

    Raises:
        ValueError: ``inference`` 小节不是映射，任一字段类型或取值不合法。
    """
    section = _inference_section(config)
    if not isinstance(section, Mapping):
        raise ValueError("inference section must be a mapping")
    budget = _finite_budget(section.get("batch_memory_budget_gb", 0.0))
    backend = section.get("preprocess_backend", "vectorized_cpu")
    if backend != "vectorized_cpu":
        raise ValueError(
            "inference.preprocess_backend only supports 'vectorized_cpu' in this version"
        )
    async_pipeline = section.get("async_pipeline", False)
    if not isinstance(async_pipeline, bool):
        raise ValueError("inference.async_pipeline must be a boolean")
    decoder_amp = section.get("decoder_amp", True)
    if not isinstance(decoder_amp, bool):
        raise ValueError("inference.decoder_amp must be a boolean")
    return InferenceConfig(
        batch_memory_budget_gb=budget,
        preprocess_backend=backend,
        async_pipeline=async_pipeline,
        decoder_amp=decoder_amp,
    )
