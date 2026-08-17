from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Final, Protocol, cast

from hiad.constants import (
    TASK_TYPE_DYNAMIC_PATCH,
    TASK_TYPE_REFINEMENT_PATCH,
    TASK_TYPE_THUMBNAIL,
)
from hiad.task.contracts import (
    DynamicPatchTask,
    RefinementPatchTask,
    TaskDefinition,
    ThumbnailTask,
)


class DetectorConfig(Protocol):
    """训练、推理和校准共享的生产配置静态视图。

    Attributes:
        backbone_name (str): ``timm`` DINOv3 主干名称。
        total_iters (int): 补丁任务配置训练迭代数。
        thumbnail_total_iters (int): 缩略图任务配置训练迭代数。
        log_per_steps (int): 训练日志间隔。
        bottleneck_dropout (float): 重建瓶颈 dropout，范围 ``[0, 1)``。
        grad_clip_norm (float): 梯度裁剪上限；``0`` 表示禁用。
        hard_mining_final (float): 困难样本挖掘最终比例。
        hard_mining_warmup_iters (int): 困难挖掘线性预热迭代数。
        easy_grad_factor (float): 易样本梯度权重。
        encoder_amp (bool): 是否在 CUDA 上用 FP16 编码。
        decoder_amp (bool): 是否在 CUDA 上用 FP16 训练重建模块。
        allow_tf32 (bool): 是否允许 CUDA TF32 矩阵和卷积计算。
        semantic_weight (float): 语义重建证据融合权重。
        memory_weight (float): 正常特征记忆证据融合权重。
        high_frequency_weight (float): 高频纹理证据融合权重。
        global_routing_weight (float): 缩略图先验在复核路由图中的权重。
        patches_per_source (int): 每轮每张正常原图最多采样的补丁数。
        score_top_k (int): 图像分数使用的最高分 token 数。
        normal_score_percentile (float): 正常图像分数校准分位数。
        normal_component_percentile (float): 正常连通组件分数校准分位数。
        normal_pixel_percentile (float): 单图异常图压缩分位数。
        normal_pixel_image_percentile (float): 跨正常图像的像素阈值分位数。
        calibration_batch_size (int): 两阶段校准批量大小。
        map_gaussian_sigma (float): 补丁拼图完成后的高斯平滑 sigma。
        decision_recheck_margin_ratio (float): 图像阈值比例形式的复检带宽。
        min_mean_luminance (float): 质量门禁最小平均亮度。
        max_mean_luminance (float): 质量门禁最大平均亮度。
        max_clipped_fraction (float): 质量门禁最大黑白截断比例。
        min_focus_variance (float): 质量门禁最小 Laplacian 方差。
        patch_size (int): 当前任务注入的正方形模型输入边长。
        use_context_conditioning (bool): 当前任务是否启用多尺度上下文条件化。
    """

    backbone_name: str
    total_iters: int
    thumbnail_total_iters: int
    log_per_steps: int
    bottleneck_dropout: float
    grad_clip_norm: float
    hard_mining_final: float
    hard_mining_warmup_iters: int
    easy_grad_factor: float
    encoder_amp: bool
    decoder_amp: bool
    allow_tf32: bool
    semantic_weight: float
    memory_weight: float
    high_frequency_weight: float
    global_routing_weight: float
    patches_per_source: int
    score_top_k: int
    normal_score_percentile: float
    normal_component_percentile: float
    normal_pixel_percentile: float
    normal_pixel_image_percentile: float
    calibration_batch_size: int
    map_gaussian_sigma: float
    decision_recheck_margin_ratio: float
    min_mean_luminance: float
    max_mean_luminance: float
    max_clipped_fraction: float
    min_focus_variance: float
    patch_size: int
    use_context_conditioning: bool

    def __getitem__(self, key: str) -> object: ...


REQUIRED_CONFIG_KEYS: Final = (
    "backbone_name",
    "total_iters",
    "thumbnail_total_iters",
    "log_per_steps",
    "bottleneck_dropout",
    "grad_clip_norm",
    "hard_mining_final",
    "hard_mining_warmup_iters",
    "easy_grad_factor",
    "encoder_amp",
    "decoder_amp",
    "allow_tf32",
    "semantic_weight",
    "memory_weight",
    "high_frequency_weight",
    "global_routing_weight",
    "patches_per_source",
    "score_top_k",
    "normal_score_percentile",
    "normal_component_percentile",
    "normal_pixel_percentile",
    "normal_pixel_image_percentile",
    "calibration_batch_size",
    "map_gaussian_sigma",
    "decision_recheck_margin_ratio",
    "min_mean_luminance",
    "max_mean_luminance",
    "max_clipped_fraction",
    "min_focus_variance",
)


def _config_value(config: object, key: str) -> object:
    """从映射或属性对象读取必需配置值，缺失时快速失败。

    Args:
        config (object): 配置映射或属性对象。
        key (str): 必需配置字段名。

    Returns:
        object: 未转换的配置值。

    Raises:
        ValueError: 配置中不存在该字段。
    """
    if isinstance(config, Mapping):
        if key not in config:
            raise ValueError(f"Missing required config setting: {key}")
        return config[key]
    if not hasattr(config, key):
        raise ValueError(f"Missing required config setting: {key}")
    return getattr(config, key)


def _positive_int(value: object, key: str) -> None:
    """校验配置值为正整数且不是布尔值。

    Args:
        value (object): 待校验配置值。
        key (str): 用于错误消息的字段名。

    Raises:
        ValueError: 值不是正整数或是布尔值。
    """
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")


def _finite_number(value: object, key: str) -> float:
    """将配置值转换为有限浮点数。

    Args:
        value (object): 支持 ``float`` 转换的配置值。
        key (str): 用于错误消息的字段名。

    Returns:
        float: 有限浮点数。

    Raises:
        ValueError: 值是布尔值、不能转换或为 NaN/无穷值。
    """
    if isinstance(value, bool):
        raise ValueError(f"{key} must be a finite number")
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{key} must be a finite number") from error
    if not math.isfinite(value):
        raise ValueError(f"{key} must be a finite number")
    return value


def validate_required_config(config: object) -> None:
    """在创建模型前完整校验当前生产配置，禁止静默使用缺省值。

    Args:
        config (object): 支持字符串键读取的映射或暴露同名属性的配置对象。

    Raises:
        ValueError: 缺少任一生产字段，或类型、数值范围、证据权重和质量阈值
            不符合约束。
    """
    values = {key: _config_value(config, key) for key in REQUIRED_CONFIG_KEYS}
    if not isinstance(values["backbone_name"], str) or not values["backbone_name"].strip():
        raise ValueError("backbone_name must be a non-empty string")
    for key in (
        "total_iters",
        "thumbnail_total_iters",
        "log_per_steps",
        "patches_per_source",
        "score_top_k",
        "calibration_batch_size",
    ):
        _positive_int(values[key], key)
    for key in ("encoder_amp", "decoder_amp", "allow_tf32"):
        if not isinstance(values[key], bool):
            raise ValueError(f"{key} must be a boolean")

    dropout = _finite_number(values["bottleneck_dropout"], "bottleneck_dropout")
    hard_mining = _finite_number(values["hard_mining_final"], "hard_mining_final")
    grad_clip = _finite_number(values["grad_clip_norm"], "grad_clip_norm")
    easy_factor = _finite_number(values["easy_grad_factor"], "easy_grad_factor")
    if not 0 <= dropout < 1:
        raise ValueError("bottleneck_dropout must be in [0, 1)")
    if grad_clip < 0:
        raise ValueError("grad_clip_norm must be non-negative")
    if not 0 <= hard_mining <= 1 or not 0 <= easy_factor <= 1:
        raise ValueError("hard-mining settings must be in [0, 1]")
    if (
        isinstance(values["hard_mining_warmup_iters"], bool)
        or not isinstance(values["hard_mining_warmup_iters"], int)
        or values["hard_mining_warmup_iters"] < 0
    ):
        raise ValueError("hard_mining_warmup_iters must be a non-negative integer")

    evidence_weights = tuple(
        _finite_number(values[key], key)
        for key in ("semantic_weight", "memory_weight", "high_frequency_weight")
    )
    if any(weight < 0 for weight in evidence_weights) or sum(evidence_weights) <= 0:
        raise ValueError("evidence weights must be non-negative with a positive sum")
    if not 0 <= _finite_number(
        values["global_routing_weight"], "global_routing_weight"
    ) <= 1:
        raise ValueError("global_routing_weight must be in [0, 1]")
    for key in (
        "normal_score_percentile",
        "normal_component_percentile",
        "normal_pixel_percentile",
        "normal_pixel_image_percentile",
    ):
        if not 0 < _finite_number(values[key], key) < 1:
            raise ValueError(f"{key} must be in the open interval (0, 1)")
    if _finite_number(values["map_gaussian_sigma"], "map_gaussian_sigma") < 0:
        raise ValueError("map_gaussian_sigma must be non-negative")
    if not 0 <= _finite_number(
        values["decision_recheck_margin_ratio"], "decision_recheck_margin_ratio"
    ) <= 1:
        raise ValueError("decision_recheck_margin_ratio must be in [0, 1]")

    min_luminance = _finite_number(values["min_mean_luminance"], "min_mean_luminance")
    max_luminance = _finite_number(values["max_mean_luminance"], "max_mean_luminance")
    clipped_fraction = _finite_number(values["max_clipped_fraction"], "max_clipped_fraction")
    focus_variance = _finite_number(values["min_focus_variance"], "min_focus_variance")
    if not 0 <= min_luminance < max_luminance <= 1:
        raise ValueError("luminance thresholds must satisfy 0 <= min < max <= 1")
    if not 0 <= clipped_fraction <= 1:
        raise ValueError("max_clipped_fraction must be in [0, 1]")
    if focus_variance < 0:
        raise ValueError("min_focus_variance must be non-negative")


def detector_config_for_task(
    config: DetectorConfig,
    task: TaskDefinition,
) -> DetectorConfig:
    """复制共享配置，并只注入当前任务决定的模型尺寸与上下文开关。

    Args:
        config (DetectorConfig): 已验证的共享生产配置。
        task (TaskDefinition): 粗扫、复核或缩略图任务。

    Returns:
        DetectorConfig: 与共享配置隔离的深拷贝。补丁任务注入 patch size 并按
        多尺度数量启用上下文；缩略图任务改用专用迭代数且关闭上下文。

    Raises:
        ValueError: 任务类型不受支持。
    """
    if task["type"] in {TASK_TYPE_DYNAMIC_PATCH, TASK_TYPE_REFINEMENT_PATCH}:
        patch_task = cast(DynamicPatchTask | RefinementPatchTask, task)
        detector_config = copy.deepcopy(config)
        detector_config.patch_size = patch_task["patch_size"]
        detector_config.use_context_conditioning = len(patch_task["ds_factors"]) > 1
        return detector_config
    if task["type"] == TASK_TYPE_THUMBNAIL:
        thumbnail_task = cast(ThumbnailTask, task)
        detector_config = copy.deepcopy(config)
        detector_config.patch_size = thumbnail_task["thumbnail_size"]
        detector_config.total_iters = detector_config.thumbnail_total_iters
        detector_config.use_context_conditioning = False
        return detector_config
    raise ValueError(f"Unsupported task type: {task}")
