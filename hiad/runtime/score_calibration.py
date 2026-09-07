from __future__ import annotations

import copy
import json
import os
from collections.abc import Mapping, Sequence
from typing import Final, cast

import numpy as np
from numpy.typing import ArrayLike

from hiad.data import HRSample

from .contracts import ScoreCalibration, ScoreVector


SCORE_CALIBRATION_FILE: Final = "score_calibration.json"


def _validate_percentile(value: float) -> float:
    """将分位数转换为浮点并限制在开区间 ``(0, 1)``。

    Args:
        value (float): 待校验分位数。

    Returns:
        float: 有限且位于 ``(0, 1)`` 的分位数。

    Raises:
        ValueError: 值非有限或超出开区间。
    """
    value = float(value)
    if not np.isfinite(value) or not 0 < value < 1:
        raise ValueError("normal_score_percentile must be in the open interval (0, 1)")
    return value


def _score_threshold(scores: ArrayLike, percentile: float) -> float:
    """计算非空有限一维分数向量的分位阈值。

    Args:
        scores (ArrayLike): 一维校准分数。
        percentile (float): 已验证的目标分位数。

    Returns:
        float: NumPy 分位数阈值。

    Raises:
        ValueError: 分数不是非空有限一维向量。
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Calibration scores must be a non-empty finite vector")
    return float(np.quantile(values, percentile))


def summarize_anomaly_map(anomaly_map: ArrayLike, percentile: float) -> float:
    """将单张原图异常图压缩为标量，控制校准阶段的内存占用。

    Args:
        anomaly_map (ArrayLike): 单张原图的异常分数图。
        percentile (float): 图内像素分位数，必须位于 ``(0, 1)``。

    Returns:
        float: 异常图在指定分位数处的分数。

    Raises:
        ValueError: ``percentile`` 无效。异常图为空或包含非有限值时，本函数不
            做额外校验，结果和异常类型遵循 NumPy 的 ``quantile`` 行为。
    """
    values = np.asarray(anomaly_map)
    return float(np.quantile(values, _validate_percentile(percentile)))


def build_score_calibration(
    samples: Sequence[HRSample],
    scores: ArrayLike,
    pixel_statistics: ArrayLike,
    *,
    percentile: float,
    pixel_percentile: float,
    pixel_image_percentile: float,
) -> ScoreCalibration:
    """仅使用正常样本建立全局及分类别图像/像素阈值。

    Args:
        samples (Sequence[HRSample]): 具有非空 ``clsname`` 的正常样本序列。
        scores (ArrayLike): 与样本同序的一维有限图像级异常分数。
        pixel_statistics (ArrayLike): 与样本同序的一维有限图内像素统计量。
        percentile (float): 图像级正常分数分位数。
        pixel_percentile (float): 生成 ``pixel_statistics`` 时采用的图内分位数，
            保存到校准产物用于追溯。
        pixel_image_percentile (float): 在正常图像之间计算像素阈值的分位数。

    Returns:
        ScoreCalibration: 包含全局阈值和按类别阈值的可落盘校准结构。

    Raises:
        ValueError: 数量、有限性、分位数或样本类别不符合校准要求。
    """
    samples = tuple(samples)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(samples),):
        raise ValueError("Calibration score count must match the normal sample count")
    pixel_statistics = np.asarray(pixel_statistics, dtype=np.float64)
    if pixel_statistics.shape != (len(samples),):
        raise ValueError("Pixel statistic count must match the normal sample count")
    percentile = _validate_percentile(percentile)
    pixel_percentile = _validate_percentile(pixel_percentile)
    pixel_image_percentile = _validate_percentile(pixel_image_percentile)

    grouped_scores: dict[str, list[float]] = {}
    grouped_pixel_statistics: dict[str, list[float]] = {}
    for sample, score, pixel_statistic in zip(samples, scores, pixel_statistics):
        category = sample.clsname
        if not isinstance(category, str) or not category.strip():
            raise ValueError("Every calibration sample must have a non-empty clsname")
        grouped_scores.setdefault(category, []).append(float(score))
        grouped_pixel_statistics.setdefault(category, []).append(float(pixel_statistic))

    return {
        "percentile": percentile,
        "pixel_percentile": pixel_percentile,
        "pixel_image_percentile": pixel_image_percentile,
        "normal_image_count": len(samples),
        "global_threshold": _score_threshold(scores, percentile),
        "global_pixel_threshold": _score_threshold(
            pixel_statistics, pixel_image_percentile
        ),
        "categories": {
            category: {
                "normal_image_count": len(category_scores),
                "threshold": _score_threshold(category_scores, percentile),
                "pixel_threshold": _score_threshold(
                    grouped_pixel_statistics[category], pixel_image_percentile
                ),
            }
            for category, category_scores in sorted(grouped_scores.items())
        },
    }


def build_component_calibration(
    calibration: ScoreCalibration,
    samples: Sequence[HRSample],
    component_scores: ArrayLike,
    *,
    percentile: float,
) -> ScoreCalibration:
    """第二阶段用最终细化图补充连通组件阈值。

    Args:
        calibration (ScoreCalibration): 已完成图像和像素阈值的基础校准。
        samples (Sequence[HRSample]): 用于基础校准的同一组正常样本。
        component_scores (ArrayLike): 最终细化图产生的有限组件分数向量。
        percentile (float): 全局及分类别组件阈值分位数。

    Returns:
        ScoreCalibration: 深拷贝后的校准结构，新增组件分位数和阈值。

    Raises:
        ValueError: 分数数量、有限性、分位数或类别名称不合法。
        KeyError: 样本类别不在基础校准的 ``categories`` 中。
    """
    samples = tuple(samples)
    scores = np.asarray(component_scores, dtype=np.float64)
    if scores.shape != (len(samples),) or not np.isfinite(scores).all():
        raise ValueError("Component score count must match finite normal samples")
    percentile = _validate_percentile(percentile)
    grouped: dict[str, list[float]] = {}
    for sample, score in zip(samples, scores):
        category = sample.clsname
        if not isinstance(category, str) or not category.strip():
            raise ValueError("Every calibration sample must have a non-empty clsname")
        grouped.setdefault(category, []).append(float(score))

    completed = copy.deepcopy(calibration)
    completed["component_percentile"] = percentile
    completed["global_component_threshold"] = _score_threshold(scores, percentile)
    for category, category_scores in grouped.items():
        completed["categories"][category]["component_threshold"] = _score_threshold(
            category_scores,
            percentile,
        )
    return completed


def save_score_calibration(
    calibration: ScoreCalibration,
    checkpoint_root: str | os.PathLike[str],
) -> str:
    """将校准结构写入检查点目录中的稳定 JSON 文件。

    Args:
        calibration (ScoreCalibration): 已验证或刚构建的校准结构。
        checkpoint_root (str | os.PathLike[str]): 已存在的检查点目录。

    Returns:
        str: 写入的 ``score_calibration.json`` 路径。

    Raises:
        OSError: 文件无法创建或写入。
        TypeError: 校准结构包含不能 JSON 序列化的值。
    """
    path = os.path.join(checkpoint_root, SCORE_CALIBRATION_FILE)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(calibration, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return path


def _validate_calibration_number(
    payload: Mapping[str, object],
    key: str,
) -> float:
    """读取校准对象中的有限数值字段并统一转换为 ``float``。

    Args:
        payload (Mapping[str, object]): 校准 JSON 对象或分类别子对象。
        key (str): 需要读取的数值字段。

    Returns:
        float: 有限浮点值。

    Raises:
        ValueError: 字段缺失、是布尔值、不是数值或为 NaN/无穷值。
    """
    value = payload.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(value)
    ):
        raise ValueError(f"Score calibration field {key!r} must be a finite number")
    return float(value)


def _validate_score_calibration(payload: object) -> ScoreCalibration:
    """校验历史与当前校准产物的公共字段，不改写其 JSON 表示。

    Args:
        payload (object): 从 JSON 读取的未验证对象。

    Returns:
        ScoreCalibration: 通过结构、数值和类别校验的原对象视图。

    Raises:
        TypeError: 顶层值不是 JSON 对象。
        ValueError: 必需字段、数值范围、类别或可选组件字段不完整。
    """
    if not isinstance(payload, dict):
        raise TypeError("Score calibration must be a JSON object")

    for key in ("percentile", "pixel_percentile", "pixel_image_percentile"):
        value = _validate_calibration_number(payload, key)
        if not 0 < value < 1:
            raise ValueError(f"Score calibration field {key!r} must be in (0, 1)")
    for key in ("global_threshold", "global_pixel_threshold"):
        _validate_calibration_number(payload, key)

    normal_image_count = payload.get("normal_image_count")
    if (
        isinstance(normal_image_count, bool)
        or not isinstance(normal_image_count, int)
        or normal_image_count <= 0
    ):
        raise ValueError("Score calibration normal_image_count must be positive")

    categories = payload.get("categories")
    if not isinstance(categories, dict) or not categories:
        raise ValueError("Score calibration categories must be a non-empty object")
    for category, values in categories.items():
        if not isinstance(category, str) or not category.strip():
            raise ValueError("Score calibration category names must be non-empty strings")
        if not isinstance(values, dict):
            raise ValueError(f"Score calibration category {category!r} must be an object")
        category_count = values.get("normal_image_count")
        if (
            isinstance(category_count, bool)
            or not isinstance(category_count, int)
            or category_count <= 0
        ):
            raise ValueError(
                f"Score calibration category {category!r} has an invalid normal_image_count"
            )
        _validate_calibration_number(values, "threshold")
        _validate_calibration_number(values, "pixel_threshold")
        if "component_threshold" in values:
            _validate_calibration_number(values, "component_threshold")

    component_keys = {
        "component_percentile",
        "global_component_threshold",
    }
    present_component_keys = component_keys & payload.keys()
    if present_component_keys and present_component_keys != component_keys:
        raise ValueError("Score calibration component fields must be present together")
    if present_component_keys:
        percentile = _validate_calibration_number(payload, "component_percentile")
        if not 0 < percentile < 1:
            raise ValueError("Score calibration component_percentile must be in (0, 1)")
        _validate_calibration_number(payload, "global_component_threshold")
    return cast(ScoreCalibration, payload)


def load_score_calibration(
    checkpoint_root: str | os.PathLike[str],
) -> ScoreCalibration:
    """从检查点目录读取并验证分数校准文件。

    Args:
        checkpoint_root (str | os.PathLike[str]): 包含校准 JSON 的检查点目录。

    Returns:
        ScoreCalibration: 通过兼容性和数值校验的校准结构。

    Raises:
        FileNotFoundError: 校准文件不存在。
        OSError: 文件无法读取。
        json.JSONDecodeError: 文件不是合法 JSON。
        TypeError: JSON 顶层结构错误。
        ValueError: 校准字段不完整或数值无效。
    """
    path = os.path.join(checkpoint_root, SCORE_CALIBRATION_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Score calibration not found: {path}")
    with open(path, "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return _validate_score_calibration(payload)


def thresholds_for_samples(
    calibration: ScoreCalibration,
    samples: Sequence[HRSample],
) -> ScoreVector:
    """为样本生成分类别图像阈值，未知类别回退到全局阈值。

    Args:
        calibration (ScoreCalibration): 已加载的分数校准结构。
        samples (Sequence[HRSample]): 需要分配阈值的样本序列。

    Returns:
        ScoreVector: 与样本同序的一维 ``float32`` 图像阈值。
    """
    global_threshold = float(calibration["global_threshold"])
    categories = calibration["categories"]
    return np.asarray(
        [
            float(categories.get(sample.clsname or "", {}).get("threshold", global_threshold))
            for sample in samples
        ],
        dtype=np.float32,
    )


def pixel_thresholds_for_samples(
    calibration: ScoreCalibration,
    samples: Sequence[HRSample],
) -> ScoreVector:
    """为样本生成分类别像素阈值，未知类别回退到全局阈值。

    Args:
        calibration (ScoreCalibration): 已加载的分数校准结构。
        samples (Sequence[HRSample]): 需要分配阈值的样本序列。

    Returns:
        ScoreVector: 与样本同序的一维 ``float32`` 像素阈值。
    """
    global_threshold = float(calibration["global_pixel_threshold"])
    categories = calibration["categories"]
    return np.asarray(
        [
            float(
                categories.get(sample.clsname or "", {}).get(
                    "pixel_threshold", global_threshold
                )
            )
            for sample in samples
        ],
        dtype=np.float32,
    )


def component_thresholds_for_samples(
    calibration: ScoreCalibration,
    samples: Sequence[HRSample],
) -> ScoreVector:
    """为样本生成分类别组件阈值，未知类别回退到全局阈值。

    Args:
        calibration (ScoreCalibration): 已包含第二阶段组件字段的校准结构。
        samples (Sequence[HRSample]): 需要分配阈值的样本序列。

    Returns:
        ScoreVector: 与样本同序的一维 ``float32`` 连通组件阈值。

    Raises:
        KeyError: 校准结构尚未包含 ``global_component_threshold``。
    """
    global_threshold = float(calibration["global_component_threshold"])
    categories = calibration["categories"]
    return np.asarray(
        [
            float(
                categories.get(sample.clsname or "", {}).get(
                    "component_threshold",
                    global_threshold,
                )
            )
            for sample in samples
        ],
        dtype=np.float32,
    )
