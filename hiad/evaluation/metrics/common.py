from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray


Float64Vector: TypeAlias = NDArray[np.float64]
Int64Vector: TypeAlias = NDArray[np.int64]
MaskPair: TypeAlias = tuple[NDArray[np.generic], NDArray[np.generic]]


def validate_binary_vectors(
    prediction_scores: ArrayLike,
    gt_labels: ArrayLike,
) -> tuple[Float64Vector, Int64Vector]:
    """校验并规范化图像级二分类分数与标签。

    Args:
        prediction_scores (ArrayLike): 可展平的一维有限异常分数。
        gt_labels (ArrayLike): 与分数等长的 ``0/1`` 或布尔标签。

    Returns:
        tuple[Float64Vector, Int64Vector]: ``float64`` 分数和 ``int64`` 标签向量。

    Raises:
        ValueError: 输入为空、数量不一致、分数非有限、标签非二值，或批次没有
        同时包含正常与异常标签。
    """
    scores = np.asarray(prediction_scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(gt_labels).reshape(-1)
    if scores.size == 0 or scores.size != labels.size:
        raise ValueError("Prediction scores and labels must be non-empty and aligned")
    if not np.isfinite(scores).all():
        raise ValueError("Prediction scores must be finite")
    if not set(np.unique(labels).tolist()).issubset({0, 1, False, True}):
        raise ValueError("Ground-truth labels must be binary")
    labels = labels.astype(np.int64, copy=False)
    if np.unique(labels).size != 2:
        raise ValueError("Binary metrics require both normal and anomalous labels")
    return scores, labels


def validate_mask_pairs(
    prediction_masks: Sequence[ArrayLike],
    gt_masks: Sequence[ArrayLike],
) -> list[MaskPair]:
    """校验可变原生分辨率的预测/真值掩码对。

    Args:
        prediction_masks (Sequence[ArrayLike]): 每张图的二维有限异常分数图。
        gt_masks (Sequence[ArrayLike]): 逐项同形状的二维 ``0/1`` 真值掩码。

    Returns:
        list[MaskPair]: 保留输入顺序和各自原生高宽的 NumPy 数组对。

    Raises:
        ValueError: 序列为空或不对齐，掩码不是二维、形状不匹配，预测包含
        非有限值，或真值不是二值。
    """
    predictions = [np.asarray(mask) for mask in prediction_masks]
    targets = [np.asarray(mask) for mask in gt_masks]
    if not predictions or len(predictions) != len(targets):
        raise ValueError("Prediction and ground-truth masks must be non-empty and aligned")

    pairs: list[MaskPair] = []
    for prediction, target in zip(predictions, targets):
        if prediction.ndim != 2 or target.ndim != 2:
            raise ValueError("Prediction and ground-truth masks must be two-dimensional")
        if prediction.shape != target.shape:
            raise ValueError("Each prediction mask must match its ground-truth mask shape")
        if not np.isfinite(prediction).all():
            raise ValueError("Prediction masks must be finite")
        if not np.logical_or(target == 0, target == 1).all():
            raise ValueError("Ground-truth masks must be binary")
        pairs.append((prediction, target))
    return pairs


def flatten_mask_pairs(
    prediction_masks: Sequence[ArrayLike],
    gt_masks: Sequence[ArrayLike],
) -> tuple[Float64Vector, Int64Vector]:
    """校验所有掩码对并展平为像素级二分类向量。

    Args:
        prediction_masks (Sequence[ArrayLike]): 可变高宽的二维异常图。
        gt_masks (Sequence[ArrayLike]): 与异常图逐项同形状的二值掩码。

    Returns:
        tuple[Float64Vector, Int64Vector]: 拼接后的像素分数和二值标签向量。
    """
    pairs = validate_mask_pairs(prediction_masks, gt_masks)
    scores = np.concatenate([prediction.reshape(-1) for prediction, _ in pairs])
    labels = np.concatenate([target.reshape(-1) for _, target in pairs])
    return validate_binary_vectors(scores, labels)
