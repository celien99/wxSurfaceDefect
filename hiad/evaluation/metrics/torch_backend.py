from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from numpy.typing import ArrayLike

from .common import validate_binary_vectors, validate_mask_pairs


def _resolve_device(device: str | torch.device | None) -> torch.device:
    """把可选设备参数规范化为 PyTorch 设备，缺省使用 CUDA。

    Args:
        device (str | torch.device | None): PyTorch 设备描述。

    Returns:
        torch.device: 规范化设备对象。
    """
    return torch.device("cuda") if device is None else torch.device(device)


def _binary_curve(
    scores: ArrayLike,
    labels: ArrayLike,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """在目标设备上按唯一分数阈值生成 ROC/PR 累积曲线。

    Args:
        scores (ArrayLike): 非空有限二分类分数。
        labels (ArrayLike): 与分数等长且同时包含 ``0/1`` 的标签。
        device (torch.device): 曲线张量的计算设备。

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 假阳性率、
        真阳性率、精确率和降序唯一分数阈值。

    Raises:
        ValueError: 分数或标签不满足二分类指标约束。
    """
    numpy_scores, numpy_labels = validate_binary_vectors(scores, labels)
    score_tensor = torch.as_tensor(
        numpy_scores,
        dtype=torch.float64,
        device=device,
    )
    label_tensor = torch.as_tensor(
        numpy_labels,
        dtype=torch.int64,
        device=device,
    )
    order = torch.argsort(score_tensor, descending=True, stable=True)
    sorted_scores = score_tensor[order]
    sorted_labels = label_tensor[order]
    distinct = torch.nonzero(
        sorted_scores[1:] != sorted_scores[:-1],
        as_tuple=True,
    )[0]
    threshold_indexes = torch.cat(
        [distinct, torch.tensor([sorted_scores.numel() - 1], device=device)]
    )
    true_positives = torch.cumsum(sorted_labels, dim=0)[threshold_indexes].double()
    predicted_positives = threshold_indexes.double() + 1.0
    false_positives = predicted_positives - true_positives
    positive_count = sorted_labels.sum().double()
    negative_count = sorted_labels.numel() - positive_count
    thresholds = sorted_scores[threshold_indexes]
    tpr = true_positives / positive_count
    fpr = false_positives / negative_count
    precision = true_positives / predicted_positives
    return fpr, tpr, precision, thresholds


def _roc_area(fpr: torch.Tensor, tpr: torch.Tensor) -> torch.Tensor:
    """从原点开始对 ROC 折线执行梯形积分。

    Args:
        fpr (torch.Tensor): 单调非降假阳性率向量。
        tpr (torch.Tensor): 与 ``fpr`` 等长的真阳性率向量。

    Returns:
        torch.Tensor: 保持输入设备和类型的标量曲线面积。
    """
    origin = torch.zeros(1, dtype=fpr.dtype, device=fpr.device)
    return torch.trapz(torch.cat([origin, tpr]), torch.cat([origin, fpr]))


def compute_imagewise_metrics(
    prediction_scores: ArrayLike,
    gt_labels: ArrayLike,
    device: str | torch.device | None = None,
    **_: object,
) -> dict[str, float]:
    """计算图像级 AUROC 及 Youden 指数最优阈值。

    Args:
        prediction_scores (ArrayLike): 一维有限图像异常分数。
        gt_labels (ArrayLike): 与分数等长且同时包含 ``0`` 和 ``1`` 的标签。
        device (str | torch.device | None): 指标计算设备；缺省使用 ``cuda``。

    Returns:
        dict[str, float]: ``image_auroc`` 和使 ``TPR - FPR`` 最大的
        ``image_threshold``。

    Raises:
        ValueError: 分数与标签不满足二分类指标约束。
    """
    device = _resolve_device(device)
    fpr, tpr, _, thresholds = _binary_curve(
        prediction_scores,
        gt_labels,
        device,
    )
    best_index = int(torch.argmax(tpr - fpr).item())
    return {
        "image_auroc": float(_roc_area(fpr, tpr).item()),
        "image_threshold": float(thresholds[best_index].item()),
    }


def compute_pixelwise_metrics(
    prediction_masks: Sequence[ArrayLike],
    gt_masks: Sequence[ArrayLike],
    device: str | torch.device | None = None,
    num_thresholds: int = 65536,
    **_: object,
) -> dict[str, float]:
    """用流式分数直方图计算有界内存的像素指标。

    该实现不拼接全部原图像素，而是把每张图累积到固定数量的全局分数桶中，
    因而 AUROC、AP、F1 和阈值是由直方图近似得到的。

    Args:
        prediction_masks (Sequence[ArrayLike]): 可变高宽的二维有限异常图。
        gt_masks (Sequence[ArrayLike]): 与预测逐项同形状的二维二值真值掩码。
        device (str | torch.device | None): 直方图和曲线计算设备；缺省使用 CUDA。
        num_thresholds (int): 全局分数桶数量，至少为 ``2``。

    Returns:
        dict[str, float]: 像素级 ``pixel_auroc``、``pixel_ap``、最佳
        ``pixel_f1`` 及其 ``seg_threshold``。

    Raises:
        TypeError: ``num_thresholds`` 不是整数或是布尔值。
        ValueError: 掩码不合法、桶数不足，或像素标签只包含一个类别。
    """
    pairs = validate_mask_pairs(prediction_masks, gt_masks)
    if isinstance(num_thresholds, bool) or not isinstance(num_thresholds, int):
        raise TypeError("num_thresholds must be an integer")
    if num_thresholds < 2:
        raise ValueError("num_thresholds must be at least 2")

    device = _resolve_device(device)
    minimum = min(float(prediction.min()) for prediction, _ in pairs)
    maximum = max(float(prediction.max()) for prediction, _ in pairs)
    positive_counts = torch.zeros(num_thresholds, dtype=torch.int64, device=device)
    negative_counts = torch.zeros_like(positive_counts)

    for prediction, target in pairs:
        scores = torch.as_tensor(
            np.ascontiguousarray(prediction),
            dtype=torch.float32,
            device=device,
        )
        labels = torch.as_tensor(
            np.ascontiguousarray(target),
            dtype=torch.bool,
            device=device,
        )
        if maximum == minimum:
            bin_indexes = torch.full_like(
                scores,
                num_thresholds - 1,
                dtype=torch.int64,
            )
        else:
            bin_indexes = ((scores - minimum) * (
                num_thresholds / (maximum - minimum)
            )).to(torch.int64)
            bin_indexes.clamp_(0, num_thresholds - 1)
        positive_counts += torch.bincount(
            bin_indexes[labels],
            minlength=num_thresholds,
        )
        negative_counts += torch.bincount(
            bin_indexes[~labels],
            minlength=num_thresholds,
        )

    positive_total = positive_counts.sum()
    negative_total = negative_counts.sum()
    if positive_total.item() == 0 or negative_total.item() == 0:
        raise ValueError("Binary metrics require both normal and anomalous labels")

    true_positives = torch.cumsum(positive_counts.flip(0), dim=0).double()
    false_positives = torch.cumsum(negative_counts.flip(0), dim=0).double()
    recall = true_positives / positive_total.double()
    fpr = false_positives / negative_total.double()
    predicted_positives = true_positives + false_positives
    precision = torch.where(
        predicted_positives > 0,
        true_positives / predicted_positives,
        torch.zeros_like(predicted_positives),
    )
    thresholds = torch.linspace(
        minimum,
        maximum,
        num_thresholds + 1,
        dtype=torch.float64,
        device=device,
    )[:-1].flip(0)
    denominator = precision + recall
    f1_scores = torch.where(
        denominator > 0,
        2 * precision * recall / denominator,
        torch.zeros_like(denominator),
    )
    best_index = int(torch.argmax(f1_scores).item())
    previous_recall = torch.cat(
        [torch.zeros(1, dtype=recall.dtype, device=device), recall[:-1]]
    )
    average_precision = torch.sum((recall - previous_recall) * precision)
    return {
        "pixel_auroc": float(_roc_area(fpr, recall).item()),
        "pixel_ap": float(average_precision.item()),
        "pixel_f1": float(f1_scores[best_index].item()),
        "seg_threshold": float(thresholds[best_index].item()),
    }
