from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TypeAlias, cast

import numpy as np
import torch
from numpy.typing import NDArray

from hiad.evaluation.inputs import EvaluationBatch
from hiad.runtime.partition import round_robin_partition


MetricValue: TypeAlias = float | int
MetricResult: TypeAlias = Mapping[str, MetricValue]
MetricEvaluator: TypeAlias = Callable[..., MetricResult]
CategoryScore: TypeAlias = dict[str, str | MetricValue]


def _compute_metrics_in_device(
    gpu_id: int,
    class_names: Sequence[str],
    evaluators: Sequence[MetricEvaluator],
    prediction_masks: Sequence[NDArray[np.generic]],
    gt_masks: Sequence[NDArray[np.generic]],
    gt_labels: NDArray[np.int64],
    prediction_scores: NDArray[np.float32],
    sample_class_names: Sequence[str],
) -> list[CategoryScore]:
    """在一个 CUDA 设备上计算一组类别的全部指标。

    Args:
        gpu_id (int): 当前工作线程绑定的 CUDA 设备编号。
        class_names (Sequence[str]): 分配给该设备的类别名称。
        evaluators (Sequence[MetricEvaluator]): 标准关键字参数指标函数序列。
        prediction_masks (Sequence[NDArray[np.generic]]): 全批次二维异常图。
        gt_masks (Sequence[NDArray[np.generic]]): 全批次二维二值真值掩码。
        gt_labels (NDArray[np.int64]): 全批次图像级二值标签。
        prediction_scores (NDArray[np.float32]): 全批次图像级异常分数。
        sample_class_names (Sequence[str]): 与全批次逐项对齐的类别名称。

    Returns:
        list[CategoryScore]: 保持 ``class_names`` 顺序的分类别指标字典。

    Raises:
        ValueError: 两个指标函数返回了重复的指标名称。
    """
    device = torch.device(f"cuda:{gpu_id}")
    scores: list[CategoryScore] = []
    for class_name in class_names:
        selected = [
            index
            for index, sample_class_name in enumerate(sample_class_names)
            if sample_class_name == class_name
        ]
        evaluator_inputs: dict[str, object] = {
            "prediction_masks": [prediction_masks[index] for index in selected],
            "gt_masks": [gt_masks[index] for index in selected],
            "prediction_scores": np.asarray(
                [prediction_scores[index] for index in selected]
            ),
            "gt_labels": np.asarray([gt_labels[index] for index in selected]),
            "device": device,
        }
        score: CategoryScore = {"clsname": class_name}
        for evaluator in evaluators:
            current_scores = evaluator(**evaluator_inputs)
            duplicate_keys = score.keys() & current_scores.keys()
            if duplicate_keys:
                raise ValueError(
                    "Evaluators produced duplicate metrics: "
                    f"{sorted(duplicate_keys)}"
                )
            score.update(current_scores)
        scores.append(score)
    return scores


def evaluate_category_metrics(
    batch: EvaluationBatch,
    gpu_ids: list[int],
    evaluators: list[MetricEvaluator],
) -> list[CategoryScore]:
    """按类别轮询分配到多个 GPU，并行计算统一指标集合。

    Args:
        batch (EvaluationBatch): 已完成顺序、形状和标签对齐的评估批次。
        gpu_ids (list[int]): 可用 CUDA 设备编号；调用前应完成设备校验。
        evaluators (list[MetricEvaluator]): 非空的可调用指标函数列表。

    Returns:
        list[CategoryScore]: 按 ``clsname`` 排序且字段模式一致的类别指标。

    Raises:
        TypeError: 任一指标对象不可调用。
        ValueError: 指标列表为空、指标键重复或分类别字段模式不一致。
        RuntimeError: 没有产生任何分类别结果。
    """
    if not isinstance(evaluators, list) or not evaluators:
        raise ValueError("evaluators must be a non-empty list")
    if any(not callable(evaluator) for evaluator in evaluators):
        raise TypeError("Every evaluator must be callable")

    all_class_names = sorted(set(batch.class_names))
    class_groups = [
        group
        for group in round_robin_partition(all_class_names, len(gpu_ids))
        if group
    ]
    # 原生分辨率掩码可能超过进程管道限制；线程共享数组，各自只绑定目标 CUDA 设备。
    with ThreadPoolExecutor(max_workers=len(class_groups)) as executor:
        pending_results = []
        for gpu_id, class_names in zip(gpu_ids, class_groups):
            pending_results.append(
                executor.submit(
                    _compute_metrics_in_device,
                    gpu_id,
                    class_names,
                    evaluators,
                    batch.prediction_masks,
                    batch.gt_masks,
                    batch.gt_labels,
                    batch.prediction_scores,
                    batch.class_names,
                )
            )
        scores: list[CategoryScore] = []
        for pending_result in pending_results:
            scores.extend(pending_result.result())
    scores.sort(key=lambda score: cast(str, score["clsname"]))
    if not scores:
        raise RuntimeError("Evaluation produced no category scores")

    expected_keys = set(scores[0])
    for score in scores[1:]:
        if set(score) != expected_keys:
            raise ValueError("Evaluators produced inconsistent category metric schemas")
    return scores
