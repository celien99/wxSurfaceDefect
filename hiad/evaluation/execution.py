import os

import numpy as np
import torch
import torch.multiprocessing as mp

from hiad.runtime.partition import round_robin_partition


def _compute_metrics_in_device(
    gpu_id,
    class_names,
    evaluators,
    prediction_masks,
    gt_masks,
    gt_labels,
    prediction_scores,
    sample_class_names,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda")
    scores = []
    for class_name in class_names:
        selected = [
            index
            for index, sample_class_name in enumerate(sample_class_names)
            if sample_class_name == class_name
        ]
        evaluator_inputs = {
            "prediction_masks": [prediction_masks[index] for index in selected],
            "gt_masks": [gt_masks[index] for index in selected],
            "prediction_scores": np.asarray(
                [prediction_scores[index] for index in selected]
            ),
            "gt_labels": np.asarray([gt_labels[index] for index in selected]),
            "device": device,
        }
        score = {"clsname": class_name}
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


def evaluate_category_metrics(batch, gpu_ids, evaluators):
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
    pending_results = []
    process_pool = mp.get_context("spawn").Pool(processes=len(class_groups))
    try:
        for gpu_id, class_names in zip(gpu_ids, class_groups):
            pending_results.append(
                process_pool.apply_async(
                    _compute_metrics_in_device,
                    (
                        gpu_id,
                        class_names,
                        evaluators,
                        batch.prediction_masks,
                        batch.gt_masks,
                        batch.gt_labels,
                        batch.prediction_scores,
                        batch.class_names,
                    ),
                )
            )
        process_pool.close()
        process_pool.join()
    except Exception:
        process_pool.terminate()
        process_pool.join()
        raise

    scores = []
    for pending_result in pending_results:
        scores.extend(pending_result.get())
    scores.sort(key=lambda score: score["clsname"])
    if not scores:
        raise RuntimeError("Evaluation produced no category scores")

    expected_keys = set(scores[0])
    for score in scores[1:]:
        if set(score) != expected_keys:
            raise ValueError("Evaluators produced inconsistent category metric schemas")
    return scores
