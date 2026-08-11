from types import SimpleNamespace

import numpy as np
import pytest

from hiad.evaluation.execution import evaluate_category_metrics


def _batch(class_names):
    sample_count = len(class_names)
    return SimpleNamespace(
        class_names=tuple(class_names),
        prediction_masks=tuple(
            np.full((2, 2), index, dtype=np.float32)
            for index in range(sample_count)
        ),
        gt_masks=tuple(
            np.zeros((2, 2), dtype=np.uint8) for _ in range(sample_count)
        ),
        gt_labels=np.arange(sample_count) % 2,
        prediction_scores=np.arange(sample_count, dtype=np.float32),
    )


def test_evaluate_category_metrics_groups_categories_on_requested_devices():
    batch = _batch(["b", "a", "c", "a"])

    def evaluator(prediction_masks, prediction_scores, device, **kwargs):
        return {
            "sample_count": len(prediction_masks),
            "score_sum": float(prediction_scores.sum()),
            "device_index": device.index,
        }

    scores = evaluate_category_metrics(batch, [2, 3], [evaluator])

    assert scores == [
        {
            "clsname": "a",
            "sample_count": 2,
            "score_sum": 4.0,
            "device_index": 2,
        },
        {
            "clsname": "b",
            "sample_count": 1,
            "score_sum": 0.0,
            "device_index": 3,
        },
        {
            "clsname": "c",
            "sample_count": 1,
            "score_sum": 2.0,
            "device_index": 2,
        },
    ]


def test_evaluate_category_metrics_propagates_evaluator_errors():
    def failing_evaluator(**kwargs):
        raise RuntimeError("metric failed")

    with pytest.raises(RuntimeError, match="metric failed"):
        evaluate_category_metrics(_batch(["a"]), [0], [failing_evaluator])
