import json
from types import SimpleNamespace

import numpy as np
from PIL import Image

from hiad.runtime.prediction import (
    has_complete_image_annotations,
    has_complete_pixel_annotations,
    save_predictions,
    threshold_anomaly_maps,
)


def test_image_metrics_require_both_labels_in_every_category():
    assert has_complete_image_annotations(
        [
            {"clsname": "a", "label": 0},
            {"clsname": "a", "label": 1},
            {"clsname": "b", "label": 0},
            {"clsname": "b", "label": 1},
        ]
    )
    assert not has_complete_image_annotations(
        [{"clsname": "a", "label": 0}, {"clsname": "a", "label": 0}]
    )
    assert not has_complete_image_annotations([{"clsname": "a"}])


def test_pixel_metrics_require_masks_for_every_anomalous_record():
    assert has_complete_pixel_annotations(
        [
            {"clsname": "a", "label": 0, "mask": None},
            {"clsname": "a", "label": 1, "mask": "mask.png"},
        ]
    )
    assert not has_complete_pixel_annotations(
        [
            {"clsname": "a", "label": 0, "mask": None},
            {"clsname": "a", "label": 1, "mask": None},
        ]
    )


def test_save_predictions_includes_calibrated_decisions(tmp_path):
    samples = [
        SimpleNamespace(
            image=SimpleNamespace(image_path="a.bmp"),
            clsname="part",
            label=1,
            label_name="scratch",
        )
    ]
    output = tmp_path / "predictions.jsonl"

    save_predictions(
        output,
        samples,
        {
            "image_scores": [0.7],
            "image_thresholds": [0.4],
            "is_defect": [True],
            "pixel_thresholds": [0.5],
            "binary_anomaly_maps": [np.asarray([[0, 1], [1, 0]], dtype=np.uint8)],
        },
    )

    assert json.loads(output.read_text()) == {
        "filename": "a.bmp",
        "clsname": "part",
        "label": 1,
        "label_name": "scratch",
        "score": 0.7,
        "threshold": 0.4,
        "is_defect": True,
        "pixel_threshold": 0.5,
        "prediction_mask": "masks/000000_a.png",
        "anomaly_pixel_count": 2,
    }
    with Image.open(tmp_path / "masks" / "000000_a.png") as mask:
        assert np.asarray(mask).tolist() == [[0, 255], [255, 0]]


def test_pixel_masks_are_not_gated_by_image_decisions():
    masks = threshold_anomaly_maps(
        [np.asarray([[0.1, 0.8]], dtype=np.float32)],
        [0.5],
    )

    assert masks[0].tolist() == [[0, 1]]


def test_save_predictions_serializes_three_state_decision_and_components(tmp_path):
    samples = [
        SimpleNamespace(
            image=SimpleNamespace(image_path="a.bmp"),
            clsname="part",
            label=0,
            label_name="good",
        )
    ]
    output = tmp_path / "predictions.jsonl"
    component_summary = {
        "component_count": 1,
        "anomalous_pixel_count": 4,
        "largest_component_area": 4,
        "strongest_component": {
            "area": 4,
            "area_fraction": 0.25,
            "mean_score": 0.4,
            "max_score": 0.72,
            "score": 0.6,
            "bbox_xywh": [1, 2, 2, 2],
        },
    }

    save_predictions(
        output,
        samples,
        {
            "image_scores": [0.6],
            "is_defect": [False],
            "decisions": ["RECHECK"],
            "decision_thresholds": [0.55],
            "decision_reasons": ["score_within_recheck_margin"],
            "component_scores": [0.6],
            "component_summaries": [component_summary],
        },
    )

    record = json.loads(output.read_text())
    assert record["is_defect"] is False
    assert record["decision"] == "RECHECK"
    assert record["decision_threshold"] == 0.55
    assert record["decision_reason"] == "score_within_recheck_margin"
    assert record["component_score"] == 0.6
    assert record["component_summary"] == component_summary
