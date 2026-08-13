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
