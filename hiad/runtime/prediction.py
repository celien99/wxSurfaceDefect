import json
import os

import numpy as np
from PIL import Image


def has_complete_image_annotations(records) -> bool:
    labels_by_category = {}
    for record in records:
        label = record.get("label")
        if isinstance(label, bool) or label not in {0, 1}:
            return False
        labels_by_category.setdefault(record.get("clsname", "default"), set()).add(label)
    return bool(labels_by_category) and all(
        labels == {0, 1} for labels in labels_by_category.values()
    )


def has_complete_pixel_annotations(records) -> bool:
    if not has_complete_image_annotations(records):
        return False
    anomalous_records = [record for record in records if record.get("label") == 1]
    return all(record.get("mask") for record in anomalous_records)


def threshold_anomaly_maps(anomaly_maps, pixel_thresholds) -> list[np.ndarray]:
    return [
        np.asarray(anomaly_map >= pixel_threshold, dtype=np.uint8)
        for anomaly_map, pixel_threshold in zip(anomaly_maps, pixel_thresholds)
    ]


def save_predictions(path, samples, inference_result) -> None:
    thresholds = inference_result.get("image_thresholds")
    decisions = inference_result.get("is_defect")
    pixel_thresholds = inference_result.get("pixel_thresholds")
    binary_maps = inference_result.get("binary_anomaly_maps")
    masks_root = os.path.join(os.path.dirname(path), "masks")
    if binary_maps is not None:
        os.makedirs(masks_root, exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="\n") as stream:
        for index, sample in enumerate(samples):
            record = {
                "filename": sample.image.image_path,
                "clsname": sample.clsname,
                "label": sample.label,
                "label_name": sample.label_name,
                "score": float(inference_result["image_scores"][index]),
            }
            if thresholds is not None:
                record["threshold"] = float(thresholds[index])
            if decisions is not None:
                record["is_defect"] = bool(decisions[index])
            if pixel_thresholds is not None:
                record["pixel_threshold"] = float(pixel_thresholds[index])
            if binary_maps is not None:
                stem = os.path.splitext(os.path.basename(sample.image.image_path))[0]
                mask_name = f"{index:06d}_{stem}.png"
                Image.fromarray(
                    np.asarray(binary_maps[index], dtype=np.uint8) * 255
                ).save(os.path.join(masks_root, mask_name))
                record["prediction_mask"] = f"masks/{mask_name}"
                record["anomaly_pixel_count"] = int(
                    np.count_nonzero(binary_maps[index])
                )
            stream.write(json.dumps(record, ensure_ascii=True) + "\n")
