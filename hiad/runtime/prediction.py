from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

from hiad.data import HRSample

from .contracts import BinaryMask, FloatMap, InferenceResult


def has_complete_image_annotations(records: Sequence[Mapping[str, object]]) -> bool:
    """确认每个类别都同时具备正常和异常图像标签。

    Args:
        records (Sequence[Mapping[str, object]]): 数据清单记录；``label`` 必须为
            整数 ``0`` 或 ``1``，``clsname`` 缺失时归入 ``default`` 类别。

    Returns:
        bool: 清单非空且每个类别的标签集合均为 ``{0, 1}`` 时为 ``True``。
    """
    labels_by_category: dict[object, set[object]] = {}
    for record in records:
        label = record.get("label")
        if isinstance(label, bool) or label not in {0, 1}:
            return False
        labels_by_category.setdefault(record.get("clsname", "default"), set()).add(label)
    return bool(labels_by_category) and all(
        labels == {0, 1} for labels in labels_by_category.values()
    )


def has_complete_pixel_annotations(records: Sequence[Mapping[str, object]]) -> bool:
    """确认图像级标签完整，且所有异常记录都具有像素级掩码。

    Args:
        records (Sequence[Mapping[str, object]]): 包含 ``label``、``clsname`` 和
            可选 ``mask`` 路径的数据清单记录。

    Returns:
        bool: 图像级标注完整且每条异常记录的 ``mask`` 值非空时为 ``True``。
    """
    if not has_complete_image_annotations(records):
        return False
    anomalous_records = [record for record in records if record.get("label") == 1]
    return all(record.get("mask") for record in anomalous_records)


def threshold_anomaly_maps(
    anomaly_maps: Sequence[FloatMap],
    pixel_thresholds: Iterable[float],
) -> list[BinaryMask]:
    """按逐样本像素阈值把异常图转换为 ``0/1`` 二值掩码。

    Args:
        anomaly_maps (Sequence[FloatMap]): 原图分辨率的二维异常图序列。
        pixel_thresholds (Iterable[float]): 与异常图同序的像素阈值。

    Returns:
        list[BinaryMask]: 二维 ``uint8`` 掩码；长度遵循 ``zip``，调用方必须确保
        异常图和阈值数量一致。
    """
    return [
        np.asarray(anomaly_map >= pixel_threshold, dtype=np.uint8)
        for anomaly_map, pixel_threshold in zip(anomaly_maps, pixel_thresholds)
    ]


def save_predictions(
    path: str | os.PathLike[str],
    samples: Sequence[HRSample],
    inference_result: InferenceResult,
) -> None:
    """保存可追溯判定记录，并在完成像素校准后落盘二值掩码。

    Args:
        path (str | os.PathLike[str]): 输出 JSONL 文件路径。
        samples (Sequence[HRSample]): 与推理结果顺序一致的源图样本。
        inference_result (InferenceResult): 至少包含图像分数和异常图的推理结果；
            所有逐样本可选字段也必须与 ``samples`` 等长。

    Raises:
        OSError: 输出 JSONL、掩码目录或 PNG 文件无法创建或写入。
        IndexError: 任一逐样本结果字段与 ``samples`` 长度不一致。
    """
    output_path = os.fspath(path)
    thresholds = inference_result.get("image_thresholds")
    decisions = inference_result.get("is_defect")
    decision_states = inference_result.get("decisions")
    decision_reasons = inference_result.get("decision_reasons")
    decision_thresholds = inference_result.get("decision_thresholds")
    component_scores = inference_result.get("component_scores")
    component_summaries = inference_result.get("component_summaries")
    refinement_statistics = inference_result.get("refinement_statistics")
    quality_results = inference_result.get("quality_results")
    pixel_thresholds = inference_result.get("pixel_thresholds")
    binary_maps = inference_result.get("binary_anomaly_maps")
    masks_root = os.path.join(os.path.dirname(output_path), "masks")
    if binary_maps is not None:
        os.makedirs(masks_root, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="\n") as stream:
        for index, sample in enumerate(samples):
            record: dict[str, object] = {
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
            if decision_states is not None:
                record["decision"] = str(decision_states[index])
            if decision_reasons is not None:
                record["decision_reason"] = str(decision_reasons[index])
            if decision_thresholds is not None:
                record["decision_threshold"] = float(decision_thresholds[index])
            if component_scores is not None:
                record["component_score"] = float(component_scores[index])
            if component_summaries is not None:
                record["component_summary"] = component_summaries[index]
            if refinement_statistics is not None:
                record["refinement"] = refinement_statistics[index]
            if quality_results is not None:
                record["quality"] = quality_results[index]
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
