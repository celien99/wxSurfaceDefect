from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from hiad.data import HRSample
from hiad.runtime.contracts import (
    BinaryMask,
    FloatMap,
    ImageArray,
    InferenceResult,
    ScoreVector,
)


@dataclass(frozen=True)
class EvaluationBatch:
    """按推理顺序对齐的原生分辨率预测、标签及可视化输入。

    Attributes:
        samples (tuple[HRSample, ...]): 保持调用顺序的评估样本。
        image_paths (tuple[str, ...]): 与样本一一对应的原图路径。
        class_names (tuple[str, ...]): 与样本一一对应的业务类别。
        prediction_scores (ScoreVector): 一维 ``float32`` 图像异常分数。
        prediction_masks (tuple[FloatMap, ...]): 每张原图的二维 ``float32``
            异常图，允许不同图像具有不同高宽。
        binary_prediction_masks (tuple[BinaryMask, ...] | None): 与异常图同形状的
            可选二维 ``uint8`` 二值预测。
        pixel_thresholds (ScoreVector | None): 与样本同序的像素阈值。
        gt_labels (NDArray[np.int64]): 一维二值图像标签。
        gt_masks (tuple[BinaryMask, ...]): 与对应异常图同形状的二维真值掩码。
        display_images (dict[str, ImageArray] | None): 原图路径到 HWC RGB
            ``uint8`` 显示图的映射。
    """

    samples: tuple[HRSample, ...]
    image_paths: tuple[str, ...]
    class_names: tuple[str, ...]
    prediction_scores: ScoreVector
    prediction_masks: tuple[FloatMap, ...]
    binary_prediction_masks: tuple[BinaryMask, ...] | None
    pixel_thresholds: ScoreVector | None
    gt_labels: NDArray[np.int64]
    gt_masks: tuple[BinaryMask, ...]
    display_images: dict[str, ImageArray] | None


def build_evaluation_batch(
    test_samples: list[HRSample],
    inference_result: InferenceResult,
) -> EvaluationBatch:
    """校验推理顺序，并把不同来源的标签和掩码对齐到统一批次。

    Args:
        test_samples (list[HRSample]): 非空评估样本，顺序必须与推理输入一致。
        inference_result (InferenceResult): 包含路径、分数和原图分辨率异常图的
            推理结果，以及可选二值图、像素阈值和显示图。

    Returns:
        EvaluationBatch: 不改变样本顺序的只读评估数据结构。缺失真值掩码时
        创建同形状全零掩码；缺失图像标签时由真值掩码推导。

    Raises:
        TypeError: 样本或推理结果顶层类型错误。
        ValueError: 路径顺序、字段数量、向量形状或掩码高宽不一致。
        OSError: 真值掩码文件无法打开。
        RuntimeError: 掩码解码后未获得像素数组。
    """
    if not isinstance(test_samples, list) or not test_samples:
        raise ValueError("test_samples must be a non-empty list")
    if any(not isinstance(sample, HRSample) for sample in test_samples):
        raise TypeError("Every test sample must be an HRSample")
    if not isinstance(inference_result, dict):
        raise TypeError("inference_result must be a mapping")

    image_paths = tuple(sample.image.image_path for sample in test_samples)
    if inference_result.get("image_paths") != list(image_paths):
        raise ValueError("Inference result image order does not match test samples")

    prediction_scores = np.asarray(
        inference_result.get("image_scores"),
        dtype=np.float32,
    )
    if prediction_scores.shape != (len(test_samples),):
        raise ValueError("Inference result image_scores have an invalid shape")
    prediction_mask_values = inference_result.get("anomaly_maps")
    if not isinstance(prediction_mask_values, list) or len(
        prediction_mask_values
    ) != len(test_samples):
        raise ValueError("Inference result anomaly_maps have an invalid length")

    prediction_masks: list[FloatMap] = []
    gt_masks: list[BinaryMask] = []
    gt_labels: list[int] = []
    for sample_index, (sample, prediction_mask_value) in enumerate(
        tqdm(
            zip(test_samples, prediction_mask_values),
            total=len(test_samples),
            desc="Loading GT Masks",
        )
    ):
        prediction_mask = np.asarray(prediction_mask_value, dtype=np.float32)
        if prediction_mask.ndim != 2:
            raise ValueError(
                f"Prediction mask at index {sample_index} must be two-dimensional"
            )
        if sample.mask is None:
            gt_mask = np.zeros_like(prediction_mask, dtype=np.uint8)
        else:
            opened_mask = sample.mask.image is None
            try:
                sample.mask.open()
                mask_image = sample.mask.image
                if mask_image is None:
                    raise RuntimeError("Ground-truth mask was not loaded")
                gt_mask = np.array(mask_image, dtype=np.uint8, copy=True)
            finally:
                if opened_mask:
                    sample.mask.close()
            gt_mask[gt_mask != 0] = 1
        if gt_mask.shape != prediction_mask.shape:
            raise ValueError(
                "Ground-truth and prediction mask shapes differ at index "
                f"{sample_index}: {gt_mask.shape} != {prediction_mask.shape}"
            )

        prediction_masks.append(prediction_mask)
        gt_masks.append(gt_mask)
        gt_labels.append(
            sample.label if sample.label is not None else int(np.max(gt_mask).item())
        )

    binary_values = inference_result.get("binary_anomaly_maps")
    binary_prediction_masks: tuple[BinaryMask, ...] | None = None
    if binary_values is not None:
        if not isinstance(binary_values, list) or len(binary_values) != len(test_samples):
            raise ValueError("Inference result binary_anomaly_maps have an invalid length")
        binary_prediction_masks = tuple(
            np.asarray(mask, dtype=np.uint8) for mask in binary_values
        )
        if any(
            mask.shape != prediction.shape
            for mask, prediction in zip(binary_prediction_masks, prediction_masks)
        ):
            raise ValueError("Binary anomaly maps must match anomaly map shapes")

    pixel_threshold_values = inference_result.get("pixel_thresholds")
    pixel_thresholds = None
    if pixel_threshold_values is not None:
        pixel_thresholds = np.asarray(pixel_threshold_values, dtype=np.float32)
        if pixel_thresholds.shape != (len(test_samples),):
            raise ValueError("Inference result pixel_thresholds have an invalid shape")

    return EvaluationBatch(
        samples=tuple(test_samples),
        image_paths=image_paths,
        class_names=tuple(
            sample.clsname if sample.clsname is not None else "unknown"
            for sample in test_samples
        ),
        prediction_scores=prediction_scores,
        prediction_masks=tuple(prediction_masks),
        binary_prediction_masks=binary_prediction_masks,
        pixel_thresholds=pixel_thresholds,
        gt_labels=np.asarray(gt_labels, dtype=np.int64),
        gt_masks=tuple(gt_masks),
        display_images=inference_result.get("display_images"),
    )
