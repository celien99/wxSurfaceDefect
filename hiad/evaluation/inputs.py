from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from hiad.data import HRSample


@dataclass(frozen=True)
class EvaluationBatch:
    samples: tuple[HRSample, ...]
    image_paths: tuple[str, ...]
    class_names: tuple[str, ...]
    prediction_scores: np.ndarray
    prediction_masks: tuple[np.ndarray, ...]
    binary_prediction_masks: tuple[np.ndarray, ...] | None
    pixel_thresholds: np.ndarray | None
    gt_labels: np.ndarray
    gt_masks: tuple[np.ndarray, ...]
    display_images: dict | None


def build_evaluation_batch(test_samples, inference_result) -> EvaluationBatch:
    if not isinstance(test_samples, list) or not test_samples:
        raise ValueError("test_samples must be a non-empty list")
    if any(not isinstance(sample, HRSample) for sample in test_samples):
        raise TypeError("Every test sample must be an HRSample")
    if not isinstance(inference_result, dict):
        raise TypeError("inference_result must be a mapping")

    image_paths = tuple(sample.image.image_path for sample in test_samples)
    if inference_result.get("image_paths") != list(image_paths):
        raise ValueError("Inference result image order does not match test samples")

    prediction_scores = np.asarray(inference_result.get("image_scores"))
    if prediction_scores.shape != (len(test_samples),):
        raise ValueError("Inference result image_scores have an invalid shape")
    prediction_mask_values = inference_result.get("anomaly_maps")
    if not isinstance(prediction_mask_values, list) or len(
        prediction_mask_values
    ) != len(test_samples):
        raise ValueError("Inference result anomaly_maps have an invalid length")

    prediction_masks = []
    gt_masks = []
    gt_labels = []
    for sample_index, (sample, prediction_mask_value) in enumerate(
        tqdm(
            zip(test_samples, prediction_mask_values),
            total=len(test_samples),
            desc="Loading GT Masks",
        )
    ):
        prediction_mask = np.asarray(prediction_mask_value)
        if prediction_mask.ndim != 2:
            raise ValueError(
                f"Prediction mask at index {sample_index} must be two-dimensional"
            )
        if sample.mask is None:
            gt_mask = np.zeros_like(prediction_mask, dtype=float)
        else:
            opened_mask = sample.mask.image is None
            try:
                sample.mask.open()
                gt_mask = np.array(sample.mask.image, copy=True)
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
            sample.label if sample.label is not None else np.max(gt_mask).item()
        )

    binary_values = inference_result.get("binary_anomaly_maps")
    binary_prediction_masks = None
    if binary_values is not None:
        if not isinstance(binary_values, list) or len(binary_values) != len(test_samples):
            raise ValueError("Inference result binary_anomaly_maps have an invalid length")
        binary_prediction_masks = tuple(np.asarray(mask, dtype=bool) for mask in binary_values)
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
        gt_labels=np.asarray(gt_labels),
        gt_masks=tuple(gt_masks),
        display_images=inference_result.get("display_images"),
    )
