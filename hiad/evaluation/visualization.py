from __future__ import annotations

import logging
import os

import cv2
import numpy as np
from numpy.typing import ArrayLike
from tqdm import tqdm

from hiad.evaluation.inputs import EvaluationBatch
from hiad.runtime.contracts import FloatMap


def scale_anomaly_map_for_display(
    prediction_mask: ArrayLike,
    threshold: float | None = None,
) -> FloatMap:
    """按像素阈值稳定缩放热图；无有效阈值时退化为单图归一化。

    Args:
        prediction_mask (ArrayLike): 二维异常分数图。
        threshold (float | None): 可选正数像素阈值；阈值映射到显示强度 ``0.5``。

    Returns:
        FloatMap: 与输入同形状、范围 ``[0, 1]`` 的 ``float32`` 显示图；常量图
        在无有效阈值时返回全零数组。
    """
    prediction_mask = np.asarray(prediction_mask, dtype=np.float32)
    if threshold is not None:
        threshold = float(threshold)
        if np.isfinite(threshold) and threshold > 0:
            return np.clip(prediction_mask / (2 * threshold), 0, 1)
    minimum = float(prediction_mask.min())
    maximum = float(prediction_mask.max())
    if maximum == minimum:
        return np.zeros_like(prediction_mask, dtype=np.float32)
    return (prediction_mask - minimum) / (maximum - minimum)


def save_evaluation_visualizations(
    batch: EvaluationBatch,
    output_root: str | os.PathLike[str],
    output_size: int | list[int] | tuple[int, int],
    logger: logging.Logger,
) -> None:
    """保存 RGB 原图、异常热图以及可用的预测/真值边界。

    Args:
        batch (EvaluationBatch): 含显示图、异常图、真值及可选二值预测的批次。
        output_root (str | os.PathLike[str]): 已存在的可视化输出目录。
        output_size (int | list[int] | tuple[int, int]): 输出 ``(width, height)``；
            整数表示正方形，并应与 ``display_images`` 的高宽一致。
        logger (logging.Logger): 记录保存阶段状态的日志器。

    Raises:
        TypeError: 输出尺寸格式错误。
        ValueError: 输出尺寸不是正整数，或推理结果没有显示图映射。
        RuntimeError: 任一样本缺少对应路径的显示图。
        OSError: 输出图像无法写入。

    Notes:
        ``cv2.applyColorMap`` 生成 BGR 热图，融合前显式转为 RGB；最终横向拼接
        的所有面板都按 RGB ``uint8`` 保存。
    """
    from PIL import Image
    from skimage.segmentation import mark_boundaries

    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    elif isinstance(output_size, (tuple, list)) and len(output_size) == 2:
        output_size = (int(output_size[0]), int(output_size[1]))
    else:
        raise TypeError("vis_size must be a positive integer or width-height pair")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in output_size
    ):
        raise ValueError("vis_size dimensions must be positive integers")
    if not isinstance(batch.display_images, dict):
        raise ValueError("Visualization requires inference(..., display_size=vis_size)")

    logger.info("Saving visualizations")
    for index, (sample, prediction_mask, gt_mask, class_name) in enumerate(
        tqdm(
            zip(
                batch.samples,
                batch.prediction_masks,
                batch.gt_masks,
                batch.class_names,
            ),
            total=len(batch.samples),
        )
    ):
        key = sample.image.image_path
        if key not in batch.display_images:
            raise RuntimeError(f"Display image is missing for {key}")
        image = batch.display_images[key]
        prediction_mask = cv2.resize(
            prediction_mask,
            output_size,
            interpolation=cv2.INTER_NEAREST,
        )
        gt_mask = cv2.resize(
            gt_mask,
            output_size,
            interpolation=cv2.INTER_NEAREST,
        )

        threshold = (
            float(batch.pixel_thresholds[index])
            if batch.pixel_thresholds is not None
            else None
        )
        normalized_mask = (
            scale_anomaly_map_for_display(prediction_mask, threshold) * 255
        ).astype(np.uint8)
        heatmap = cv2.applyColorMap(normalized_mask, cv2.COLORMAP_JET)
        # OpenCV 生成 BGR 热图，和业务侧 RGB 原图融合前必须转换颜色顺序。
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heat = cv2.addWeighted(image.astype(np.uint8), 0.5, heatmap, 0.5, 0)

        if batch.binary_prediction_masks is None:
            panels = [image, heat]
        else:
            binary_prediction = cv2.resize(
                batch.binary_prediction_masks[index].astype(np.uint8),
                output_size,
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            image_with_prediction = mark_boundaries(
                image / 255,
                binary_prediction,
                color=(1, 0, 0),
                mode="inner",
            )
            panels = [image_with_prediction * 255, heat]
        if sample.mask is not None:
            image_with_mask = mark_boundaries(
                image / 255,
                gt_mask,
                color=(1, 0, 0),
                mode="inner",
            )
            panels.append(image_with_mask * 255)

        image_name = os.path.basename(sample.image.image_path)
        Image.fromarray(np.concatenate(panels, axis=1).astype(np.uint8)).save(
            os.path.join(output_root, f"{class_name}_{index}_{image_name}")
        )
