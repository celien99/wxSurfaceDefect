import os

import cv2
import numpy as np
from tqdm import tqdm


def save_evaluation_visualizations(
    batch,
    scores,
    output_root,
    output_size,
    logger,
) -> None:
    from PIL import Image
    from skimage.segmentation import mark_boundaries

    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    elif isinstance(output_size, (tuple, list)) and len(output_size) == 2:
        output_size = tuple(output_size)
    else:
        raise TypeError("vis_size must be a positive integer or width-height pair")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in output_size
    ):
        raise ValueError("vis_size dimensions must be positive integers")
    if not isinstance(batch.display_images, dict):
        raise ValueError("Visualization requires inference(..., display_size=vis_size)")

    segmentation_thresholds = (
        {score["clsname"]: score["seg_threshold"] for score in scores}
        if "seg_threshold" in scores[0]
        else None
    )
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

        normalized_mask = cv2.normalize(
            prediction_mask,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        ).astype(np.uint8)
        heatmap = cv2.applyColorMap(normalized_mask, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heat = cv2.addWeighted(image.astype(np.uint8), 0.5, heatmap, 0.5, 0)

        image_with_mask = mark_boundaries(
            image / 255,
            gt_mask,
            color=(1, 0, 0),
            mode="inner",
        )
        if segmentation_thresholds is None:
            panels = [image, heat, image_with_mask * 255]
        else:
            binary_prediction = (
                prediction_mask >= segmentation_thresholds[class_name]
            )
            image_with_prediction = mark_boundaries(
                image / 255,
                binary_prediction,
                color=(1, 0, 0),
                mode="inner",
            )
            panels = [image_with_prediction * 255, heat, image_with_mask * 255]

        image_name = os.path.basename(sample.image.image_path)
        Image.fromarray(np.concatenate(panels, axis=1).astype(np.uint8)).save(
            os.path.join(output_root, f"{class_name}_{index}_{image_name}")
        )
