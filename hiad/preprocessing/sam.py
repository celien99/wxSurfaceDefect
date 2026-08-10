from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from transformers import Sam2Model, Sam2Processor

from .constants import MAX_SAM_POSITIVE_POINTS
from .masks import MaskRejected


def sam2_longest_edge(processor: Sam2Processor) -> int:
    size = processor.image_processor.size
    if isinstance(size, int):
        longest_edge = size
    elif isinstance(size, Mapping) and "longest_edge" in size:
        longest_edge = size["longest_edge"]
    elif isinstance(size, Mapping) and {"height", "width"}.issubset(size):
        longest_edge = max(size["height"], size["width"])
    else:
        raise ValueError(f"Unsupported SAM2 processor image size: {size}")
    if (
        isinstance(longest_edge, bool)
        or not isinstance(longest_edge, int)
        or longest_edge <= 0
    ):
        raise ValueError(f"Invalid SAM2 processor longest edge: {longest_edge}")
    return longest_edge


def load_sam2_components(model_id: str) -> tuple[Sam2Model, Sam2Processor]:
    processor = Sam2Processor.from_pretrained(model_id)
    model = Sam2Model.from_pretrained(model_id)
    model.requires_grad_(False)
    model.eval()
    model.cpu()
    return model, processor


def run_sam2(
    rgb: np.ndarray,
    box_xyxy: np.ndarray,
    positive_points_xy: np.ndarray,
    *,
    model: Sam2Model,
    processor: Sam2Processor,
    device: torch.device,
    runtime_dtype: torch.dtype,
) -> np.ndarray:
    image_height, image_width = rgb.shape[:2]
    if box_xyxy.shape != (4,) or not np.isfinite(box_xyxy).all():
        raise MaskRejected("invalid_sam_box")
    if (
        box_xyxy[0] < 0
        or box_xyxy[1] < 0
        or box_xyxy[2] >= image_width
        or box_xyxy[3] >= image_height
        or box_xyxy[2] <= box_xyxy[0]
        or box_xyxy[3] <= box_xyxy[1]
    ):
        raise MaskRejected("out_of_frame_sam_box")
    if (
        positive_points_xy.ndim != 2
        or positive_points_xy.shape[1] != 2
        or positive_points_xy.shape[0] == 0
        or positive_points_xy.shape[0] > MAX_SAM_POSITIVE_POINTS
        or not np.isfinite(positive_points_xy).all()
    ):
        raise MaskRejected("invalid_sam_positive_points")

    rounded_points = np.rint(positive_points_xy).astype(np.int64)
    if (
        np.any(rounded_points[:, 0] < 0)
        or np.any(rounded_points[:, 0] >= image_width)
        or np.any(rounded_points[:, 1] < 0)
        or np.any(rounded_points[:, 1] >= image_height)
    ):
        raise MaskRejected("out_of_frame_sam_positive_points")

    processor_inputs = None
    model_inputs = None
    outputs = None
    postprocessed_masks = None
    selected_mask = None
    try:
        processor_inputs = processor(
            images=rgb,
            input_boxes=[[box_xyxy.tolist()]],
            input_points=[[positive_points_xy.tolist()]],
            input_labels=[[[1] * positive_points_xy.shape[0]]],
            return_tensors="pt",
        )
        model_inputs = {}
        for name, value in processor_inputs.items():
            if not isinstance(value, torch.Tensor):
                model_inputs[name] = value
            elif value.is_floating_point():
                model_inputs[name] = value.to(
                    device=device,
                    dtype=runtime_dtype,
                    non_blocking=False,
                )
            else:
                model_inputs[name] = value.to(device=device, non_blocking=False)

        model.requires_grad_(False)
        model.eval()
        model.to(device=device, dtype=runtime_dtype)
        with torch.inference_mode():
            outputs = model(**model_inputs)
        postprocessed_masks = processor.post_process_masks(
            outputs.pred_masks.detach().cpu(),
            model_inputs["original_sizes"].detach().cpu(),
        )
        if not isinstance(postprocessed_masks, (tuple, list)) or len(
            postprocessed_masks
        ) != 1:
            raise MaskRejected("invalid_sam_batch_output")

        mask_tensor = postprocessed_masks[0]
        score_tensor = outputs.iou_scores.detach().float().cpu().reshape(-1)
        if not isinstance(mask_tensor, torch.Tensor) or mask_tensor.ndim < 2:
            raise MaskRejected("invalid_sam_mask_dimensions")
        if tuple(mask_tensor.shape[-2:]) != (image_height, image_width):
            raise MaskRejected("invalid_sam_mask_size")
        candidate_masks = mask_tensor.reshape(-1, image_height, image_width)
        if candidate_masks.shape[0] == 0 or candidate_masks.shape[0] != score_tensor.numel():
            raise MaskRejected("invalid_sam_candidate_count")

        finite_scores = torch.isfinite(score_tensor)
        if not finite_scores.any():
            raise MaskRejected("nonfinite_sam_iou_scores")
        best_index = int(
            torch.argmax(
                torch.where(
                    finite_scores,
                    score_tensor,
                    torch.full_like(score_tensor, -torch.inf),
                )
            ).item()
        )
        candidate = candidate_masks[best_index]
        if candidate.is_floating_point() and not torch.isfinite(candidate).all():
            raise MaskRejected("nonfinite_sam_mask")
        selected_mask = np.ascontiguousarray(candidate.numpy() != 0)
        if selected_mask.shape != (image_height, image_width) or not selected_mask.any():
            raise MaskRejected("empty_sam_mask")
        if not np.all(selected_mask[rounded_points[:, 1], rounded_points[:, 0]]):
            raise MaskRejected("sam_mask_misses_positive_prompt")
    finally:
        model.requires_grad_(False)
        model.eval()
        model.cpu()
        del processor_inputs, model_inputs, outputs, postprocessed_masks
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return selected_mask
