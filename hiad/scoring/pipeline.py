from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import numpy as np

from hiad.constants import TASK_TYPE_DYNAMIC_PATCH, TASK_TYPE_THUMBNAIL
from hiad.data.preparation import PreparedInputRecord

from .calibration import MULTIRISK_CALIBRATION_DOMAIN, calibrate_batch
from .contracts import (
    ContextEvidence,
    DetectorEvidence,
    PatchEvidence,
    RawImageScores,
    valid_token_mask,
)
from .multirisk import MultiRiskScorer


@dataclass(frozen=True)
class ImageEvidence:
    image_size: tuple[int, int]
    patches: tuple[PatchEvidence, ...]
    context: ContextEvidence
    display_image: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not self.patches:
            raise ValueError("Image evidence requires local patches")
        if self.context.image_size != self.image_size:
            raise ValueError("Context and image evidence sizes must match")
        if any(patch.image_size != self.image_size for patch in self.patches):
            raise ValueError("Local and image evidence sizes must match")


def _matched_records_and_outputs(records, outputs, task_type: str):
    records = tuple(records)
    outputs = tuple(outputs)
    if len(records) != len(outputs):
        raise RuntimeError(
            f"{task_type} evidence count {len(outputs)} does not match input count "
            f"{len(records)}"
        )
    for record, output in zip(records, outputs):
        if not isinstance(record, PreparedInputRecord) or record.task_type != task_type:
            raise TypeError(f"{task_type} assembly received an invalid input record")
        if not isinstance(output, DetectorEvidence):
            raise TypeError(f"{task_type} assembly requires DetectorEvidence outputs")
        yield record, output


def assemble_patch_evidence(
    records: Sequence[PreparedInputRecord],
    outputs: Sequence[DetectorEvidence],
) -> dict[str, list[PatchEvidence]]:
    assembled = {}
    for record, output in _matched_records_and_outputs(
        records,
        outputs,
        TASK_TYPE_DYNAMIC_PATCH,
    ):
        evidence = PatchEvidence(
            source_xywh=record.source_xywh,
            valid_source_hw=record.valid_source_hw,
            image_size=record.image_size,
            raw_token_map=output.raw_token_map,
            valid_token_mask=valid_token_mask(
                output.raw_token_map.shape,
                record.valid_source_hw,
            ),
            raw_pixel_map=output.raw_pixel_map,
        )
        assembled.setdefault(record.image_path, []).append(evidence)
    return assembled


def assemble_context_evidence(
    records: Sequence[PreparedInputRecord],
    outputs: Sequence[DetectorEvidence],
) -> dict[str, ContextEvidence]:
    assembled = {}
    for record, output in _matched_records_and_outputs(
        records,
        outputs,
        TASK_TYPE_THUMBNAIL,
    ):
        if record.image_path in assembled:
            raise ValueError(f"Duplicate Context evidence for {record.image_path}")
        assembled[record.image_path] = ContextEvidence(
            image_size=record.image_size,
            model_input_size=record.model_input_size,
            raw_token_map=output.raw_token_map,
            raw_pixel_map=output.raw_pixel_map,
        )
    return assembled


def merge_worker_evidence(
    worker_results: Sequence[Mapping[str, Mapping]],
    image_paths: Sequence[str],
    *,
    display_images: Mapping[str, np.ndarray] | None = None,
) -> dict[str, ImageEvidence]:
    ordered_paths = tuple(image_paths)
    if not ordered_paths or len(ordered_paths) != len(set(ordered_paths)):
        raise ValueError("image_paths must be non-empty and unique")
    expected_paths = set(ordered_paths)
    merged = {
        path: {"image_size": None, "patches": [], "context": None}
        for path in ordered_paths
    }

    for worker_result in worker_results:
        if not isinstance(worker_result, Mapping) or set(worker_result) != expected_paths:
            raise ValueError("Every worker result must contain the exact source image set")
        for path, payload in worker_result.items():
            if not isinstance(payload, Mapping):
                raise TypeError("Worker image evidence must be a mapping")
            unknown_fields = set(payload).difference({"image_size", "patches", "context"})
            if unknown_fields:
                raise ValueError(f"Unknown worker evidence fields: {sorted(unknown_fields)}")
            image_size = payload.get("image_size")
            if image_size is None:
                raise ValueError(f"Worker image size is missing for {path}")
            current_size = merged[path]["image_size"]
            if current_size is not None and current_size != image_size:
                raise ValueError(f"Worker image sizes disagree for {path}")
            merged[path]["image_size"] = image_size

            patches = payload.get("patches", ())
            if not isinstance(patches, (list, tuple)):
                raise TypeError("Worker patches must be a sequence")
            if any(not isinstance(patch, PatchEvidence) for patch in patches):
                raise TypeError("Worker patches must contain PatchEvidence")
            merged[path]["patches"].extend(patches)

            context = payload.get("context")
            if context is not None:
                if not isinstance(context, ContextEvidence):
                    raise TypeError("Worker Context output must be ContextEvidence")
                if merged[path]["context"] is not None:
                    raise ValueError(f"Duplicate Context evidence for {path}")
                merged[path]["context"] = context

    display_images = {} if display_images is None else dict(display_images)
    unknown_displays = set(display_images).difference(expected_paths)
    if unknown_displays:
        raise ValueError(f"Display images contain unknown paths: {sorted(unknown_displays)}")

    result = {}
    for path in ordered_paths:
        payload = merged[path]
        if payload["image_size"] is None:
            raise ValueError(f"Image size is missing for {path}")
        if not payload["patches"]:
            raise ValueError(f"Local patch evidence is missing for {path}")
        if payload["context"] is None:
            raise ValueError(f"Context evidence is missing for {path}")
        patch_keys = [patch.source_xywh for patch in payload["patches"]]
        if len(patch_keys) != len(set(patch_keys)):
            raise ValueError(f"Duplicate local patch geometry for {path}")
        result[path] = ImageEvidence(
            image_size=payload["image_size"],
            patches=tuple(payload["patches"]),
            context=payload["context"],
            display_image=display_images.get(path),
        )
    return result


def _raised_hann_weights(height: int, width: int, floor: float = 0.05) -> np.ndarray:
    row_hann = np.hanning(height) if height > 1 else np.ones(1, dtype=np.float64)
    column_hann = np.hanning(width) if width > 1 else np.ones(1, dtype=np.float64)
    weights = np.outer(row_hann, column_hann)
    return floor + (1.0 - floor) * weights


def render_local_anomaly_map(patches: Sequence[PatchEvidence]) -> np.ndarray:
    patches = tuple(patches)
    if not patches:
        raise ValueError("Local anomaly-map reconstruction requires patches")
    image_size = patches[0].image_size
    if any(patch.image_size != image_size for patch in patches):
        raise ValueError("Local patches have inconsistent image sizes")
    image_width, image_height = image_size
    accumulated = np.zeros((image_height, image_width), dtype=np.float64)
    weight_map = np.zeros((image_height, image_width), dtype=np.float64)

    for patch in patches:
        patch_x, patch_y, patch_width, patch_height = patch.source_xywh
        valid_height, valid_width = patch.valid_source_hw
        weights = _raised_hann_weights(patch_height, patch_width)
        source_y = slice(patch_y, patch_y + valid_height)
        source_x = slice(patch_x, patch_x + valid_width)
        valid_weights = weights[:valid_height, :valid_width]
        accumulated[source_y, source_x] += (
            patch.raw_pixel_map[:valid_height, :valid_width] * valid_weights
        )
        weight_map[source_y, source_x] += valid_weights

    if np.any(weight_map <= 0):
        raise ValueError("Local patch evidence does not cover the complete source image")
    return (accumulated / weight_map).astype(np.float32)


def score_batch_evidence(
    evidence_by_path: Mapping[str, ImageEvidence],
    image_paths: Sequence[str],
    scorer: MultiRiskScorer,
) -> list[RawImageScores]:
    if not isinstance(scorer, MultiRiskScorer):
        raise TypeError("scorer must be MultiRiskScorer")
    image_paths = tuple(image_paths)
    if not image_paths or len(image_paths) != len(set(image_paths)):
        raise ValueError("Batch scoring image_paths must be non-empty and unique")
    scores = []
    for path in image_paths:
        if path not in evidence_by_path:
            raise KeyError(f"Image evidence is missing for {path}")
        evidence = evidence_by_path[path]
        scores.append(scorer.score_raw(evidence.patches, evidence.context))
    return scores


def build_batch_output(
    evidence_by_path: Mapping[str, ImageEvidence],
    image_paths: Sequence[str],
    scorer: MultiRiskScorer,
    calibration,
) -> dict:
    ordered_paths = tuple(image_paths)
    raw_scores = score_batch_evidence(evidence_by_path, ordered_paths, scorer)
    calibrated = calibrate_batch(raw_scores, calibration)
    anomaly_maps = [
        render_local_anomaly_map(evidence_by_path[path].patches)
        for path in ordered_paths
    ]
    return {
        "domain": MULTIRISK_CALIBRATION_DOMAIN,
        "image_paths": list(ordered_paths),
        "image_scores": np.asarray(
            [result.joint_percentile for result in calibrated],
            dtype=np.float64,
        ),
        "anomaly_maps": anomaly_maps,
        "is_defect": [result.is_defect for result in calibrated],
        "decision_percentile": calibrated[0].decision_percentile,
        "joint_percentile": [result.joint_percentile for result in calibrated],
        "defect_margin": [result.defect_margin for result in calibrated],
        "dominant_branch": [result.dominant_branch for result in calibrated],
        "branch_percentiles": [result.branch_percentiles for result in calibrated],
        "raw_branch_scores": [result.raw_branch_scores for result in calibrated],
        "candidate_location": [
            result.candidate_location.to_dict()
            for result in calibrated
        ],
    }
