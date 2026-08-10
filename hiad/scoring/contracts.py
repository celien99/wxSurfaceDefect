from dataclasses import dataclass
from typing import Optional

import numpy as np

from hiad.constants import DINO_PATCH_SIZE


def _validate_float_map(value: np.ndarray, name: str) -> None:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.float32
        or value.ndim != 2
        or value.shape[0] <= 0
        or value.shape[1] <= 0
        or not np.isfinite(value).all()
    ):
        raise ValueError(f"{name} must be a finite non-empty float32 2D array")


def _validate_size(size, name: str) -> tuple[int, int]:
    if (
        not isinstance(size, tuple)
        or len(size) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in size)
    ):
        raise ValueError(f"{name} must be a positive integer (width, height) tuple")
    return size


def _validate_integer_tuple(
    value,
    name: str,
    length: int,
    *,
    minimum: int = 0,
) -> tuple[int, ...]:
    if (
        not isinstance(value, tuple)
        or len(value) != length
        or any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or item < minimum
            for item in value
        )
    ):
        raise ValueError(
            f"{name} must be a length-{length} integer tuple with values >= {minimum}"
        )
    return value


@dataclass(frozen=True)
class DetectorEvidence:
    raw_token_map: np.ndarray
    raw_pixel_map: np.ndarray

    def __post_init__(self) -> None:
        _validate_float_map(self.raw_token_map, "raw_token_map")
        _validate_float_map(self.raw_pixel_map, "raw_pixel_map")


@dataclass(frozen=True)
class PatchEvidence:
    source_xywh: tuple[int, int, int, int]
    valid_source_hw: tuple[int, int]
    image_size: tuple[int, int]
    raw_token_map: np.ndarray
    valid_token_mask: np.ndarray
    raw_pixel_map: np.ndarray

    def __post_init__(self) -> None:
        _validate_float_map(self.raw_token_map, "raw_token_map")
        _validate_float_map(self.raw_pixel_map, "raw_pixel_map")
        image_width, image_height = _validate_size(self.image_size, "image_size")
        _validate_integer_tuple(self.source_xywh, "source_xywh", 4)
        source_x, source_y, patch_width, patch_height = self.source_xywh
        if source_x < 0 or source_y < 0 or patch_width <= 0 or patch_height <= 0:
            raise ValueError("source_xywh contains invalid patch geometry")
        if source_x >= image_width or source_y >= image_height:
            raise ValueError("Patch origin lies outside the source image")
        valid_height, valid_width = _validate_integer_tuple(
            self.valid_source_hw,
            "valid_source_hw",
            2,
            minimum=1,
        )
        if (
            valid_height > patch_height
            or valid_width > patch_width
            or source_y + valid_height > image_height
            or source_x + valid_width > image_width
        ):
            raise ValueError("valid_source_hw is inconsistent with patch and image geometry")
        if self.raw_pixel_map.shape != (patch_height, patch_width):
            raise ValueError("raw_pixel_map shape must match source_xywh patch size")
        if (
            not isinstance(self.valid_token_mask, np.ndarray)
            or self.valid_token_mask.dtype != np.bool_
            or self.valid_token_mask.shape != self.raw_token_map.shape
        ):
            raise ValueError("valid_token_mask must be a boolean raw-token-shaped array")
        expected_token_shape = (
            patch_height // DINO_PATCH_SIZE,
            patch_width // DINO_PATCH_SIZE,
        )
        if self.raw_token_map.shape != expected_token_shape:
            raise ValueError("raw_token_map shape is inconsistent with the DINO patch size")
        expected_valid = valid_token_mask(expected_token_shape, self.valid_source_hw)
        if not np.array_equal(self.valid_token_mask, expected_valid):
            raise ValueError("valid_token_mask does not match valid_source_hw")


@dataclass(frozen=True)
class ContextEvidence:
    image_size: tuple[int, int]
    model_input_size: tuple[int, int]
    raw_token_map: np.ndarray
    raw_pixel_map: np.ndarray

    def __post_init__(self) -> None:
        _validate_float_map(self.raw_token_map, "raw_token_map")
        _validate_float_map(self.raw_pixel_map, "raw_pixel_map")
        _validate_size(self.image_size, "image_size")
        model_width, model_height = _validate_size(self.model_input_size, "model_input_size")
        if self.raw_pixel_map.shape != (model_height, model_width):
            raise ValueError("Context raw_pixel_map shape must match model_input_size")
        if self.raw_token_map.shape != (
            model_height // DINO_PATCH_SIZE,
            model_width // DINO_PATCH_SIZE,
        ):
            raise ValueError("Context raw_token_map shape is inconsistent with model_input_size")


@dataclass(frozen=True)
class TokenSupport:
    task: str
    subscale: str
    patch_origin_xy: tuple[int, int]
    token_bbox_xyxy: tuple[int, int, int, int]
    contributing_tokens_xy: tuple[tuple[int, int], ...]
    trace_token_bbox_xyxy: Optional[tuple[int, int, int, int]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.task, str) or not isinstance(self.subscale, str) or not self.task or not self.subscale:
            raise ValueError("Token support task and subscale must be non-empty")
        _validate_integer_tuple(self.patch_origin_xy, "patch_origin_xy", 2)
        token_bbox = _validate_integer_tuple(
            self.token_bbox_xyxy,
            "token_bbox_xyxy",
            4,
        )
        x0, y0, x1, y1 = token_bbox
        if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
            raise ValueError("token_bbox_xyxy must be a positive half-open box")
        if not isinstance(self.contributing_tokens_xy, tuple) or not self.contributing_tokens_xy:
            raise ValueError("Token support must contain at least one contributing token")
        for index, point in enumerate(self.contributing_tokens_xy):
            _validate_integer_tuple(
                point,
                f"contributing_tokens_xy[{index}]",
                2,
            )
        if self.trace_token_bbox_xyxy is not None:
            trace_x0, trace_y0, trace_x1, trace_y1 = _validate_integer_tuple(
                self.trace_token_bbox_xyxy,
                "trace_token_bbox_xyxy",
                4,
            )
            if trace_x1 <= trace_x0 or trace_y1 <= trace_y0:
                raise ValueError("trace_token_bbox_xyxy must be a positive half-open box")

    def to_dict(self) -> dict:
        payload = {
            "task": self.task,
            "subscale": self.subscale,
            "patch_origin_xy": list(self.patch_origin_xy),
            "token_bbox_xyxy": list(self.token_bbox_xyxy),
            "contributing_tokens_xy": [list(point) for point in self.contributing_tokens_xy],
        }
        if self.trace_token_bbox_xyxy is not None:
            payload["trace_token_bbox_xyxy"] = list(self.trace_token_bbox_xyxy)
        return payload


@dataclass(frozen=True)
class CandidateLocation:
    source_point_xy: tuple[float, float]
    source_bbox_xyxy: tuple[float, float, float, float]
    token_support: TokenSupport

    def __post_init__(self) -> None:
        point = np.asarray(self.source_point_xy, dtype=np.float64)
        box = np.asarray(self.source_bbox_xyxy, dtype=np.float64)
        if point.shape != (2,) or box.shape != (4,) or not np.isfinite(point).all() or not np.isfinite(box).all():
            raise ValueError("Candidate source coordinates must be finite")
        x0, y0, x1, y1 = box.tolist()
        if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
            raise ValueError("source_bbox_xyxy must be a positive half-open box")
        if not (x0 <= point[0] < x1 and y0 <= point[1] < y1):
            raise ValueError("source_point_xy must lie inside source_bbox_xyxy")

    def to_dict(self) -> dict:
        return {
            "coordinate_space": "source_image",
            "source_point_xy": list(self.source_point_xy),
            "source_bbox_xyxy": list(self.source_bbox_xyxy),
            "token_support": self.token_support.to_dict(),
        }


@dataclass(frozen=True)
class RawSubscore:
    key: str
    branch: str
    value: float
    candidate: CandidateLocation

    def __post_init__(self) -> None:
        if not self.key or not self.branch or not np.isfinite(self.value):
            raise ValueError("Raw subscore key, branch, and finite value are required")


@dataclass(frozen=True)
class RawImageScores:
    subscores: tuple[RawSubscore, ...]

    def __post_init__(self) -> None:
        keys = [subscore.key for subscore in self.subscores]
        if not keys or len(keys) != len(set(keys)):
            raise ValueError("Raw image subscores must be non-empty and uniquely keyed")

    def by_key(self) -> dict[str, RawSubscore]:
        return {subscore.key: subscore for subscore in self.subscores}

    def values(self) -> dict[str, float]:
        return {subscore.key: float(subscore.value) for subscore in self.subscores}


@dataclass(frozen=True)
class CalibratedImageResult:
    is_defect: bool
    decision_percentile: float
    joint_percentile: float
    defect_margin: float
    dominant_branch: str
    branch_percentiles: dict[str, float]
    raw_branch_scores: dict[str, dict[str, float]]
    candidate_location: CandidateLocation

    def to_dict(self) -> dict:
        return {
            "is_defect": self.is_defect,
            "decision_percentile": self.decision_percentile,
            "joint_percentile": self.joint_percentile,
            "defect_margin": self.defect_margin,
            "dominant_branch": self.dominant_branch,
            "branch_percentiles": dict(self.branch_percentiles),
            "raw_branch_scores": {
                branch: dict(values) for branch, values in self.raw_branch_scores.items()
            },
            "candidate_location": self.candidate_location.to_dict(),
        }


def valid_token_mask(
    token_shape: tuple[int, int],
    valid_source_hw: tuple[int, int],
) -> np.ndarray:
    token_height, token_width = token_shape
    valid_height, valid_width = valid_source_hw
    rows = np.arange(token_height, dtype=np.int32) * DINO_PATCH_SIZE < valid_height
    columns = np.arange(token_width, dtype=np.int32) * DINO_PATCH_SIZE < valid_width
    return np.logical_and(rows[:, None], columns[None, :])
