from dataclasses import dataclass
import hashlib
import json
from collections.abc import Mapping, Sequence

import numpy as np

from hiad.constants import (
    DINO_PATCH_SIZE,
    SUPPORTED_ANOMALY_DISTANCES,
    TASK_TYPE_DYNAMIC_PATCH,
    TASK_TYPE_THUMBNAIL,
)


def _integer_tuple(value, name: str) -> tuple[int, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value)
    ):
        raise ValueError(f"{name} must be a non-empty sequence of positive integers")
    result = tuple(value)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _normalized_fusion_weights(fusion_weights, count: int) -> tuple[float, ...]:
    if fusion_weights is None:
        return tuple(1.0 / count for _ in range(count))
    if (
        not isinstance(fusion_weights, Sequence)
        or isinstance(fusion_weights, (str, bytes))
        or len(fusion_weights) != count
    ):
        raise ValueError("fusion_weights must match ds_factors")
    values = np.asarray(fusion_weights, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0) or values.sum() <= 0:
        raise ValueError("fusion_weights must be finite, non-negative, and have a positive sum")
    return tuple(float(value) for value in values / values.sum())


@dataclass(frozen=True)
class MultiRiskConfig:
    patch_size: int
    stride: int
    ds_factors: tuple[int, ...]
    thumbnail_size: int
    anomaly_distance: str
    use_fp16: bool
    fusion_weights: tuple[float, ...]
    peak_top_k: int
    region_kernels: tuple[int, ...]
    line_lengths: tuple[int, ...]
    line_weight_floor: float
    line_center_weight: float
    line_peak_weight: float
    context_kernels: tuple[int, ...]
    decision_percentile: float
    branch_order: tuple[str, ...]

    @classmethod
    def from_runtime(
        cls,
        scoring_config,
        tasks,
        *,
        anomaly_distance: str,
        use_fp16: bool,
        fusion_weights=None,
    ) -> "MultiRiskConfig":
        if not isinstance(scoring_config, Mapping):
            raise TypeError("scoring_config must be a mapping")
        dynamic_tasks = [
            task
            for task in tasks
            if task.get("type") == TASK_TYPE_DYNAMIC_PATCH
        ]
        thumbnail_tasks = [
            task
            for task in tasks
            if task.get("type") == TASK_TYPE_THUMBNAIL
        ]
        if len(dynamic_tasks) != 1 or len(thumbnail_tasks) != 1:
            raise ValueError("Multi-risk scoring requires one dynamic patch and one thumbnail task")
        dynamic = dynamic_tasks[0]
        thumbnail = thumbnail_tasks[0]
        patch_size = dynamic["patch_size"]
        stride = patch_size if dynamic["stride"] is None else dynamic["stride"]
        ds_factors = tuple(dynamic["ds_factors"])
        thumbnail_size = thumbnail["thumbnail_size"]
        if patch_size % DINO_PATCH_SIZE or thumbnail_size % DINO_PATCH_SIZE:
            raise ValueError("Patch and thumbnail sizes must be multiples of the DINO patch size")
        if anomaly_distance not in SUPPORTED_ANOMALY_DISTANCES:
            raise ValueError("anomaly_distance must be normalized_l2 or cosine")
        if not isinstance(use_fp16, bool):
            raise TypeError("use_fp16 must be a boolean")
        peak_top_k = scoring_config.get("peak_top_k")
        if isinstance(peak_top_k, bool) or not isinstance(peak_top_k, int) or peak_top_k <= 0:
            raise ValueError("peak_top_k must be a positive integer")
        region_kernels = _integer_tuple(scoring_config.get("region_kernels"), "region_kernels")
        line_lengths = _integer_tuple(scoring_config.get("line_lengths"), "line_lengths")
        context_kernels = _integer_tuple(scoring_config.get("context_kernels"), "context_kernels")
        if min(line_lengths) < 2:
            raise ValueError("Line kernel lengths must be at least 2")
        line_weight_floor = float(scoring_config.get("line_weight_floor"))
        line_center_weight = float(scoring_config.get("line_center_weight"))
        line_peak_weight = float(scoring_config.get("line_peak_weight"))
        if (
            not np.isfinite([line_weight_floor, line_center_weight, line_peak_weight]).all()
            or not 0 < line_weight_floor <= 1
            or line_center_weight < 0
            or line_peak_weight < 0
            or not np.isclose(line_center_weight + line_peak_weight, 1.0)
        ):
            raise ValueError("Line weights are invalid")
        decision_percentile = float(scoring_config.get("decision_percentile"))
        if not np.isfinite(decision_percentile) or not 0 < decision_percentile < 1:
            raise ValueError("decision_percentile must be finite and in (0, 1)")
        branch_order = tuple(scoring_config.get("branch_order", ()))
        if set(branch_order) != {"peak", "region", "line", "context"} or len(branch_order) != 4:
            raise ValueError("branch_order must contain peak, region, line, and context exactly once")
        thumbnail_tokens = thumbnail_size // DINO_PATCH_SIZE
        patch_tokens = patch_size // DINO_PATCH_SIZE
        if peak_top_k > patch_tokens * patch_tokens:
            raise ValueError("peak_top_k must not exceed the patch token count")
        if max(region_kernels) > patch_tokens:
            raise ValueError("Region kernels must fit the patch token grid")
        if max(line_lengths) > patch_tokens or patch_tokens < 3:
            raise ValueError(
                "Line kernels must fit the patch token grid with room for both flanks"
            )
        if max(context_kernels) > thumbnail_tokens:
            raise ValueError("Context kernels must fit the thumbnail token grid")
        return cls(
            patch_size=patch_size,
            stride=stride,
            ds_factors=ds_factors,
            thumbnail_size=thumbnail_size,
            anomaly_distance=anomaly_distance,
            use_fp16=use_fp16,
            fusion_weights=_normalized_fusion_weights(fusion_weights, len(ds_factors)),
            peak_top_k=peak_top_k,
            region_kernels=region_kernels,
            line_lengths=line_lengths,
            line_weight_floor=line_weight_floor,
            line_center_weight=line_center_weight,
            line_peak_weight=line_peak_weight,
            context_kernels=context_kernels,
            decision_percentile=decision_percentile,
            branch_order=branch_order,
        )

    @property
    def subscore_order(self) -> tuple[str, ...]:
        return (
            "peak",
            *(f"region.{kernel}" for kernel in self.region_kernels),
            *(f"line.h{length}" for length in self.line_lengths),
            *(f"line.v{length}" for length in self.line_lengths),
            *(f"context.{kernel}" for kernel in self.context_kernels),
        )

    def to_dict(self) -> dict:
        return {
            "schema_version": 2,
            "dino_patch_size": DINO_PATCH_SIZE,
            "patch_size": self.patch_size,
            "stride": self.stride,
            "ds_factors": list(self.ds_factors),
            "thumbnail_size": self.thumbnail_size,
            "anomaly_distance": self.anomaly_distance,
            "use_fp16": self.use_fp16,
            "fusion_weights": list(self.fusion_weights),
            "peak_top_k": self.peak_top_k,
            "region_kernels": list(self.region_kernels),
            "line_lengths": list(self.line_lengths),
            "line_weight_floor": self.line_weight_floor,
            "line_center_weight": self.line_center_weight,
            "line_peak_weight": self.line_peak_weight,
            "context_kernels": list(self.context_kernels),
            "decision_percentile": self.decision_percentile,
            "branch_order": list(self.branch_order),
            "subscore_order": list(self.subscore_order),
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
