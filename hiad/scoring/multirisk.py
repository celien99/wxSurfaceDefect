from dataclasses import dataclass
from typing import Sequence

import numpy as np

from hiad.constants import (
    DINO_PATCH_SIZE,
    TASK_TYPE_DYNAMIC_PATCH,
    TASK_TYPE_THUMBNAIL,
)

from .config import MultiRiskConfig
from .contracts import (
    CandidateLocation,
    ContextEvidence,
    PatchEvidence,
    RawImageScores,
    RawSubscore,
    TokenSupport,
)


@dataclass(frozen=True)
class PeakPatchScore:
    value: float
    primary_token_xy: tuple[int, int]
    contributing_tokens_xy: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class _WindowScore:
    value: float
    token_bbox_xyxy: tuple[int, int, int, int]
    contributing_tokens_xy: tuple[tuple[int, int], ...]
    trace_token_bbox_xyxy: tuple[int, int, int, int] | None = None


def peak_patch_score(
    raw_token_map: np.ndarray,
    valid_token_mask: np.ndarray,
    top_k: int,
) -> PeakPatchScore:
    if (
        not isinstance(raw_token_map, np.ndarray)
        or raw_token_map.dtype != np.float32
        or raw_token_map.ndim != 2
        or not np.isfinite(raw_token_map).all()
    ):
        raise ValueError("Peak token map must be a finite float32 2D array")
    if (
        not isinstance(valid_token_mask, np.ndarray)
        or valid_token_mask.dtype != np.bool_
        or valid_token_mask.shape != raw_token_map.shape
    ):
        raise ValueError("Peak valid mask must be a boolean token-shaped array")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("Peak top_k must be a positive integer")
    valid_y, valid_x = np.nonzero(valid_token_mask)
    if valid_y.size == 0:
        raise ValueError("Peak scoring requires at least one valid token")
    count = min(top_k, valid_y.size)
    values = raw_token_map[valid_y, valid_x]
    ranked = np.argsort(-values, kind="stable")[:count]
    tokens = tuple((int(valid_x[index]), int(valid_y[index])) for index in ranked)
    return PeakPatchScore(
        value=float(np.mean(values[ranked], dtype=np.float64)),
        primary_token_xy=tokens[0],
        contributing_tokens_xy=tokens,
    )


def _window_tokens(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> tuple[tuple[int, int], ...]:
    return tuple((x, y) for y in range(y0, y1) for x in range(x0, x1))


def _average_pool_max(
    raw_token_map: np.ndarray,
    valid_token_mask: np.ndarray,
    kernel: int,
) -> _WindowScore | None:
    height, width = raw_token_map.shape
    if kernel > height or kernel > width:
        return None

    values = np.where(valid_token_mask, raw_token_map, 0).astype(np.float64, copy=False)
    integral = np.pad(values, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    sums = (
        integral[kernel:, kernel:]
        - integral[:-kernel, kernel:]
        - integral[kernel:, :-kernel]
        + integral[:-kernel, :-kernel]
    )
    mask_integral = np.pad(
        valid_token_mask.astype(np.int32),
        ((1, 0), (1, 0)),
    ).cumsum(0).cumsum(1)
    counts = (
        mask_integral[kernel:, kernel:]
        - mask_integral[:-kernel, kernel:]
        - mask_integral[kernel:, :-kernel]
        + mask_integral[:-kernel, :-kernel]
    )
    applicable = counts == kernel * kernel
    if not np.any(applicable):
        return None

    means = sums / (kernel * kernel)
    means[~applicable] = -np.inf
    y0, x0 = np.unravel_index(int(np.argmax(means)), means.shape)
    x1 = int(x0 + kernel)
    y1 = int(y0 + kernel)
    return _WindowScore(
        value=float(means[y0, x0]),
        token_bbox_xyxy=(int(x0), int(y0), x1, y1),
        contributing_tokens_xy=_window_tokens(int(x0), int(y0), x1, y1),
    )


def _line_weights(length: int, floor: float) -> np.ndarray:
    positions = np.arange(length, dtype=np.float64)
    hann = 0.5 * (1.0 - np.cos(2.0 * np.pi * positions / (length - 1)))
    weights = floor + (1.0 - floor) * hann
    return weights / weights.sum()


def _line_pool_max(
    raw_token_map: np.ndarray,
    valid_token_mask: np.ndarray,
    length: int,
    orientation: str,
    *,
    weight_floor: float,
    center_weight: float,
    peak_weight: float,
) -> _WindowScore | None:
    weights = _line_weights(length, weight_floor)
    if orientation == "h":
        if raw_token_map.shape[0] < 3 or raw_token_map.shape[1] < length:
            return None
        windows = np.lib.stride_tricks.sliding_window_view(
            raw_token_map,
            (3, length),
        )
        mask_windows = np.lib.stride_tricks.sliding_window_view(
            valid_token_mask,
            (3, length),
        )
        applicable = mask_windows.all(axis=(-2, -1))
        center = windows[:, :, 1, :]
        flanks = (
            windows[:, :, 0, :].mean(axis=-1, dtype=np.float64)
            + windows[:, :, 2, :].mean(axis=-1, dtype=np.float64)
        ) / 2.0
        center_response = (
            center_weight * np.sum(center * weights, axis=-1, dtype=np.float64)
            + peak_weight * center.max(axis=-1)
        )
    elif orientation == "v":
        if raw_token_map.shape[0] < length or raw_token_map.shape[1] < 3:
            return None
        windows = np.lib.stride_tricks.sliding_window_view(
            raw_token_map,
            (length, 3),
        )
        mask_windows = np.lib.stride_tricks.sliding_window_view(
            valid_token_mask,
            (length, 3),
        )
        applicable = mask_windows.all(axis=(-2, -1))
        center = windows[:, :, :, 1]
        flanks = (
            windows[:, :, :, 0].mean(axis=-1, dtype=np.float64)
            + windows[:, :, :, 2].mean(axis=-1, dtype=np.float64)
        ) / 2.0
        center_response = (
            center_weight * np.sum(center * weights, axis=-1, dtype=np.float64)
            + peak_weight * center.max(axis=-1)
        )
    else:
        raise ValueError(f"Unsupported line orientation: {orientation}")

    if not np.any(applicable):
        return None
    responses = np.maximum(0.0, center_response - flanks)
    responses[~applicable] = -np.inf
    window_y, window_x = np.unravel_index(int(np.argmax(responses)), responses.shape)

    if orientation == "h":
        center_y = int(window_y + 1)
        x0 = int(window_x)
        x1 = x0 + length
        token_bbox = (x0, center_y, x1, center_y + 1)
        trace_bbox = (x0, int(window_y), x1, int(window_y + 3))
    else:
        center_x = int(window_x + 1)
        y0 = int(window_y)
        y1 = y0 + length
        token_bbox = (center_x, y0, center_x + 1, y1)
        trace_bbox = (int(window_x), y0, int(window_x + 3), y1)

    return _WindowScore(
        value=float(responses[window_y, window_x]),
        token_bbox_xyxy=token_bbox,
        contributing_tokens_xy=_window_tokens(*token_bbox),
        trace_token_bbox_xyxy=trace_bbox,
    )


def _patch_candidate(
    patch: PatchEvidence,
    *,
    subscale: str,
    window: _WindowScore,
) -> CandidateLocation:
    patch_x, patch_y, _, _ = patch.source_xywh
    valid_height, valid_width = patch.valid_source_hw
    image_width, image_height = patch.image_size
    token_x0, token_y0, token_x1, token_y1 = window.token_bbox_xyxy
    source_x0 = float(max(0, min(image_width, patch_x + token_x0 * DINO_PATCH_SIZE)))
    source_y0 = float(max(0, min(image_height, patch_y + token_y0 * DINO_PATCH_SIZE)))
    source_x1 = float(max(
        source_x0 + 1.0,
        min(image_width, patch_x + valid_width, patch_x + token_x1 * DINO_PATCH_SIZE),
    ))
    source_y1 = float(max(
        source_y0 + 1.0,
        min(image_height, patch_y + valid_height, patch_y + token_y1 * DINO_PATCH_SIZE),
    ))
    support = TokenSupport(
        task=TASK_TYPE_DYNAMIC_PATCH,
        subscale=subscale,
        patch_origin_xy=(patch_x, patch_y),
        token_bbox_xyxy=window.token_bbox_xyxy,
        contributing_tokens_xy=window.contributing_tokens_xy,
        trace_token_bbox_xyxy=window.trace_token_bbox_xyxy,
    )
    return CandidateLocation(
        source_point_xy=(
            (source_x0 + source_x1) / 2.0,
            (source_y0 + source_y1) / 2.0,
        ),
        source_bbox_xyxy=(source_x0, source_y0, source_x1, source_y1),
        token_support=support,
    )


def _context_candidate(
    context: ContextEvidence,
    *,
    subscale: str,
    window: _WindowScore,
) -> CandidateLocation:
    image_width, image_height = context.image_size
    token_height, token_width = context.raw_token_map.shape
    token_x0, token_y0, token_x1, token_y1 = window.token_bbox_xyxy
    source_x0 = token_x0 / token_width * image_width
    source_y0 = token_y0 / token_height * image_height
    source_x1 = token_x1 / token_width * image_width
    source_y1 = token_y1 / token_height * image_height
    support = TokenSupport(
        task=TASK_TYPE_THUMBNAIL,
        subscale=subscale,
        patch_origin_xy=(0, 0),
        token_bbox_xyxy=window.token_bbox_xyxy,
        contributing_tokens_xy=window.contributing_tokens_xy,
    )
    return CandidateLocation(
        source_point_xy=(
            (source_x0 + source_x1) / 2.0,
            (source_y0 + source_y1) / 2.0,
        ),
        source_bbox_xyxy=(source_x0, source_y0, source_x1, source_y1),
        token_support=support,
    )


class MultiRiskScorer:
    def __init__(self, config: MultiRiskConfig):
        self.config = config

    def score_raw(
        self,
        local_evidence: Sequence[PatchEvidence],
        context_evidence: ContextEvidence,
    ) -> RawImageScores:
        patches = tuple(local_evidence)
        if not patches:
            raise ValueError("Multi-risk scoring requires local patch evidence")
        image_size = context_evidence.image_size
        if any(patch.image_size != image_size for patch in patches):
            raise ValueError("Local and Context evidence image sizes must match")

        subscores = [self._score_peak(patches)]
        region_scores = self._score_regions(patches)
        line_scores = self._score_lines(patches)
        if not region_scores:
            raise ValueError("No Region scale is applicable to the local evidence")
        if not line_scores:
            raise ValueError("No Line scale is applicable to the local evidence")
        subscores.extend(region_scores)
        subscores.extend(line_scores)
        subscores.extend(self._score_context(context_evidence))
        by_key = {subscore.key: subscore for subscore in subscores}
        ordered = tuple(
            by_key[key]
            for key in self.config.subscore_order
            if key in by_key
        )
        if len(ordered) != len(subscores):
            raise RuntimeError("Scorer produced an unconfigured subscore")
        return RawImageScores(ordered)

    def _score_peak(self, patches: tuple[PatchEvidence, ...]) -> RawSubscore:
        winning_patch = patches[0]
        winning = peak_patch_score(
            winning_patch.raw_token_map,
            winning_patch.valid_token_mask,
            self.config.peak_top_k,
        )
        for patch in patches[1:]:
            candidate = peak_patch_score(
                patch.raw_token_map,
                patch.valid_token_mask,
                self.config.peak_top_k,
            )
            if candidate.value > winning.value:
                winning_patch = patch
                winning = candidate

        primary_x, primary_y = winning.primary_token_xy
        window = _WindowScore(
            value=winning.value,
            token_bbox_xyxy=(primary_x, primary_y, primary_x + 1, primary_y + 1),
            contributing_tokens_xy=winning.contributing_tokens_xy,
        )
        return RawSubscore(
            key="peak",
            branch="peak",
            value=winning.value,
            candidate=_patch_candidate(
                winning_patch,
                subscale="peak",
                window=window,
            ),
        )

    def _score_regions(
        self,
        patches: tuple[PatchEvidence, ...],
    ) -> list[RawSubscore]:
        scores = []
        for kernel in self.config.region_kernels:
            winner = None
            winning_patch = None
            for patch in patches:
                candidate = _average_pool_max(
                    patch.raw_token_map,
                    patch.valid_token_mask,
                    kernel,
                )
                if candidate is not None and (
                    winner is None or candidate.value > winner.value
                ):
                    winner = candidate
                    winning_patch = patch
            if winner is not None:
                key = f"region.{kernel}"
                scores.append(
                    RawSubscore(
                        key=key,
                        branch="region",
                        value=winner.value,
                        candidate=_patch_candidate(
                            winning_patch,
                            subscale=key,
                            window=winner,
                        ),
                    )
                )
        return scores

    def _score_lines(
        self,
        patches: tuple[PatchEvidence, ...],
    ) -> list[RawSubscore]:
        scores = []
        for orientation in ("h", "v"):
            for length in self.config.line_lengths:
                winner = None
                winning_patch = None
                for patch in patches:
                    candidate = _line_pool_max(
                        patch.raw_token_map,
                        patch.valid_token_mask,
                        length,
                        orientation,
                        weight_floor=self.config.line_weight_floor,
                        center_weight=self.config.line_center_weight,
                        peak_weight=self.config.line_peak_weight,
                    )
                    if candidate is not None and (
                        winner is None or candidate.value > winner.value
                    ):
                        winner = candidate
                        winning_patch = patch
                if winner is not None:
                    key = f"line.{orientation}{length}"
                    scores.append(
                        RawSubscore(
                            key=key,
                            branch="line",
                            value=winner.value,
                            candidate=_patch_candidate(
                                winning_patch,
                                subscale=key,
                                window=winner,
                            ),
                        )
                    )
        return scores

    def _score_context(self, context: ContextEvidence) -> list[RawSubscore]:
        valid_mask = np.ones(context.raw_token_map.shape, dtype=np.bool_)
        scores = []
        for kernel in self.config.context_kernels:
            winner = _average_pool_max(
                context.raw_token_map,
                valid_mask,
                kernel,
            )
            if winner is None:
                raise ValueError(f"Context scale {kernel} is not applicable")
            key = f"context.{kernel}"
            scores.append(
                RawSubscore(
                    key=key,
                    branch="context",
                    value=winner.value,
                    candidate=_context_candidate(
                        context,
                        subscale=key,
                        window=winner,
                    ),
                )
            )
        return scores
