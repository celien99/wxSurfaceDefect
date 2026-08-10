from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from PIL import Image

from hiad.models import TimmDinoV3Encoder

from .artifacts import (
    load_preprocessing_artifacts,
    sha256_state_dict,
    validate_preprocessing_bundle,
)
from .config import require_nonempty_string
from .dino import build_frozen_dino_encoder
from .images import (
    inverse_normalize_image,
    normalize_with_foreground,
    validate_image_array,
)
from .masks import MaskRejected, validate_warped_mask
from .registration import register_and_warp_mask


class ForegroundPreprocessor:
    def __init__(
        self,
        bundle: Mapping[str, Any],
        prototypes: Mapping[str, Any],
        template: Mapping[str, Any],
        reference_mask: np.ndarray,
        device: torch.device,
        logger=None,
    ):
        self.checkpoint_root = bundle["checkpoint_root"]
        self.config = bundle["config"]
        self.manifest = bundle["manifest"]
        self.category = self.manifest["category"]
        self.device = torch.device(device)
        self.logger = logger
        self.prototypes = dict(prototypes)
        self.template = dict(template)
        self.reference_mask = reference_mask
        self.dino_encoder: TimmDinoV3Encoder | None = None
        self._closed = False

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_root: str,
        device: torch.device,
        runtime_config: Mapping[str, Any] | None = None,
        expected_category: str | None = None,
        logger=None,
        eager_runtime_load: bool = True,
    ) -> "ForegroundPreprocessor":
        bundle = validate_preprocessing_bundle(
            checkpoint_root,
            runtime_config=runtime_config,
            expected_category=expected_category,
        )
        prototypes, template, reference_mask = load_preprocessing_artifacts(bundle)
        instance = cls(
            bundle,
            prototypes,
            template,
            reference_mask,
            device,
            logger=logger,
        )
        if eager_runtime_load:
            try:
                instance._ensure_dino()
            except Exception:
                instance.close()
                raise
        return instance

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("ForegroundPreprocessor is closed")

    def _ensure_dino(self) -> TimmDinoV3Encoder:
        self._ensure_open()
        if self.dino_encoder is None:
            encoder = build_frozen_dino_encoder(self.config)
            if encoder.patch_size != self.template["patch_size"]:
                raise ValueError("Loaded DINO patch size does not match preprocessing bundle")
            weights_hash = sha256_state_dict(encoder.state_dict())
            if weights_hash != self.manifest["dino"]["weights_sha256"]:
                raise ValueError("Loaded DINO weights do not match preprocessing bundle")
            self.dino_encoder = encoder
        self.dino_encoder.requires_grad_(False)
        self.dino_encoder.eval()
        return self.dino_encoder

    def _validate_dino_metrics(self, metrics: Mapping[str, Any]) -> None:
        required_dino_metrics = ("match_count", "inlier_ratio", "reprojection_ratio")
        if any(name not in metrics for name in required_dino_metrics):
            raise MaskRejected("missing_dino_metrics")
        if (
            not isinstance(metrics["match_count"], int)
            or metrics["match_count"] < self.config["min_dino_matches"]
        ):
            raise MaskRejected("dino_match_count_below_threshold")
        if (
            not isinstance(metrics["inlier_ratio"], (int, float))
            or not math.isfinite(float(metrics["inlier_ratio"]))
            or metrics["inlier_ratio"] < self.config["min_dino_inlier_ratio"]
        ):
            raise MaskRejected("dino_inlier_ratio_below_threshold")
        if (
            not isinstance(metrics["reprojection_ratio"], (int, float))
            or not math.isfinite(float(metrics["reprojection_ratio"]))
            or metrics["reprojection_ratio"]
            > self.config["max_dino_reprojection_ratio"]
        ):
            raise MaskRejected("dino_reprojection_ratio_above_threshold")

    def _effective_mask(self, rgb: np.ndarray, source_identity: str) -> np.ndarray:
        metrics: dict[str, Any] = {}
        try:
            warped_mask, dino_metrics = register_and_warp_mask(
                rgb,
                encoder=self._ensure_dino(),
                prototypes=self.prototypes,
                template=self.template,
                reference_mask=self.reference_mask,
                config=self.config,
                device=self.device,
            )
            metrics.update(dino_metrics)
            metrics.pop("reason", None)
            self._validate_dino_metrics(metrics)

            cleaned_mask, mask_metrics = validate_warped_mask(
                warped_mask,
                self.reference_mask,
                self.config,
            )
            metrics.update(mask_metrics)
            return cleaned_mask
        except MaskRejected as error:
            metrics.update(error.metrics)
            if self.logger is not None:
                finite_metrics = {}
                for name, value in metrics.items():
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        continue
                    if math.isfinite(float(value)):
                        finite_metrics[name] = value
                self.logger.warning(
                    json.dumps(
                        {
                            "category": self.category,
                            "event": "foreground_mask_rejected",
                            "metrics": finite_metrics,
                            "reason": str(error),
                            "source": str(source_identity),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
            raise

    def _process_rgb(self, rgb: np.ndarray, source_identity: str) -> np.ndarray:
        effective_mask = self._effective_mask(rgb, source_identity)
        if effective_mask.dtype != np.bool_ or effective_mask.shape != rgb.shape[:2]:
            raise RuntimeError("Effective foreground mask has invalid geometry")
        return normalize_with_foreground(rgb, effective_mask, self.config)

    def _validate_category(self, category: str | None) -> None:
        if category is not None and require_nonempty_string(category, "category") != self.category:
            raise ValueError(
                f"Image category {category!r} does not match bound category "
                f"{self.category!r}"
            )

    def process_file(self, image_path: str, category: str | None = None) -> np.ndarray:
        self._ensure_open()
        self._validate_category(category)
        resolved_path = os.path.abspath(os.fspath(image_path))
        with Image.open(resolved_path) as image_file:
            converted = image_file.convert("RGB")
            try:
                rgb = validate_image_array(
                    np.asarray(converted, dtype=np.uint8),
                    self.config["input_scale"],
                    "RGB",
                )
                processed = self._process_rgb(rgb, resolved_path)
                del rgb
            finally:
                converted.close()
        return processed

    def process_array(self, image: np.ndarray, category: str | None = None) -> np.ndarray:
        self._ensure_open()
        self._validate_category(category)
        rgb = validate_image_array(
            image,
            self.config["input_scale"],
            self.config["array_color_space"],
        )
        source_identity = f"array:{rgb.shape[1]}x{rgb.shape[0]}"
        processed = self._process_rgb(rgb, source_identity)
        del rgb
        return processed

    def inverse_normalize(
        self,
        image: np.ndarray,
        output_size: int | tuple[int, int] | list[int] | None = None,
    ) -> np.ndarray:
        self._ensure_open()
        return inverse_normalize_image(image, self.config, output_size)

    def release_gpu(self) -> None:
        if self.dino_encoder is not None:
            self.dino_encoder.requires_grad_(False)
            self.dino_encoder.eval()
            self.dino_encoder.cpu()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def close(self) -> None:
        if self._closed:
            return
        self.release_gpu()
        self.dino_encoder = None
        self.prototypes = None
        self.template = None
        self.reference_mask = None
        self._closed = True
