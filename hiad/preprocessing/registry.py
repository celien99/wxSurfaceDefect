from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from .artifacts import validate_preprocessing_registry
from .config import require_nonempty_string
from .runtime import ForegroundPreprocessor


class ForegroundPreprocessorRegistry:
    def __init__(
        self,
        registry: Mapping[str, Any],
        device: torch.device,
        logger=None,
    ):
        self.generation_root = registry["generation_root"]
        self.categories = tuple(registry["categories"])
        self.bundles = dict(registry["bundles"])
        self.config = dict(registry["config"])
        self.device = torch.device(device)
        self.logger = logger
        self._preprocessors: dict[str, ForegroundPreprocessor] = {}
        self._active_category: str | None = None
        self._closed = False

    @classmethod
    def from_checkpoint(
        cls,
        generation_root: str,
        device: torch.device,
        runtime_config: Mapping[str, Any] | None = None,
        logger=None,
    ) -> "ForegroundPreprocessorRegistry":
        registry = validate_preprocessing_registry(
            generation_root,
            runtime_config=runtime_config,
        )
        return cls(registry, device, logger=logger)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("ForegroundPreprocessorRegistry is closed")

    def get(self, category: str) -> ForegroundPreprocessor:
        self._ensure_open()
        category = require_nonempty_string(category, "sample clsname")
        if category not in self.bundles:
            raise KeyError(
                f"No foreground preprocessing bundle is published for clsname {category!r}"
            )
        if self._active_category is not None and self._active_category != category:
            self._preprocessors[self._active_category].release_gpu()
        preprocessor = self._preprocessors.get(category)
        if preprocessor is None:
            preprocessor = ForegroundPreprocessor.from_checkpoint(
                self.bundles[category]["checkpoint_root"],
                self.device,
                runtime_config=self.config,
                expected_category=category,
                logger=self.logger,
                eager_runtime_load=False,
            )
            self._preprocessors[category] = preprocessor
        self._active_category = category
        return preprocessor

    def release_gpu(self) -> None:
        for preprocessor in self._preprocessors.values():
            preprocessor.release_gpu()
        self._active_category = None

    def close(self) -> None:
        if self._closed:
            return
        for preprocessor in self._preprocessors.values():
            preprocessor.close()
        self._preprocessors.clear()
        self._active_category = None
        self._closed = True
