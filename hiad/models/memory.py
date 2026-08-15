from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class NormalFeatureMemory(nn.Module):
    """Streaming diagonal-Gaussian model of normal feature tokens."""

    def __init__(self, embed_dim: int, layers: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        if embed_dim <= 0 or layers <= 0:
            raise ValueError("embed_dim and layers must be positive")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.embed_dim = int(embed_dim)
        self.layers = int(layers)
        self.epsilon = float(epsilon)
        self.register_buffer("counts", torch.zeros(layers, dtype=torch.float32))
        self.register_buffer("means", torch.zeros(layers, embed_dim, dtype=torch.float32))
        self.register_buffer("m2", torch.zeros(layers, embed_dim, dtype=torch.float32))

    def _validate(self, features: Sequence[torch.Tensor]) -> None:
        if not isinstance(features, (list, tuple)) or len(features) != self.layers:
            raise ValueError(f"features must contain {self.layers} layers")
        for index, feature in enumerate(features):
            if feature.ndim != 4 or feature.shape[1] != self.embed_dim:
                raise ValueError(
                    f"features[{index}] must have shape "
                    f"[batch, {self.embed_dim}, height, width]"
                )

    @torch.no_grad()
    def reset(self) -> None:
        self.counts.zero_()
        self.means.zero_()
        self.m2.zero_()

    @torch.no_grad()
    def update(self, features: Sequence[torch.Tensor]) -> None:
        self._validate(features)
        for index, feature in enumerate(features):
            values = (
                feature.detach()
                .permute(0, 2, 3, 1)
                .reshape(-1, self.embed_dim)
                .to(dtype=torch.float32)
            )
            batch_count = float(values.shape[0])
            batch_mean = values.mean(dim=0)
            batch_m2 = (values - batch_mean).square().sum(dim=0)
            current_count = self.counts[index]
            total_count = current_count + batch_count
            delta = batch_mean - self.means[index]
            self.means[index].add_(delta * (batch_count / total_count))
            self.m2[index].add_(
                batch_m2
                + delta.square() * current_count * batch_count / total_count
            )
            self.counts[index] = total_count

    def score(self, features: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        self._validate(features)
        if torch.any(self.counts < 2):
            raise RuntimeError("normal feature memory is not fitted")
        scores = []
        for index, feature in enumerate(features):
            mean = self.means[index].to(device=feature.device, dtype=feature.dtype)
            variance = (
                self.m2[index] / (self.counts[index] - 1)
            ).to(device=feature.device, dtype=feature.dtype)
            normalized = (
                feature - mean.view(1, -1, 1, 1)
            ).square() / variance.clamp_min(self.epsilon).view(1, -1, 1, 1)
            scores.append(normalized.mean(dim=1, keepdim=True).sqrt())
        return scores
