from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConditionalLayer(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden_channels = max(channels // 4, 16)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.residual = nn.Sequential(
            nn.Conv2d(channels * 2, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
        )

    def forward(self, main: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if context.shape[-2:] != main.shape[-2:]:
            context = F.interpolate(
                context,
                size=main.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        combined = torch.cat((main, context), dim=1)
        summary = torch.cat(
            (
                main.mean(dim=(-2, -1), keepdim=True),
                context.mean(dim=(-2, -1), keepdim=True),
            ),
            dim=1,
        )
        return main + self.gate(summary) * self.residual(combined)


class ConditionalFeatureFusion(nn.Module):
    """Condition local features on aligned multi-scale context features."""

    def __init__(self, embed_dim: int, layers: int) -> None:
        super().__init__()
        if isinstance(embed_dim, bool) or not isinstance(embed_dim, int) or embed_dim <= 0:
            raise ValueError("embed_dim must be a positive integer")
        if isinstance(layers, bool) or not isinstance(layers, int) or layers <= 0:
            raise ValueError("layers must be a positive integer")
        self.embed_dim = embed_dim
        self.layers = nn.ModuleList(
            _ConditionalLayer(embed_dim) for _ in range(layers)
        )

    def _validate(self, features: Sequence[torch.Tensor], name: str) -> None:
        if not isinstance(features, (list, tuple)) or len(features) != len(self.layers):
            raise ValueError(f"{name} must contain {len(self.layers)} feature layers")
        for index, feature in enumerate(features):
            if feature.ndim != 4 or feature.shape[1] != self.embed_dim:
                raise ValueError(
                    f"{name}[{index}] must have shape [batch, {self.embed_dim}, height, width]"
                )

    def forward(
        self,
        main_features: Sequence[torch.Tensor],
        context_features: Sequence[torch.Tensor] | None,
    ) -> list[torch.Tensor]:
        self._validate(main_features, "main_features")
        if context_features is None:
            return list(main_features)
        if len(context_features) != len(main_features):
            raise ValueError("main and context features must contain the same number of layers")
        self._validate(context_features, "context_features")
        outputs = []
        for index, (main, context, layer) in enumerate(
            zip(main_features, context_features, self.layers)
        ):
            if main.shape[:2] != context.shape[:2]:
                raise ValueError(
                    f"feature layer {index} has mismatched batch or channel dimensions"
                )
            outputs.append(layer(main, context))
        return outputs
