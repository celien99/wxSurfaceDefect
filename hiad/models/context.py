from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConditionalLayer(nn.Module):
    """用单层上下文门控残差修正主特征图。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden_channels = max(channels // 4, 16)
        self.gate: nn.Sequential = nn.Sequential(
            nn.Conv2d(channels * 2, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.residual: nn.Sequential = nn.Sequential(
            nn.Conv2d(channels * 2, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
        )

    def forward(self, main: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """对齐上下文空间尺寸并输出与主特征同形状的融合结果。

        Args:
            main (torch.Tensor): ``(batch, channels, height, width)`` 主补丁特征。
            context (torch.Tensor): 批量和通道相同、空间尺寸可不同的上下文特征。

        Returns:
            torch.Tensor: 与 ``main`` 形状一致的门控残差融合特征。
        """
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
    """用对齐的多尺度上下文对局部特征做逐层条件化融合。

    Attributes:
        embed_dim (int): 每层特征的通道数。
        layers (nn.ModuleList): 与编码器特征层一一对应的融合层。
    """

    def __init__(self, embed_dim: int, layers: int) -> None:
        super().__init__()
        if isinstance(embed_dim, bool) or not isinstance(embed_dim, int) or embed_dim <= 0:
            raise ValueError("embed_dim must be a positive integer")
        if isinstance(layers, bool) or not isinstance(layers, int) or layers <= 0:
            raise ValueError("layers must be a positive integer")
        self.embed_dim: int = embed_dim
        self.layers: nn.ModuleList = nn.ModuleList(
            _ConditionalLayer(embed_dim) for _ in range(layers)
        )

    def _validate(self, features: Sequence[torch.Tensor], name: str) -> None:
        """校验特征层数量及每层 BCHW 通道数。

        Args:
            features (Sequence[torch.Tensor]): 待校验的多层特征。
            name (str): 用于错误消息的参数名称。

        Raises:
            ValueError: 特征不是列表/元组、层数错误或任一层不是配置通道的 BCHW。
        """
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
        """逐层融合主补丁和上下文特征。

        Args:
            main_features (Sequence[torch.Tensor]): 每项为
                ``(batch, embed_dim, height, width)`` 的主补丁特征。
            context_features (Sequence[torch.Tensor] | None): 层数、批量和通道与
                主特征一致的上下文特征；空间尺寸可不同，``None`` 表示跳过融合。

        Returns:
            list[torch.Tensor]: 与主特征层数和各层形状一致的融合特征。

        Raises:
            ValueError: 层数、维度、批量或通道不匹配。
        """
        self._validate(main_features, "main_features")
        if context_features is None:
            return list(main_features)
        if len(context_features) != len(main_features):
            raise ValueError("main and context features must contain the same number of layers")
        self._validate(context_features, "context_features")
        outputs: list[torch.Tensor] = []
        for index, (main, context, layer) in enumerate(
            zip(main_features, context_features, self.layers)
        ):
            if main.shape[:2] != context.shape[:2]:
                raise ValueError(
                    f"feature layer {index} has mismatched batch or channel dimensions"
                )
            outputs.append(layer(main, context))
        return outputs
