from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class NormalFeatureMemory(nn.Module):
    """以流式对角高斯统计描述正常样本的多层特征 token。

    Attributes:
        embed_dim (int): 每层特征通道数。
        layers (int): 参与统计的特征层数。
        epsilon (float): 计算标准化距离时的最小方差。
        counts (torch.Tensor): ``(layers,)`` 每层累计 token 数。
        means (torch.Tensor): ``(layers, embed_dim)`` 每层逐通道均值。
        m2 (torch.Tensor): ``(layers, embed_dim)`` Welford 二阶中心矩累积量。
    """

    counts: torch.Tensor
    means: torch.Tensor
    m2: torch.Tensor

    def __init__(self, embed_dim: int, layers: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        if embed_dim <= 0 or layers <= 0:
            raise ValueError("embed_dim and layers must be positive")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.embed_dim: int = int(embed_dim)
        self.layers: int = int(layers)
        self.epsilon: float = float(epsilon)
        self.register_buffer("counts", torch.zeros(layers, dtype=torch.float32))
        self.register_buffer("means", torch.zeros(layers, embed_dim, dtype=torch.float32))
        self.register_buffer("m2", torch.zeros(layers, embed_dim, dtype=torch.float32))

    def _validate(self, features: Sequence[torch.Tensor]) -> None:
        """校验特征层数量及每层 BCHW 通道数。

        Args:
            features (Sequence[torch.Tensor]): 待更新或打分的多层特征。

        Raises:
            ValueError: 特征不是列表/元组、层数错误或任一层不是配置通道的 BCHW。
        """
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
        """原地清空所有层的 token 数、均值和二阶矩统计。"""
        self.counts.zero_()
        self.means.zero_()
        self.m2.zero_()

    @torch.no_grad()
    def update(self, features: Sequence[torch.Tensor]) -> None:
        """用一批正常特征流式更新逐层对角高斯统计。

        Args:
            features (Sequence[torch.Tensor]): 每项为
                ``(batch, embed_dim, height, width)`` 的正常特征；批量和空间维合并为
                独立 token。

        Raises:
            ValueError: 特征层数、维度或通道数不符合内存配置。
        """
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
        """计算各 token 到正常对角高斯分布的标准化距离。

        Args:
            features (Sequence[torch.Tensor]): 与内存层数和通道一致的 BCHW 特征。

        Returns:
            list[torch.Tensor]: 每层一个 ``(batch, 1, height, width)`` 异常距离图。

        Raises:
            ValueError: 特征层数、维度或通道不匹配。
            RuntimeError: 任一层累计 token 少于两个，尚不能估计样本方差。
        """
        self._validate(features)
        if torch.any(self.counts < 2):
            raise RuntimeError("normal feature memory is not fitted")
        scores: list[torch.Tensor] = []
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
