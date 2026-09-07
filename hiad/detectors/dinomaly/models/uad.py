"""Dinomaly2 非对称自编码重构模型。

与官方 ``Dinomaly2/models/uad.py`` 一致：冻结编码器在含 cls/register 前缀的
完整 token 序列上取中间层，两段式窄瓶颈 + 随机初始化解码器在整序列上重构，
随后按层分组输出 patch-token 的重构目标。教师侧做 context-aware recentering：
按组减去全局 cls 锚（整图缩略前向得到，或同前向自身 cls）并 LayerNorm。
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from hiad.models.dinov3 import fused_group_anchors

DINOMALY_TARGET_LAYERS = (2, 3, 4, 5, 6, 7, 8, 9)
DINOMALY_FUSE_LAYER_ENCODER = ((0, 1, 2, 3), (4, 5, 6, 7))
DINOMALY_FUSE_LAYER_DECODER = ((0, 1, 2, 3), (4, 5, 6, 7))


class Dinomaly(nn.Module):
    """冻结 DINOv3 教师 + 窄噪声瓶颈 + 学生解码器的重构检测器。

    Attributes:
        encoder (nn.Module): 返回每层完整 token 序列（含前缀）的冻结编码器。
        bottleneck (nn.ModuleList): 重建瓶颈模块序列。
        decoder (nn.ModuleList): 按层重建的学生 ViT 块。
        num_prefix_tokens (int): 编码器序列开头的 cls + register token 数。
        target_layers (tuple[int, ...]): 取中间层的编码器层号。
        fuse_layer_encoder (tuple[tuple[int, ...], ...]): 教师层分组。
        fuse_layer_decoder (tuple[tuple[int, ...], ...]): 学生层分组。
        fuse_layer_bottleneck (tuple[int, ...]): 进入瓶颈前融合的层下标。
    """

    def __init__(
        self,
        encoder: nn.Module,
        bottleneck: nn.ModuleList,
        decoder: nn.ModuleList,
        target_layers: Sequence[int] | None = None,
        fuse_layer_encoder: Sequence[Sequence[int]] | None = None,
        fuse_layer_decoder: Sequence[Sequence[int]] | None = None,
        fuse_layer_bottleneck: Sequence[int] | None = None,
        num_prefix_tokens: int | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = tuple(
            target_layers if target_layers is not None else DINOMALY_TARGET_LAYERS
        )
        self.fuse_layer_encoder = tuple(
            tuple(group) for group in (
                fuse_layer_encoder if fuse_layer_encoder is not None
                else DINOMALY_FUSE_LAYER_ENCODER
            )
        )
        self.fuse_layer_decoder = tuple(
            tuple(group) for group in (
                fuse_layer_decoder if fuse_layer_decoder is not None
                else DINOMALY_FUSE_LAYER_DECODER
            )
        )
        self.fuse_layer_bottleneck = tuple(
            fuse_layer_bottleneck if fuse_layer_bottleneck is not None
            else tuple(range(len(self.target_layers)))
        )
        if num_prefix_tokens is None:
            num_prefix_tokens = int(getattr(encoder, "num_prefix_tokens", 0))
        if num_prefix_tokens <= 0:
            raise ValueError("num_prefix_tokens must be a positive integer")
        self.num_prefix_tokens: int = num_prefix_tokens

    @torch.no_grad()
    def global_anchor(self, image: torch.Tensor) -> torch.Tensor:
        """编码一张缩略图并返回每组的 cls 锚。

        Args:
            image (torch.Tensor): ImageNet 标准化的 ``(batch, 3, height, width)``
                整图缩略输入。

        Returns:
            torch.Tensor: ``(batch, len(fuse_layer_encoder), embed_dim)`` 每图每组的
            cls 锚。
        """
        sequences = self.encoder(image)
        return fused_group_anchors(sequences, self.fuse_layer_encoder)

    def forward(
        self,
        x: torch.Tensor,
        global_anchor: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """在补丁（或整图）上执行重构，返回每组的教师/学生 patch-token 图。

        Args:
            x (torch.Tensor): ImageNet 标准化的 ``(batch, 3, height, width)``
                模型输入，宽高应能被 patch size 整除。
            global_anchor (torch.Tensor | None): 每组 recenter 锚，形状
                ``(batch, len(groups), embed_dim)`` 或 ``(len(groups), embed_dim)``；
                ``None`` 时使用本前向自身的组 cls（整图单次前向的官方语义）。

        Returns:
            tuple[list[torch.Tensor], list[torch.Tensor]]: ``(en, de)`` 各含
            ``len(fuse_layer_encoder)`` 个 ``(batch, embed_dim, side, side)``
            patch-token 图；``en`` 已 recenter + LayerNorm，``de`` 为学生原始输出。
        """
        if x.ndim != 4:
            raise ValueError(f"Expected a BCHW image, got shape {tuple(x.shape)}")

        with torch.no_grad():
            sequences = self.encoder(x)

        tokens = self.fuse_feature(
            [sequences[layer] for layer in self.fuse_layer_bottleneck]
        ).detach()
        for block in self.bottleneck:
            tokens = block(tokens)

        decoder_sequences = []
        for block in self.decoder:
            tokens = block(tokens)
            decoder_sequences.append(tokens)
        decoder_sequences = decoder_sequences[::-1]

        if global_anchor is None:
            with torch.no_grad():
                global_anchor = fused_group_anchors(
                    sequences, self.fuse_layer_encoder
                )
        elif global_anchor.ndim == 2:
            global_anchor = global_anchor.unsqueeze(0)
        if (
            global_anchor.ndim != 3
            or global_anchor.shape[1] != len(self.fuse_layer_encoder)
        ):
            raise ValueError(
                "global_anchor must be (batch, groups, embed_dim), got "
                f"{tuple(global_anchor.shape)}"
            )

        encoder_groups = [
            self.fuse_feature([sequences[layer] for layer in group])
            for group in self.fuse_layer_encoder
        ]
        decoder_groups = [
            self.fuse_feature([decoder_sequences[layer] for layer in group])
            for group in self.fuse_layer_decoder
        ]

        en_maps: list[torch.Tensor] = []
        de_maps: list[torch.Tensor] = []
        for group_index, (encoder_group, decoder_group) in enumerate(
            zip(encoder_groups, decoder_groups)
        ):
            encoder_patches = encoder_group[:, self.num_prefix_tokens:, :]
            decoder_patches = decoder_group[:, self.num_prefix_tokens:, :]
            encoder_patches = (
                encoder_patches
                - global_anchor[:, group_index, :].unsqueeze(1)
            )
            encoder_patches = F.layer_norm(
                encoder_patches,
                normalized_shape=(encoder_patches.shape[-1],),
                eps=1e-8,
            )
            batch, _, embed_dim = encoder_patches.shape
            side = math.isqrt(encoder_patches.shape[1])
            if side * side != encoder_patches.shape[1]:
                raise ValueError(
                    "Patch token count must be a perfect square for grid reshape"
                )
            en_maps.append(
                encoder_patches.permute(0, 2, 1)
                .reshape(batch, embed_dim, side, side)
                .contiguous()
            )
            de_maps.append(
                decoder_patches.permute(0, 2, 1)
                .reshape(batch, embed_dim, side, side)
                .contiguous()
            )
        return en_maps, de_maps

    @staticmethod
    def fuse_feature(features: Sequence[torch.Tensor]) -> torch.Tensor:
        """沿层维对完整序列做均值融合。"""
        return torch.stack(list(features), dim=1).mean(dim=1)
