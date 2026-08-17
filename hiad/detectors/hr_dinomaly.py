from __future__ import annotations

import logging
import os
import time
from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any, TypeAlias, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from hiad.models import (
    ConditionalFeatureFusion,
    NormalFeatureMemory,
    TimmDinoV3Encoder,
)
from hiad.runtime.contracts import DetectorPrediction
from hiad.runtime.evidence import (
    denormalize_imagenet_batch,
    fuse_evidence_tensors,
    high_frequency_map,
)

from .base import BaseDetector
from .dinomaly.models.uad import ViTill
from .dinomaly.models.vision_transformer import Block as VitBlock
from .dinomaly.models.vision_transformer import LinearAttention2, bMlp
from .dinomaly.optimizers import StableAdamW
from .dinomaly.utils import WarmCosineScheduler, global_cosine_hm_percent


RunningMoments: TypeAlias = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
FeatureLayers: TypeAlias = Sequence[torch.Tensor]
DetectorBatch: TypeAlias = Mapping[str, Any]


def _positive_int(value: object, name: str) -> int:
    """校验检测器参数为正整数且不是布尔值。

    Args:
        value (object): 待校验参数。
        name (str): 用于错误消息的参数名。

    Returns:
        int: 通过校验的原整数。

    Raises:
        ValueError: 值不是正整数或是布尔值。
    """
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _empty_moments(device: torch.device) -> RunningMoments:
    """在目标设备创建 ``(count, mean, m2)`` 流式统计初值。

    Args:
        device (torch.device): 标量统计张量所在设备。

    Returns:
        RunningMoments: 三个相互独立的 ``float32`` 零维零张量。
    """
    zero = torch.zeros((), dtype=torch.float32, device=device)
    return zero.clone(), zero.clone(), zero.clone()


def _update_moments(
    moments: RunningMoments,
    values: torch.Tensor,
) -> RunningMoments:
    """用一批张量元素通过并行 Welford 公式更新全局矩统计。

    Args:
        moments (RunningMoments): 当前元素数、均值和二阶中心矩累积量。
        values (torch.Tensor): 任意形状的新证据值，展平后按 ``float32`` 统计。

    Returns:
        RunningMoments: 更新后的元素数、均值和二阶中心矩。
    """
    count, mean, m2 = moments
    values = values.reshape(-1).to(dtype=torch.float32)
    batch_count = torch.as_tensor(values.numel(), dtype=torch.float32, device=values.device)
    batch_mean = values.mean()
    batch_m2 = (values - batch_mean).square().sum()
    total = count + batch_count
    delta = batch_mean - mean
    return (
        total,
        mean + delta * batch_count / total,
        m2 + batch_m2 + delta.square() * count * batch_count / total,
    )


def _finalize_moments(
    moments: RunningMoments,
    name: str,
) -> tuple[float, float]:
    """把流式矩转换为证据标准化使用的均值和样本标准差。

    Args:
        moments (RunningMoments): 已累计的元素数、均值和二阶中心矩。
        name (str): 用于异常消息的证据分支名称。

    Returns:
        tuple[float, float]: Python 浮点均值和不小于 ``1e-6`` 的标准差。

    Raises:
        ValueError: 累计值少于两个，无法估计样本标准差。
    """
    count, mean, m2 = moments
    if count < 2:
        raise ValueError(f"{name} fitting requires at least two values")
    scale = torch.sqrt(m2 / (count - 1)).clamp_min(1e-6)
    return float(mean.item()), float(scale.item())


class HRDinomaly(BaseDetector):
    """DINOv3 + Dinomaly 高分辨率检测器及其多证据融合实现。

    Args:
        backbone_name (str): ``timm`` DINOv3 主干名称。
        total_iters (int): 配置训练迭代数；实际至少覆盖一个完整采样轮次。
        log_per_steps (int): 训练状态日志间隔。
        patch_size (int): 正方形模型输入边长，单位为像素。
        logger (logging.Logger | None): 可选任务日志器。
        device (torch.device): 模型驻留设备。
        seed (int): 检测器随机种子。
        bottleneck_dropout (float): 瓶颈 dropout，范围 ``[0, 1)``。
        grad_clip_norm (float): 梯度范数裁剪上限；``0`` 表示禁用。
        hard_mining_final (float): 困难 token 挖掘最终比例，范围 ``[0, 1]``。
        hard_mining_warmup_iters (int): 困难挖掘线性预热迭代数。
        easy_grad_factor (float): 易 token 梯度因子，范围 ``[0, 1]``。
        score_top_k (int): 图像分数聚合的最高异常 token 数。
        encoder_amp (bool): CUDA 编码器是否使用 FP16 autocast。
        decoder_amp (bool): CUDA 重建训练是否使用 FP16 和 GradScaler。
        allow_tf32 (bool): 是否允许 CUDA TF32 计算。
        semantic_weight (float): 语义重建证据权重。
        memory_weight (float): 正常特征记忆证据权重。
        high_frequency_weight (float): 高频纹理证据权重。
        use_context_conditioning (bool): 是否用多尺度上下文条件化主补丁特征。

    Attributes:
        encoder (TimmDinoV3Encoder): 冻结的多层 DINOv3 编码器。
        context_conditioner (ConditionalFeatureFusion | None): 可训练上下文融合模块。
        feature_memory (NormalFeatureMemory): 正常特征对角高斯记忆。
        bottleneck (nn.ModuleList): Dinomaly 可训练瓶颈。
        decoder (nn.ModuleList): Dinomaly 可训练重建解码块。
        model (ViTill): 组合编码器、瓶颈和解码器的重建模型。
        evidence_weights (tuple[float, float, float]): 语义、记忆和高频分支权重。
    """

    def __init__(
        self,
        backbone_name: str,
        total_iters: int,
        log_per_steps: int,
        patch_size: int,
        logger: logging.Logger | None,
        device: torch.device,
        seed: int = 0,
        bottleneck_dropout: float = 0.1,
        grad_clip_norm: float = 1.0,
        hard_mining_final: float = 0.0,
        hard_mining_warmup_iters: int = 1000,
        easy_grad_factor: float = 0.1,
        score_top_k: int = 4,
        encoder_amp: bool = True,
        decoder_amp: bool = True,
        allow_tf32: bool = True,
        semantic_weight: float = 0.6,
        memory_weight: float = 0.3,
        high_frequency_weight: float = 0.1,
        use_context_conditioning: bool = True,
        **_: object,
    ) -> None:
        total_iters = _positive_int(total_iters, "total_iters")
        log_per_steps = _positive_int(log_per_steps, "log_per_steps")
        super().__init__(patch_size, device, logger, seed)

        bottleneck_dropout = float(bottleneck_dropout)
        grad_clip_norm = float(grad_clip_norm)
        hard_mining_final = float(hard_mining_final)
        easy_grad_factor = float(easy_grad_factor)
        if not 0 <= bottleneck_dropout < 1:
            raise ValueError(f"bottleneck_dropout must be in [0, 1), got {bottleneck_dropout}")
        if not np.isfinite(grad_clip_norm):
            raise ValueError(f"grad_clip_norm must be finite, got {grad_clip_norm}")
        if not 0 <= hard_mining_final <= 1:
            raise ValueError(f"hard_mining_final must be in [0, 1], got {hard_mining_final}")
        if (
            isinstance(hard_mining_warmup_iters, bool)
            or not isinstance(hard_mining_warmup_iters, int)
            or hard_mining_warmup_iters < 0
        ):
            raise ValueError(
                "hard_mining_warmup_iters must be a non-negative integer, "
                f"got {hard_mining_warmup_iters}"
            )
        if not 0 <= easy_grad_factor <= 1:
            raise ValueError(f"easy_grad_factor must be in [0, 1], got {easy_grad_factor}")
        if not isinstance(encoder_amp, bool):
            raise TypeError("encoder_amp must be a boolean")
        if not isinstance(decoder_amp, bool):
            raise TypeError("decoder_amp must be a boolean")
        if not isinstance(allow_tf32, bool):
            raise TypeError("allow_tf32 must be a boolean")
        if not isinstance(use_context_conditioning, bool):
            raise TypeError("use_context_conditioning must be a boolean")
        evidence_weights = (
            float(semantic_weight),
            float(memory_weight),
            float(high_frequency_weight),
        )
        if any(not np.isfinite(weight) or weight < 0 for weight in evidence_weights):
            raise ValueError("evidence weights must be finite and non-negative")
        if sum(evidence_weights) <= 0:
            raise ValueError("at least one evidence weight must be positive")
        score_top_k = _positive_int(score_top_k, "score_top_k")
        self.total_iters: int = total_iters
        self.grad_clip_norm: float = grad_clip_norm
        self.hard_mining_final: float = hard_mining_final
        self.hard_mining_warmup_iters: int = hard_mining_warmup_iters
        self.easy_grad_factor: float = easy_grad_factor
        self.score_top_k: int = score_top_k
        self.encoder_amp: bool = encoder_amp
        self.decoder_amp: bool = decoder_amp
        self.allow_tf32: bool = allow_tf32
        self.evidence_weights: tuple[float, float, float] = evidence_weights
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            torch.backends.cudnn.allow_tf32 = allow_tf32
            torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
        self.target_layers: list[int] = [2, 3, 4, 5, 6, 7, 8, 9]
        self.fuse_layer_encoder: list[list[int]] = [[0, 1, 2, 3], [4, 5, 6, 7]]
        self.fuse_layer_decoder: list[list[int]] = [[0, 1, 2, 3], [4, 5, 6, 7]]
        self.encoder: TimmDinoV3Encoder = TimmDinoV3Encoder(
            model_name=backbone_name,
            intermediate_layers=self.target_layers,
            use_fp16=self.encoder_amp,
        )
        embed_dim = self.encoder.embed_dim
        if embed_dim == 384:
            num_heads = 6
        elif embed_dim == 768:
            num_heads = 12
        elif embed_dim == 1024:
            num_heads = 16
        else:
            raise ValueError(f"Unsupported DINOv3 embedding dimension: {embed_dim}")

        self.context_conditioner: ConditionalFeatureFusion | None = (
            ConditionalFeatureFusion(
                embed_dim=embed_dim,
                layers=len(self.target_layers),
            )
            if use_context_conditioning
            else None
        )
        self.feature_memory: NormalFeatureMemory = NormalFeatureMemory(
            embed_dim=embed_dim,
            layers=len(self.target_layers),
        )
        self.high_frequency_center: float = 0.0
        self.high_frequency_scale: float = 1.0
        self.semantic_center: float = 0.0
        self.semantic_scale: float = 1.0
        self.memory_center: float = 0.0
        self.memory_scale: float = 1.0

        self.bottleneck: nn.ModuleList = nn.ModuleList(
            [bMlp(embed_dim, embed_dim * 4, embed_dim, drop=bottleneck_dropout)]
        )

        decoder_blocks: list[nn.Module] = []
        for _ in range(8):
            blk = VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                           qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),

                           attn=LinearAttention2)

            decoder_blocks.append(blk)
        self.decoder: nn.ModuleList = nn.ModuleList(decoder_blocks)

        self.model: ViTill = ViTill(
            encoder=self.encoder,
            bottleneck=self.bottleneck,
            decoder=self.decoder,
            fuse_layer_encoder=self.fuse_layer_encoder,
            fuse_layer_decoder=self.fuse_layer_decoder,
        )
        self.to_device(device)
        self.log_per_steps: int = log_per_steps

    @torch.no_grad()
    def embedding(self, input_tensor: torch.Tensor) -> list[torch.Tensor]:
        """在检测器设备上提取冻结编码器的多层 BCHW 特征。

        Args:
            input_tensor (torch.Tensor): ImageNet 标准化的 BCHW RGB 输入。

        Returns:
            list[torch.Tensor]: 八个目标层的 ``float32`` BCHW 特征。
        """
        return self.model.encoder_image(input_tensor.to(self.device))

    def to_device(self, device: torch.device) -> None:
        """把重建模型、上下文模块和正常特征内存移动到目标设备。

        Args:
            device (torch.device): 目标 CPU 或 CUDA 设备。
        """
        self.model = self.model.to(device)
        if self.context_conditioner is not None:
            self.context_conditioner = self.context_conditioner.to(device)
        self.feature_memory = self.feature_memory.to(device)
        self.device = device

    @torch.no_grad()
    def fit_normal_evidence(self, train_dataloader: DataLoader[Any]) -> None:
        """仅用正常补丁拟合语义、记忆和高频证据的标准化参数。

        Args:
            train_dataloader (DataLoader[Any]): 产生主补丁及可选多尺度上下文的
                正常训练加载器。

        Raises:
            ValueError: 任一证据分支累计值少于两个。
            RuntimeError: 正常特征内存未成功拟合或模型前向失败。

        Notes:
            第一遍同时拟合正常特征内存、语义和高频统计；第二遍在固定内存上
            计算记忆距离统计，避免边更新边打分造成分布漂移。
        """
        self.model.eval()
        if self.context_conditioner is not None:
            self.context_conditioner.eval()
        self.feature_memory.reset()
        semantic_moments = _empty_moments(self.device)
        frequency_moments = _empty_moments(self.device)
        for data in train_dataloader:
            main_features, context_features = self.get_multi_resolution_embeddings(data)
            conditioned = self._condition_features(main_features, context_features)
            self.feature_memory.update(conditioned)

            semantic_encoder, semantic_decoder = self.model.distillation(
                list(conditioned)
            )
            semantic_token = torch.cat(
                self._layer_anomaly_token_maps(semantic_encoder, semantic_decoder),
                dim=1,
            ).amax(dim=1, keepdim=True)
            semantic_moments = _update_moments(semantic_moments, semantic_token)

            image = data["image"].to(self.device, non_blocking=True)
            frequency_moments = _update_moments(
                frequency_moments,
                high_frequency_map(denormalize_imagenet_batch(image)),
            )

        memory_moments = _empty_moments(self.device)
        for data in train_dataloader:
            main_features, context_features = self.get_multi_resolution_embeddings(data)
            conditioned = self._condition_features(main_features, context_features)
            semantic_encoder, semantic_decoder = self.model.distillation(
                list(conditioned)
            )
            semantic_token = torch.cat(
                self._layer_anomaly_token_maps(semantic_encoder, semantic_decoder),
                dim=1,
            ).amax(dim=1, keepdim=True)
            memory_token = self._memory_token_map(conditioned, semantic_token.shape[-2:])
            memory_moments = _update_moments(memory_moments, memory_token)

        self.semantic_center, self.semantic_scale = _finalize_moments(
            semantic_moments,
            "semantic evidence",
        )
        self.memory_center, self.memory_scale = _finalize_moments(
            memory_moments,
            "memory evidence",
        )
        self.high_frequency_center, self.high_frequency_scale = _finalize_moments(
            frequency_moments,
            "high-frequency evidence",
        )

    def _memory_token_map(
        self,
        features: FeatureLayers,
        output_size: Sequence[int],
    ) -> torch.Tensor:
        """把多层正常记忆距离对齐并聚合为单通道 token 图。

        Args:
            features (FeatureLayers): 与正常特征内存层数和通道一致的 BCHW 特征。
            output_size (Sequence[int]): 输出 ``(height, width)``。

        Returns:
            torch.Tensor: ``(batch, 1, height, width)`` 的最大层记忆距离。

        Raises:
            RuntimeError: 正常特征内存尚未拟合。
            ValueError: 特征层契约不匹配。
        """
        memory_layers = self.feature_memory.score(features)
        return torch.cat(
            [
                F.interpolate(
                    layer,
                    size=output_size,
                    mode="bilinear",
                    align_corners=False,
                )
                for layer in memory_layers
            ],
            dim=1,
        ).amax(dim=1, keepdim=True)

    @staticmethod
    def _positive_normalize(
        values: torch.Tensor,
        center: float,
        scale: float,
    ) -> torch.Tensor:
        """把证据转为只保留高于正常中心的非负标准分数。

        Args:
            values (torch.Tensor): 任意形状证据张量。
            center (float): 正常证据均值。
            scale (float): 正常证据标准差，必须已拟合为正数。

        Returns:
            torch.Tensor: 与输入同形状的截断标准分数。
        """
        return ((values - center) / scale).clamp_min(0.0)

    def _condition_features(
        self,
        main_features: FeatureLayers,
        context_features: FeatureLayers | None,
    ) -> list[torch.Tensor]:
        """按当前任务配置决定是否用上下文条件化主特征。

        Args:
            main_features (FeatureLayers): 主补丁多层 BCHW 特征。
            context_features (FeatureLayers | None): 对齐后的多层上下文特征。

        Returns:
            list[torch.Tensor]: 条件化结果；模块禁用时返回主特征的列表副本。
        """
        if self.context_conditioner is None:
            return list(main_features)
        return self.context_conditioner(main_features, context_features)

    def _fused_evidence(
        self,
        data: DetectorBatch,
        memory_features: FeatureLayers,
        semantic_encoder_features: FeatureLayers,
        semantic_decoder_features: FeatureLayers,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """计算并融合语义、正常记忆和高频纹理三类异常证据。

        Args:
            data (DetectorBatch): 含 ImageNet 标准化 BCHW ``image`` 的批次。
            memory_features (FeatureLayers): 用于正常特征内存打分的条件化特征。
            semantic_encoder_features (FeatureLayers): 重建蒸馏的编码器特征。
            semantic_decoder_features (FeatureLayers): 与编码器逐层对应的解码特征。

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 模型输入分辨率的
            ``(batch, 1, patch_height, patch_width)`` 像素证据，以及编码器 token
            分辨率的 ``(batch, 1, token_height, token_width)`` 证据。

        Raises:
            KeyError: 批次缺少 ``image``。
            ValueError: 特征层或证据权重不满足融合契约。
            RuntimeError: 正常特征内存尚未拟合。
        """
        semantic_pixel, semantic_token = self.cal_anomaly_maps(
            semantic_encoder_features,
            semantic_decoder_features,
            self.patch_size,
        )
        semantic_pixel = self._positive_normalize(
            semantic_pixel, self.semantic_center, self.semantic_scale
        )
        semantic_token = self._positive_normalize(
            semantic_token, self.semantic_center, self.semantic_scale
        )
        memory_token = self._memory_token_map(
            memory_features,
            semantic_token.shape[-2:],
        )
        memory_token = self._positive_normalize(
            memory_token, self.memory_center, self.memory_scale
        )
        memory_pixel = F.interpolate(
            memory_token,
            size=semantic_pixel.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        image = data["image"].to(self.device, non_blocking=True)
        frequency_pixel = (
            (
                high_frequency_map(denormalize_imagenet_batch(image))
                - self.high_frequency_center
            )
            / self.high_frequency_scale
        ).clamp_min(0.0)
        if frequency_pixel.shape[-2:] != semantic_pixel.shape[-2:]:
            frequency_pixel = F.interpolate(
                frequency_pixel,
                size=semantic_pixel.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        frequency_token = F.adaptive_avg_pool2d(
            frequency_pixel,
            output_size=semantic_token.shape[-2:],
        )
        fused_pixel = fuse_evidence_tensors(
            [semantic_pixel, memory_pixel, frequency_pixel],
            self.evidence_weights,
        )
        fused_token = fuse_evidence_tensors(
            [semantic_token, memory_token, frequency_token],
            self.evidence_weights,
        )
        return fused_pixel, fused_token

    def train_step(
        self,
        train_dataloader: DataLoader[Any],
        task_name: str,
    ) -> None:
        """训练当前任务的瓶颈、解码器及可选上下文融合模块。

        Args:
            train_dataloader (DataLoader[Any]): 按源图公平采样的正常补丁加载器。
            task_name (str): 用于日志和数值异常消息的任务名称。

        Raises:
            FloatingPointError: 损失或裁剪前梯度出现非有限值。
            RuntimeError: 模型前向、反向或优化器步骤失败。

        Notes:
            冻结 DINOv3 编码器；实际迭代数取配置值与一个完整采样轮次中的较大值，
            从而保证每张正常原图至少为当前任务贡献过补丁。
        """
        trainable_modules: list[nn.Module] = [self.bottleneck, self.decoder]
        if self.context_conditioner is not None:
            trainable_modules.insert(0, self.context_conditioner)
        trainable = nn.ModuleList(trainable_modules)

        for m in trainable.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        optimizer = StableAdamW([{'params': trainable.parameters()}], lr=2e-3,
                                betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=False, eps=1e-10)
        # 至少跑完首个采样轮次，确保每张正常原图都为当前任务提供过补丁。
        batches_per_epoch = len(train_dataloader)
        training_iters = max(self.total_iters, batches_per_epoch)
        warmup_iters = min(100, max(training_iters - 1, 0))
        lr_scheduler = WarmCosineScheduler(
            optimizer,
            base_value=2e-3,
            final_value=2e-4,
            total_iters=training_iters,
            warmup_iters=warmup_iters,
        )
        if self.logger is not None:
            self.logger.info(
                "Task %s effective training iterations: %d "
                "(configured=%d, sampled_epoch_batches=%d)",
                task_name,
                training_iters,
                self.total_iters,
                len(train_dataloader),
            )

        decoder_amp_enabled = self.decoder_amp and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=decoder_amp_enabled)
        self.model.train()
        self.model.encoder.eval()
        if self.context_conditioner is not None:
            self.context_conditioner.train()
        it = 0
        step_started = time.perf_counter()
        for _epoch in range(int(np.ceil(training_iters / batches_per_epoch))):
            for data in train_dataloader:
                if it >= training_iters:
                    break

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=decoder_amp_enabled,
                ):
                    main_features, context_features = self.get_multi_resolution_embeddings(data)
                    en = self._condition_features(main_features, context_features)
                    en, de = self.model.distillation(en)

                    if self.hard_mining_warmup_iters == 0:
                        p = self.hard_mining_final
                    else:
                        p = min(
                            self.hard_mining_final * it / self.hard_mining_warmup_iters,
                            self.hard_mining_final,
                        )
                    loss = global_cosine_hm_percent(
                        en,
                        de,
                        p=p,
                        factor=self.easy_grad_factor,
                    )

                if not torch.isfinite(loss).all():
                    raise FloatingPointError(
                        f"Task {task_name} produced a non-finite loss at iteration {it + 1}"
                    )
                if decoder_amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                grad_norm = None
                if self.grad_clip_norm > 0:
                    try:
                        grad_norm = nn.utils.clip_grad_norm_(
                            trainable.parameters(),
                            max_norm=self.grad_clip_norm,
                            error_if_nonfinite=True,
                        )
                    except RuntimeError as error:
                        raise FloatingPointError(
                            f"Task {task_name} produced non-finite gradients at iteration {it + 1}"
                        ) from error

                if decoder_amp_enabled:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                lr_scheduler.step()
                it += 1

                if it == 1 or it % self.log_per_steps == 0 or it == training_iters:
                    elapsed = time.perf_counter() - step_started
                    step_time = elapsed / it
                    log_message = "iter [{}/{}], loss:{:.4f}, avg_step_sec:{:.2f}".format(
                        it, training_iters, loss.item(), step_time
                    )
                    if grad_norm is not None:
                        log_message += ", grad_norm:{:.4f}".format(grad_norm.item())
                    if self.logger is not None:
                        self.logger.info(log_message)

                if it >= training_iters:
                    break

    @torch.no_grad()
    def inference_step(
        self,
        test_dataloader: DataLoader[Any],
    ) -> list[DetectorPrediction]:
        """按 DataLoader 顺序生成融合异常图和 Top-K token 图像分数。

        Args:
            test_dataloader (DataLoader[Any]): 不打乱的模型输入批次加载器。

        Returns:
            list[DetectorPrediction]: 每个输入对应一个模型输入分辨率二维
            ``float32`` 异常图和标量 Top-K token 分数。

        Raises:
            RuntimeError: 正常证据或特征内存未从检查点恢复。
            ValueError: 输入批次、特征层或 token 图形状不符合契约。
        """
        self.model.eval()
        if self.context_conditioner is not None:
            self.context_conditioner.eval()
        predictions: list[DetectorPrediction] = []
        for data in test_dataloader:
            main_features, context_features = self.get_multi_resolution_embeddings(data)
            conditioned_features = self._condition_features(
                main_features,
                context_features,
            )
            semantic_encoder, semantic_decoder = self.model.distillation(
                list(conditioned_features)
            )
            anomaly_map, token_map = self._fused_evidence(
                data,
                conditioned_features,
                semantic_encoder,
                semantic_decoder,
            )
            pixel_batch = anomaly_map[:, 0].cpu().numpy()
            score_batch = self._top_k_token_scores(token_map, self.score_top_k).cpu().numpy()
            predictions.extend(
                {
                    "anomaly_map": pixel_map,
                    "score": float(score),
                }
                for pixel_map, score in zip(
                    pixel_batch,
                    score_batch,
                )
            )
        return predictions

    @staticmethod
    def _layer_anomaly_token_maps(
        encoder_features: FeatureLayers,
        decoder_features: FeatureLayers,
    ) -> list[torch.Tensor]:
        """计算逐层编码/重建特征的非负余弦距离 token 图。

        Args:
            encoder_features (FeatureLayers): 非空多层 BCHW 编码特征。
            decoder_features (FeatureLayers): 数量和各层形状对应的重建特征。

        Returns:
            list[torch.Tensor]: 每层一个 ``(batch, 1, height, width)``、范围
            ``[0, 2]`` 的余弦距离图。

        Raises:
            ValueError: 特征列表为空或层数不一致。
        """
        if not encoder_features or len(encoder_features) != len(decoder_features):
            raise ValueError("Encoder and decoder feature lists must be non-empty and aligned")
        return [
            torch.clamp(
                1 - F.cosine_similarity(encoder_feature.float(), decoder_feature.float()),
                min=0.0,
                max=2.0,
            ).unsqueeze(1)
            for encoder_feature, decoder_feature in zip(encoder_features, decoder_features)
        ]

    def cal_anomaly_maps(
        self,
        encoder_features: FeatureLayers,
        decoder_features: FeatureLayers,
        output_size: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """把逐层语义 token 距离聚合为 token 图和模型像素图。

        Args:
            encoder_features (FeatureLayers): 多层编码器 BCHW 特征。
            decoder_features (FeatureLayers): 对应的多层重建 BCHW 特征。
            output_size (Sequence[int]): 目标模型输入 ``(width, height)``。

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(batch, 1, height, width)`` 像素图，
            以及编码器空间尺寸的单通道 token 图；两者均跨层取最大响应。
        """
        token_layer_maps = self._layer_anomaly_token_maps(
            encoder_features,
            decoder_features,
        )
        token_map = torch.cat(token_layer_maps, dim=1).amax(dim=1, keepdim=True)
        pixel_layer_maps = [
            F.interpolate(
                token_layer_map,
                size=(output_size[1], output_size[0]),
                mode="bilinear",
                align_corners=True,
            )
            for token_layer_map in token_layer_maps
        ]
        anomaly_map = torch.cat(pixel_layer_maps, dim=1).amax(dim=1, keepdim=True)
        return anomaly_map, token_map

    @staticmethod
    def _top_k_token_scores(token_maps: torch.Tensor, top_k: int) -> torch.Tensor:
        """按样本聚合最高异常 token 的均值。

        Args:
            token_maps (torch.Tensor): ``(batch, 1, height, width)`` token 异常图。
            top_k (int): 参与均值的最高分 token 数；超过总数时使用全部 token。

        Returns:
            torch.Tensor: ``(batch,)`` 图像级异常分数。

        Raises:
            ValueError: ``top_k`` 不是正整数或 token 图不是单通道 BCHW。
        """
        top_k = _positive_int(top_k, "top_k")
        if token_maps.ndim != 4 or token_maps.shape[1] != 1:
            raise ValueError("token_maps must have shape [batch, 1, height, width]")
        values = token_maps.flatten(start_dim=1)
        count = min(top_k, values.shape[1])
        return torch.topk(values, k=count, dim=1).values.mean(dim=1)

    def save_checkpoint(
        self,
        checkpoint_path: str | os.PathLike[str],
    ) -> None:
        """保存当前任务推理所需的可训练模块、正常内存和证据统计。

        Args:
            checkpoint_path (str | os.PathLike[str]): 目标 PyTorch 检查点路径。

        Raises:
            OSError: 检查点无法写入。
        """
        state: dict[str, object] = {
            "feature_memory": self.feature_memory.state_dict(),
            "bottleneck": self.bottleneck.state_dict(),
            "decoder": self.decoder.state_dict(),
            "high_frequency_center": self.high_frequency_center,
            "high_frequency_scale": self.high_frequency_scale,
            "semantic_center": self.semantic_center,
            "semantic_scale": self.semantic_scale,
            "memory_center": self.memory_center,
            "memory_scale": self.memory_scale,
        }
        if self.context_conditioner is not None:
            state["context_conditioner"] = self.context_conditioner.state_dict()
        torch.save(state, checkpoint_path)

    def load_checkpoint(
        self,
        checkpoint_path: str | os.PathLike[str],
    ) -> None:
        """从检查点恢复当前任务推理状态。

        Args:
            checkpoint_path (str | os.PathLike[str]): 与当前模型结构匹配的检查点。

        Raises:
            OSError: 检查点无法读取。
            KeyError: 检查点缺少当前结构必需的模块或证据字段。
            RuntimeError: 模块参数形状与当前模型不兼容。
        """
        state_dict = cast(
            dict[str, Any],
            torch.load(checkpoint_path, map_location=self.device),
        )
        if self.context_conditioner is not None:
            self.context_conditioner.load_state_dict(state_dict["context_conditioner"])
        self.feature_memory.load_state_dict(state_dict["feature_memory"])
        self.bottleneck.load_state_dict(state_dict["bottleneck"])
        self.decoder.load_state_dict(state_dict["decoder"])
        self.high_frequency_center = float(state_dict["high_frequency_center"])
        self.high_frequency_scale = float(state_dict["high_frequency_scale"])
        self.semantic_center = float(state_dict["semantic_center"])
        self.semantic_scale = float(state_dict["semantic_scale"])
        self.memory_center = float(state_dict["memory_center"])
        self.memory_scale = float(state_dict["memory_scale"])
