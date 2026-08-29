from __future__ import annotations

import logging
import os
import time
from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any, TypeAlias

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from hiad.models import TimmDinoV3Encoder

from .base import BaseDetector
from .dinomaly.models.uad import ViTill
from .dinomaly.models.vision_transformer import Block as VitBlock
from .dinomaly.models.vision_transformer import LinearAttention2, bMlp
from .dinomaly.optimizers import StableAdamW
from .dinomaly.utils import WarmCosineScheduler, global_cosine_hm_percent

FeatureLayers: TypeAlias = Sequence[torch.Tensor]
DetectorBatch: TypeAlias = Mapping[str, Any]


def _positive_int(value: object, name: str) -> int:
    """校验检测器参数为正整数且不是布尔值。"""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


class HRDinomaly(BaseDetector):
    """DINOv3 + Dinomaly 高分辨率检测器。

    用冻结的 DINOv3 编码器配合可训练瓶颈与解码器做重建，异常证据为逐层
    ``1 - cosine_similarity`` 的跨层最大值；多尺度上下文特征按 ``fusion_weights``
    加权融合后进入重建，图像分数取最高 ``score_top_k`` 个 token 的均值。
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
        fusion_weights: Sequence[float] | None = None,
        bottleneck_dropout: float = 0.1,
        grad_clip_norm: float = 1.0,
        hard_mining_final: float = 0.0,
        hard_mining_warmup_iters: int = 1000,
        easy_grad_factor: float = 0.1,
        score_top_k: int = 4,
        encoder_amp: bool = True,
        decoder_amp: bool = True,
        allow_tf32: bool = True,
        backbone_weights_path: str | None = None,
        **_: object,
    ) -> None:
        total_iters = _positive_int(total_iters, "total_iters")
        log_per_steps = _positive_int(log_per_steps, "log_per_steps")
        super().__init__(
            patch_size,
            device,
            logger=logger,
            seed=seed,
            fusion_weights=fusion_weights,
        )

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
        for name, value in (("encoder_amp", encoder_amp), ("decoder_amp", decoder_amp), ("allow_tf32", allow_tf32)):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a boolean")
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
        self.decoder_inference_amp: bool = False

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
            weights_path=backbone_weights_path or None,
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

        self.bottleneck: nn.ModuleList = nn.ModuleList(
            [bMlp(embed_dim, embed_dim * 4, embed_dim, drop=bottleneck_dropout)]
        )

        decoder_blocks: list[nn.Module] = []
        for _ in range(8):
            blk = VitBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-8),
                attn=LinearAttention2,
            )
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
        """在检测器设备上提取冻结编码器的多层 BCHW 特征。"""
        return self.model.encoder_image(input_tensor.to(self.device))

    def to_device(self, device: torch.device) -> None:
        """把重建模型移动到目标设备。"""
        self.model = self.model.to(device)
        self.device = device

    def set_decoder_precision(self, amp: bool) -> None:
        """推理时是否用 FP16 autocast 跑重建解码器。

        只影响 :meth:`inference_batch`，不影响 :meth:`train_step`。
        """
        if not isinstance(amp, bool):
            raise TypeError("amp must be a boolean")
        self.decoder_inference_amp = amp

    def train_step(self, train_dataloader: DataLoader[Any], task_name: str) -> None:
        """训练当前任务的瓶颈与解码器重建模块。"""
        trainable = nn.ModuleList([self.bottleneck, self.decoder])

        for m in trainable.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        optimizer = StableAdamW(
            [{"params": trainable.parameters()}],
            lr=2e-3,
            betas=(0.9, 0.999),
            weight_decay=1e-4,
            amsgrad=False,
            eps=1e-10,
        )
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
                    en = self.get_multi_resolution_fusion_embeddings(data)
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

    @torch.inference_mode()
    def inference_batch(self, data: DetectorBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """在检测器设备上计算异常图与 token 图，不做任何 CPU 往返。

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(anomaly_map, token_map)``，
            均为设备驻留张量；前者形状 ``(batch, 1, patch_h, patch_w)``，后者
            为编码器 token 分辨率。由调用方决定何时拷贝回 CPU。
        """
        self.model.eval()
        en = self.get_multi_resolution_fusion_embeddings(data)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.decoder_inference_amp and self.device.type == "cuda",
        ):
            en, de = self.model.distillation(en)
        anomaly_map, token_map = self.cal_anomaly_maps(en, de, self.patch_size)
        return anomaly_map, token_map

    @staticmethod
    def _layer_anomaly_token_maps(
        encoder_features: FeatureLayers,
        decoder_features: FeatureLayers,
    ) -> list[torch.Tensor]:
        """计算逐层编码/重建特征的非负余弦距离 token 图。"""
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
        """把逐层语义 token 距离聚合为 token 图和模型像素图。"""
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

    def save_checkpoint(self, checkpoint_path: str | os.PathLike[str]) -> None:
        """保存当前任务推理所需的瓶颈、解码器与分数配置。"""
        for module_name, module in (
            ("bottleneck", self.bottleneck),
            ("decoder", self.decoder),
        ):
            for parameter_name, parameter in module.named_parameters():
                if not torch.isfinite(parameter).all():
                    raise FloatingPointError(
                        f"Refusing to save a non-finite {module_name} parameter: {parameter_name}"
                    )
        torch.save(
            {
                "bottleneck": self.bottleneck.state_dict(),
                "decoder": self.decoder.state_dict(),
                "fusion_weights": self.fusion_weights,
                "score_top_k": self.score_top_k,
                "layer_aggregation": "max",
                "encoder_amp": self.encoder_amp,
                "decoder_amp": self.decoder_amp,
                "allow_tf32": self.allow_tf32,
            },
            checkpoint_path,
        )

    def load_checkpoint(self, checkpoint_path: str | os.PathLike[str]) -> None:
        """从检查点恢复当前任务推理状态。"""
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.bottleneck.load_state_dict(state_dict["bottleneck"])
        self.decoder.load_state_dict(state_dict["decoder"])
        self.fusion_weights = state_dict.get("fusion_weights")
        aggregation = state_dict.get("layer_aggregation", "max")
        if aggregation != "max":
            raise ValueError(f"Unsupported checkpoint layer aggregation: {aggregation}")
        if state_dict.get("encoder_amp", self.encoder_amp) != self.encoder_amp:
            raise ValueError("Checkpoint encoder_amp does not match runtime configuration")
        self.score_top_k = _positive_int(
            state_dict.get("score_top_k", self.score_top_k),
            "checkpoint score_top_k",
        )
