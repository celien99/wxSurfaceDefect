import logging
import time
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List

from .base import BaseDetector
from .dinomaly.models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2
from .dinomaly.models.uad import ViTill
from .dinomaly.optimizers import StableAdamW
from .dinomaly.utils import global_cosine_hm_percent, WarmCosineScheduler
from hiad.models import (
    ConditionalFeatureFusion,
    NormalFeatureMemory,
    TimmDinoV3Encoder,
)
from hiad.runtime.evidence import (
    denormalize_imagenet_batch,
    fuse_evidence_tensors,
    high_frequency_map,
)


def _positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _empty_moments(device):
    zero = torch.zeros((), dtype=torch.float32, device=device)
    return zero.clone(), zero.clone(), zero.clone()


def _update_moments(moments, values):
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


def _finalize_moments(moments, name):
    count, mean, m2 = moments
    if count < 2:
        raise ValueError(f"{name} fitting requires at least two values")
    scale = torch.sqrt(m2 / (count - 1)).clamp_min(1e-6)
    return float(mean.item()), float(scale.item())


class HRDinomaly(BaseDetector):
    def __init__(self,
                 backbone_name,
                 total_iters,
                 log_per_steps,
                 patch_size: int,  # base
                 logger: logging.Logger,  # base
                 device: torch.device,  # base
                 seed: int = 0,  #base
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
                 **kwargs):

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
                f"hard_mining_warmup_iters must be a non-negative integer, got {hard_mining_warmup_iters}"
            )
        if not 0 <= easy_grad_factor <= 1:
            raise ValueError(f"easy_grad_factor must be in [0, 1], got {easy_grad_factor}")
        if not isinstance(encoder_amp, bool):
            raise TypeError("encoder_amp must be a boolean")
        if not isinstance(decoder_amp, bool):
            raise TypeError("decoder_amp must be a boolean")
        if not isinstance(allow_tf32, bool):
            raise TypeError("allow_tf32 must be a boolean")
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
        self.total_iters = total_iters
        self.grad_clip_norm = grad_clip_norm
        self.hard_mining_final = hard_mining_final
        self.hard_mining_warmup_iters = hard_mining_warmup_iters
        self.easy_grad_factor = easy_grad_factor
        self.score_top_k = score_top_k
        self.encoder_amp = encoder_amp
        self.decoder_amp = decoder_amp
        self.allow_tf32 = allow_tf32
        self.evidence_weights = evidence_weights
        self.backbone_name = backbone_name
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            torch.backends.cudnn.allow_tf32 = allow_tf32
            torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
        self.target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
        self.fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
        self.fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
        self.encoder = TimmDinoV3Encoder(
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

        self.context_conditioner = ConditionalFeatureFusion(
            embed_dim=embed_dim,
            layers=len(self.target_layers),
        )
        self.feature_memory = NormalFeatureMemory(
            embed_dim=embed_dim,
            layers=len(self.target_layers),
        )
        self.high_frequency_center = 0.0
        self.high_frequency_scale = 1.0
        self.semantic_center = 0.0
        self.semantic_scale = 1.0
        self.memory_center = 0.0
        self.memory_scale = 1.0

        self.bottleneck = []
        self.bottleneck.append(bMlp(embed_dim, embed_dim * 4, embed_dim, drop=bottleneck_dropout))
        self.bottleneck = nn.ModuleList(self.bottleneck)

        self.decoder = []
        for i in range(8):
            blk = VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                           qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),

                           attn=LinearAttention2)

            self.decoder.append(blk)
        self.decoder = nn.ModuleList(self.decoder)

        self.model = ViTill(encoder=self.encoder, bottleneck=self.bottleneck, decoder=self.decoder,
                            fuse_layer_encoder=self.fuse_layer_encoder,
                            fuse_layer_decoder=self.fuse_layer_decoder)
        self.to_device(device)
        self.log_per_steps = log_per_steps

    @torch.no_grad()
    def embedding(self, input_tensor: torch.Tensor ) -> List[torch.Tensor]:
        return self.model.encoder_image(input_tensor.to(self.device))

    def to_device(self, device):
        self.model = self.model.to(device)
        self.context_conditioner = self.context_conditioner.to(device)
        self.feature_memory = self.feature_memory.to(device)
        self.device = device

    @torch.no_grad()
    def fit_normal_evidence(self, train_dataloader: DataLoader) -> None:
        """Fit normal token statistics and high-frequency normalization."""
        self.model.eval()
        self.context_conditioner.eval()
        self.feature_memory.reset()
        semantic_moments = _empty_moments(self.device)
        frequency_moments = _empty_moments(self.device)
        for data in train_dataloader:
            main_features, context_features = self.get_multi_resolution_embeddings(data)
            conditioned = self.context_conditioner(main_features, context_features)
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
            conditioned = self.context_conditioner(main_features, context_features)
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

    def _memory_token_map(self, features, output_size):
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
    def _positive_normalize(values, center, scale):
        return ((values - center) / scale).clamp_min(0.0)

    def _fused_evidence(
        self,
        data,
        memory_features,
        semantic_encoder_features,
        semantic_decoder_features,
    ):
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
        fused_pixel, max_pixel = fuse_evidence_tensors(
            [semantic_pixel, memory_pixel, frequency_pixel],
            self.evidence_weights,
        )
        fused_token, _ = fuse_evidence_tensors(
            [semantic_token, memory_token, frequency_token],
            self.evidence_weights,
        )
        return fused_pixel, fused_token, max_pixel

    def train_step(self,
                   train_dataloader: DataLoader,
                   task_name: str) -> None:

        trainable = nn.ModuleList([
            self.context_conditioner,
            self.bottleneck,
            self.decoder,
        ])

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
        # Keep the configured budget, but never skip the first sampled epoch:
        # with one or more sampled patches per source that epoch contains every
        # normal training image.
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
        self.context_conditioner.train()
        it = 0
        step_started = time.perf_counter()
        for epoch in range(int(np.ceil(training_iters / batches_per_epoch))):
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
                    en = self.context_conditioner(main_features, context_features)
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
                    log_message = 'iter [{}/{}], loss:{:.4f}, avg_step_sec:{:.2f}'.format(
                        it, training_iters, loss.item(), step_time
                    )
                    if grad_norm is not None:
                        log_message += ', grad_norm:{:.4f}'.format(grad_norm.item())
                    if self.logger is not None:
                        self.logger.info(log_message)

                if it >= training_iters:
                    break
        return None


    @torch.no_grad()
    def inference_step(
        self,
        test_dataloader: DataLoader,
    ) -> list[dict]:
        self.model.eval()
        self.context_conditioner.eval()
        predictions = []
        for data in test_dataloader:
            main_features, context_features = self.get_multi_resolution_embeddings(data)
            conditioned_features = self.context_conditioner(main_features, context_features)
            semantic_encoder, semantic_decoder = self.model.distillation(
                list(conditioned_features)
            )
            anomaly_map, token_map, max_evidence_map = self._fused_evidence(
                data,
                conditioned_features,
                semantic_encoder,
                semantic_decoder,
            )
            pixel_batch = anomaly_map[:, 0].cpu().numpy()
            max_batch = max_evidence_map[:, 0].cpu().numpy()
            score_batch = self._top_k_token_scores(token_map, self.score_top_k).cpu().numpy()
            predictions.extend(
                {
                    "anomaly_map": pixel_map,
                    "max_evidence_map": max_map,
                    "score": float(score),
                }
                for pixel_map, max_map, score in zip(
                    pixel_batch,
                    max_batch,
                    score_batch,
                )
            )
        return predictions

    @staticmethod
    def _layer_anomaly_token_maps(encoder_features, decoder_features):
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

    def cal_anomaly_maps(self, encoder_features, decoder_features, output_size):
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
        top_k = _positive_int(top_k, "top_k")
        if token_maps.ndim != 4 or token_maps.shape[1] != 1:
            raise ValueError("token_maps must have shape [batch, 1, height, width]")
        values = token_maps.flatten(start_dim=1)
        count = min(top_k, values.shape[1])
        return torch.topk(values, k=count, dim=1).values.mean(dim=1)

    def save_checkpoint(self, checkpoint_path: str):
        torch.save({
            "context_conditioner": self.context_conditioner.state_dict(),
            "feature_memory": self.feature_memory.state_dict(),
            "bottleneck": self.bottleneck.state_dict(),
            "decoder": self.decoder.state_dict(),
            "high_frequency_center": self.high_frequency_center,
            "high_frequency_scale": self.high_frequency_scale,
            "semantic_center": self.semantic_center,
            "semantic_scale": self.semantic_scale,
            "memory_center": self.memory_center,
            "memory_scale": self.memory_scale,
        }, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str):
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.context_conditioner.load_state_dict(state_dict['context_conditioner'])
        self.feature_memory.load_state_dict(state_dict['feature_memory'])
        self.bottleneck.load_state_dict(state_dict['bottleneck'])
        self.decoder.load_state_dict(state_dict['decoder'])
        self.high_frequency_center = state_dict['high_frequency_center']
        self.high_frequency_scale = state_dict['high_frequency_scale']
        self.semantic_center = state_dict['semantic_center']
        self.semantic_scale = state_dict['semantic_scale']
        self.memory_center = state_dict['memory_center']
        self.memory_scale = state_dict['memory_scale']

    @staticmethod
    def get_image_score(task_score_groups):
        scores = []
        for task_scores in task_score_groups:
            values = np.asarray(task_scores, dtype=np.float32).reshape(-1)
            if values.size == 0 or not np.isfinite(values).all():
                raise ValueError("Every image requires finite task anomaly scores")
            scores.append(float(values.max()))
        return np.asarray(scores, dtype=np.float32)
