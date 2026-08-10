import logging
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List

from .base import BaseDetector
from hiad.checkpoints import atomic_torch_save, safe_torch_load
from hiad.checkpoint_schema import (
    DETECTOR_CHECKPOINT_KEYS,
    validate_detector_checkpoint,
)
from hiad.constants import (
    ANOMALY_DISTANCE_NORMALIZED_L2,
    SUPPORTED_ANOMALY_DISTANCES,
)
from hiad.scoring.contracts import DetectorEvidence
from .dinomaly.models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2
from .dinomaly.models.uad import ViTill
from .dinomaly.optimizers import StableAdamW
from .dinomaly.utils import global_cosine_hm_percent, WarmCosineScheduler
from hiad.models import TimmDinoV3Encoder


def _positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


class HRDinomaly(BaseDetector):

    CHECKPOINT_SCHEMA_VERSION = 4

    def __init__(self,
                 backbone_name,
                 total_iters,
                 eval_per_steps,
                 log_per_steps,
                 patch_size: int,  # base
                 logger: logging.Logger,  # base
                 device: torch.device,  # base
                 seed: int = 0,  #base
                 fusion_weights = None,
                 use_fp16: bool = False,
                 bottleneck_dropout: float = 0.1,
                 grad_clip_norm: float = 1.0,
                 hard_mining_final: float = 0.0,
                 hard_mining_warmup_iters: int = 1000,
                 easy_grad_factor: float = 0.1,
                 anomaly_distance: str = ANOMALY_DISTANCE_NORMALIZED_L2,
                 **kwargs):

        total_iters = _positive_int(total_iters, "total_iters")
        eval_per_steps = _positive_int(eval_per_steps, "eval_per_steps")
        log_per_steps = _positive_int(log_per_steps, "log_per_steps")
        if not isinstance(use_fp16, bool):
            raise TypeError("use_fp16 must be a boolean")

        super().__init__(patch_size, device, fusion_weights, logger, seed)

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
        if anomaly_distance not in SUPPORTED_ANOMALY_DISTANCES:
            raise ValueError(
                f"anomaly_distance must be 'normalized_l2' or 'cosine', got {anomaly_distance}"
            )

        self.total_iters = total_iters
        self.grad_clip_norm = grad_clip_norm
        self.hard_mining_final = hard_mining_final
        self.hard_mining_warmup_iters = hard_mining_warmup_iters
        self.easy_grad_factor = easy_grad_factor
        self.anomaly_distance = anomaly_distance
        self.use_fp16 = use_fp16
        self.target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
        self.fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
        self.fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
        self.encoder = TimmDinoV3Encoder(
            model_name=backbone_name,
            intermediate_layers=self.target_layers,
            use_fp16=self.use_fp16,
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
        self.eval_per_steps = eval_per_steps
        self.log_per_steps = log_per_steps

    @torch.no_grad()
    def embedding(self, input_tensor: torch.Tensor ) -> List[torch.Tensor]:
        return self.model.encoder_image(input_tensor.to(self.device))

    def to_device(self, device):
        self.model = self.model.to(device)
        self.device = device

    def train_step(self,
                   train_dataloader: DataLoader,
                   task_name: str,
                   validation_callback=None) -> None:

        trainable = nn.ModuleList([self.bottleneck, self.decoder])

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
        lr_scheduler = WarmCosineScheduler(optimizer, base_value=2e-3, final_value=2e-4, total_iters=self.total_iters, warmup_iters=100)

        it = 0
        last_validation_iteration = None
        for epoch in range(int(np.ceil(self.total_iters / len(train_dataloader)))):
            torch.cuda.empty_cache()

            for data in train_dataloader:

                self.model.train()
                self.model.encoder.eval()

                en = self.get_multi_resolution_fusion_embeddings(data)
                en, de = self.model.distillation(en)

                if self.hard_mining_warmup_iters == 0:
                    p = self.hard_mining_final
                else:
                    p = min(
                        self.hard_mining_final * it / self.hard_mining_warmup_iters,
                        self.hard_mining_final,
                    )
                loss = global_cosine_hm_percent(en, de, p=p, factor=self.easy_grad_factor)

                optimizer.zero_grad()
                loss.backward()
                grad_norm = None
                if self.grad_clip_norm > 0:
                    grad_norm = nn.utils.clip_grad_norm_(
                        trainable.parameters(), max_norm=self.grad_clip_norm
                    )

                optimizer.step()
                lr_scheduler.step()
                it += 1

                if it % self.log_per_steps == 0:
                    log_message = 'iter [{}/{}], loss:{:.4f}'.format(it, self.total_iters, loss.item())
                    if grad_norm is not None:
                        log_message += ', grad_norm:{:.4f}'.format(grad_norm.item())
                    self.logger.info(log_message)

                if it % self.eval_per_steps == 0 and validation_callback is not None:
                    last_validation_iteration = it
                    if validation_callback():
                        return

                if it == self.total_iters:
                    break

        if (
            validation_callback is not None
            and last_validation_iteration != it
        ):
            validation_callback()
        return None


    @torch.no_grad()
    def predict_evidence(
        self,
        test_dataloader: DataLoader,
        task_name: str,
    ) -> list[DetectorEvidence]:
        self.model.eval()
        evidence = []
        for data in test_dataloader:
            en = self.get_multi_resolution_fusion_embeddings(data)
            en, de = self.model.distillation(en)
            raw_token_maps, raw_pixel_maps = self._build_anomaly_evidence_maps(
                en,
                de,
                self.patch_size,
            )
            token_batch = raw_token_maps.cpu().numpy()
            pixel_batch = raw_pixel_maps.cpu().numpy()
            for index in range(token_batch.shape[0]):
                evidence.append(DetectorEvidence(
                    raw_token_map=np.ascontiguousarray(
                        token_batch[index, 0],
                        dtype=np.float32,
                    ),
                    raw_pixel_map=np.ascontiguousarray(
                        pixel_batch[index, 0],
                        dtype=np.float32,
                    ),
                ))
        return evidence

    def _build_anomaly_evidence_maps(
        self,
        encoder_features,
        decoder_features,
        output_size,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not encoder_features or len(encoder_features) != len(decoder_features):
            raise ValueError("Encoder and decoder feature lists must be non-empty and aligned")

        token_layer_maps = []
        pixel_layer_maps = []
        token_shape = None
        for encoder_feature, decoder_feature in zip(encoder_features, decoder_features):
            fs = encoder_feature.float()
            ft = decoder_feature.float()
            if self.anomaly_distance == ANOMALY_DISTANCE_NORMALIZED_L2:
                fs = F.normalize(fs, p=2, dim=1)
                ft = F.normalize(ft, p=2, dim=1)
                distance = torch.linalg.vector_norm(fs - ft, ord=2, dim=1)
            else:
                distance = torch.clamp(
                    1 - F.cosine_similarity(fs, ft),
                    min=0.0,
                    max=2.0,
                )
            distance = distance.unsqueeze(1)
            if token_shape is None:
                token_shape = distance.shape[-2:]
            elif distance.shape[-2:] != token_shape:
                raise ValueError("Selected detector layers must share one token geometry")
            token_layer_maps.append(distance)
            pixel_layer_maps.append(F.interpolate(
                distance,
                size=(output_size[1], output_size[0]),
                mode='bilinear',
                align_corners=True,
            ))

        raw_token_maps = torch.cat(token_layer_maps, dim=1).max(
            dim=1,
            keepdim=True,
        ).values
        raw_pixel_maps = torch.cat(pixel_layer_maps, dim=1).max(
            dim=1,
            keepdim=True,
        ).values
        return raw_token_maps, raw_pixel_maps


    def save_checkpoint(self, checkpoint_path: str):
        atomic_torch_save(
            {
                'schema_version': self.CHECKPOINT_SCHEMA_VERSION,
                'bottleneck': self.bottleneck.state_dict(),
                'decoder': self.decoder.state_dict(),
                'anomaly_distance': self.anomaly_distance,
                'fusion_weights': self.fusion_weights,
                'use_fp16': self.use_fp16,
            },
            checkpoint_path,
        )


    def load_checkpoint(self, checkpoint_path: str):
        state_dict = safe_torch_load(
            checkpoint_path,
            required_keys=DETECTOR_CHECKPOINT_KEYS,
            map_location=self.device,
        )
        state_dict = validate_detector_checkpoint(
            state_dict,
            expected_version=self.CHECKPOINT_SCHEMA_VERSION,
        )
        if state_dict['use_fp16'] != self.use_fp16:
            raise ValueError(
                "Detector checkpoint use_fp16 does not match runtime configuration"
            )
        self.bottleneck.load_state_dict(state_dict['bottleneck'])
        self.decoder.load_state_dict(state_dict['decoder'])
        self.anomaly_distance = state_dict['anomaly_distance']
        self.fusion_weights = state_dict['fusion_weights']
