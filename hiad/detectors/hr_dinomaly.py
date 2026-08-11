import logging
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
from hiad.models import TimmDinoV3Encoder


def _positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def effective_training_iters(configured_iters: int, dataloader_length: int) -> int:
    configured_iters = _positive_int(configured_iters, "configured_iters")
    dataloader_length = _positive_int(dataloader_length, "dataloader_length")
    return max(configured_iters, dataloader_length)


class HRDinomaly(BaseDetector):
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
                 bottleneck_dropout: float = 0.1,
                 grad_clip_norm: float = 1.0,
                 hard_mining_final: float = 0.0,
                 hard_mining_warmup_iters: int = 1000,
                 easy_grad_factor: float = 0.1,
                 score_top_k: int = 4,
                 **kwargs):

        total_iters = _positive_int(total_iters, "total_iters")
        eval_per_steps = _positive_int(eval_per_steps, "eval_per_steps")
        log_per_steps = _positive_int(log_per_steps, "log_per_steps")
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
        score_top_k = _positive_int(score_top_k, "score_top_k")
        self.total_iters = total_iters
        self.grad_clip_norm = grad_clip_norm
        self.hard_mining_final = hard_mining_final
        self.hard_mining_warmup_iters = hard_mining_warmup_iters
        self.easy_grad_factor = easy_grad_factor
        self.score_top_k = score_top_k
        self.target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
        self.fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
        self.fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
        self.encoder = TimmDinoV3Encoder(
            model_name=backbone_name,
            intermediate_layers=self.target_layers,
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
        self.max_anomaly_score = None
        self.min_anomaly_score = None

    @torch.no_grad()
    def embedding(self, input_tensor: torch.Tensor ) -> List[torch.Tensor]:
        return self.model.encoder_image(input_tensor.to(self.device))

    def to_device(self, device):
        self.model = self.model.to(device)
        self.device = device

    def train_step(self,
                   train_dataloader: DataLoader,
                   task_name: str) -> None:

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
        training_iters = effective_training_iters(self.total_iters, len(train_dataloader))
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
                "(configured=%d, full_epoch=%d)",
                task_name,
                training_iters,
                self.total_iters,
                len(train_dataloader),
            )

        it = 0
        for epoch in range(int(np.ceil(training_iters / len(train_dataloader)))):
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
                    log_message = 'iter [{}/{}], loss:{:.4f}'.format(it, training_iters, loss.item())
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
        task_name: str,
    ) -> list[dict]:
        self.model.eval()
        predictions = []
        for data in test_dataloader:
            en = self.get_multi_resolution_fusion_embeddings(data)
            en, de = self.model.distillation(en)
            anomaly_map, token_map = self.cal_anomaly_maps(en, de, self.patch_size)
            pixel_batch = anomaly_map[:, 0].cpu().numpy()
            score_batch = self._top_k_token_scores(token_map, self.score_top_k).cpu().numpy()
            predictions.extend(
                {
                    "anomaly_map": self.patch_post_processing(pixel_map),
                    "score": float(score),
                }
                for pixel_map, score in zip(pixel_batch, score_batch)
            )
        return predictions

    def cal_anomaly_maps(self, encoder_features, decoder_features, output_size):
        if not encoder_features or len(encoder_features) != len(decoder_features):
            raise ValueError("Encoder and decoder feature lists must be non-empty and aligned")
        token_layer_maps = []
        pixel_layer_maps = []
        for encoder_feature, decoder_feature in zip(encoder_features, decoder_features):
            distance = torch.clamp(
                1 - F.cosine_similarity(encoder_feature.float(), decoder_feature.float()),
                min=0.0,
                max=2.0,
            )
            distance = distance.unsqueeze(1)
            token_layer_maps.append(distance)
            pixel_layer_maps.append(F.interpolate(
                distance,
                size=(output_size[1], output_size[0]),
                mode="bilinear",
                align_corners=True,
            ))
        token_map = torch.cat(token_layer_maps, dim=1).amax(dim=1, keepdim=True)
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

    def patch_post_processing(self, anomaly_map, eps=1e-4):
        if self.min_anomaly_score is None or self.max_anomaly_score is None:
            return anomaly_map
        anomaly_map = (anomaly_map - self.min_anomaly_score) / (
            self.max_anomaly_score - self.min_anomaly_score + eps
        )
        return np.clip(anomaly_map, 0, 1)


    def save_checkpoint(self, checkpoint_path: str):
        torch.save({
            "bottleneck": self.bottleneck.state_dict(),
            "decoder": self.decoder.state_dict(),
            "max_anomaly_score": self.max_anomaly_score,
            "min_anomaly_score": self.min_anomaly_score,
            "fusion_weights": self.fusion_weights,
            "score_top_k": self.score_top_k,
            "layer_aggregation": "max",
        }, checkpoint_path)


    def load_checkpoint(self, checkpoint_path: str):
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.bottleneck.load_state_dict(state_dict['bottleneck'])
        self.decoder.load_state_dict(state_dict['decoder'])
        self.min_anomaly_score = state_dict.get("min_anomaly_score")
        self.max_anomaly_score = state_dict.get("max_anomaly_score")
        self.fusion_weights = state_dict.get("fusion_weights")
        aggregation = state_dict.get("layer_aggregation", "max")
        if aggregation != "max":
            raise ValueError(f"Unsupported checkpoint layer aggregation: {aggregation}")
        self.score_top_k = _positive_int(
            state_dict.get("score_top_k", self.score_top_k),
            "checkpoint score_top_k",
        )

    @staticmethod
    def get_image_score(task_score_groups):
        scores = []
        for task_scores in task_score_groups:
            values = np.asarray(task_scores, dtype=np.float32).reshape(-1)
            if values.size == 0 or not np.isfinite(values).all():
                raise ValueError("Every image requires finite task anomaly scores")
            scores.append(float(values.max()))
        return np.asarray(scores, dtype=np.float32)
