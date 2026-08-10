from typing import List, Sequence

import timm
import torch
import torch.nn as nn

from hiad.constants import DINO_PATCH_SIZE


class TimmDinoV3Encoder(nn.Module):
    """Frozen DINOv3 feature encoder backed by timm."""

    def __init__(
        self,
        model_name: str,
        intermediate_layers: Sequence[int],
        use_fp16: bool = False,
    ):
        super().__init__()
        if "dinov3" not in model_name:
            raise ValueError(f"Expected a timm DINOv3 model name, got: {model_name}")

        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
        )
        patch_size = self.model.patch_embed.patch_size
        patch_size = (
            (patch_size, patch_size)
            if isinstance(patch_size, int)
            else tuple(patch_size)
        )
        if patch_size != (DINO_PATCH_SIZE, DINO_PATCH_SIZE):
            raise ValueError(
                f"HiAD requires a DINOv3 patch size of {DINO_PATCH_SIZE}, "
                f"got: {patch_size}"
            )

        self.patch_size = patch_size[0]
        self.embed_dim = self.model.num_features
        self.intermediate_layers = tuple(intermediate_layers)
        self.use_fp16 = use_fp16
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.model.eval()

    @torch.no_grad()
    def forward(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        with torch.autocast(
            device_type=inputs.device.type,
            dtype=torch.float16,
            enabled=self.use_fp16 and inputs.device.type == "cuda",
        ):
            features = self.model.forward_intermediates(
                inputs,
                indices=self.intermediate_layers,
                norm=False,
                output_fmt="NCHW",
                intermediates_only=True,
            )
        return [feature.float() for feature in features]
