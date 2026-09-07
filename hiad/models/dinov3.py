from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import timm
import torch
import torch.nn as nn

from hiad.constants import DINO_PATCH_SIZE


class TimmDinoV3Encoder(nn.Module):
    """由 timm 提供的冻结 DINOv3 多层特征编码器。

    Attributes:
        model (Any): ``timm`` 创建并固定为评估模式的 DINOv3 主干。
        patch_size (int): 模型 patch 边长，必须等于项目常量 ``DINO_PATCH_SIZE``。
        embed_dim (int): 每层输出特征通道数。
        intermediate_layers (tuple[int, ...]): 需要提取的主干中间层编号。
        use_fp16 (bool): CUDA 上是否在自动混合精度上下文中编码。
    """

    def __init__(
        self,
        model_name: str,
        intermediate_layers: Sequence[int],
        use_fp16: bool = False,
        weights_path: str | None = None,
    ) -> None:
        super().__init__()
        if "dinov3" not in model_name:
            raise ValueError(f"Expected a timm DINOv3 model name, got: {model_name}")

        # 工业机离线部署：显式本地权重文件，运行时不触发任何 timm/HF 下载。
        # 仅在未提供路径的开发回退下才允许 timm 联网取预训练权重。
        self.model: Any = timm.create_model(
            model_name,
            pretrained=not bool(weights_path),
            num_classes=0,
        )
        if weights_path:
            self._load_weights(weights_path)
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

        self.patch_size: int = patch_size[0]
        self.embed_dim: int = int(self.model.num_features)
        self.intermediate_layers: tuple[int, ...] = tuple(intermediate_layers)
        self.use_fp16: bool = use_fp16
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.model.eval()

    def _load_weights(self, weights_path: str) -> None:
        """从本地 ``.pth`` 状态字典文件加载冻结主干权重。

        Args:
            weights_path (str): ``runs/export_backbone.py`` 导出的主干状态字典
                文件路径。

        Raises:
            FileNotFoundError: 权重文件不存在。
            RuntimeError: 权重与当前主干结构不匹配，无法严格装载。
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Backbone weights file not found: {weights_path}")
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict, strict=True)

    @torch.no_grad()
    def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        """提取指定 DINOv3 中间层的 NCHW 浮点特征。

        Args:
            inputs (torch.Tensor): ImageNet 标准化的
                ``(batch, 3, height, width)`` RGB 张量，高宽应能被 patch size 整除。

        Returns:
            list[torch.Tensor]: 按 ``intermediate_layers`` 顺序排列的 ``float32``
            BCHW 特征，空间高宽约为输入高宽除以 ``patch_size``。
        """
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
