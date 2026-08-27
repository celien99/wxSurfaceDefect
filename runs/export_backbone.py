"""在联网机器上导出 DINOv3 主干权重，供离线工业部署打包。

在能访问网络的机器上运行一次，把 ``timm`` 下载的 DINOv3 预训练权重保存为本地
``.pth`` 状态字典文件；把该文件随部署包交付到不联网的工业机，并在
``configs/dinomaly.yaml`` 的 ``backbone_weights_path`` 指向它。之后推理进程
在加载主干时不再发起任何网络请求。

用法::

    python runs/export_backbone.py --output models/dinov3_backbone.pth
"""

from __future__ import annotations

import argparse
import os

import timm
import torch


def parse_args() -> argparse.Namespace:
    """解析模型名称与输出路径参数。"""
    parser = argparse.ArgumentParser(
        description="Export a frozen DINOv3 backbone for offline deployment"
    )
    parser.add_argument(
        "--model",
        default="vit_base_patch16_dinov3.lvd1689m",
        help="timm DINOv3 model name; must match backbone_name in the config",
    )
    parser.add_argument(
        "--output",
        default="models/dinov3_backbone.pth",
        help="output state-dict path; ship this file with the deployment package",
    )
    return parser.parse_args()


def main() -> None:
    """创建带预训练权重的 DINOv3 主干并把状态字典保存到本地文件。"""
    args = parse_args()
    if "dinov3" not in args.model:
        raise ValueError(f"Expected a timm DINOv3 model name, got: {args.model}")

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)

    print(f"Creating {args.model} with pretrained weights (requires network)...")
    model = timm.create_model(args.model, pretrained=True, num_classes=0)
    torch.save(model.state_dict(), args.output)
    print(f"Exported backbone state dict to {args.output}")


if __name__ == "__main__":
    main()
