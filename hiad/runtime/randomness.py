from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """固定主要随机源，同时保留适合固定尺寸训练的 cuDNN 算法选择。

    Args:
        seed (int): 写入 Python 哈希、``random``、NumPy、PyTorch CPU 和所有
            CUDA 设备的随机种子。

    Notes:
        为避免显著降低 Dinomaly 固定尺寸训练速度，本函数启用 cuDNN benchmark
        且不强制确定性卷积；因此它保证采样随机源可复现，而非逐位确定的 GPU 结果。
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 强制确定性卷积会显著拖慢固定尺寸 Dinomaly；采样随机源仍由上方种子控制。
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
