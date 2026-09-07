from __future__ import annotations

from typing import cast

import torch


def validate_gpu_ids(gpu_ids: object) -> list[int]:
    """校验 CUDA 设备编号，并在启动工作进程前快速失败。

    Args:
        gpu_ids (object): 非空且不重复的非负整数列表。

    Returns:
        list[int]: 保持调用方顺序的设备编号副本。

    Raises:
        ValueError: 参数不是有效列表，存在重复编号，或编号超出可用设备范围。
        RuntimeError: 当前 PyTorch 运行时不可使用 CUDA。
    """
    if not isinstance(gpu_ids, list) or not gpu_ids:
        raise ValueError("gpu_ids must be a non-empty list")
    if any(
        isinstance(gpu_id, bool)
        or not isinstance(gpu_id, int)
        or gpu_id < 0
        for gpu_id in gpu_ids
    ):
        raise ValueError("gpu_ids must contain non-negative integers")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("gpu_ids must not contain duplicates")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Dinomaly training and inference")
    device_count = torch.cuda.device_count()
    if any(gpu_id >= device_count for gpu_id in gpu_ids):
        raise ValueError(
            f"Requested GPU ids {gpu_ids} exceed the {device_count} available CUDA device(s)"
        )
    return list(cast(list[int], gpu_ids))
