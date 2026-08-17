from __future__ import annotations

from typing import Any

__all__ = ["HRTrainer"]


def __getattr__(name: str) -> Any:
    """延迟导入训练器，避免仅加载数据工具时初始化训练依赖。"""
    if name == "HRTrainer":
        from .trainer import HRTrainer

        return HRTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
