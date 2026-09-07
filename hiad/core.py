"""外部可调用的核心算法检测模块。

一次构造、反复调用：传入一帧内存图像与业务类别，立即返回 OK/NG 判定。

边界：本模块只做"图像 → 判定"，不包含相机采集、可视化、PLC 通信、
重试/看门狗等任何集成层或防御性措施。
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hiad.data.samples import HRSample, HRImage
from hiad.detectors import HRDinomaly
from hiad.inferencer import HRInferencer
from hiad.runtime.contracts import InferenceResult

logger = logging.getLogger("hiad.core")


def _elapsed_milliseconds(timing: dict[str, float]) -> dict[str, float]:
    """把各阶段秒级耗时转为毫秒，去掉 ``_seconds`` 后缀。"""
    return {
        key.removesuffix("_seconds"): round(value * 1000.0, 2)
        for key, value in timing.items()
    }


class HiADDetector:
    """常驻模型的高分辨率异常检测核心。

    Args:
        config: 与训练一致的检测配置（``configs/dinomaly.yaml`` 或等价字典）。
        checkpoint_root: 包含 ``tasks.json``、三任务权重与
            ``score_calibration.json`` 的检查点目录。
        gpu_ids: 非空、不重复的 CUDA 设备编号序列。
        batch_size: 每任务推理批量大小。
        warmup: 是否在构造末尾用假图完整跑一次检测，吸收冷启动的 CUDA 上下文
            创建、cuDNN 自动调优与权重加载开销，保证第一帧真实检测速度稳定。

    Raises:
        FileNotFoundError: 检查点目录缺少 ``tasks.json`` 或
            ``score_calibration.json``（判定依赖正常样本校准阈值）。
    """

    def __init__(
        self,
        *,
        config: dict[str, Any],
        checkpoint_root: str | os.PathLike[str],
        gpu_ids: Sequence[int] = (0,),
        batch_size: int = 1,
        warmup: bool = True,
    ) -> None:
        # HRInferencer 构造即加载全部任务权重并常驻显存。
        self._inferencer = HRInferencer(
            detector_class=HRDinomaly,
            config=config,
            checkpoint_root=checkpoint_root,
            gpu_ids=list(gpu_ids),
            batch_size=batch_size,
            require_score_calibration=True,
        )
        if warmup:
            self._warm_up()

    def detect(
        self,
        image: NDArray[np.uint8],
        *,
        clsname: str,
        foreground: NDArray[np.uint8] | None = None,
    ) -> dict[str, Any]:
        """对单帧图像执行完整粗到细检测并立即返回判定。

        Args:
            image: ``(H, W, 3)`` RGB ``uint8`` 原图。
            clsname: 业务类别，用于选择分类别校准阈值。
            foreground: 可选单通道 ``(H, W)`` ``uint8`` 前景掩码；非零像素保留、
                零像素压黑，合成完全在内存中进行。

        Returns:
            dict: ``decision`` (OK/NG)、``score``、``threshold``、
            ``raw_image_score``、``component``（最强组件，可为 None）、
            ``quality``（采集质量）与 ``elapsed_ms``（各阶段耗时）。
        """
        sample = HRSample(
            image=HRImage.from_array(image),
            foreground=(
                HRImage.from_array(foreground, is_mask=True)
                if foreground is not None
                else None
            ),
            clsname=clsname,
        )
        result = self._inferencer.inference([sample])
        return _verdict(result)

    def _warm_up(self) -> None:
        """构造后跑一次假图检测，吸收冷启动开销。

        工业机重启后第一帧变慢的主要来源是 CUDA 上下文创建、cuDNN benchmark
        首次对每个输入 shape 的自动调优以及主干权重加载。预热一次后，后续每一帧
        （含第一帧）的速度保持一致；预热结果直接丢弃。
        """
        rng = np.random.default_rng(0)
        dummy = rng.integers(0, 256, size=(1024, 1024, 3), dtype=np.uint8)
        started = time.perf_counter()
        # 未校准的哨兵类别会回退全局阈值，仍然走完整粗到细管线。
        self.detect(dummy, clsname="__warmup__")
        logger.info("HiAD warm-up completed in %.2f seconds", time.perf_counter() - started)

    def close(self) -> None:
        """释放推理线程池与全部常驻模型引用。"""
        self._inferencer.close()

    def __enter__(self) -> HiADDetector:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> None:
        self.close()


def _verdict(result: InferenceResult) -> dict[str, Any]:
    """把单样本推理结果裁剪成最小判定字典。"""
    if "decisions" not in result:
        raise RuntimeError(
            "Verdicts require score calibration; ensure score_calibration.json "
            "exists in the checkpoint root"
        )
    return {
        "decision": result["decisions"][0],
        "score": float(result["component_scores"][0]),
        "threshold": float(result["decision_thresholds"][0]),
        "raw_image_score": float(result["raw_image_scores"][0]),
        "component": result["component_summaries"][0]["strongest_component"],
        "quality": result["quality_results"][0],
        "elapsed_ms": _elapsed_milliseconds(result["inference_timing"]),
    }
