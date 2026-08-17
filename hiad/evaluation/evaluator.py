from __future__ import annotations

import logging
import os
from typing import TypedDict

from hiad.data import HRSample
from hiad.evaluation.execution import (
    CategoryScore,
    MetricEvaluator,
    evaluate_category_metrics,
)
from hiad.evaluation.inputs import build_evaluation_batch
from hiad.evaluation.report import summarize_category_scores
from hiad.evaluation.visualization import save_evaluation_visualizations
from hiad.runtime.contracts import InferenceResult
from hiad.runtime.devices import validate_gpu_ids
from hiad.runtime.logging import create_logger


class EvaluationResult(TypedDict):
    """评估入口返回的分类别与均值指标。

    Attributes:
        per_category (list[CategoryScore]): 按类别名称排序的指标记录。
        mean (dict[str, float]): 排除阈值字段后按类别等权平均的指标。
    """

    per_category: list[CategoryScore]
    mean: dict[str, float]


class HREvaluator:
    """编排评估、报告和可视化，不持有推理器或指标实现。

    Attributes:
        log_root (str): 评估日志输出目录。
        vis_root (str | None): 可视化输出目录；``None`` 表示不保存可视化。
    """

    def __init__(
        self,
        log_root: str | os.PathLike[str],
        vis_root: str | os.PathLike[str] | None = None,
    ) -> None:
        self.log_root: str = os.fspath(log_root)
        self.vis_root: str | None = os.fspath(vis_root) if vis_root is not None else None
        os.makedirs(self.log_root, exist_ok=True)
        if self.vis_root is not None:
            os.makedirs(self.vis_root, exist_ok=True)

    def evaluate(
        self,
        test_samples: list[HRSample],
        inference_result: InferenceResult,
        gpu_ids: list[int],
        evaluators: list[MetricEvaluator],
        *,
        main_logger: logging.Logger | None = None,
        vis_size: int | list[int] | tuple[int, int] = 1024,
    ) -> EvaluationResult:
        """对已完成的推理结果计算指标并按需保存可视化。

        Args:
            test_samples (list[HRSample]): 与推理结果路径顺序严格一致的评估样本。
            inference_result (InferenceResult): 原图分辨率异常图、图像分数及可选
                二值掩码、像素阈值和显示图。
            gpu_ids (list[int]): 用于分类别指标计算的 CUDA 设备编号。
            evaluators (list[MetricEvaluator]): 接收标准评估批次字段的指标函数。
            main_logger (logging.Logger | None): 可选主日志器；未提供时在
                ``log_root`` 创建独立日志。
            vis_size (int | list[int] | tuple[int, int]): 可视化宽高；整数表示
                正方形，二元序列按 ``(width, height)`` 解释。

        Returns:
            EvaluationResult: 分类别指标及其等权平均；未提供指标函数时均为空。

        Raises:
            ValueError: 设备、样本顺序、结果形状或可视化尺寸不符合约定。
            RuntimeError: CUDA 不可用、评估无输出或显示图缺失。
        """
        gpu_ids = validate_gpu_ids(gpu_ids)
        batch = build_evaluation_batch(test_samples, inference_result)
        if main_logger is None:
            main_logger = create_logger(
                "evaluation",
                os.path.join(self.log_root, "evaluation.log"),
                print_console=True,
            )

        scores: list[CategoryScore] = []
        mean_metrics: dict[str, float] = {}
        if evaluators:
            main_logger.info("Computing metrics")
            scores = evaluate_category_metrics(batch, gpu_ids, evaluators)
            mean_metrics, report = summarize_category_scores(scores)
            main_logger.info("\n%s", report)

        if self.vis_root is not None:
            save_evaluation_visualizations(
                batch,
                self.vis_root,
                vis_size,
                main_logger,
            )

        main_logger.info("End evaluation")
        return {
            "per_category": scores,
            "mean": mean_metrics,
        }
