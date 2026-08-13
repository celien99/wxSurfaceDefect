import os
from typing import List

from hiad.data import HRSample
from hiad.evaluation.execution import evaluate_category_metrics
from hiad.evaluation.inputs import build_evaluation_batch
from hiad.evaluation.report import summarize_category_scores
from hiad.evaluation.visualization import save_evaluation_visualizations
from hiad.runtime.devices import validate_gpu_ids
from hiad.runtime.logging import create_logger


class HREvaluator:
    """Orchestrate evaluation without owning inference or metric implementations."""

    def __init__(self, log_root: str, vis_root: str | None = None):
        self.log_root = log_root
        self.vis_root = vis_root
        os.makedirs(self.log_root, exist_ok=True)
        if self.vis_root is not None:
            os.makedirs(self.vis_root, exist_ok=True)

    def evaluate(
        self,
        test_samples: List[HRSample],
        inference_result: dict,
        gpu_ids: List[int],
        evaluators: List,
        *,
        main_logger=None,
        vis_size: int | List[int] = 1024,
    ) -> dict:
        gpu_ids = validate_gpu_ids(gpu_ids)
        batch = build_evaluation_batch(test_samples, inference_result)
        if main_logger is None:
            main_logger = create_logger(
                "evaluation",
                os.path.join(self.log_root, "evaluation.log"),
                print_console=True,
            )

        scores = []
        mean_metrics = {}
        if evaluators:
            main_logger.info("Computing metrics")
            scores = evaluate_category_metrics(batch, gpu_ids, evaluators)
            mean_metrics, report = summarize_category_scores(scores)
            main_logger.info(f"\n{report}")

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
