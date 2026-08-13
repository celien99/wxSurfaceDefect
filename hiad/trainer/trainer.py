import os
import copy

import numpy as np

import torch.multiprocessing as mp
from easydict import EasyDict

from hiad.runtime.partition import round_robin_partition
from hiad.runtime.logging import create_logger
from hiad.runtime.devices import validate_gpu_ids
from hiad.runtime.score_calibration import (
    build_score_calibration,
    save_score_calibration,
    summarize_anomaly_map,
)
from hiad.task import save_tasks, validate_tasks
from hiad.trainer.sources import validate_unified_training_samples
from hiad.trainer.worker import train_tasks_in_device


class HRTrainer:
    """Simple HiAD-style task trainer.

    Every configured task gets its own detector and checkpoint. Normal samples
    are passed to every task without a validation holdout or anomaly synthesis.
    """

    def __init__(
        self,
        detector_class,
        config,
        batch_size: int,
        checkpoint_root: str,
        log_root: str,
        tasks,
        seed: int = 0,
        fusion_weights=None,
    ):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if tasks is None:
            raise ValueError("tasks must not be None")

        self.detector_class = detector_class
        self.config = EasyDict(config) if isinstance(config, dict) else config
        self.batch_size = batch_size
        self.checkpoint_root = checkpoint_root
        self.log_root = log_root
        self.tasks = validate_tasks(tasks)
        self.seed = seed
        self.fusion_weights = fusion_weights
        os.makedirs(self.checkpoint_root, exist_ok=True)
        os.makedirs(self.log_root, exist_ok=True)
        mp.set_start_method("spawn", force=True)

    def train(self, train_samples, gpu_ids, main_logger=None):
        sources = validate_unified_training_samples(train_samples)
        gpu_ids = validate_gpu_ids(gpu_ids)

        if main_logger is None:
            main_logger = create_logger(
                "main",
                os.path.join(self.log_root, "main.log"),
                print_console=True,
            )

        tasks_path = os.path.join(self.checkpoint_root, "tasks.json")
        save_tasks(self.tasks, tasks_path)
        main_logger.info("Start training, devices: %s", gpu_ids)
        main_logger.info("Tasks config is saved as: %s", tasks_path)
        main_logger.info(
            "Normal training samples: %d (100%% used; no validation holdout)",
            len(sources.samples),
        )
        for index, task in enumerate(self.tasks, start=1):
            if task["type"] == "dynamic_patch":
                main_logger.info(
                    "[%d/%d] Task %s, patch_size=%s, stride=%s, ds_factors=%s",
                    index,
                    len(self.tasks),
                    task["name"],
                    task["patch_size"],
                    task["stride"],
                    task["ds_factors"],
                )
            else:
                main_logger.info(
                    "[%d/%d] Task %s, thumbnail_size=%s",
                    index,
                    len(self.tasks),
                    task["name"],
                    task["thumbnail_size"],
                )
        main_logger.info("The training progress can be monitored in: %s", self.log_root)

        tasks_in_device = [
            task_group
            for task_group in round_robin_partition(self.tasks, len(gpu_ids))
            if task_group
        ]
        results = []
        process_pool = mp.Pool(processes=len(tasks_in_device))
        try:
            for gpu_id, task_group in zip(gpu_ids, tasks_in_device):
                results.append(process_pool.apply_async(
                    train_tasks_in_device,
                    args=(
                        gpu_id,
                        self.detector_class,
                        self.config,
                        copy.deepcopy(list(sources.samples)),
                        task_group,
                        self.batch_size,
                        self.checkpoint_root,
                        self.log_root,
                        self.seed,
                        self.fusion_weights,
                    ),
                ))
            process_pool.close()
            process_pool.join()
        except Exception:
            process_pool.terminate()
            process_pool.join()
            raise

        for result in results:
            message = result.get()
            if message:
                main_logger.info(message)

        from hiad.inferencer import HRInferencer

        main_logger.info(
            "Calibrating image and pixel thresholds from all normal training images"
        )
        calibration_scores = []
        calibration_pixel_statistics = []
        calibration_batch_size = int(
            getattr(self.config.patch, "calibration_batch_size", 1)
        )
        pixel_percentile = float(
            getattr(self.config.patch, "normal_pixel_percentile", 0.9999)
        )
        with HRInferencer(
            detector_class=self.detector_class,
            config=self.config,
            checkpoint_root=self.checkpoint_root,
            gpu_ids=gpu_ids,
            models_per_gpu=-1,
            batch_size=calibration_batch_size,
            require_score_calibration=False,
        ) as inferencer:
            for start in range(0, len(sources.samples), calibration_batch_size):
                batch_samples = list(
                    sources.samples[start:start + calibration_batch_size]
                )
                calibration_result = inferencer.inference(batch_samples)
                calibration_scores.extend(calibration_result["image_scores"].tolist())
                calibration_pixel_statistics.extend(
                    summarize_anomaly_map(anomaly_map, pixel_percentile)
                    for anomaly_map in calibration_result["anomaly_maps"]
                )

        percentile = float(getattr(self.config.patch, "normal_score_percentile", 0.99))
        pixel_image_percentile = float(
            getattr(self.config.patch, "normal_pixel_image_percentile", 0.99)
        )
        score_top_k = int(getattr(self.config.patch, "score_top_k", 4))
        calibration = build_score_calibration(
            sources.samples,
            np.asarray(calibration_scores, dtype=np.float64),
            calibration_pixel_statistics,
            percentile=percentile,
            pixel_percentile=pixel_percentile,
            pixel_image_percentile=pixel_image_percentile,
            score_top_k=score_top_k,
        )
        calibration_path = save_score_calibration(calibration, self.checkpoint_root)
        main_logger.info(
            "Score calibration saved as %s: image_percentile=%.4f, "
            "global_image_threshold=%.6f, pixel_percentile=%.6f, "
            "pixel_image_percentile=%.4f, global_pixel_threshold=%.6f",
            calibration_path,
            calibration["percentile"],
            calibration["global_threshold"],
            calibration["pixel_percentile"],
            calibration["pixel_image_percentile"],
            calibration["global_pixel_threshold"],
        )
        for category, payload in calibration["categories"].items():
            main_logger.info(
                "Category %s thresholds: image=%.6f, pixel=%.6f "
                "(normal_images=%d)",
                category,
                payload["threshold"],
                payload["pixel_threshold"],
                payload["normal_image_count"],
            )

        main_logger.info("End training")
        main_logger.info("Checkpoints are saved as: %s", self.checkpoint_root)
