import os
import uuid
from typing import Dict, List

import torch
import torch.multiprocessing as mp
from easydict import EasyDict

from hiad.checkpoints import begin_generation, publish_generation
from hiad.constants import TASK_TYPE_DYNAMIC_PATCH, TASK_TYPE_THUMBNAIL
from hiad.data import HRSample
from hiad.preprocessing import (
    PREPROCESSING_DIRECTORY,
    PREPROCESSING_CONFIG_FILE,
    PREPROCESSING_MANIFEST_FILE,
    PREPROCESSING_REGISTRY_FILE,
    PROTOTYPES_FILE,
    REFERENCE_MASK_FILE,
    REFERENCE_TEMPLATE_FILE,
    calibrate_preprocessing_registry,
    validate_preprocessing_registry,
)
from hiad.runtime.devices import validate_gpu_ids
from hiad.runtime.logging import create_logger
from hiad.runtime.partition import round_robin_partition
from hiad.scoring import (
    MULTIRISK_CALIBRATION_DOMAIN,
    MULTIRISK_CALIBRATION_FILE,
    MultiRiskConfig,
    MultiRiskScorer,
    build_calibration,
    save_calibration,
    score_batch_evidence,
)
from hiad.task import save_tasks, validate_tasks
from hiad.trainer.checkpoint_evidence import collect_checkpoint_evidence
from hiad.trainer.sources import validate_unified_training_samples
from hiad.trainer.worker import train_tasks_in_device


class HRTrainer:
    def __init__(
        self,
        detector_class,
        config,
        batch_size: int,
        checkpoint_root: str,
        log_root: str,
        tasks: List[Dict],
        seed: int = 0,
        fusion_weights: List = None,
    ):
        r"""High-resolution detector trainer.

        Args:
            detector_class: Detector class instantiated in each training worker.
            config: Detector, preprocessing, and scoring configuration.
            batch_size: Training and calibration batch size.
            checkpoint_root: Shared checkpoint root.
            log_root: Training log directory.
            tasks: Tasks created by ``DynamicTaskGenerator``.
            seed: Random seed.
            fusion_weights: Optional multi-resolution feature fusion weights.
        """
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if tasks is None:
            raise ValueError("Training requires explicit tasks")
        self.detector_class = detector_class
        self.batch_size = batch_size
        self.checkpoint_root = checkpoint_root
        self.log_root = log_root
        self.seed = seed
        self.tasks = tasks
        self.fusion_weights = fusion_weights

        os.makedirs(self.checkpoint_root, exist_ok=True)
        os.makedirs(self.log_root, exist_ok=True)

        self.config = EasyDict(config) if type(config) == dict else config
        self.scoring_config = None
        mp.set_start_method("spawn", force=True)

    def _configure_tasks(self, tasks):
        tasks = validate_tasks(tasks)
        dynamic_tasks = [
            task for task in tasks if task["type"] == TASK_TYPE_DYNAMIC_PATCH
        ]
        thumbnail_tasks = [
            task for task in tasks if task["type"] == TASK_TYPE_THUMBNAIL
        ]
        if len(dynamic_tasks) != 1 or len(thumbnail_tasks) != 1:
            raise ValueError(
                "Multi-risk scoring requires exactly one dynamic patch task "
                "and one thumbnail task"
            )
        dynamic_task = dynamic_tasks[0]
        self.config.patch.patch_size = dynamic_task["patch_size"]
        if self.fusion_weights is not None and (
            len(self.fusion_weights) != len(dynamic_task["ds_factors"])
            or any(weight < 0 for weight in self.fusion_weights)
            or sum(self.fusion_weights) <= 0
        ):
            raise ValueError(
                "fusion_weights must match ds_factors, be non-negative, and have a positive sum"
            )

        self.config.thumbnail.thumbnail_size = thumbnail_tasks[0]["thumbnail_size"]
        scoring_config = getattr(self.config, "scoring", None)
        if scoring_config is None:
            raise ValueError("Training requires scoring config")
        self.scoring_config = MultiRiskConfig.from_runtime(
            scoring_config,
            tasks,
            anomaly_distance=self.config.patch.anomaly_distance,
            use_fp16=getattr(self.config.patch, "use_fp16", False),
            fusion_weights=self.fusion_weights,
        )
        return tasks

    def train(
        self,
        train_samples: List[HRSample],
        gpu_ids: List[int],
        main_logger=None,
    ):
        r"""Train all tasks and publish a calibrated checkpoint generation.

        Args:
            train_samples: Normal high-resolution samples used for fitting.
            gpu_ids: CUDA device identifiers.
            main_logger: Optional logger. A file logger is created when omitted.
        """
        sources = validate_unified_training_samples(train_samples)
        gpu_ids = validate_gpu_ids(gpu_ids)
        self.tasks = self._configure_tasks(self.tasks)

        training_samples = list(sources.samples)
        categories = sources.categories

        if main_logger is None:
            main_logger = create_logger(
                "main",
                os.path.join(self.log_root, "main.log"),
                print_console=True,
            )

        main_logger.info(f"Start training, devices: {gpu_ids}")

        preprocessing_config = getattr(self.config, "preprocessing", None)
        if preprocessing_config is None:
            raise ValueError("Training requires preprocessing config")
        preprocessing_backbone = getattr(
            preprocessing_config,
            "dino_backbone_name",
            None,
        )
        detector_backbone = getattr(self.config.patch, "backbone_name", None)
        if preprocessing_backbone != detector_backbone:
            raise ValueError(
                "Preprocessing and detector DINO backbone names must match"
            )

        generation_root = begin_generation(
            self.checkpoint_root,
            generation_id=uuid.uuid4().hex,
        )
        main_logger.info(
            "Unified normal sources: "
            f"training_and_calibration={len(training_samples)}, "
            f"categories={list(categories)}"
        )

        calibrate_preprocessing_registry(
            preprocessing_config,
            categories,
            str(generation_root),
            torch.device(f"cuda:{gpu_ids[0]}"),
            logger=main_logger,
        )
        validate_preprocessing_registry(
            str(generation_root),
            runtime_config=preprocessing_config,
        )

        tasks_save_path = os.path.join(generation_root, "tasks.json")
        save_tasks(self.tasks, tasks_save_path)
        main_logger.info(f"Tasks config is saved as: {tasks_save_path}")

        for index, task in enumerate(self.tasks):
            if task["type"] == TASK_TYPE_DYNAMIC_PATCH:
                main_logger.info(
                    f"[{index + 1}/{len(self.tasks)}] Task: {task['name']}, "
                    f"patch_size: {task['patch_size']}, stride: {task['stride']}, "
                    f"ds_factors: {task['ds_factors']}"
                )
            elif task["type"] == TASK_TYPE_THUMBNAIL:
                main_logger.info(
                    f"[{index + 1}/{len(self.tasks)}] Task: {task['name']}, "
                    f"thumbnail_size: {task['thumbnail_size']}"
                )
            else:
                raise ValueError(f"Unsupported task type: {task}")

        main_logger.info(
            f"The training progress can be monitored in: {self.log_root}."
        )
        tasks_in_device = [
            task_group
            for task_group in round_robin_partition(self.tasks, len(gpu_ids))
            if task_group
        ]

        results = []
        process_pool = mp.Pool(processes=len(tasks_in_device))
        try:
            for gpu_id, task_group in zip(gpu_ids, tasks_in_device):
                results.append(
                    process_pool.apply_async(
                        train_tasks_in_device,
                        args=(
                            gpu_id,
                            self.detector_class,
                            self.config,
                            training_samples,
                            task_group,
                            self.batch_size,
                            str(generation_root),
                            self.log_root,
                            self.seed,
                            self.fusion_weights,
                        ),
                    )
                )
            process_pool.close()
            process_pool.join()
        except Exception:
            process_pool.terminate()
            process_pool.join()
            raise
        for result in results:
            message = result.get()
            if message is not None:
                main_logger.info(message)

        calibration_evidence, scoring_identity = collect_checkpoint_evidence(
            samples=training_samples,
            gpu_ids=gpu_ids,
            tasks=self.tasks,
            detector_class=self.detector_class,
            config=self.config,
            batch_size=self.batch_size,
            checkpoint_root=generation_root,
            log_root=self.log_root,
            seed=self.seed,
        )
        checkpoint_scoring_config = MultiRiskConfig.from_runtime(
            self.config.scoring,
            self.tasks,
            anomaly_distance=scoring_identity["anomaly_distance"],
            use_fp16=scoring_identity["use_fp16"],
            fusion_weights=scoring_identity["fusion_weights"],
        )
        if checkpoint_scoring_config.fingerprint != self.scoring_config.fingerprint:
            raise ValueError(
                "Trained dynamic checkpoint identity differs from the scoring configuration"
            )
        self.scoring_config = checkpoint_scoring_config
        calibration_paths = [
            sample.image.image_path for sample in training_samples
        ]
        scorer = MultiRiskScorer(self.scoring_config)
        raw_scores = score_batch_evidence(
            calibration_evidence,
            calibration_paths,
            scorer,
        )
        calibration = build_calibration(raw_scores, self.scoring_config)
        save_calibration(calibration, generation_root)
        main_logger.info(
            "Multi-risk calibration complete: "
            f"domain={MULTIRISK_CALIBRATION_DOMAIN}, "
            f"normal_images={calibration['normal_image_count']}, "
            f"decision_percentile={calibration['decision_percentile']:.4f}, "
            f"fingerprint={calibration['scoring_config_sha256']}"
        )
        for key, diagnostic in calibration["raw_diagnostics"].items():
            main_logger.info(
                f"Raw calibration {key}: "
                f"count={diagnostic['count']}, "
                f"p99.9={diagnostic['p99_9']:.6f}"
            )
        for warning in calibration["warnings"]:
            main_logger.warning(f"Calibration warning: {warning}")

        required_files = {
            "tasks.json",
            PREPROCESSING_REGISTRY_FILE,
            MULTIRISK_CALIBRATION_FILE,
            *(f"{task['name']}_weight.pt" for task in self.tasks),
            *(
                os.path.join(PREPROCESSING_DIRECTORY, category, artifact)
                for category in categories
                for artifact in (
                    PREPROCESSING_CONFIG_FILE,
                    PREPROCESSING_MANIFEST_FILE,
                    PROTOTYPES_FILE,
                    REFERENCE_TEMPLATE_FILE,
                    REFERENCE_MASK_FILE,
                )
            ),
        }
        publish_generation(
            self.checkpoint_root,
            generation_root,
            required_files=required_files,
        )

        main_logger.info("End training")
        main_logger.info(f"Checkpoint generation published: {generation_root}")
        return generation_root
