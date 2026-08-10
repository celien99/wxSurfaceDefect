import os
from concurrent.futures import ThreadPoolExecutor, wait
from threading import Lock
from typing import List

import torch
from easydict import EasyDict
from tqdm import tqdm

from hiad.constants import TASK_TYPE_DYNAMIC_PATCH, TASK_TYPE_THUMBNAIL
from hiad.data import HRSample
from hiad.inferencer.modelmanager import ModelManager
from hiad.checkpoints import resolve_current_generation
from hiad.preprocessing import (
    ForegroundPreprocessorRegistry,
    validate_preprocessing_registry,
)
from hiad.runtime.devices import validate_gpu_ids
from hiad.runtime.evidence import collect_task_evidence
from hiad.runtime.partition import round_robin_partition
from hiad.scoring import (
    MultiRiskConfig,
    MultiRiskScorer,
    build_batch_output,
    load_calibration,
    merge_worker_evidence,
)
from hiad.task import load_tasks


def inference_in_device(test_samples, tasks, model_manager, batch_size, preprocessors):
    sorted_names = (
        model_manager.get_device_task_names(gpu=True)
        + model_manager.get_device_task_names(gpu=False)
    )
    device_results, _ = collect_task_evidence(
        test_samples,
        tasks,
        preprocessors,
        lambda task: model_manager.get_detector(task["name"]),
        batch_size,
        task_names=sorted_names,
    )
    return device_results


class HRInferencer:
    def __init__(
        self,
        detector_class,
        config,
        checkpoint_root: str,
        gpu_ids: List[int],
        models_per_gpu: int = -1,
        batch_size: int | None = None,
    ):
        self.detector_class = detector_class
        self.checkpoint_root = str(resolve_current_generation(checkpoint_root))
        self.config = EasyDict(config) if type(config) == dict else config
        self.gpu_ids = validate_gpu_ids(gpu_ids)
        self.models_per_gpu = models_per_gpu
        if self.models_per_gpu <= 0 and self.models_per_gpu != -1:
            raise ValueError("models_per_gpu must be positive or -1")
        if batch_size is not None and (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer or None")
        self.batch_size = batch_size

        runtime_preprocessing_config = getattr(self.config, 'preprocessing', None)
        preprocessing_registry = validate_preprocessing_registry(
            self.checkpoint_root,
            runtime_config=runtime_preprocessing_config,
        )
        self.preprocessing_config = preprocessing_registry['config']
        preprocessing_backbone = self.preprocessing_config['dino_backbone_name']
        detector_backbone = getattr(self.config.patch, 'backbone_name', None)
        if preprocessing_backbone != detector_backbone:
            raise ValueError(
                "Preprocessing and detector DINO backbone names must match"
            )

        tasks_path = os.path.join(self.checkpoint_root, 'tasks.json')
        if not os.path.exists(tasks_path):
            raise FileNotFoundError(f"Task configuration not found: {tasks_path}")
        self.tasks = load_tasks(tasks_path)
        dynamic_tasks = [
            task
            for task in self.tasks
            if task['type'] == TASK_TYPE_DYNAMIC_PATCH
        ]
        thumbnail_tasks = [
            task
            for task in self.tasks
            if task['type'] == TASK_TYPE_THUMBNAIL
        ]
        if len(dynamic_tasks) != 1 or len(thumbnail_tasks) != 1:
            raise ValueError(
                "Multi-risk inference requires exactly one dynamic patch task "
                "and one thumbnail task"
            )
        dynamic_task = dynamic_tasks[0]
        self.config.patch.patch_size = dynamic_task['patch_size']
        self.config.thumbnail.thumbnail_size = thumbnail_tasks[0]['thumbnail_size']

        self.tasks_in_devices = [
            task_group
            for task_group in round_robin_partition(self.tasks, len(self.gpu_ids))
            if task_group
        ]
        if self.models_per_gpu == -1:
            self.models_per_gpu = max(len(task_group) for task_group in self.tasks_in_devices)

        self.model_managers = []
        for index, tasks in tqdm(
            enumerate(self.tasks_in_devices),
            total=len(self.tasks_in_devices),
            desc="Loading checkpoints...",
        ):
            self.model_managers.append(ModelManager(
                tasks,
                self.detector_class,
                self.config,
                self.checkpoint_root,
                self.gpu_ids[index],
                self.models_per_gpu,
            ))

        self.preprocessors = ForegroundPreprocessorRegistry.from_checkpoint(
            self.checkpoint_root,
            torch.device(f"cuda:{self.gpu_ids[0]}"),
            runtime_config=self.preprocessing_config,
        )
        scoring_identities = [
            identity
            for manager in self.model_managers
            if (identity := manager.get_dynamic_scoring_identity()) is not None
        ]
        if len(scoring_identities) != 1:
            raise ValueError("Exactly one dynamic detector scoring identity is required")
        identity = scoring_identities[0]
        self.scoring_config = MultiRiskConfig.from_runtime(
            self.config.scoring,
            self.tasks,
            anomaly_distance=identity["anomaly_distance"],
            use_fp16=identity["use_fp16"],
            fusion_weights=identity["fusion_weights"],
        )
        self.scorer = MultiRiskScorer(self.scoring_config)
        self.calibration = load_calibration(
            self.checkpoint_root,
            self.scoring_config,
        )
        self._executor = ThreadPoolExecutor(max_workers=len(self.tasks_in_devices))
        self._inference_lock = Lock()
        self._closed = False

    def _validate_samples(self, test_samples: List[HRSample]) -> None:
        if not isinstance(test_samples, list) or not test_samples:
            raise ValueError("test_samples must be a non-empty list")
        for sample in test_samples:
            if not isinstance(sample, HRSample):
                raise TypeError("Every test sample must be an HRSample")
            if not isinstance(sample.clsname, str) or not sample.clsname.strip():
                raise ValueError(
                    "Every inference sample must have a non-empty clsname"
                )
            if sample.image.is_mask:
                raise ValueError("The local RGB input cannot be a mask image")
            if sample.mask is not None:
                raise ValueError("Local inference does not accept mask images")

    def _build_display_images(self, test_samples: List[HRSample], display_size):
        display_images = {}
        for sample in test_samples:
            preprocessor = self.preprocessors.get(sample.clsname)
            processed = preprocessor.process_file(
                sample.image.image_path,
                sample.clsname,
            )
            display_images[sample.image.image_path] = preprocessor.inverse_normalize(
                processed,
                output_size=display_size,
            )
        return display_images

    def inference(self, test_samples: List[HRSample], *, display_size=None):
        with self._inference_lock:
            if self._closed:
                raise RuntimeError("HRInferencer is closed")
            self._validate_samples(test_samples)
            batch_size = self.batch_size or len(test_samples)
            pending_results = [
                self._executor.submit(
                    inference_in_device,
                    test_samples,
                    tasks,
                    model_manager,
                    batch_size,
                    self.preprocessors,
                )
                for model_manager, tasks in zip(
                    self.model_managers,
                    self.tasks_in_devices,
                )
            ]
            wait(pending_results)
            worker_results = [pending_result.result() for pending_result in pending_results]
            image_paths = [sample.image.image_path for sample in test_samples]
            evidence_by_path = merge_worker_evidence(worker_results, image_paths)
            result = build_batch_output(
                evidence_by_path,
                image_paths,
                self.scorer,
                self.calibration,
            )
            if display_size is not None:
                result["display_images"] = self._build_display_images(
                    test_samples,
                    display_size,
                )
            return result

    def close(self) -> None:
        with self._inference_lock:
            if self._closed:
                return
            self._executor.shutdown(wait=True, cancel_futures=True)
            self.preprocessors.close()
            self.model_managers.clear()
            self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError("HRInferencer is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
