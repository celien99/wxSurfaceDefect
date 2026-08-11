import copy
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import cv2
import numpy as np
import torch
from easydict import EasyDict
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from hiad.constants import TASK_TYPE_DYNAMIC_PATCH, TASK_TYPE_THUMBNAIL
from hiad.data import HRSample
from hiad.datasets import StreamingTaskDataset
from hiad.inferencer.modelmanager import ModelManager
from hiad.runtime.devices import validate_gpu_ids
from hiad.runtime.partition import round_robin_partition
from hiad.runtime.score_calibration import (
    load_score_calibration,
    thresholds_for_samples,
)
from hiad.task import load_tasks


def _gather_patch_predictions(patches, image_size):
    image_width, image_height = image_size
    accumulated = np.zeros((image_height, image_width), dtype=np.float64)
    weight_map = np.zeros((image_height, image_width), dtype=np.float64)

    for record, prediction in patches:
        x, y, width, height = record["source_xywh"]
        valid_height, valid_width = record["valid_source_hw"]
        prediction = np.asarray(prediction, dtype=np.float32)
        row_hann = np.hanning(height) if height > 1 else np.ones(1, dtype=np.float64)
        column_hann = np.hanning(width) if width > 1 else np.ones(1, dtype=np.float64)
        weights = 0.05 + 0.95 * np.outer(row_hann, column_hann)
        valid_weights = weights[:valid_height, :valid_width]
        accumulated[y:y + valid_height, x:x + valid_width] += (
            prediction[:valid_height, :valid_width] * valid_weights
        )
        weight_map[y:y + valid_height, x:x + valid_width] += valid_weights

    if np.any(weight_map <= 0):
        raise ValueError("Patch predictions do not cover the complete source image")
    return (accumulated / weight_map).astype(np.float32)


def inference_in_device(
    test_samples,
    task_group,
    model_manager,
    batch_size,
    *,
    include_anomaly_maps: bool = True,
):
    paths = [sample.image.image_path for sample in test_samples]
    results = {
        path: {"image_size": None, "patches": [], "thumbnail": None, "scores": []}
        for path in paths
    }

    for task in task_group:
        task_name = task["name"]
        detector = model_manager.get_detector(task_name)
        dataset = StreamingTaskDataset(
            copy.deepcopy(test_samples),
            task,
            training=False,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        predictions = detector.inference_step(
            dataloader,
            task_name,
            include_anomaly_maps=include_anomaly_maps,
        )
        if len(predictions) != len(dataset.records):
            raise RuntimeError(
                f"Task {task_name} returned {len(predictions)} predictions for "
                f"{len(dataset.records)} inputs"
            )

        if task["type"] == TASK_TYPE_DYNAMIC_PATCH:
            for record, prediction in zip(dataset.records, predictions):
                path = record["image_path"]
                results[path]["image_size"] = record["image_size"]
                if include_anomaly_maps:
                    results[path]["patches"].append((record, prediction["anomaly_map"]))
                results[path]["scores"].append(prediction["score"])
        elif task["type"] == TASK_TYPE_THUMBNAIL:
            for record, prediction in zip(dataset.records, predictions):
                path = record["image_path"]
                results[path]["image_size"] = record["image_size"]
                if include_anomaly_maps:
                    results[path]["thumbnail"] = prediction["anomaly_map"]
                results[path]["scores"].append(prediction["score"])
        else:
            raise ValueError(f"Unsupported task type: {task}")

    return results


class HRInferencer:
    """Simple HiAD-style inference over task checkpoints."""

    def __init__(
        self,
        detector_class,
        config,
        checkpoint_root: str,
        gpu_ids,
        models_per_gpu: int = -1,
        batch_size: int | None = None,
        require_score_calibration: bool = True,
    ):
        gpu_ids = validate_gpu_ids(gpu_ids)
        if batch_size is not None and (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer or None")
        if not isinstance(require_score_calibration, bool):
            raise TypeError("require_score_calibration must be a boolean")

        self.detector_class = detector_class
        self.checkpoint_root = os.path.abspath(checkpoint_root)
        self.config = EasyDict(config) if isinstance(config, dict) else config
        self.gpu_ids = gpu_ids
        self.batch_size = batch_size
        tasks_path = os.path.join(self.checkpoint_root, "tasks.json")
        if not os.path.isfile(tasks_path):
            raise FileNotFoundError(f"Task configuration not found: {tasks_path}")
        self.tasks = load_tasks(tasks_path)

        for task in self.tasks:
            if task["type"] == TASK_TYPE_DYNAMIC_PATCH:
                self.config.patch.patch_size = task["patch_size"]
            elif task["type"] == TASK_TYPE_THUMBNAIL:
                self.config.thumbnail.thumbnail_size = task["thumbnail_size"]

        task_groups = [group for group in round_robin_partition(self.tasks, len(self.gpu_ids)) if group]
        if models_per_gpu == -1:
            models_per_gpu = max(len(group) for group in task_groups)
        if models_per_gpu <= 0:
            raise ValueError("models_per_gpu must be positive or -1")

        self.tasks_in_devices = task_groups
        self.model_managers = [
            ModelManager(
                tasks,
                detector_class,
                self.config,
                self.checkpoint_root,
                self.gpu_ids[index],
                models_per_gpu,
            )
            for index, tasks in enumerate(tqdm(task_groups, desc="Loading checkpoints..."))
        ]
        score_top_k_values = set().union(
            *(manager.score_top_k_values() for manager in self.model_managers)
        )
        if len(score_top_k_values) != 1:
            raise ValueError("All task checkpoints must use the same score_top_k")
        self.score_top_k = score_top_k_values.pop()
        self.score_calibration = load_score_calibration(
            self.checkpoint_root,
            required=require_score_calibration,
        )
        if (
            self.score_calibration is not None
            and self.score_calibration["score_top_k"] != self.score_top_k
        ):
            raise ValueError("Score calibration does not match checkpoint score_top_k")
        self.map_gaussian_sigma = float(
            getattr(self.config.patch, "map_gaussian_sigma", 0.0)
        )
        if not np.isfinite(self.map_gaussian_sigma) or self.map_gaussian_sigma < 0:
            raise ValueError("map_gaussian_sigma must be a finite non-negative number")
        self._executor = ThreadPoolExecutor(max_workers=len(task_groups))
        self._inference_lock = Lock()
        self._closed = False

    def inference(self, test_samples, *, display_size=None):
        with self._inference_lock:
            if self._closed:
                raise RuntimeError("HRInferencer is closed")
            self._validate_samples(test_samples)

            batch_size = self.batch_size or len(test_samples)
            pending = [
                self._executor.submit(
                    inference_in_device,
                    test_samples,
                    task_group,
                    manager,
                    batch_size,
                    include_anomaly_maps=True,
                )
                for task_group, manager in zip(self.tasks_in_devices, self.model_managers)
            ]
            worker_results = [future.result() for future in pending]

            merged = {
                sample.image.image_path: {
                    "image_size": None,
                    "patches": [],
                    "thumbnail": None,
                    "scores": [],
                }
                for sample in test_samples
            }
            for worker_result in worker_results:
                for path, result in worker_result.items():
                    if result["image_size"] is not None:
                        current_size = merged[path]["image_size"]
                        if current_size is not None and current_size != result["image_size"]:
                            raise ValueError(f"Task image sizes disagree for {path}")
                        merged[path]["image_size"] = result["image_size"]
                    merged[path]["patches"].extend(result["patches"])
                    merged[path]["scores"].extend(result["scores"])
                    if result["thumbnail"] is not None:
                        if merged[path]["thumbnail"] is not None:
                            raise ValueError(f"Duplicate thumbnail prediction for {path}")
                        merged[path]["thumbnail"] = result["thumbnail"]

            anomaly_maps = []
            task_score_groups = []
            for sample in test_samples:
                path = sample.image.image_path
                result = merged[path]
                if result["image_size"] is None or not result["patches"]:
                    raise ValueError(f"Incomplete dynamic patch prediction for {path}")
                patch_map = _gather_patch_predictions(result["patches"], result["image_size"])
                final_map = patch_map
                if result["thumbnail"] is not None:
                    image_width, image_height = result["image_size"]
                    thumbnail_map = cv2.resize(
                        np.asarray(result["thumbnail"], dtype=np.float32),
                        (image_width, image_height),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    final_map = np.maximum(final_map, thumbnail_map)
                if self.map_gaussian_sigma > 0:
                    final_map = gaussian_filter(
                        final_map,
                        sigma=self.map_gaussian_sigma,
                    )
                anomaly_maps.append(np.asarray(final_map, dtype=np.float32))
                task_score_groups.append(result["scores"])

            image_scores = self.detector_class.get_image_score(task_score_groups)
            output = {
                "image_paths": [sample.image.image_path for sample in test_samples],
                "image_scores": image_scores,
                "anomaly_maps": anomaly_maps,
                "display_images": self._build_display_images(test_samples, display_size),
            }
            if self.score_calibration is not None:
                thresholds = thresholds_for_samples(self.score_calibration, test_samples)
                output["image_thresholds"] = thresholds
                output["is_defect"] = (image_scores > thresholds).tolist()
            return output

    def score_samples(self, test_samples) -> np.ndarray:
        """Compute image scores without reconstructing full-resolution maps."""
        with self._inference_lock:
            if self._closed:
                raise RuntimeError("HRInferencer is closed")
            self._validate_samples(test_samples)

            batch_size = self.batch_size or len(test_samples)
            pending = [
                self._executor.submit(
                    inference_in_device,
                    test_samples,
                    task_group,
                    manager,
                    batch_size,
                    include_anomaly_maps=False,
                )
                for task_group, manager in zip(self.tasks_in_devices, self.model_managers)
            ]
            worker_results = [future.result() for future in pending]
            scores_by_path = {
                sample.image.image_path: []
                for sample in test_samples
            }
            for worker_result in worker_results:
                for path, result in worker_result.items():
                    scores_by_path[path].extend(result["scores"])
            return self.detector_class.get_image_score(
                [scores_by_path[sample.image.image_path] for sample in test_samples]
            )

    @staticmethod
    def _validate_samples(test_samples) -> None:
        if not isinstance(test_samples, list) or not test_samples:
            raise ValueError("test_samples must be a non-empty list")
        if any(not isinstance(sample, HRSample) for sample in test_samples):
            raise TypeError("Every test sample must be an HRSample")
        paths = [os.path.abspath(sample.image.image_path) for sample in test_samples]
        if len(paths) != len(set(paths)):
            raise ValueError("Inference sample image paths must be unique")
        if any(not isinstance(sample.clsname, str) or not sample.clsname.strip() for sample in test_samples):
            raise ValueError("Every inference sample must have a non-empty clsname")

    @staticmethod
    def _build_display_images(test_samples, display_size):
        if display_size is None:
            return None
        display_size = (
            (display_size, display_size)
            if isinstance(display_size, int) and not isinstance(display_size, bool)
            else tuple(display_size)
        )
        if len(display_size) != 2 or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in display_size
        ):
            raise ValueError("display_size must contain two positive integers")
        display_images = {}
        for sample in test_samples:
            sample.open()
            try:
                display_images[sample.image.image_path] = cv2.resize(
                    sample.image.image,
                    display_size,
                    interpolation=cv2.INTER_LINEAR,
                )
            finally:
                sample.close()
        return display_images

    def close(self):
        with self._inference_lock:
            if self._closed:
                return
            self._executor.shutdown(wait=True, cancel_futures=True)
            for manager in self.model_managers:
                manager.close()
            self.model_managers.clear()
            self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError("HRInferencer is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
