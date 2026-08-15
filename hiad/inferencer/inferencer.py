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

from hiad.constants import (
    TASK_TYPE_DYNAMIC_PATCH,
    TASK_TYPE_REFINEMENT_PATCH,
    TASK_TYPE_THUMBNAIL,
)
from hiad.data import HRImageIndex, HRSample
from hiad.datasets import StreamingTaskDataset
from hiad.inferencer.modelmanager import ModelManager
from hiad.detectors.config import validate_required_config
from hiad.inferencer.refinement import (
    build_routing_map,
    merge_refinement_maps,
    select_refinement_regions,
)
from hiad.runtime.devices import validate_gpu_ids
from hiad.runtime.decision import (
    apply_quality_gate,
    classify_score,
    component_statistics,
    image_score_from_statistics,
    top_k_map_score,
)
from hiad.runtime.partition import round_robin_partition
from hiad.runtime.prediction import threshold_anomaly_maps
from hiad.runtime.quality import assess_image_quality
from hiad.runtime.score_calibration import (
    component_thresholds_for_samples,
    load_score_calibration,
    pixel_thresholds_for_samples,
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
    regions_by_path=None,
):
    paths = [sample.image.image_path for sample in test_samples]
    results = {
        path: {
            "image_size": None,
            "patches": [],
            "thumbnail": None,
            "thumbnail_score": None,
        }
        for path in paths
    }

    for task in task_group:
        task_name = task["name"]
        detector = model_manager.get_detector(task_name)
        dataset = StreamingTaskDataset(
            copy.deepcopy(test_samples),
            task,
            training=False,
            regions_by_path=regions_by_path,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        predictions = detector.inference_step(dataloader)
        if len(predictions) != len(dataset.records):
            raise RuntimeError(
                f"Task {task_name} returned {len(predictions)} predictions for "
                f"{len(dataset.records)} inputs"
            )

        if task["type"] in {TASK_TYPE_DYNAMIC_PATCH, TASK_TYPE_REFINEMENT_PATCH}:
            for record, prediction in zip(dataset.records, predictions):
                path = record["image_path"]
                results[path]["image_size"] = record["image_size"]
                results[path]["patches"].append((record, prediction["anomaly_map"]))
        elif task["type"] == TASK_TYPE_THUMBNAIL:
            for record, prediction in zip(dataset.records, predictions):
                path = record["image_path"]
                results[path]["image_size"] = record["image_size"]
                results[path]["thumbnail"] = prediction["anomaly_map"]
                results[path]["thumbnail_score"] = prediction["score"]
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
        validate_required_config(self.config)

        task_groups = [group for group in round_robin_partition(self.tasks, len(self.gpu_ids)) if group]

        self.coarse_tasks_in_devices = [
            [task for task in task_group if task["type"] != TASK_TYPE_REFINEMENT_PATCH]
            for task_group in task_groups
        ]
        self.refinement_tasks_in_devices = [
            [task for task in task_group if task["type"] == TASK_TYPE_REFINEMENT_PATCH]
            for task_group in task_groups
        ]
        refinement_tasks = [
            task for task in self.tasks if task["type"] == TASK_TYPE_REFINEMENT_PATCH
        ]
        self.refinement_task = refinement_tasks[0]
        self.model_managers = [
            ModelManager(
                tasks,
                detector_class,
                self.config,
                self.checkpoint_root,
                self.gpu_ids[index],
            )
            for index, tasks in enumerate(tqdm(task_groups, desc="Loading checkpoints..."))
        ]
        self.score_calibration = (
            load_score_calibration(self.checkpoint_root)
            if require_score_calibration
            else None
        )
        self.map_gaussian_sigma = float(self.config.map_gaussian_sigma)
        self.decision_recheck_margin_ratio = float(
            self.config.decision_recheck_margin_ratio
        )
        self.quality_thresholds = {
            key: float(self.config[key])
            for key in (
                "min_mean_luminance",
                "max_mean_luminance",
                "max_clipped_fraction",
                "min_focus_variance",
            )
        }
        self.global_routing_weight = float(
            self.config.global_routing_weight
        )
        self.score_top_k = int(self.config.score_top_k)
        self._executor = ThreadPoolExecutor(max_workers=len(task_groups))
        self._inference_lock = Lock()
        self._closed = False

    def inference(self, test_samples, *, display_size=None):
        with self._inference_lock:
            if self._closed:
                raise RuntimeError("HRInferencer is closed")
            self._validate_samples(test_samples)
            quality_results = self._assess_quality(test_samples)

            batch_size = self.batch_size or len(test_samples)
            worker_results = self._run_inference_groups(
                test_samples,
                self.coarse_tasks_in_devices,
                batch_size,
            )

            merged = {
                sample.image.image_path: {
                    "image_size": None,
                    "patches": [],
                    "thumbnail": None,
                    "thumbnail_score": None,
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
                    if result["thumbnail"] is not None:
                        if merged[path]["thumbnail"] is not None:
                            raise ValueError(f"Duplicate thumbnail prediction for {path}")
                        merged[path]["thumbnail"] = result["thumbnail"]
                        merged[path]["thumbnail_score"] = result["thumbnail_score"]

            anomaly_maps = []
            global_scores = []
            refinement_task = self.refinement_task
            regions_by_path = {}
            for sample in test_samples:
                path = sample.image.image_path
                result = merged[path]
                if result["image_size"] is None or not result["patches"]:
                    raise ValueError(f"Incomplete dynamic patch prediction for {path}")
                patch_map = _gather_patch_predictions(result["patches"], result["image_size"])
                final_map = patch_map
                if (
                    result["thumbnail"] is None
                    or result["thumbnail_score"] is None
                ):
                    raise ValueError(f"Incomplete global context prediction for {path}")
                image_width, image_height = result["image_size"]
                global_context_map = cv2.resize(
                    np.asarray(result["thumbnail"], dtype=np.float32),
                    (image_width, image_height),
                    interpolation=cv2.INTER_LINEAR,
                )
                if self.map_gaussian_sigma > 0:
                    final_map = gaussian_filter(
                        final_map,
                        sigma=self.map_gaussian_sigma,
                    )
                final_map = np.asarray(final_map, dtype=np.float32)
                routing_map = build_routing_map(
                    final_map,
                    global_context_map,
                    self.global_routing_weight,
                )
                regions_by_path[path] = select_refinement_regions(
                    routing_map,
                    threshold=float(
                        np.quantile(
                            routing_map,
                            refinement_task["refinement_quantile"],
                        )
                    ),
                    tile_size=refinement_task["patch_size"],
                    min_area=refinement_task["refinement_min_area"],
                    safety_fraction=refinement_task["refinement_safety_fraction"],
                )
                anomaly_maps.append(final_map)
                global_scores.append(float(result["thumbnail_score"]))

            anomaly_maps = self._apply_refinement(
                test_samples,
                anomaly_maps,
                regions_by_path,
                batch_size,
            )
            image_scores = np.asarray(
                [
                    max(top_k_map_score(anomaly_map, self.score_top_k), global_score)
                    for anomaly_map, global_score in zip(anomaly_maps, global_scores)
                ],
                dtype=np.float32,
            )
            output = {
                "image_paths": [sample.image.image_path for sample in test_samples],
                "image_scores": image_scores,
                "anomaly_maps": anomaly_maps,
                "display_images": self._build_display_images(test_samples, display_size),
                "quality_results": quality_results,
            }
            if self.score_calibration is not None:
                thresholds = thresholds_for_samples(self.score_calibration, test_samples)
                pixel_thresholds = pixel_thresholds_for_samples(
                    self.score_calibration, test_samples
                )
                component_summaries = [
                    component_statistics(anomaly_map, pixel_threshold)
                    for anomaly_map, pixel_threshold in zip(anomaly_maps, pixel_thresholds)
                ]
                component_scores = [
                    image_score_from_statistics(summary, image_score)
                    for summary, image_score in zip(component_summaries, image_scores)
                ]
                decision_thresholds = (
                    component_thresholds_for_samples(
                        self.score_calibration,
                        test_samples,
                    )
                    if "global_component_threshold" in self.score_calibration
                    else thresholds
                )
                decisions = [
                    classify_score(
                        score,
                        threshold,
                        threshold * self.decision_recheck_margin_ratio,
                    )
                    for score, threshold in zip(component_scores, decision_thresholds)
                ]
                output["decision_thresholds"] = decision_thresholds
                output["component_scores"] = component_scores
                output["raw_image_scores"] = image_scores
                output["image_scores"] = np.asarray(component_scores, dtype=np.float32)
                output["image_thresholds"] = decision_thresholds
                output["decisions"] = decisions
                output["decision_reasons"] = [
                    "score_at_or_below_threshold"
                    if decision == "OK"
                    else "score_within_recheck_margin"
                    if decision == "RECHECK"
                    else "score_above_recheck_margin"
                    for decision in decisions
                ]
                output["component_summaries"] = component_summaries
                for index, quality in enumerate(quality_results):
                    decisions[index], quality_reason = apply_quality_gate(
                        decisions[index],
                        quality["reasons"],
                    )
                    if quality_reason is not None:
                        output["decision_reasons"][index] = quality_reason
                output["is_defect"] = [decision == "NG" for decision in decisions]
                output["pixel_thresholds"] = pixel_thresholds
                output["binary_anomaly_maps"] = threshold_anomaly_maps(
                    anomaly_maps, pixel_thresholds
                )
            return output

    def score_samples(self, test_samples) -> np.ndarray:
        """Compute image scores through the mandatory coarse-to-fine path."""
        return self.inference(test_samples)["image_scores"]

    def _run_inference_groups(
        self,
        test_samples,
        task_groups,
        batch_size,
        *,
        regions_by_path=None,
    ):
        pending = [
            self._executor.submit(
                inference_in_device,
                test_samples,
                task_group,
                manager,
                batch_size,
                regions_by_path=regions_by_path,
            )
            for task_group, manager in zip(task_groups, self.model_managers)
            if task_group
        ]
        return [future.result() for future in pending]

    def _apply_refinement(
        self,
        test_samples,
        base_maps,
        regions_by_path,
        batch_size,
    ):
        worker_results = self._run_inference_groups(
            test_samples,
            self.refinement_tasks_in_devices,
            batch_size,
            regions_by_path=regions_by_path,
        )
        refinements_by_path = {
            sample.image.image_path: []
            for sample in test_samples
        }
        for worker_result in worker_results:
            for path, result in worker_result.items():
                refinements_by_path[path].extend(result["patches"])

        refined_maps = []
        for sample, base_map in zip(test_samples, base_maps):
            path = sample.image.image_path
            refinements = []
            for record, anomaly_map in refinements_by_path[path]:
                x, y, width, height = record["source_xywh"]
                refinements.append((
                    HRImageIndex(
                        x=x,
                        y=y,
                        width=width,
                        height=height,
                    ),
                    anomaly_map,
                ))
            refined_maps.append(
                merge_refinement_maps(
                    base_map,
                    refinements,
                    image_size=(base_map.shape[1], base_map.shape[0]),
                )
            )
        return refined_maps

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

    def _assess_quality(self, test_samples) -> list[dict]:
        results = []
        for sample in test_samples:
            sample.open()
            try:
                results.append(
                    assess_image_quality(
                        sample.image.image,
                        self.quality_thresholds,
                        sample.foreground.image if sample.foreground is not None else None,
                    )
                )
            finally:
                sample.close()
        return results

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
