import os

import torch

from hiad.detectors.config import detector_config_for_task


class ModelManager:
    def __init__(
        self,
        tasks,
        detector_class,
        config,
        checkpoint_root: str,
        gpu_id: int,
        models_per_gpu: int,
    ):
        self.tasks = tasks
        self.detector_class = detector_class
        self.config = config
        self.gpu_device = torch.device(f"cuda:{gpu_id}")
        self.models_per_gpu = models_per_gpu
        if self.models_per_gpu <= 0:
            raise ValueError("models_per_gpu must be positive")
        if self.models_per_gpu < len(self.tasks):
            raise ValueError(
                "Production inference requires every assigned model to remain on GPU; "
                f"models_per_gpu={self.models_per_gpu}, assigned_tasks={len(self.tasks)}"
            )
        self.models = []

        for task in self.tasks:
            task_name = task['name']
            detector_config = detector_config_for_task(config, task)

            detector = detector_class(
                **detector_config,
                device=self.gpu_device,
                logger=None,
                seed=0,
            )
            checkpoint_path = os.path.join(checkpoint_root, f'{task_name}_weight.pkl')
            detector.load_checkpoint(checkpoint_path)
            self.models.append({
                "name": task_name,
                "detector": detector,
                "gpu": True,
            })

    def get_detector(self, task_name, must_in_gpu=True):
        for model in self.models:
            if model['name'] != task_name:
                continue
            detector = model['detector']
            if must_in_gpu and (
                not model['gpu']
                or getattr(detector, "device", None) is None
                or detector.device.type != "cuda"
            ):
                raise RuntimeError(f"Task {task_name} is not resident on GPU")
            return detector
        raise KeyError(f"Unknown task: {task_name}")

    def get_device_task_names(self, gpu: bool):
        return [model["name"] for model in self.models if model["gpu"] == gpu]

    def score_top_k_values(self) -> set[int]:
        return {model["detector"].score_top_k for model in self.models}

    def close(self) -> None:
        self.models.clear()
