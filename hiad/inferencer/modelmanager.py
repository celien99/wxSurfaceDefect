import os

import torch

from hiad.constants import TASK_TYPE_DYNAMIC_PATCH
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
        self.cpu_device = torch.device("cpu")
        self.models_per_gpu = models_per_gpu
        if self.models_per_gpu <= 0:
            raise ValueError("models_per_gpu must be positive")
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
            checkpoint_path = os.path.join(checkpoint_root, f'{task_name}_weight.pt')
            detector.load_checkpoint(checkpoint_path)
            detector.to_device(self.cpu_device)
            self.models.append({
                "name": task_name,
                "detector": detector,
                "gpu": False,
            })

        load_task_names = self.get_device_task_names(gpu=False)[:self.models_per_gpu]
        for load_task_name in load_task_names:
            self.change_model_device(load_task_name, target_device='gpu')

    def get_detector(self, task_name, must_in_gpu=True):
        for model in self.models:
            if model['name'] != task_name:
                continue
            if not must_in_gpu or model['gpu']:
                return model['detector']

            gpu_tasks = self.get_device_task_names(gpu=True)
            if not gpu_tasks:
                raise RuntimeError("No GPU model slot is available")
            self.change_model_device(gpu_tasks[0], target_device='cpu')
            self.change_model_device(task_name, target_device='gpu')
            return model['detector']
        raise KeyError(f"Unknown task: {task_name}")

    def get_device_task_names(self, gpu: bool):
        return [model["name"] for model in self.models if model["gpu"] == gpu]

    def get_dynamic_scoring_identity(self):
        dynamic_names = [
            task["name"]
            for task in self.tasks
            if task["type"] == TASK_TYPE_DYNAMIC_PATCH
        ]
        if not dynamic_names:
            return None
        if len(dynamic_names) != 1:
            raise ValueError("A model manager cannot own multiple dynamic patch tasks")
        detector = self.get_detector(dynamic_names[0], must_in_gpu=False)
        fusion_weights = detector.fusion_weights
        return {
            "anomaly_distance": detector.anomaly_distance,
            "use_fp16": detector.use_fp16,
            "fusion_weights": None if fusion_weights is None else list(fusion_weights),
        }

    def change_model_device(self, task_name, target_device='gpu'):
        if target_device not in {'gpu', 'cpu'}:
            raise ValueError("target_device must be 'gpu' or 'cpu'")
        for model in self.models:
            if model['name'] != task_name:
                continue
            device = self.gpu_device if target_device == 'gpu' else self.cpu_device
            model["detector"].to_device(device)
            model['gpu'] = target_device == 'gpu'
            return
        raise KeyError(f"Unknown task: {task_name}")

    def offload_all(self) -> list[str]:
        gpu_task_names = self.get_device_task_names(gpu=True)
        for task_name in gpu_task_names:
            self.change_model_device(task_name, target_device='cpu')
        return gpu_task_names

    def restore_gpu_tasks(self, task_names: list[str]) -> None:
        if not isinstance(task_names, list):
            raise TypeError("task_names must be a list")
        if any(not isinstance(task_name, str) for task_name in task_names):
            raise TypeError("Every task name must be a string")
        if len(set(task_names)) != len(task_names):
            raise ValueError("task_names must not contain duplicates")
        if len(task_names) > self.models_per_gpu:
            raise ValueError("Requested GPU tasks exceed the configured model slot limit")

        known_task_names = {model['name'] for model in self.models}
        unknown_task_names = [
            task_name for task_name in task_names if task_name not in known_task_names
        ]
        if unknown_task_names:
            raise KeyError(f"Unknown tasks: {unknown_task_names}")

        target_task_names = set(task_names)
        for task_name in self.get_device_task_names(gpu=True):
            if task_name not in target_task_names:
                self.change_model_device(task_name, target_device='cpu')
        current_gpu_tasks = set(self.get_device_task_names(gpu=True))
        for task_name in task_names:
            if task_name not in current_gpu_tasks:
                self.change_model_device(task_name, target_device='gpu')
