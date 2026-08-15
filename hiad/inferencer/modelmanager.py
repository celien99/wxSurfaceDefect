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
    ):
        gpu_device = torch.device(f"cuda:{gpu_id}")
        self.detectors = {}

        for task in tasks:
            task_name = task['name']
            detector_config = detector_config_for_task(config, task)

            detector = detector_class(
                **detector_config,
                device=gpu_device,
                logger=None,
                seed=0,
            )
            checkpoint_path = os.path.join(checkpoint_root, f'{task_name}_weight.pkl')
            detector.load_checkpoint(checkpoint_path)
            self.detectors[task_name] = detector

    def get_detector(self, task_name):
        return self.detectors[task_name]

    def close(self) -> None:
        self.detectors.clear()
