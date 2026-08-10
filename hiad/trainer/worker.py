import os

import torch

from hiad.datasets import StreamingTaskDataset
from hiad.detectors.config import detector_config_for_task
from hiad.preprocessing import ForegroundPreprocessorRegistry
from hiad.runtime.logging import create_logger
from hiad.runtime.randomness import seed_everything


def train_tasks_in_device(
    gpu_id,
    detector_class,
    config,
    train_samples,
    tasks,
    batch_size,
    checkpoint_root,
    log_root,
    seed,
    fusion_weights=None,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda")
    cpu_device = torch.device("cpu")
    seed_everything(seed)
    logger = create_logger(
        f"train_logger_device{gpu_id}",
        os.path.join(log_root, f"train_log_device{gpu_id}.log"),
    )
    logger.info(f"Device {gpu_id} start training")

    preprocessing_config = getattr(config, "preprocessing", None)
    if preprocessing_config is None:
        raise ValueError("Training worker requires preprocessing config")

    preprocessors = ForegroundPreprocessorRegistry.from_checkpoint(
        str(checkpoint_root),
        device,
        runtime_config=preprocessing_config,
        logger=logger,
    )
    try:
        for task in tasks:
            task_name = task["name"]
            logger.info(f"Task {task_name} streaming dataset")

            detector_config = detector_config_for_task(config, task)
            dataset = StreamingTaskDataset(
                train_samples,
                task,
                preprocessors,
                training=True,
            )
            detector = detector_class(
                **detector_config,
                logger=logger,
                device=device,
                seed=seed,
                fusion_weights=fusion_weights,
            )
            logger.info(f"Task {task_name}: {len(dataset)} patches for training")
            train_dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
                drop_last=len(dataset) >= batch_size,
            )
            if len(train_dataloader) == 0:
                raise ValueError(f"Task {task_name} has no training batches")

            checkpoint_path = os.path.join(
                checkpoint_root,
                f"{task_name}_weight.pt",
            )
            detector.train_step(train_dataloader, task_name)
            detector.save_checkpoint(checkpoint_path)
            logger.info(f"Task {task_name} checkpoint saved as {checkpoint_path}")
            detector.to_device(cpu_device)
            del detector
            torch.cuda.empty_cache()
    finally:
        preprocessors.close()

    logger.info("All tasks are done.")
