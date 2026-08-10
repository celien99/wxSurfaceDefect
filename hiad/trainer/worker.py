import os

import torch

from hiad.detectors.config import detector_config_for_task
from hiad.data.preparation import build_task_inputs_from_open_samples
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

    task_inputs, _ = build_task_inputs_from_open_samples(
        train_samples,
        tasks,
        logger=logger,
    )

    for task in tasks:
        task_name = task["name"]
        logger.info(f"Task {task_name} loading images")

        detector_config = detector_config_for_task(config, task)
        patches = task_inputs[task_name]["patches"]
        detector = detector_class(
            **detector_config,
            logger=logger,
            device=device,
            seed=seed,
            fusion_weights=fusion_weights,
        )
        logger.info(f"Task {task_name}: {len(patches)} patches for training")
        train_dataset = detector.create_dataset(
            patches,
            training=True,
            task_name=task_name,
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=len(train_dataset) >= batch_size,
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

    logger.info("All tasks are done.")
