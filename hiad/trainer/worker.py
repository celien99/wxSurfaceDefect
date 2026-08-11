import os

import torch

from hiad.datasets import SourceGroupedRandomSampler, StreamingTaskDataset
from hiad.detectors.config import detector_config_for_task
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
    seed_everything(seed)
    logger = create_logger(
        f"train_logger_device{gpu_id}",
        os.path.join(log_root, f"train_log_device{gpu_id}.log"),
    )
    logger.info("Device %s start training", gpu_id)
    logger.info("Task Num: %d", len(tasks))

    for index, task in enumerate(tasks, start=1):
        task_name = task["name"]
        logger.info("[%d/%d] Task %s start loading images", index, len(tasks), task_name)

        dataset = StreamingTaskDataset(
            list(train_samples),
            task,
            training=True,
        )
        if len(dataset) == 0:
            raise ValueError(f"Task {task_name} has no training samples")

        detector_config = detector_config_for_task(config, task)
        detector = detector_class(
            **detector_config,
            logger=logger,
            device=device,
            seed=seed,
            fusion_weights=fusion_weights,
        )
        sampler_generator = torch.Generator()
        sampler_generator.manual_seed(seed)
        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=SourceGroupedRandomSampler(
                dataset,
                generator=sampler_generator,
            ),
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )

        logger.info("Task %s train dataset len is: %d", task_name, len(dataset))
        checkpoint_path = os.path.join(checkpoint_root, f"{task_name}_weight.pkl")
        detector.train_step(train_dataloader, task_name)
        detector.save_checkpoint(checkpoint_path)
        logger.info("Task %s checkpoint saved as %s", task_name, checkpoint_path)

        detector.to_device(torch.device("cpu"))
        del detector
        torch.cuda.empty_cache()

    logger.info("All tasks are done.")
