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
        patches_per_source = int(detector_config.patches_per_source)
        logger.info(
            "Task %s index ready: source_images=%d, candidate_patches=%d; "
            "initializing DINOv3 detector",
            task_name,
            len(train_samples),
            len(dataset),
        )
        detector = detector_class(
            **detector_config,
            logger=logger,
            device=device,
            seed=seed,
        )
        logger.info("Task %s detector is resident on %s", task_name, device)
        sampler_generator = torch.Generator()
        sampler_generator.manual_seed(seed)
        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=SourceGroupedRandomSampler(
                dataset,
                patches_per_source=patches_per_source,
                generator=sampler_generator,
            ),
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )

        logger.info(
            "Task %s training patches: total=%d, sampled_per_epoch=%d, batches_per_epoch=%d",
            task_name,
            len(dataset),
            len(train_dataloader.sampler),
            len(train_dataloader),
        )
        checkpoint_path = os.path.join(checkpoint_root, f"{task_name}_weight.pkl")
        detector.train_step(train_dataloader, task_name)
        detector.fit_normal_evidence(train_dataloader)
        detector.save_checkpoint(checkpoint_path)
        logger.info("Task %s checkpoint saved as %s", task_name, checkpoint_path)

        del detector
        # Release the task model in-place. Moving a full DINOv3 model to CPU
        # between tasks is slow and can hide an accidental CPU inference path.
        torch.cuda.empty_cache()

    logger.info("All tasks are done.")
