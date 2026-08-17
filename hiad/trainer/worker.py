from __future__ import annotations

import os
from collections.abc import Mapping
from typing import cast

import torch

from hiad.data import HRSample
from hiad.datasets import SourceGroupedRandomSampler, StreamingTaskDataset
from hiad.detectors.base import BaseDetector
from hiad.detectors.config import DetectorConfig, detector_config_for_task
from hiad.runtime.logging import create_logger
from hiad.runtime.randomness import seed_everything
from hiad.task.contracts import TaskDefinition


def train_tasks_in_device(
    gpu_id: int,
    detector_class: type[BaseDetector],
    config: DetectorConfig,
    train_samples: list[HRSample],
    tasks: list[TaskDefinition],
    batch_size: int,
    checkpoint_root: str | os.PathLike[str],
    log_root: str | os.PathLike[str],
    seed: int,
) -> None:
    """在一个 CUDA 进程内顺序训练任务，避免多个大模型同时驻留显存。

    Args:
        gpu_id (int): 写入 ``CUDA_VISIBLE_DEVICES`` 的物理 CUDA 设备编号。
        detector_class (type[BaseDetector]): 当前任务使用的检测器类型。
        config (DetectorConfig): 共享且已验证的检测器配置。
        train_samples (list[HRSample]): 统一正常训练样本。
        tasks (list[TaskDefinition]): 分配给当前设备顺序执行的任务。
        batch_size (int): DataLoader 批量大小。
        checkpoint_root (str | os.PathLike[str]): 每任务权重输出目录。
        log_root (str | os.PathLike[str]): 当前设备训练日志目录。
        seed (int): 采样器及 Python/NumPy/PyTorch 随机种子。

    Raises:
        ValueError: 任一任务没有候选训练补丁。
        OSError: 日志、图像或检查点无法读取或写入。
        RuntimeError: 模型训练、正常证据拟合或 CUDA 执行失败。

    Notes:
        每个任务依次执行重建训练、正常证据拟合和权重保存，随后立即释放模型
        并清理 CUDA 缓存，确保进程内不同时驻留多个大模型。
    """
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
            **cast(Mapping[str, object], detector_config),
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
        # 原地释放任务模型；先搬回 CPU 会拖慢切换，也可能掩盖误走 CPU 推理的问题。
        torch.cuda.empty_cache()

    logger.info("All tasks are done.")
