import argparse
import copy
import json
import os
import shutil
import sys

import yaml
from easydict import EasyDict

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from hiad.constants import DINO_PATCH_SIZE
from hiad.detectors import HRDinomaly
from hiad.runtime.logging import create_logger
from hiad.task import DynamicTaskGenerator, print_task_summary
from hiad.trainer import HRTrainer
from hiad.trainer.sources import load_unified_training_samples


def parse_args():
    parser = argparse.ArgumentParser(description="HiAD Dinomaly training")
    parser.add_argument("--data-root", default="data/MVTec-2K")
    parser.add_argument("--config", default="configs/dinomaly.yaml")
    parser.add_argument("--patch-size", default=512, type=int)
    parser.add_argument("--stride", default=-1, type=int)
    parser.add_argument("--ds-factors", default=[0, 1], nargs="+", type=int)
    parser.add_argument("--fusion-weights", default=None, nargs="+", type=float)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--thumbnail-size", default=512, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--checkpoint-root", default="results/dinomaly_checkpoints")
    parser.add_argument("--log-root", default="results/dinomaly_logs")
    parser.add_argument("--gpus", default="0")
    args = parser.parse_args()

    if args.patch_size <= 0 or args.patch_size % DINO_PATCH_SIZE:
        parser.error(f"--patch-size must be a positive multiple of {DINO_PATCH_SIZE}")
    if args.stride != -1 and (args.stride <= 0 or args.stride > args.patch_size):
        parser.error("--stride must be -1 or in [1, patch-size]")
    if not args.ds_factors or args.ds_factors[0] != 0 or args.ds_factors != sorted(set(args.ds_factors)):
        parser.error("--ds-factors must be unique, sorted, non-negative, and start with 0")
    if args.fusion_weights is not None and (
        len(args.fusion_weights) != len(args.ds_factors)
        or any(weight < 0 for weight in args.fusion_weights)
        or sum(args.fusion_weights) <= 0
    ):
        parser.error("--fusion-weights must match --ds-factors and have a positive sum")
    if args.batch_size <= 0 or args.thumbnail_size <= 0:
        parser.error("--batch-size and --thumbnail-size must be positive")
    if args.thumbnail_size % DINO_PATCH_SIZE:
        parser.error(f"--thumbnail-size must be a multiple of {DINO_PATCH_SIZE}")
    return args


if __name__ == "__main__":
    args = parse_args()
    gpu_ids = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
    if not gpu_ids:
        raise ValueError("At least one GPU id is required")

    os.makedirs(args.checkpoint_root, exist_ok=True)
    os.makedirs(args.log_root, exist_ok=True)
    shutil.copy(args.config, os.path.join(args.checkpoint_root, os.path.basename(args.config)))
    with open(os.path.join(args.checkpoint_root, "args.json"), "w", encoding="utf-8") as stream:
        json.dump(vars(args), stream, indent=2)

    with open(args.config, encoding="utf-8") as stream:
        loaded_config = yaml.safe_load(stream)
    if not isinstance(loaded_config, dict):
        raise TypeError("Training config must be a mapping")

    patch_config = EasyDict(copy.deepcopy(loaded_config))
    thumbnail_config = EasyDict(copy.deepcopy(loaded_config))
    config = EasyDict(patch=patch_config, thumbnail=thumbnail_config)

    main_logger = create_logger(
        "main",
        os.path.join(args.log_root, "main.log"),
        print_console=True,
    )
    main_logger.info(args)
    train_samples, categories = load_unified_training_samples(args.data_root)
    main_logger.info("Training categories: %s", ", ".join(categories))
    main_logger.info("Training samples: %d (all samples are used)", len(train_samples))

    tasks = DynamicTaskGenerator(
        patch_size=args.patch_size,
        ds_factors=args.ds_factors,
        stride=None if args.stride == -1 else args.stride,
    ).create_tasks(thumbnail_size=args.thumbnail_size)
    print_task_summary(tasks)

    trainer = HRTrainer(
        detector_class=HRDinomaly,
        config=config,
        batch_size=args.batch_size,
        checkpoint_root=args.checkpoint_root,
        log_root=args.log_root,
        tasks=tasks,
        seed=args.seed,
        fusion_weights=args.fusion_weights,
    )
    trainer.train(train_samples=train_samples, gpu_ids=gpu_ids, main_logger=main_logger)
    main_logger.info("Training done. Checkpoints saved to %s", args.checkpoint_root)
