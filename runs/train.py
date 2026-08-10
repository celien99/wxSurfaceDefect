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

from hiad.detectors import HRDinomaly
from hiad.constants import DINO_PATCH_SIZE
from hiad.task import DynamicTaskGenerator
from hiad.trainer import HRTrainer
from hiad.trainer.sources import load_unified_training_samples
from hiad.runtime.logging import create_logger
from hiad.task import print_task_summary


def parse_args():
    parser = argparse.ArgumentParser(description="HiAD Training")
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
    parser.add_argument("--gpus", default="0,1")
    args = parser.parse_args()

    if args.patch_size <= 0 or args.patch_size % DINO_PATCH_SIZE != 0:
        parser.error(
            f"--patch-size must be a positive multiple of {DINO_PATCH_SIZE}"
        )
    if args.stride != -1 and (args.stride <= 0 or args.stride > args.patch_size):
        parser.error("--stride must be -1 or in [1, patch-size]")
    if not args.ds_factors or args.ds_factors[0] != 0 or args.ds_factors != sorted(set(args.ds_factors)):
        parser.error("--ds-factors must be unique, sorted, non-negative, and start with 0")
    if args.fusion_weights is not None and (
        len(args.fusion_weights) != len(args.ds_factors)
        or any(w < 0 for w in args.fusion_weights)
        or sum(args.fusion_weights) <= 0
    ):
        parser.error("--fusion-weights must match --ds-factors, be non-negative, and have a positive sum")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.thumbnail_size <= 0 or args.thumbnail_size % DINO_PATCH_SIZE != 0:
        parser.error(
            f"--thumbnail-size must be a positive multiple of {DINO_PATCH_SIZE}"
        )
    return args


if __name__ == "__main__":
    args = parse_args()
    gpu_ids = [int(g.strip()) for g in args.gpus.split(",") if g.strip()]
    if not gpu_ids:
        raise ValueError("At least one GPU id is required")

    os.makedirs(args.checkpoint_root, exist_ok=True)
    os.makedirs(args.log_root, exist_ok=True)

    with open(args.config, encoding="utf-8") as config_file:
        loaded_config = yaml.safe_load(config_file)
    if not isinstance(loaded_config, dict):
        raise TypeError("Training config must be a mapping")
    if "preprocessing" not in loaded_config or "scoring" not in loaded_config:
        raise ValueError("Training config requires preprocessing and scoring mappings")
    preprocessing_config = loaded_config.pop("preprocessing")
    scoring_config = loaded_config.pop("scoring")
    if not isinstance(preprocessing_config, dict):
        raise TypeError("preprocessing config must be a mapping")
    if not isinstance(scoring_config, dict):
        raise TypeError("scoring config must be a mapping")

    patch_config = EasyDict(copy.deepcopy(loaded_config))
    shutil.copy(args.config, os.path.join(args.log_root, "config.yaml"))

    thumbnail_config = EasyDict(copy.deepcopy(loaded_config))
    thumbnail_config.thumbnail_size = args.thumbnail_size
    config = EasyDict(
        patch=patch_config,
        thumbnail=thumbnail_config,
        preprocessing=EasyDict(preprocessing_config),
        scoring=EasyDict(scoring_config),
    )

    with open(os.path.join(args.log_root, "args.json"), "w") as args_file:
        json.dump(vars(args), args_file, indent=4)

    main_logger = create_logger("main", os.path.join(args.log_root, "train.log"), print_console=True)
    main_logger.info(args)

    train_samples, categories = load_unified_training_samples(args.data_root)
    main_logger.info("Unified training categories: %s", ", ".join(categories))

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

    trainer.train(
        train_samples=train_samples,
        gpu_ids=gpu_ids,
        main_logger=main_logger,
    )

    main_logger.info(f"Training done. Checkpoints saved to {args.checkpoint_root}")
