import argparse
import copy
import os
import sys

import yaml
from easydict import EasyDict

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from hiad.data import HRSample, read_jsonl_records
from hiad.detectors import HRDinomaly
from hiad.detectors.config import validate_required_config
from hiad.evaluation import HREvaluator
from hiad.evaluation.metrics import compute_pro
from hiad.evaluation.metrics.torch_backend import (
    compute_imagewise_metrics,
    compute_pixelwise_metrics,
)
from hiad.inferencer import HRInferencer
from hiad.runtime.logging import create_logger
from hiad.runtime.prediction import (
    has_complete_image_annotations,
    has_complete_pixel_annotations,
    save_predictions,
)


def parse_args():
    parser = argparse.ArgumentParser(description="HiAD Dinomaly inference")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", default="test_uni.jsonl")
    parser.add_argument("--config", default="configs/dinomaly.yaml")
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--checkpoint-root", default="results/dinomaly_checkpoints")
    parser.add_argument("--log-root", default="results/dinomaly_logs")
    parser.add_argument("--vis-root", default="results/dinomaly_vis")
    parser.add_argument("--vis-size", default=1024, type=int)
    parser.add_argument("--gpus", default="0")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.vis_size <= 0:
        parser.error("--batch-size and --vis-size must be positive")
    return args


if __name__ == "__main__":
    args = parse_args()
    gpu_ids = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
    if not gpu_ids:
        raise ValueError("At least one GPU id is required")

    os.makedirs(args.log_root, exist_ok=True)
    os.makedirs(args.vis_root, exist_ok=True)
    with open(args.config, encoding="utf-8") as stream:
        loaded_config = yaml.safe_load(stream)
    if not isinstance(loaded_config, dict):
        raise TypeError("Inference config must be a mapping")
    validate_required_config(loaded_config)
    config = EasyDict(
        patch=EasyDict(copy.deepcopy(loaded_config)),
        refinement=EasyDict(copy.deepcopy(loaded_config)),
        thumbnail=EasyDict(copy.deepcopy(loaded_config)),
    )

    main_logger = create_logger(
        "main",
        os.path.join(args.log_root, "inference.log"),
        print_console=True,
    )
    main_logger.info(args)
    main_logger.info("Loading checkpoints from %s", args.checkpoint_root)

    test_meta = read_jsonl_records(os.path.join(args.data_root, args.manifest))
    test_samples = [
        HRSample(
            image=os.path.join(args.data_root, record["filename"]),
            foreground=(
                os.path.join(args.data_root, record["foreground"])
                if isinstance(record.get("foreground"), str) and record["foreground"]
                else None
            ),
            mask=os.path.join(args.data_root, record["mask"]) if record.get("mask") else None,
            clsname=record.get("clsname", "default"),
            label=record.get("label"),
            label_name=record.get("label_name"),
        )
        for record in test_meta
    ]
    inference_samples = [
        HRSample(
            image=os.path.join(args.data_root, record["filename"]),
            foreground=(
                os.path.join(args.data_root, record["foreground"])
                if isinstance(record.get("foreground"), str) and record["foreground"]
                else None
            ),
            clsname=record.get("clsname", "default"),
        )
        for record in test_meta
    ]

    with HRInferencer(
        detector_class=HRDinomaly,
        config=config,
        checkpoint_root=args.checkpoint_root,
        gpu_ids=gpu_ids,
        batch_size=args.batch_size,
    ) as inferencer:
        inference_result = inferencer.inference(
            inference_samples,
            display_size=args.vis_size,
        )

    predictions_path = os.path.join(args.log_root, "predictions.jsonl")
    save_predictions(predictions_path, test_samples, inference_result)
    main_logger.info("Per-image predictions saved to %s", predictions_path)

    evaluators = []
    if has_complete_image_annotations(test_meta):
        evaluators.append(compute_imagewise_metrics)
    else:
        main_logger.info("Image labels are incomplete; skipping image metrics")
    if has_complete_pixel_annotations(test_meta):
        evaluators.extend([compute_pixelwise_metrics, compute_pro])
    elif evaluators:
        main_logger.info(
            "Pixel masks are incomplete; skipping pixel AUROC/AP/F1 and PRO metrics"
        )

    evaluator = HREvaluator(log_root=args.log_root, vis_root=args.vis_root)
    evaluator.evaluate(
        test_samples=test_samples,
        inference_result=inference_result,
        gpu_ids=gpu_ids,
        evaluators=evaluators,
        main_logger=main_logger,
        vis_size=args.vis_size,
    )
    main_logger.info("Inference done. Results logged to %s", args.log_root)
