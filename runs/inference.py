from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping

import yaml

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from hiad.data import HRSample, read_jsonl_records
from hiad.detectors import HRDinomaly
from hiad.evaluation import HREvaluator
from hiad.evaluation.execution import MetricEvaluator
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


def parse_args() -> argparse.Namespace:
    """解析并校验推理脚本的路径、批量、可视化和 GPU 参数。

    Returns:
        argparse.Namespace: 通过正数批量大小和显示尺寸校验的参数对象。
    """
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


def _sample_from_record(
    data_root: str,
    record: Mapping[str, object],
    *,
    include_annotations: bool,
) -> HRSample:
    """把统一清单记录转换为推理或评估使用的样本。

    Args:
        data_root (str): 清单相对路径的根目录。
        record (Mapping[str, object]): 包含文件名、类别及可选标注的清单记录。
        include_annotations (bool): 是否加载掩码、图像标签和标签名称；纯推理样本
            设为 ``False`` 以防标注信息进入推理链路。

    Returns:
        HRSample: 路径已拼接到 ``data_root`` 的延迟加载样本。

    Raises:
        ValueError: 文件名、类别或任一可选标注字段类型不合法。
    """
    filename = record.get("filename")
    if not isinstance(filename, str) or not filename:
        raise ValueError("Every inference record must contain a filename")
    category = record.get("clsname", "default")
    if not isinstance(category, str) or not category.strip():
        raise ValueError("Every inference record must contain a valid clsname")
    foreground = record.get("foreground")
    mask = record.get("mask") if include_annotations else None
    label = record.get("label") if include_annotations else None
    label_name = record.get("label_name") if include_annotations else None
    if foreground is not None and not isinstance(foreground, str):
        raise ValueError("foreground must be a relative path or null")
    if mask is not None and not isinstance(mask, str):
        raise ValueError("mask must be a relative path or null")
    if label is not None and (
        isinstance(label, bool) or not isinstance(label, int)
    ):
        raise ValueError("label must be an integer or null")
    if label_name is not None and not isinstance(label_name, str):
        raise ValueError("label_name must be a string or null")
    return HRSample(
        image=os.path.join(data_root, filename),
        foreground=(
            os.path.join(data_root, foreground) if foreground else None
        ),
        mask=os.path.join(data_root, mask) if mask else None,
        clsname=category,
        label=label,
        label_name=label_name,
    )


def main() -> None:
    """执行完整推理、预测落盘，并在标注完整时计算对应评估指标。"""
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

    main_logger = create_logger(
        "main",
        os.path.join(args.log_root, "inference.log"),
        print_console=True,
    )
    main_logger.info(args)
    main_logger.info("Loading checkpoints from %s", args.checkpoint_root)

    test_meta = read_jsonl_records(os.path.join(args.data_root, args.manifest))
    test_samples = [
        _sample_from_record(args.data_root, record, include_annotations=True)
        for record in test_meta
    ]
    inference_samples = [
        _sample_from_record(args.data_root, record, include_annotations=False)
        for record in test_meta
    ]

    with HRInferencer(
        detector_class=HRDinomaly,
        config=loaded_config,
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

    evaluators: list[MetricEvaluator] = []
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


if __name__ == "__main__":
    main()
