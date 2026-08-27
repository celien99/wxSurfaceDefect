from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import yaml

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from hiad.data import HRSample, read_jsonl_records
from hiad.detectors import HRDinomaly
from hiad.inferencer import HRInferencer


def parse_args(argv=None) -> argparse.Namespace:
    """解析 parity 校验参数；``argv=None`` 时读命令行。"""
    parser = argparse.ArgumentParser(
        description="Compare legacy vs pipeline inference for decision parity"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", default="test_uni.jsonl")
    parser.add_argument("--config", default="configs/dinomaly.yaml")
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--checkpoint-root", default="results/dinomaly_checkpoints")
    parser.add_argument("--log-root", default="results/parity")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--tolerance", default=1e-4, type=float)
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.tolerance <= 0:
        parser.error("--batch-size and --tolerance must be positive")
    return args


def _sample(data_root: str, record: dict) -> HRSample:
    return HRSample(
        image=os.path.join(data_root, record["filename"]),
        clsname=record.get("clsname", "default"),
    )


def _compare(results_new, results_legacy, tolerance: float) -> dict:
    decisions_new = results_new.get("decisions")
    decisions_legacy = results_legacy.get("decisions")
    flips = 0
    if decisions_new is not None and decisions_legacy is not None:
        for path, new, legacy in zip(
            results_new["image_paths"], decisions_new, decisions_legacy
        ):
            if new != legacy:
                flips += 1
                print(f"DECISION FLIP {path}: legacy={legacy} pipeline={new}")
    maps_new = results_new["anomaly_maps"]
    maps_legacy = results_legacy["anomaly_maps"]
    max_abs_diff = 0.0
    for path, new, legacy in zip(
        results_new["image_paths"], maps_new, maps_legacy
    ):
        if new.shape != legacy.shape:
            raise ValueError(f"Shape mismatch for {path}")
        max_abs_diff = max(max_abs_diff, float(np.abs(new - legacy).max()))
    return {
        "decision_flips": flips,
        "max_abs_diff": max_abs_diff,
        "passed": flips == 0 and max_abs_diff < tolerance,
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    gpu_ids = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
    if not gpu_ids:
        raise ValueError("At least one GPU id is required")
    with open(args.config, encoding="utf-8") as stream:
        loaded_config = yaml.safe_load(stream)
    records = read_jsonl_records(os.path.join(args.data_root, args.manifest))
    samples = [_sample(args.data_root, record) for record in records]

    results = {}
    for use_pipeline in (False, True):
        with HRInferencer(
            detector_class=HRDinomaly,
            config=loaded_config,
            checkpoint_root=args.checkpoint_root,
            gpu_ids=gpu_ids,
            batch_size=args.batch_size,
        ) as inferencer:
            key = "pipeline" if use_pipeline else "legacy"
            results[key] = inferencer.inference(
                samples, use_pipeline=use_pipeline
            )

    os.makedirs(args.log_root, exist_ok=True)
    report = _compare(results["pipeline"], results["legacy"], args.tolerance)
    report_path = os.path.join(args.log_root, "parity_report.json")
    with open(report_path, "w", encoding="utf-8") as stream:
        json.dump({**report, "gpus": gpu_ids}, stream, indent=2)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
