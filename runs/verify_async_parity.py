"""训练机门槛：serial vs P0 async 逐位一致（判定 0 flip、max_abs_diff == 0）。

P0 只重排 GPU 提交顺序，不改算子，理论上输出 bit-identical；本脚本是它的
集成验证。通过后执行方按 Task 5 Step 6 翻转 ``async_pipeline`` 默认值。
"""
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
        description="Compare serial vs P0 async pipeline for bit-identical parity"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", default="test_uni.jsonl")
    parser.add_argument("--config", default="configs/dinomaly.yaml")
    parser.add_argument("--checkpoint-root", default="results/dinomaly_checkpoints")
    parser.add_argument("--log-root", default="results/parity")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--batch-size", default=16, type=int)
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    gpu_ids = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
    if not gpu_ids:
        raise ValueError("At least one GPU id is required")
    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config.setdefault("inference", {})
    records = read_jsonl_records(os.path.join(args.data_root, args.manifest))
    samples = [
        HRSample(
            image=os.path.join(args.data_root, record["filename"]),
            clsname=record.get("clsname", "default"),
        )
        for record in records
    ]

    results = {}
    for use_async in (False, True):
        config["inference"]["async_pipeline"] = use_async
        key = "async" if use_async else "serial"
        with HRInferencer(
            detector_class=HRDinomaly,
            config=config,
            checkpoint_root=args.checkpoint_root,
            gpu_ids=gpu_ids,
            batch_size=args.batch_size,
        ) as inferencer:
            results[key] = inferencer.inference(samples)

    serial = results["serial"]
    async_ = results["async"]
    decisions_serial = serial.get("decisions")
    decisions_async = async_.get("decisions")
    flips = 0
    if decisions_serial is not None and decisions_async is not None:
        for path, legacy, new in zip(
            serial["image_paths"], decisions_serial, decisions_async
        ):
            if legacy != new:
                flips += 1
                print(f"DECISION FLIP {path}: serial={legacy} async={new}")
    max_abs_diff = 0.0
    for path, a, b in zip(
        serial["image_paths"], serial["anomaly_maps"], async_["anomaly_maps"]
    ):
        if a.shape != b.shape:
            raise ValueError(f"Anomaly map shape mismatch for {path}")
        max_abs_diff = max(max_abs_diff, float(np.abs(a - b).max()))
    scores_equal = bool(
        np.array_equal(serial["image_scores"], async_["image_scores"])
    )
    passed = flips == 0 and max_abs_diff == 0.0 and scores_equal
    report = {
        "decision_flips": flips,
        "max_abs_diff": max_abs_diff,
        "scores_bit_identical": scores_equal,
        "passed": passed,
    }
    os.makedirs(args.log_root, exist_ok=True)
    report_path = os.path.join(args.log_root, "async_parity_report.json")
    with open(report_path, "w", encoding="utf-8") as stream:
        json.dump({**report, "gpus": gpu_ids}, stream, indent=2)
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
