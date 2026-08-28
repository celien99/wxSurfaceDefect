"""训练机门槛：decoder_amp=false（基线）vs true（FP16 解码）独立容差 parity。

FP16 ≠ bit-identical，因此不用逐位断言；按预注册标准门判定：
  decision_flips == 0              （硬）
  image_scores 最大相对漂移 ≤ 1e-2 （容，相对基线分数尺度）
  image_scores Spearman 秩 ≥ 0.99  （容）
通过后执行方按 plan Task 7 翻转 ``inference.decoder_amp`` 默认值。
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

MAX_REL_DRIFT = 1e-2
MIN_RANK = 0.99


def parse_args(argv=None) -> argparse.Namespace:
    """解析 parity 校验参数；``argv=None`` 时读命令行。"""
    parser = argparse.ArgumentParser(
        description="Compare decoder_amp=false vs true for tolerance parity (P1)"
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


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman 秩相关系数（无 scipy 依赖；连续分数很少并列，用简单位序）。"""
    def rank(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=float)
        ranks[order] = np.arange(len(values))
        return ranks

    if len(a) < 2:
        return float("nan")
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def main(argv=None) -> int:
    args = parse_args(argv)
    gpu_ids = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
    if not gpu_ids:
        raise ValueError("At least one GPU id is required")
    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config.setdefault("inference", {})
    # 两遍都固定串行，隔离 P0 异步状态的干扰。
    config["inference"]["async_pipeline"] = False
    records = read_jsonl_records(os.path.join(args.data_root, args.manifest))
    samples = [
        HRSample(
            image=os.path.join(args.data_root, record["filename"]),
            clsname=record.get("clsname", "default"),
        )
        for record in records
    ]

    results = {}
    for use_amp in (False, True):
        config["inference"]["decoder_amp"] = use_amp
        key = "fp16" if use_amp else "baseline"
        with HRInferencer(
            detector_class=HRDinomaly,
            config=config,
            checkpoint_root=args.checkpoint_root,
            gpu_ids=gpu_ids,
            batch_size=args.batch_size,
        ) as inferencer:
            results[key] = inferencer.inference(samples)

    baseline = results["baseline"]
    fp16 = results["fp16"]
    paths = baseline["image_paths"]

    flips = 0
    for path, legacy, new in zip(paths, baseline["decisions"], fp16["decisions"]):
        if legacy != new:
            flips += 1
            print(f"DECISION FLIP {path}: baseline={legacy} fp16={new}")

    base_scores = np.asarray(baseline["image_scores"], dtype=float)
    fp16_scores = np.asarray(fp16["image_scores"], dtype=float)
    denom = max(float(np.abs(base_scores).max()), 1e-12)
    per_image_rel = np.abs(fp16_scores - base_scores) / denom
    max_rel_drift = float(per_image_rel.max())
    rank = _spearman(base_scores, fp16_scores)

    max_abs_diff = 0.0
    worst_map_path = ""
    for path, a, b in zip(paths, baseline["anomaly_maps"], fp16["anomaly_maps"]):
        if a.shape != b.shape:
            raise ValueError(f"Anomaly map shape mismatch for {path}")
        diff = float(np.abs(a - b).max())
        if diff > max_abs_diff:
            max_abs_diff = diff
            worst_map_path = path

    passed = (
        flips == 0
        and max_rel_drift <= MAX_REL_DRIFT
        and rank >= MIN_RANK
    )
    report = {
        "decision_flips": flips,
        "max_rel_drift_image_scores": max_rel_drift,
        "max_rel_drift_threshold": MAX_REL_DRIFT,
        "spearman_image_scores": rank,
        "spearman_threshold": MIN_RANK,
        "max_abs_diff_anomaly_map": max_abs_diff,
        "worst_map_path": worst_map_path,
        "worst_score_path": str(paths[int(per_image_rel.argmax())]),
        "passed": passed,
    }
    os.makedirs(args.log_root, exist_ok=True)
    report_path = os.path.join(args.log_root, "decoder_fp16_parity_report.json")
    with open(report_path, "w", encoding="utf-8") as stream:
        json.dump({**report, "gpus": gpu_ids}, stream, indent=2)
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
