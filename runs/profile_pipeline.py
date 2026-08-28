"""逐阶段 CPU/GPU 归因测量：报告每阶段的 CPU 墙钟与 GPU 忙碌时间占比。

串行基线（默认）给出每张图的架构级气泡证据；P0 后 ``--async-pipeline`` 复跑
比较端到端墙钟。按方法存在性给 ``DeviceImagePipeline`` 打计时补丁，同一脚本
兼容 P0 前后两版代码。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

import yaml

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import torch

from hiad.data import HRSample, read_jsonl_records
from hiad.detectors import HRDinomaly
from hiad.inferencer import HRInferencer
from hiad.inferencer.pipeline import DeviceImagePipeline


def parse_args(argv=None) -> argparse.Namespace:
    """解析归因参数；``argv=None`` 时读命令行。"""
    parser = argparse.ArgumentParser(
        description="Profile per-stage CPU/GPU attribution of the image pipeline"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", default="test_uni.jsonl")
    parser.add_argument("--config", default="configs/dinomaly.yaml")
    parser.add_argument("--checkpoint-root", default="results/dinomaly_checkpoints")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument(
        "--async-pipeline",
        action="store_true",
        help="profile the P0 async double-buffer loop instead of the serial baseline",
    )
    parser.add_argument("--report", default="results/profile_pipeline_report.txt")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


@dataclass
class StageStats:
    """单个命名阶段的 CPU 墙钟与 GPU 忙碌归因累加（毫秒）。"""

    name: str
    cpu_wall_ms: float = 0.0
    gpu_busy_ms: float = 0.0
    calls: int = 0

    def add(self, cpu_wall: float, gpu_busy: float) -> None:
        self.cpu_wall_ms += cpu_wall * 1000.0
        self.gpu_busy_ms += gpu_busy * 1000.0
        self.calls += 1


_STATS: dict[str, StageStats] = {}
_INSTALLED = False


def _install_timing(async_pipeline: bool = False) -> None:
    """把 pipeline 命名阶段包一层计时器（CPU 墙钟 + GPU busy 归因）。

    映射按方法存在性选择，兼容 P0 前后：P0 前 ``_coarse_forward``/``_route``，
    P0 后 ``_submit_coarse``/``_finish_coarse``/``_refine_and_merge``；后两版
    的 ``_submit_coarse`` 与 ``_finish_coarse`` 共享 ``coarse`` 归因桶。

    ``async_pipeline=True`` 时只记录每阶段 CPU 墙钟，不注入 CUDA 事件与同步：
    串行基线用事件归因找出架构级气泡，但事件 ``synchronize`` 会强制等待当前
    stream，恰好把双缓冲的重叠窗口吞掉，使 P4 的端到端对比失真。
    """
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    _STATS.clear()
    mapping = {
        "_coarse_forward": "coarse",
        "_submit_coarse": "coarse",
        "_finish_coarse": "coarse",
        "_route": "routing",
        "_refine_and_merge": "refinement",
    }
    for method_name, stage in mapping.items():
        if not hasattr(DeviceImagePipeline, method_name):
            continue
        original = getattr(DeviceImagePipeline, method_name)
        if getattr(original, "_profiled", False):
            continue
        stats = _STATS.setdefault(stage, StageStats(stage))

        def timed(self, *args, method=original, stats=stats, **kwargs):
            cpu_started = time.perf_counter()
            if not async_pipeline and self.device.type == "cuda":
                stream = torch.cuda.current_stream()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record(stream)
            else:
                start_event = end_event = stream = None
            result = method(self, *args, **kwargs)
            if start_event is not None:
                end_event.record(stream)
                end_event.synchronize()
                wall_ms = (time.perf_counter() - cpu_started) * 1000.0
                gpu_busy = start_event.elapsed_time(end_event)
                # elapsed_time 的单位在各 torch/CUDA/驱动组合下并不一致（文档为
                # 毫秒，但训练机实测为微秒乃至纳秒）。毫秒下 GPU busy 不可能超过
                # 调用墙钟（CPU 在同步处等 GPU），故用墙钟做锚点：把报告值反复
                # 除以 1000 直到落在墙钟的合理倍数内，一步归一掉 μs/ns 单位。
                # 上限 4 次覆盖 ms→μs→ns→ps，也防住无界单位导致死循环。
                for _ in range(4):
                    if gpu_busy <= wall_ms * 5:
                        break
                    gpu_busy = gpu_busy / 1000.0
            else:
                gpu_busy = 0.0
            stats.add(time.perf_counter() - cpu_started, gpu_busy)
            return result

        timed._profiled = True
        setattr(DeviceImagePipeline, method_name, timed)


def render_report(
    stats_by_stage: dict[str, StageStats],
    images: int,
    total_wall_s: float,
    *,
    async_mode: bool = False,
) -> str:
    """渲染每阶段归因表与串行/流水理论界。

    ``async_mode`` 时阶段归因只有 CPU 墙钟（GPU 事件会注入同步、破坏双缓冲
    重叠），报告额外标注；P4 对比只看端到端 ``total_wall``。
    """
    total_gpu = sum(s.gpu_busy_ms for s in stats_by_stage.values())
    total_cpu = sum(s.cpu_wall_ms for s in stats_by_stage.values())
    mode_note = "（async 模式：阶段归因仅 CPU 墙钟，GPU busy 不适用）" if async_mode else ""
    lines = ["=== 每阶段归因（跨图累加） ===" + mode_note]
    lines.append(
        f"{'stage':<12} {'calls':>6} {'cpu_ms':>10} {'gpu_ms':>10} "
        f"{'gpu/cpu':>7} {'cpu_share':>9}"
    )
    for stage in ("coarse", "routing", "refinement"):
        s = stats_by_stage.get(stage)
        if s is None or s.calls == 0:
            continue
        gpu_frac = s.gpu_busy_ms / max(s.cpu_wall_ms, 1e-9) * 100
        cpu_share = s.cpu_wall_ms / max(total_cpu, 1e-9) * 100
        lines.append(
            f"{stage:<12} {s.calls:>6} {s.cpu_wall_ms:>10.1f} {s.gpu_busy_ms:>10.1f} "
            f"{gpu_frac:>6.0f}% {cpu_share:>8.0f}%"
        )
    wall_ms = total_wall_s * 1000.0
    serial_bound = total_cpu + total_gpu
    pipeline_bound = max(total_cpu, total_gpu)
    lines.append("")
    lines.append(
        f"images={images}  total_wall={wall_ms:.1f}ms  Σcpu={total_cpu:.1f}ms  "
        f"Σgpu={total_gpu:.1f}ms  gpu_busy_frac={total_gpu / max(wall_ms, 1e-9) * 100:.0f}%"
    )
    lines.append(
        f"串行下限=Σcpu+Σgpu={serial_bound:.1f}ms；"
        f"流水上限=max(Σcpu,Σgpu)={pipeline_bound:.1f}ms "
        f"→ 理论提速={pipeline_bound / max(serial_bound, 1e-9):.2f}×"
    )
    return "\n".join(lines)


def main(argv=None) -> int:
    args = parse_args(argv)
    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if args.async_pipeline:
        config.setdefault("inference", {})["async_pipeline"] = True
    gpu_ids = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
    if not gpu_ids:
        raise ValueError("At least one GPU id is required")
    records = read_jsonl_records(os.path.join(args.data_root, args.manifest))
    samples = [
        HRSample(
            image=os.path.join(args.data_root, record["filename"]),
            clsname=record.get("clsname", "default"),
        )
        for record in records
    ]
    _install_timing(async_pipeline=args.async_pipeline)
    wall_started = time.perf_counter()
    with HRInferencer(
        detector_class=HRDinomaly,
        config=config,
        checkpoint_root=args.checkpoint_root,
        gpu_ids=gpu_ids,
        batch_size=args.batch_size,
    ) as inferencer:
        inferencer.inference(samples)
    total_wall = time.perf_counter() - wall_started
    report_text = render_report(
        _STATS, len(samples), total_wall, async_mode=args.async_pipeline
    )
    print(report_text)
    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as stream:
            stream.write(report_text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
