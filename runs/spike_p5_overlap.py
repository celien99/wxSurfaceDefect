"""P5 双 stream 重叠可行性 spike（throwaway，不进产品代码）。

问题：串行基线里 GPU 只忙 43%（gpu_busy_frac=43%），但总 GPU 工作量固定
（Σgpu≈3.7s）；P5 赌「coarse(N+1) 与 refine(N) 双 stream 并发」能把临界路径
从 ~618ms/图 降到 ~471ms/图。可行与否取决于单次 coarse 前向是否把 SM 打满
——这是硬件经验问题，故用真实模型 + 真实批次量化。

本脚本只量墙钟（``time.perf_counter`` + ``stream.synchronize``），不注入
CUDA Event，避免 profiler 那次「毫秒/微秒」单位歧义。OOM 本身算一个发现
（双 stream 并发让 GPU 驻留张量翻倍）。

配置（每个都取 ``--repeats`` 次的最小值）：
  solo_c  单 stream 跑 coarse 前向（粗扫批 + 缩略图）
  solo_r  单 stream 跑 refine 前向
  seq_cr  同 stream 先后跑 coarse→refine（sanity 锚点，应 ≈ solo_c + solo_r）
  par_cr  coarse ∥ refine 双 stream 并发（P5 的核心假设）
  par2    两个 coarse 前向双 stream 并发（SM 饱和检查）

指标：
  饱和指数   = (par2 − solo_c) / solo_c   0=完美重叠/SM 有头，≈1=完全串行/SM 打满
  重叠效率   = 1 − par_cr / (solo_c + solo_r)   越大越好（1=coarse 完全吞掉 refine）
  临界路径比 = par_cr / max(solo_c, solo_r)    1.0=双 stream 不劣于单 piece 最佳
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import yaml

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import torch

from hiad.data import (
    HRSample,
    HRImageIndex,
    build_multiresolution_region,
    read_jsonl_records,
)
from hiad.data.patch_builder import build_patch_batch
from hiad.detectors import HRDinomaly
from hiad.inferencer import HRInferencer
from hiad.inferencer.pipeline import DeviceImagePipeline
from hiad.inferencer.refinement import select_refinement_regions


def parse_args(argv=None) -> argparse.Namespace:
    """解析 spike 参数；``argv=None`` 时读命令行。"""
    parser = argparse.ArgumentParser(
        description=(
            "P5 spike: measure coarse||refine dual-stream overlap on real "
            "coarse/refine batches built from the first eval image"
        )
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", default="test_uni.jsonl")
    parser.add_argument("--config", default="configs/dinomaly.yaml")
    parser.add_argument("--checkpoint-root", default="results/dinomaly_checkpoints")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--repeats", default=5, type=int)
    parser.add_argument("--report", default="results/spike_p5_overlap.txt")
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.repeats <= 0:
        parser.error("--batch-size and --repeats must be positive")
    return args


def _make_pipeline(inferencer: HRInferencer, batch_size: int) -> DeviceImagePipeline:
    """按 ``HRInferencer._run_inference`` 完全相同的参数构造单设备流水线。"""
    manager = inferencer.model_managers[0]
    return DeviceImagePipeline(
        manager.detectors,
        inferencer.coarse_task_definitions,
        inferencer.refinement_task,
        inference_config=inferencer.inference_config,
        global_routing_weight=inferencer.global_routing_weight,
        score_top_k=inferencer.score_top_k,
        refinement_bridge_gap_tiles=inferencer.refinement_bridge_gap_tiles,
        map_gaussian_sigma=inferencer.map_gaussian_sigma,
        batch_cap=batch_size,
        async_pipeline=False,
    )


def _build_spike_batches(
    pipeline: DeviceImagePipeline,
    sample: HRSample,
) -> tuple[object, int, int, object]:
    """用真实数据构建 coarse 批（``_build_item``）与真实 refine 批。

    refine 批取生产路由路径选出的候选区域（``_submit_coarse``→``_finish_coarse``
    →``select_refinement_regions``），并额外保证包含图像中心区域（探针设计
    「取中心区域构建真实 refine batch」），避免第一张 eval 图太干净时 refine
    前向过小、把重叠效率虚高。返回 ``(item, refine_region_count, coarse_records,
    refine_batch)``。
    """
    item = pipeline._build_item(sample)
    try:
        image_size = (int(item.image.shape[1]), int(item.image.shape[0]))
        state = pipeline._submit_coarse(item)  # 真实 coarse 前向，兼作 warmup
        routing_np, threshold, _ = pipeline._finish_coarse(state)
        task = pipeline.refinement_task
        regions = select_refinement_regions(
            routing_np,
            threshold=threshold,
            tile_size=task["patch_size"],
            min_area=task["refinement_min_area"],
            safety_fraction=task["refinement_safety_fraction"],
            max_bridge_gap_tiles=pipeline.refinement_bridge_gap_tiles,
        )
        patch = task["patch_size"]
        image_w, image_h = image_size
        center_x = min(max(0, image_w // 2 - patch // 2), max(0, image_w - patch))
        center_y = min(max(0, image_h // 2 - patch // 2), max(0, image_h - patch))
        center = HRImageIndex(x=center_x, y=center_y, width=patch, height=patch)
        seen = {(r.x, r.y, r.width, r.height) for r in regions}
        if (center.x, center.y, center.width, center.height) not in seen:
            regions = [*regions, center]

        base = {
            "task_name": task["name"],
            "task_type": task["type"],
            "image_path": item.sample.image.image_path,
            "image_size": image_size,
            "model_input_size": (patch, patch),
        }
        indexes = [
            build_multiresolution_region(image_size, region, task["ds_factors"])
            for region in regions
        ]
        refine_batch, _records = build_patch_batch(
            item.image, indexes, patch, base
        )
        return item, len(regions), len(item.coarse_records), refine_batch
    except BaseException:
        item.sample.close()
        raise


def _measure(
    pipeline: DeviceImagePipeline,
    item: object,
    refine_batch: object,
    repeats: int,
) -> tuple[dict[str, float | None], dict[str, str], dict[str, list[float]]]:
    """测 5 个配置的墙钟，返回 ``(mins, oom, per_rep)``。

    ``mins`` 每配置取 ``--repeats`` 次的最小值；OOM 一次即该配置不可行，记录后
    不再重复。返回的 GPU 结果张量被 ``_hold`` 持有，防止测量中途被回收。
    """
    coarse_task = pipeline._coarse_task()
    thumbnail_task = pipeline._thumbnail_task()
    coarse_detector = pipeline.detectors[coarse_task["name"]]
    thumbnail_detector = pipeline.detectors[thumbnail_task["name"]]
    refine_detector = pipeline.detectors[pipeline.refinement_task["name"]]
    coarse_patch = coarse_task["patch_size"]
    refine_patch = pipeline.refinement_task["patch_size"]

    _hold: list[object] = []

    def fwd_c() -> None:
        _hold.append(
            pipeline._forward_and_collect(
                coarse_detector, item.coarse_batch, coarse_patch
            )
        )
        _hold.append(thumbnail_detector.inference_batch(item.thumbnail_batch))

    def fwd_r() -> None:
        _hold.append(
            pipeline._forward_and_collect(refine_detector, refine_batch, refine_patch)
        )

    # stream 必须建在 pipeline 所在设备上，否则 ``--gpus`` 非当前设备时，
    # ``stream.synchronize()`` 等的是空 stream，测到的是 CPU 启动时间。
    s_c = torch.cuda.Stream(device=pipeline.device)
    s_r = torch.cuda.Stream(device=pipeline.device)

    def run(plan: list[tuple[object, torch.cuda.Stream]], streams: list) -> float:
        torch.cuda.synchronize(pipeline.device)
        started = time.perf_counter()
        for fn, stream in plan:
            with torch.cuda.stream(stream):
                fn()
        for stream in streams:
            stream.synchronize()
        torch.cuda.synchronize(pipeline.device)
        elapsed = time.perf_counter() - started
        # 释放本 run 持有的 GPU 结果，避免跨 repeat/config 累积：否则空闲显存
        # 单调下降、_records_per_batch 分块随之变细（solo_c 与 par2 分块不同，
        # 饱和指数失真），且泄漏到 par2 时还可能制造假 OOM。
        _hold.clear()
        return elapsed

    # 先各跑一次，把 coarse/refine 两套 kernel 都预热，避免首轮编译/加载噪声。
    run([(fwd_c, s_c)], [s_c])
    run([(fwd_r, s_r)], [s_r])
    _hold.clear()

    configs: dict[str, tuple[list, list]] = {
        "solo_c": ([(fwd_c, s_c)], [s_c]),
        "solo_r": ([(fwd_r, s_r)], [s_r]),
        "seq_cr": ([(fwd_c, s_c), (fwd_r, s_c)], [s_c]),
        "par_cr": ([(fwd_c, s_c), (fwd_r, s_r)], [s_c, s_r]),
        "par2": ([(fwd_c, s_c), (fwd_c, s_r)], [s_c, s_r]),
    }
    results: dict[str, list[float]] = {name: [] for name in configs}
    oom: dict[str, str] = {}
    for name, (plan, streams) in configs.items():
        for _ in range(repeats):
            try:
                results[name].append(run(plan, streams))
            except torch.cuda.OutOfMemoryError as error:
                oom[name] = str(error)
                torch.cuda.empty_cache()
                break  # OOM 一次即该配置不可行，不用再重复
    mins = {name: min(values) if values else None for name, values in results.items()}
    return mins, oom, results


def _render_report(
    mins: dict[str, float | None],
    oom: dict[str, str],
    per_rep: dict[str, list[float]],
    *,
    refine_regions: int,
    coarse_records: int,
    repeats: int,
    device_name: str,
) -> str:
    """渲染 spike 报告：配置表 + 两个指标 + OOM 发现。"""
    ms = lambda v: v * 1000.0 if v is not None else float("nan")
    lines = ["=== P5 双 stream 重叠 spike（perf_counter 墙钟，min over %d repeats） ===" % repeats]
    lines.append(
        f"device={device_name}  coarse_records={coarse_records}  "
        f"refine_regions={refine_regions}（含中心区域）"
    )
    lines.append(f"{'config':<8} {'min_ms':>10} {'per-rep ms':>40}")
    for name in ("solo_c", "solo_r", "seq_cr", "par_cr", "par2"):
        value = mins.get(name)
        if value is None:
            lines.append(f"{name:<8} {'OOM':>10}")
            continue
        reps = ", ".join(f"{v * 1000.0:7.1f}" for v in per_rep.get(name, []))
        lines.append(f"{name:<8} {ms(value):>10.1f} {reps:>40}")
    lines.append("")

    solo_c, solo_r = mins.get("solo_c"), mins.get("solo_r")
    seq_cr, par_cr, par2 = mins.get("seq_cr"), mins.get("par_cr"), mins.get("par2")

    if seq_cr is not None and solo_c is not None and solo_r is not None:
        ratio = seq_cr / (solo_c + solo_r)
        lines.append(
            f"sanity: seq_cr/(solo_c+solo_r) = {ratio:.3f}  "
            "（应≈1.0；显著<1 说明测量有误）"
        )
    if par2 is not None and solo_c is not None and solo_c > 0:
        sat = (par2 - solo_c) / solo_c
        lines.append(
            f"饱和指数 (par2−solo_c)/solo_c = {sat:.3f}  "
            "（0=完美重叠/SM 有头，≈1=完全串行/SM 打满）"
        )
    if par_cr is not None and solo_c is not None and solo_r is not None:
        eff = 1.0 - par_cr / (solo_c + solo_r)
        crit = par_cr / max(solo_c, solo_r)
        lines.append(
            f"重叠效率 1−par_cr/(solo_c+solo_r) = {eff:.3f}  "
            f"（越大越好；临界路径比 par_cr/max(solo_c,solo_r) = {crit:.3f}）"
        )

    if par2 is not None and solo_c is not None and solo_c > 0:
        sat = (par2 - solo_c) / solo_c
        if sat <= 0.3:
            verdict = (
                "饱和指数 ≤ 0.3：单 coarse 前向没有打满 SM，双 stream 重叠有真实"
                "收益空间 → P5 值得推进设计。"
            )
        elif sat >= 0.7:
            verdict = (
                "饱和指数 ≥ 0.7：单 coarse 前向已接近打满 SM，双 stream 帮不上忙"
                " → P5 放弃，结论写入 spec §6.1。"
            )
        else:
            verdict = (
                "饱和指数 0.3~0.7：收益不确定，结合重叠效率与 OOM 情况再判断。"
            )
        lines.append(verdict)
    if oom:
        lines.append("OOM 发现（双 stream 并发显存翻倍）：")
        for name, message in oom.items():
            lines.append(f"  {name}: {message}")
    return "\n".join(lines)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        print("spike requires CUDA (training machine); aborting", file=sys.stderr)
        return 2
    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    gpu_ids = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
    if not gpu_ids:
        raise ValueError("At least one GPU id is required")
    records = read_jsonl_records(os.path.join(args.data_root, args.manifest))
    if not records:
        raise ValueError("No eval records loaded")
    sample = HRSample(
        image=os.path.join(args.data_root, records[0]["filename"]),
        clsname=records[0].get("clsname", "default"),
    )

    with HRInferencer(
        detector_class=HRDinomaly,
        config=config,
        checkpoint_root=args.checkpoint_root,
        gpu_ids=gpu_ids,
        batch_size=args.batch_size,
    ) as inferencer:
        pipeline = _make_pipeline(inferencer, args.batch_size)
        if pipeline.device.type != "cuda":
            print("pipeline device is not CUDA; aborting", file=sys.stderr)
            return 2
        item, refine_regions, coarse_records, refine_batch = _build_spike_batches(
            pipeline, sample
        )
        try:
            mins, oom, per_rep = _measure(
                pipeline, item, refine_batch, args.repeats
            )
        finally:
            item.sample.close()
    report_text = _render_report(
        mins,
        oom,
        per_rep,
        refine_regions=refine_regions,
        coarse_records=coarse_records,
        repeats=args.repeats,
        device_name=torch.cuda.get_device_name(pipeline.device),
    )
    print(report_text)
    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as stream:
            stream.write(report_text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
