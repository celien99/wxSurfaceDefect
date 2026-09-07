import pytest

from runs.profile_pipeline import StageStats, parse_args, render_report


def test_parse_args_rejects_non_positive_batch_size():
    with pytest.raises(SystemExit):
        parse_args(["--data-root", "d", "--batch-size", "0"])


def test_parse_args_accepts_minimal_invocation():
    args = parse_args(["--data-root", "d"])
    assert args.data_root == "d"
    assert args.manifest == "test_uni.jsonl"
    assert args.async_pipeline is False


def test_stage_stats_accumulates_cpu_and_gpu():
    stats = StageStats("coarse")
    stats.add(1.0, 0.9)
    stats.add(2.0, 1.5)
    assert stats.calls == 2
    assert stats.cpu_wall_ms == pytest.approx(3000.0)
    assert stats.gpu_busy_ms == pytest.approx(2400.0)


def test_render_report_lists_stage_rows_and_bounds():
    stats = {
        "coarse": StageStats("coarse", cpu_wall_ms=100.0, gpu_busy_ms=90.0, calls=2),
        "routing": StageStats("routing", cpu_wall_ms=80.0, gpu_busy_ms=0.0, calls=2),
        "refinement": StageStats("refinement", cpu_wall_ms=60.0, gpu_busy_ms=40.0, calls=2),
    }
    text = render_report(stats, images=2, total_wall_s=0.3)
    assert "coarse" in text and "routing" in text and "refinement" in text
    assert "total_wall=300.0ms" in text
    assert "流水上限" in text and "理论提速" in text
