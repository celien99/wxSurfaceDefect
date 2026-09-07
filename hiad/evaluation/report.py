from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import tabulate

from hiad.evaluation.execution import CategoryScore


class MetricReport:
    """保存固定列数的分类别 Markdown 指标表。

    Attributes:
        headers (tuple[str, ...]): 输出列名；空元组表示不显示表头。
        records (list[Sequence[object]]): 按添加顺序保存的表格行。
    """

    def __init__(self, headers: Sequence[object] | None = None) -> None:
        self.headers: tuple[str, ...] = tuple(map(str, headers)) if headers else ()
        self.records: list[Sequence[object]] = []

    def add_record(self, record: Sequence[object]) -> None:
        """追加一行，并在存在表头时校验列数。

        Args:
            record (Sequence[object]): 待追加的表格单元格序列。

        Raises:
            ValueError: 记录列数与表头列数不一致。
        """
        if self.headers and len(record) != len(self.headers):
            raise ValueError(
                f"Record length ({len(record)}) must equal header length "
                f"({len(self.headers)})"
            )
        self.records.append(record)

    def __str__(self) -> str:
        return tabulate.tabulate(
            self.records,
            self.headers,
            tablefmt="pipe",
            numalign="center",
            stralign="center",
        )


def summarize_category_scores(
    scores: Sequence[CategoryScore],
) -> tuple[dict[str, float], MetricReport]:
    """汇总分类别指标并生成包含均值行的报告。

    Args:
        scores (Sequence[CategoryScore]): 字段模式一致的分类别指标；包含
            ``clsname``，阈值字段名需包含 ``threshold``。

    Returns:
        tuple[dict[str, float], MetricReport]: 排除类别名和阈值后的等权平均指标，
        以及包含各类别和均值行的 Markdown 报告对象。

    Raises:
        ValueError: 没有类别记录或没有可报告指标。
        KeyError: 分类别记录缺少首条记录定义的指标字段。
    """
    if not scores:
        raise ValueError("At least one category score is required")
    metric_names: list[str] = [
        key
        for key in scores[0]
        if "threshold" not in key and key != "clsname"
    ]
    if not metric_names:
        raise ValueError("Evaluation produced no reportable metrics")

    mean_metrics = {
        metric: float(np.mean([float(score[metric]) for score in scores]))
        for metric in metric_names
    }
    report = MetricReport(["clsname"] + metric_names)
    for score in scores:
        report.add_record(
            [score["clsname"]] + [score[metric] for metric in metric_names]
        )
    report.add_record(["mean"] + [mean_metrics[metric] for metric in metric_names])
    return mean_metrics, report
