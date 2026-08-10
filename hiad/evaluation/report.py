import numpy as np
import tabulate


class MetricReport:
    def __init__(self, headers=None):
        self.headers = tuple(map(str, headers)) if headers else ()
        self.records = []

    def add_record(self, record):
        if self.headers and len(record) != len(self.headers):
            raise ValueError(
                f"Record length ({len(record)}) must equal header length "
                f"({len(self.headers)})"
            )
        self.records.append(record)

    def __str__(self):
        return tabulate.tabulate(
            self.records,
            self.headers,
            tablefmt="pipe",
            numalign="center",
            stralign="center",
        )


def summarize_category_scores(scores):
    if not scores:
        raise ValueError("At least one category score is required")
    metric_names = [
        key
        for key in scores[0]
        if "threshold" not in key and key != "clsname"
    ]
    if not metric_names:
        raise ValueError("Evaluation produced no reportable metrics")

    mean_metrics = {
        metric: float(np.mean([score[metric] for score in scores]))
        for metric in metric_names
    }
    report = MetricReport(["clsname"] + metric_names)
    for score in scores:
        report.add_record(
            [score["clsname"]] + [score[metric] for metric in metric_names]
        )
    report.add_record(["mean"] + [mean_metrics[metric] for metric in metric_names])
    return mean_metrics, report
