import numpy as np
import torch

from .common import validate_binary_vectors, validate_mask_pairs


def _resolve_device(device):
    return torch.device("cuda") if device is None else torch.device(device)


def _binary_curve(scores, labels, device):
    numpy_scores, numpy_labels = validate_binary_vectors(scores, labels)
    score_tensor = torch.as_tensor(
        numpy_scores,
        dtype=torch.float64,
        device=device,
    )
    label_tensor = torch.as_tensor(
        numpy_labels,
        dtype=torch.int64,
        device=device,
    )
    order = torch.argsort(score_tensor, descending=True, stable=True)
    sorted_scores = score_tensor[order]
    sorted_labels = label_tensor[order]
    distinct = torch.nonzero(
        sorted_scores[1:] != sorted_scores[:-1],
        as_tuple=True,
    )[0]
    threshold_indexes = torch.cat(
        [distinct, torch.tensor([sorted_scores.numel() - 1], device=device)]
    )
    true_positives = torch.cumsum(sorted_labels, dim=0)[threshold_indexes].double()
    predicted_positives = threshold_indexes.double() + 1.0
    false_positives = predicted_positives - true_positives
    positive_count = sorted_labels.sum().double()
    negative_count = sorted_labels.numel() - positive_count
    thresholds = sorted_scores[threshold_indexes]
    tpr = true_positives / positive_count
    fpr = false_positives / negative_count
    precision = true_positives / predicted_positives
    return fpr, tpr, precision, thresholds


def _roc_area(fpr, tpr):
    origin = torch.zeros(1, dtype=fpr.dtype, device=fpr.device)
    return torch.trapz(torch.cat([origin, tpr]), torch.cat([origin, fpr]))


def compute_imagewise_metrics(
    prediction_scores,
    gt_labels,
    device=None,
    **kwargs,
):
    device = _resolve_device(device)
    fpr, tpr, _, thresholds = _binary_curve(
        prediction_scores,
        gt_labels,
        device,
    )
    best_index = int(torch.argmax(tpr - fpr).item())
    return {
        "image_auroc": float(_roc_area(fpr, tpr).item()),
        "image_threshold": float(thresholds[best_index].item()),
    }


def compute_pixelwise_metrics(
    prediction_masks,
    gt_masks,
    device=None,
    num_thresholds: int = 65536,
    **kwargs,
):
    """Compute bounded-memory pixel metrics from a streaming score histogram."""
    pairs = validate_mask_pairs(prediction_masks, gt_masks)
    if isinstance(num_thresholds, bool) or not isinstance(num_thresholds, int):
        raise TypeError("num_thresholds must be an integer")
    if num_thresholds < 2:
        raise ValueError("num_thresholds must be at least 2")

    device = _resolve_device(device)
    minimum = min(float(prediction.min()) for prediction, _ in pairs)
    maximum = max(float(prediction.max()) for prediction, _ in pairs)
    positive_counts = torch.zeros(num_thresholds, dtype=torch.int64, device=device)
    negative_counts = torch.zeros_like(positive_counts)

    for prediction, target in pairs:
        scores = torch.as_tensor(
            np.ascontiguousarray(prediction),
            dtype=torch.float32,
            device=device,
        )
        labels = torch.as_tensor(
            np.ascontiguousarray(target),
            dtype=torch.bool,
            device=device,
        )
        if maximum == minimum:
            bin_indexes = torch.full_like(
                scores,
                num_thresholds - 1,
                dtype=torch.int64,
            )
        else:
            bin_indexes = ((scores - minimum) * (
                num_thresholds / (maximum - minimum)
            )).to(torch.int64)
            bin_indexes.clamp_(0, num_thresholds - 1)
        positive_counts += torch.bincount(
            bin_indexes[labels],
            minlength=num_thresholds,
        )
        negative_counts += torch.bincount(
            bin_indexes[~labels],
            minlength=num_thresholds,
        )

    positive_total = positive_counts.sum()
    negative_total = negative_counts.sum()
    if positive_total.item() == 0 or negative_total.item() == 0:
        raise ValueError("Binary metrics require both normal and anomalous labels")

    true_positives = torch.cumsum(positive_counts.flip(0), dim=0).double()
    false_positives = torch.cumsum(negative_counts.flip(0), dim=0).double()
    recall = true_positives / positive_total.double()
    fpr = false_positives / negative_total.double()
    predicted_positives = true_positives + false_positives
    precision = torch.where(
        predicted_positives > 0,
        true_positives / predicted_positives,
        torch.zeros_like(predicted_positives),
    )
    thresholds = torch.linspace(
        minimum,
        maximum,
        num_thresholds + 1,
        dtype=torch.float64,
        device=device,
    )[:-1].flip(0)
    denominator = precision + recall
    f1_scores = torch.where(
        denominator > 0,
        2 * precision * recall / denominator,
        torch.zeros_like(denominator),
    )
    best_index = int(torch.argmax(f1_scores).item())
    previous_recall = torch.cat(
        [torch.zeros(1, dtype=recall.dtype, device=device), recall[:-1]]
    )
    average_precision = torch.sum((recall - previous_recall) * precision)
    return {
        "pixel_auroc": float(_roc_area(fpr, recall).item()),
        "pixel_ap": float(average_precision.item()),
        "pixel_f1": float(f1_scores[best_index].item()),
        "seg_threshold": float(thresholds[best_index].item()),
    }


def compute_tpr_at_fpr(
    prediction_scores,
    gt_labels,
    target_fpr=0.01,
    device=None,
    **kwargs,
):
    device = _resolve_device(device)
    fpr, tpr, _, thresholds = _binary_curve(
        prediction_scores,
        gt_labels,
        device,
    )
    valid = torch.nonzero(fpr <= target_fpr, as_tuple=True)[0]
    if valid.numel() == 0:
        return {
            "tpr_at_fpr": 0.0,
            "fpr_achieved": 0.0,
            "threshold_at_fpr": float("inf"),
            "target_fpr": float(target_fpr),
        }
    index = int(valid[-1].item())
    return {
        "tpr_at_fpr": float(tpr[index].item()),
        "fpr_achieved": float(fpr[index].item()),
        "threshold_at_fpr": float(thresholds[index].item()),
        "target_fpr": float(target_fpr),
    }


def compute_fpr_at_tpr(
    prediction_scores,
    gt_labels,
    target_tpr=0.95,
    device=None,
    **kwargs,
):
    device = _resolve_device(device)
    fpr, tpr, _, thresholds = _binary_curve(
        prediction_scores,
        gt_labels,
        device,
    )
    valid = torch.nonzero(tpr >= target_tpr, as_tuple=True)[0]
    index = int(valid[0].item()) if valid.numel() else len(tpr) - 1
    return {
        "fpr_at_tpr": float(fpr[index].item()),
        "tpr_achieved": float(tpr[index].item()),
        "threshold_at_tpr": float(thresholds[index].item()),
        "target_tpr": float(target_tpr),
    }
