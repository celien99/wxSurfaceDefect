import numpy as np
from sklearn import metrics
from sklearn.metrics import average_precision_score, precision_recall_curve

from .common import flatten_mask_pairs, validate_binary_vectors


def compute_imagewise_metrics(prediction_scores, gt_labels, **kwargs):
    scores, labels = validate_binary_vectors(prediction_scores, gt_labels)
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
    best_index = int(np.argmax(tpr - fpr))
    return {
        "image_auroc": float(metrics.roc_auc_score(labels, scores)),
        "image_threshold": float(thresholds[best_index]),
    }


def compute_pixelwise_metrics(prediction_masks, gt_masks, **kwargs):
    scores, labels = flatten_mask_pairs(prediction_masks, gt_masks)
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        raise ValueError("Pixel metrics require at least one prediction threshold")
    denominator = precision + recall
    f1_scores = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )[:-1]
    best_index = int(np.argmax(f1_scores))
    return {
        "pixel_auroc": float(metrics.roc_auc_score(labels, scores)),
        "pixel_ap": float(average_precision_score(labels, scores)),
        "pixel_f1": float(f1_scores[best_index]),
        "seg_threshold": float(thresholds[best_index]),
    }


def compute_tpr_at_fpr(
    prediction_scores,
    gt_labels,
    target_fpr=0.01,
    **kwargs,
):
    scores, labels = validate_binary_vectors(prediction_scores, gt_labels)
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
    valid = np.flatnonzero(fpr <= target_fpr)
    index = int(valid[-1]) if valid.size else 0
    return {
        "tpr_at_fpr": float(tpr[index]),
        "fpr_achieved": float(fpr[index]),
        "threshold_at_fpr": float(thresholds[index]),
        "target_fpr": float(target_fpr),
    }


def compute_fpr_at_tpr(
    prediction_scores,
    gt_labels,
    target_tpr=0.95,
    **kwargs,
):
    scores, labels = validate_binary_vectors(prediction_scores, gt_labels)
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
    valid = np.flatnonzero(tpr >= target_tpr)
    index = int(valid[0]) if valid.size else len(tpr) - 1
    return {
        "fpr_at_tpr": float(fpr[index]),
        "tpr_achieved": float(tpr[index]),
        "threshold_at_tpr": float(thresholds[index]),
        "target_tpr": float(target_tpr),
    }
