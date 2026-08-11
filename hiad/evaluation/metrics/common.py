import numpy as np


def validate_binary_vectors(prediction_scores, gt_labels):
    scores = np.asarray(prediction_scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(gt_labels).reshape(-1)
    if scores.size == 0 or scores.size != labels.size:
        raise ValueError("Prediction scores and labels must be non-empty and aligned")
    if not np.isfinite(scores).all():
        raise ValueError("Prediction scores must be finite")
    if not set(np.unique(labels).tolist()).issubset({0, 1, False, True}):
        raise ValueError("Ground-truth labels must be binary")
    labels = labels.astype(np.int64, copy=False)
    if np.unique(labels).size != 2:
        raise ValueError("Binary metrics require both normal and anomalous labels")
    return scores, labels


def validate_mask_pairs(prediction_masks, gt_masks):
    predictions = [np.asarray(mask) for mask in prediction_masks]
    targets = [np.asarray(mask) for mask in gt_masks]
    if not predictions or len(predictions) != len(targets):
        raise ValueError("Prediction and ground-truth masks must be non-empty and aligned")

    pairs = []
    for prediction, target in zip(predictions, targets):
        if prediction.ndim != 2 or target.ndim != 2:
            raise ValueError("Prediction and ground-truth masks must be two-dimensional")
        if prediction.shape != target.shape:
            raise ValueError("Each prediction mask must match its ground-truth mask shape")
        if not np.isfinite(prediction).all():
            raise ValueError("Prediction masks must be finite")
        if not np.logical_or(target == 0, target == 1).all():
            raise ValueError("Ground-truth masks must be binary")
        pairs.append((prediction, target))
    return pairs


def flatten_mask_pairs(prediction_masks, gt_masks):
    pairs = validate_mask_pairs(prediction_masks, gt_masks)
    scores = np.concatenate([prediction.reshape(-1) for prediction, _ in pairs])
    labels = np.concatenate([target.reshape(-1) for _, target in pairs])
    return validate_binary_vectors(scores, labels)
