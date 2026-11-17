import numpy as np


def binary_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("Etiket vektorlerinin boyutu esit ve bos olmamali")

    actual_ones = y_true == 1
    predicted_ones = y_pred == 1
    tn = int(np.sum(~actual_ones & ~predicted_ones))
    fp = int(np.sum(~actual_ones & predicted_ones))
    fn = int(np.sum(actual_ones & ~predicted_ones))
    tp = int(np.sum(actual_ones & predicted_ones))

    predicted_positive_count = tp + fp
    positive_count = tp + fn
    negative_count = tn + fp
    f1_denominator = 2 * tp + fp + fn

    precision = tp / predicted_positive_count if predicted_positive_count else 0.0
    recall = tp / positive_count if positive_count else 0.0
    specificity = tn / negative_count if negative_count else 0.0
    f1 = 2 * tp / f1_denominator if f1_denominator else 0.0

    return {
        "accuracy": (tp + tn) / len(y_true),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "n": len(y_true),
    }


def roc_curve_auc(y_true, scores):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)

    order = np.argsort(-scores, kind="stable")
    sorted_labels = y_true[order]
    sorted_scores = scores[order]

    change_points = np.flatnonzero(np.diff(sorted_scores) != 0)
    group_ends = np.append(change_points, len(scores) - 1)

    true_positives = np.cumsum(sorted_labels == 1)[group_ends]
    false_positives = (group_ends + 1) - true_positives

    tpr = np.append(0.0, true_positives / np.sum(y_true == 1))
    fpr = np.append(0.0, false_positives / np.sum(y_true == 0))
    thresholds = np.append(np.inf, sorted_scores[group_ends])

    auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
    return fpr, tpr, thresholds, auc
