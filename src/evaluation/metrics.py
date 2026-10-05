"""Evaluation helpers for Experiment 1 binary and Experiment 2 multiclass tasks."""

from __future__ import annotations

from typing import Any

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, y_pred) -> dict[str, object]:
    """Return common classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
    }


def binary_classification_metrics(
    y_true, y_pred, positive_scores
) -> dict[str, Any]:
    """Calculate baseline metrics for class 1 without changing its threshold.

    ``positive_scores`` must contain the model's probability or score for class
    1. This helper evaluates already-made predictions and never fits a model.
    """
    return {
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_class_1": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "recall_class_1": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_class_1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, positive_scores)),
        "average_precision": float(average_precision_score(y_true, positive_scores)),
        "classification_report": classification_report(
            y_true, y_pred, labels=[0, 1], digits=4, zero_division=0
        ),
    }


def multiclass_classification_metrics(y_true, y_pred, probabilities, classes) -> dict[str, Any]:
    """Evaluate fixed class predictions and correctly ordered probabilities.

    Macro averaging gives each fault class equal weight. ROC-AUC is undefined
    if any expected class is missing from the evaluation data.
    """
    import numpy as np

    classes = list(classes)
    scores = np.asarray(probabilities)
    if scores.shape != (len(y_true), len(classes)):
        raise ValueError("Probability columns must match the supplied class order")
    if classes != sorted(set(classes)):
        raise ValueError("Class order must be sorted and unique")
    if not np.isfinite(scores).all() or (scores < 0).any() or not np.allclose(scores.sum(axis=1), 1):
        raise ValueError("Probabilities must be finite, nonnegative and sum to one")
    if not set(y_true).issubset(classes) or not set(y_pred).issubset(classes):
        raise ValueError("Unexpected target or prediction class")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
        "roc_auc_macro_ovr": float(roc_auc_score(y_true, scores, labels=classes, multi_class="ovr", average="macro"))
        if set(y_true) == set(classes) else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
        "per_class": classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0),
    }
