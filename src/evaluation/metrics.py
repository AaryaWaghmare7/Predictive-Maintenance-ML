"""Model evaluation helpers for the locked binary failure baseline."""

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
