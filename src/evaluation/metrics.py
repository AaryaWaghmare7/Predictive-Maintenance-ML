"""Model evaluation helpers."""

from __future__ import annotations

from sklearn.metrics import accuracy_score, classification_report


def classification_metrics(y_true, y_pred) -> dict[str, object]:
    """Return common classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
    }
