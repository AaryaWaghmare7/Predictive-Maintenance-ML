"""Tests for model-evaluation metrics."""

from __future__ import annotations

import pytest

from src.evaluation.metrics import binary_classification_metrics


def test_binary_classification_metrics_reports_class_one_metrics() -> None:
    metrics = binary_classification_metrics(
        y_true=[0, 0, 1, 1],
        y_pred=[0, 1, 0, 1],
        positive_scores=[0.1, 0.8, 0.2, 0.9],
    )

    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["precision_class_1"] == pytest.approx(0.5)
    assert metrics["recall_class_1"] == pytest.approx(0.5)
    assert metrics["f1_class_1"] == pytest.approx(0.5)
    assert metrics["roc_auc"] == pytest.approx(0.75)
    assert "precision" in metrics["classification_report"]
