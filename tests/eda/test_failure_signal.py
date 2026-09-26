"""Tests for descriptive failure-signal diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.eda.failure_signal import analyze_failure_signal
from src.preprocessing.cleaning import MODEL_1_OPERATIONAL_FEATURES


def test_failure_signal_analysis_uses_chronology_and_does_not_mutate_input() -> None:
    rows = 60
    generator = np.random.default_rng(42)
    dataframe = pd.DataFrame(
        {
            feature: generator.normal(index, 1.0, rows)
            for index, feature in enumerate(MODEL_1_OPERATIONAL_FEATURES)
        }
    )
    dataframe["Timestamp"] = pd.date_range("2024-01-01", periods=rows, freq="15min")
    dataframe["Failure_Probability"] = [int(index % 5 == 0) for index in range(rows)]
    dataframe["Maintenance_Type"] = 0
    dataframe["RUL"] = 100.0
    dataframe["TTF"] = 50.0
    dataframe["Component_Health_Score"] = 0.8
    original = dataframe.copy(deep=True)

    analysis = analyze_failure_signal(dataframe)

    assert analysis.train_rows == 48
    assert analysis.test_rows == 12
    assert analysis.feature_summary["feature"].tolist() != []
    assert set(analysis.feature_summary["feature"]) == set(MODEL_1_OPERATIONAL_FEATURES)
    assert {
        "no_failure_q05",
        "no_failure_q25",
        "no_failure_q75",
        "no_failure_q95",
        "failure_q05",
        "failure_q25",
        "failure_q75",
        "failure_q95",
    }.issubset(analysis.feature_summary.columns)
    assert len(analysis.pair_summary) == 276
    assert set(analysis.target_autocorrelations) == {1, 4, 96, 672}
    assert analysis.monthly_target_rate.empty
    pd.testing.assert_frame_equal(dataframe, original)
