"""Tests for descriptive Model 2 maintenance-type signal diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.eda.maintenance_type_signal import analyze_maintenance_type_signal
from src.preprocessing.cleaning import MODEL_1_OPERATIONAL_FEATURES, MODEL_2_EXCLUDED_COLUMNS


def test_maintenance_type_signal_uses_chronology_and_preserves_input() -> None:
    rows = 80
    generator = np.random.default_rng(42)
    dataframe = pd.DataFrame(
        {
            feature: generator.normal(index, 1.0, rows)
            for index, feature in enumerate(MODEL_1_OPERATIONAL_FEATURES)
        }
    )
    dataframe["Timestamp"] = pd.date_range("2024-01-01", periods=rows, freq="15min")
    dataframe["Maintenance_Type"] = np.tile([0, 1, 2, 3], rows // 4)
    dataframe["Failure_Probability"] = np.tile([0, 1], rows // 2)
    dataframe["RUL"] = 100.0
    dataframe["TTF"] = 50.0
    dataframe["Component_Health_Score"] = 0.8
    original = dataframe.copy(deep=True)

    analysis = analyze_maintenance_type_signal(dataframe)

    assert analysis.train_rows == 64
    assert analysis.test_rows == 16
    assert set(analysis.class_distribution["class"]) == {0, 1, 2, 3}
    assert set(analysis.feature_summary["feature"]) == set(MODEL_1_OPERATIONAL_FEATURES)
    assert len(analysis.class_feature_summary) == 24 * 4
    assert len(analysis.pair_summary) == 276
    assert set(analysis.binned_class_rates) == set(MODEL_1_OPERATIONAL_FEATURES)
    assert analysis.monthly_class_rates.empty
    assert set(analysis.class_autocorrelations["lag"]) == {1, 4, 96, 672}
    assert not analysis.leakage_summary["present_in_X"].any()
    assert set(analysis.leakage_summary["column"]) == set(MODEL_2_EXCLUDED_COLUMNS)
    pd.testing.assert_frame_equal(dataframe, original)
