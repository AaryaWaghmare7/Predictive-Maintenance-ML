"""Exact chart data, bounded payloads, reference scope and upload integration."""

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from scripts.smoke_test_localhost import SAMPLE_ROWS, csv_bytes
from src.api.app import create_app
from src.data.nev_fault import NEV_CLASSES, NEV_FEATURES
from src.eda.dashboard import (build_dashboard_data, binned_reference_scatter,
                               load_dashboard_reference, SCATTER_LIMIT)


def frame(rows=SAMPLE_ROWS):
    return pd.DataFrame(rows, columns=NEV_FEATURES)


@pytest.fixture
def client():
    with TestClient(create_app()) as active:
        yield active


def test_histograms_boxes_and_class_counts_are_exact():
    data = build_dashboard_data(frame(), labels=[0,1,2,3])
    assert data["rows"] == 4 and data["class_counts"] == {str(c): 1 for c in NEV_CLASSES}
    for feature in NEV_FEATURES:
        stats = data["groups"]["all"]["features"][feature]
        values = frame()[feature]
        assert stats["count"] == sum(stats["histogram"]) == 4
        assert stats["min"] == values.min() and stats["max"] == values.max()
        assert stats["q1"] == values.quantile(.25) and stats["q3"] == values.quantile(.75)
        assert stats["median"] == values.median()
    assert data["groups"]["1"]["features"][NEV_FEATURES[4]]["median"] == .921


def test_constant_or_insufficient_correlation_is_null_not_zero():
    data = build_dashboard_data(frame([SAMPLE_ROWS[0]]*4))
    assert all(value is None for row in data["groups"]["all"]["correlation"] for value in row)
    assert data["groups"]["all"]["pair_counts"][0][1] == 4
    assert data["class_counts"] is None and not data["associations"]
    assert build_dashboard_data(frame(SAMPLE_ROWS[:2]))["groups"]["all"]["correlation"][0][1] is None


def test_correlation_is_sensor_only_pairwise_and_matches_pandas():
    x = frame(); x.loc[0,NEV_FEATURES[0]] = np.nan
    x["Fault Label"] = [3,2,1,0]
    data = build_dashboard_data(x)
    expected = x[list(NEV_FEATURES)].corr(min_periods=3)
    assert data["features"] == list(NEV_FEATURES)
    assert data["groups"]["all"]["correlation"][0][1] == pytest.approx(expected.iloc[0,1])
    assert data["groups"]["all"]["pair_counts"][0][1] == 3
    json.dumps(data, allow_nan=False)


def test_missing_nonnumeric_and_infinite_cells_are_counted_not_imputed():
    x = frame(); x.loc[0,NEV_FEATURES[0]] = np.nan
    x[NEV_FEATURES[1]] = ["bad", .597, .275, .453]
    x.loc[1,NEV_FEATURES[2]] = np.inf
    data = build_dashboard_data(x)
    for feature in NEV_FEATURES[:3]:
        assert data["groups"]["all"]["features"][feature]["count"] == 3
        assert data["quality"][feature]["missing_or_invalid_for_charts"] == 1
    assert data["rows"] == 4


def test_under_overflow_and_roundoff_are_visible_without_changing_values():
    x = frame(); x[NEV_FEATURES[0]] = [-.1, 0, 1.0000000000000002, 20.0]
    before = x.copy(deep=True)
    stats = build_dashboard_data(x)["groups"]["all"]["features"][NEV_FEATURES[0]]
    assert stats["below_zero"] == stats["above_one"] == 1
    assert sum(stats["histogram"]) == 2 and stats["histogram"][-1] == 1
    assert stats["max"] == 20.0
    assert sum(stats["actual_range_histogram"]["counts"]) == 4
    pd.testing.assert_frame_equal(x,before)


def test_all_rows_summarized_but_scatter_is_deterministic_and_bounded():
    x = frame(list(SAMPLE_ROWS)*600)
    labels = [0,1,2,3]*600
    data = build_dashboard_data(x, labels=labels)
    assert data["rows"] == 2400
    assert data["groups"]["all"]["features"][NEV_FEATURES[0]]["count"] == 2400
    assert len(data["scatter"]["points"]) == SCATTER_LIMIT
    assert data["scatter"] == build_dashboard_data(x,labels=labels)["scatter"]
    for point in data["scatter"]["points"]:
        assert point["values"] == x.iloc[point["row"]-1].tolist()
        assert point["class"] == str(labels[point["row"]-1])


@pytest.mark.parametrize("labels", [[0,1,2,4], [0,1,2,1.5], [0,1]])
def test_unknown_or_misaligned_labels_are_never_silently_cast(labels):
    with pytest.raises(ValueError):
        build_dashboard_data(frame(),labels=labels)


def test_reference_matches_saved_scope_and_contains_no_raw_observation_vectors(client):
    response = client.get("/analytics/reference")
    assert response.status_code == 200
    data = response.json(); train=data["datasets"]["training"]; full=data["datasets"]["full"]
    assert train["rows"] == 7700 and train["class_counts"] == {"0":3500,"1":1400,"2":1400,"3":1400}
    assert full["rows"] == 11000 and set(full["groups"]) == {"all"}
    assert full["scatter"] is None and full["associations"] == {}
    assert train["scatter"]["kind"] == "binned_reference" and "points" not in train["scatter"]
    assert full["groups"]["all"]["correlation"][0][3] == pytest.approx(-.49813169134362073)
    assert data["historical_associations"]["Voltage (V)"]["eta_squared"] == pytest.approx(.755029,abs=1e-6)
    assert data["historical_importance"]["validation_rows"] == 1540
    for groups in train["scatter"]["pairs"].values():
        for code,bins in groups.items():
            assert sum(cell[2] for cell in bins) == train["class_counts"][code]
    assert len(response.content) < 250_000


def test_comparison_matches_training_and_flags_only_documented_heuristics():
    reference=load_dashboard_reference()
    x=frame([SAMPLE_ROWS[1]]*40)
    x[NEV_FEATURES[5]] = 0.0  # Endpoint absent from training; retain, warn, don't clip.
    data=build_dashboard_data(x, reference=reference, normalized=False)
    comparison=data["comparison"]
    assert comparison["available"] and not comparison["normalization_confirmed"]
    assert comparison["features"][NEV_FEATURES[5]]["outside_training_range"] == 40
    assert comparison["features"][NEV_FEATURES[4]]["shape_review"]
    small=build_dashboard_data(frame(),reference=reference)["comparison"]
    assert not any(s["shape_review"] for s in small["features"].values())
    assert "not statistical significance" in comparison["method"]


def test_legacy_csv_uses_all_rows_and_new_upload_replaces_statistics(client):
    first=client.post("/analyze-csv",files={"file":("first.csv",csv_bytes(list(SAMPLE_ROWS)*300))}).json()
    assert first["predictions_returned"] == 1000
    assert first["analytics"]["rows"] == 1200
    assert first["analytics"]["class_counts"] == {str(c):300 for c in NEV_CLASSES}
    second=client.post("/analyze-csv",files={"file":("second.csv",csv_bytes([SAMPLE_ROWS[1]]*3))}).json()
    assert second["analytics"]["rows"] == 3
    assert second["analytics"]["class_counts"]["1"] == 3
    assert second["analytics"]["groups"]["all"]["features"][NEV_FEATURES[0]]["median"] == .805
    assert second["analytics"]["label_basis"] == "model_predictions"


def test_inspection_can_chart_raw_numeric_values_without_running_inference(client,monkeypatch):
    from src.api import datasets
    def forbidden(*args):
        raise AssertionError("Inspect-only analytics must not call the model")
    monkeypatch.setattr(datasets,"predict_nev_faults",forbidden)
    x=frame(); x[NEV_FEATURES[0]] = [300,400,450,500]
    response=client.post("/analyze-dataset",files={"file":("physical.csv",x.to_csv(index=False).encode())})
    assert response.status_code == 200
    result=response.json()
    assert not result["inspection"]["compatible"]
    assert "prediction" not in result and result["analytics"]["class_counts"] is None
    assert result["analytics"]["quality"][NEV_FEATURES[0]]["outside_normalized_range"] == 4
    assert result["analytics"]["groups"]["all"]["features"][NEV_FEATURES[0]]["max"] == 500
    actual=result["analytics"]["groups"]["all"]["features"][NEV_FEATURES[0]]["actual_range_histogram"]
    assert actual["edges"][0] == 300 and actual["edges"][-1] == 500
    assert sum(actual["counts"]) == 4


def test_ground_truth_and_prediction_grouping_are_distinct(client):
    x=frame(); x["Fault Label"]=[3,2,1,0]
    content=x.to_csv(index=False).encode()
    prediction=client.post("/analyze-dataset",files={"file":("x.csv",content)},data={"mode":"predict","normalized":"true"}).json()
    evaluated=client.post("/analyze-dataset",files={"file":("x.csv",content)},data={"mode":"evaluate","normalized":"true","labels_confirmed":"true"}).json()
    assert prediction["analytics"]["label_basis"] == "model_predictions"
    assert evaluated["analytics"]["label_basis"] == "user_confirmed_labels"
    assert prediction["analytics"]["groups"]["0"]["features"][NEV_FEATURES[0]]["median"] == .939
    assert evaluated["analytics"]["groups"]["0"]["features"][NEV_FEATURES[0]]["median"] == .180
    assert prediction["prediction"]["predictions"] == evaluated["prediction"]["predictions"]
    assert "analytics" not in evaluated["prediction"]


def test_reference_unavailable_does_not_fabricate_training_numbers(client,monkeypatch):
    from src.api import datasets, app as app_module
    monkeypatch.setattr(app_module,"load_dashboard_reference",lambda:None)
    monkeypatch.setattr(datasets,"load_dashboard_reference",lambda:None)
    assert client.get("/analytics/reference").status_code == 503
    result=client.post("/analyze-dataset",files={"file":("x.csv",csv_bytes())}).json()
    assert not result["analytics"]["comparison"]["available"]
    assert result["analytics"]["rows"] == 4
