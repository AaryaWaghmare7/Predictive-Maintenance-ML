import pandas as pd

from src.preprocessing.cleaning import (
    normalize_column_names,
    parse_and_sort_timestamp,
    remove_duplicate_rows,
)


def test_normalize_column_names() -> None:
    dataframe = pd.DataFrame({" Motor Temperature (C) ": [48.0]})

    result = normalize_column_names(dataframe)

    assert result.columns.tolist() == ["motor_temperature_c"]


def test_parse_and_sort_timestamp() -> None:
    dataframe = pd.DataFrame({"timestamp": ["2020-01-02", "2020-01-01"]})

    result = parse_and_sort_timestamp(dataframe)

    assert result["timestamp"].tolist() == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]


def test_remove_duplicate_rows() -> None:
    dataframe = pd.DataFrame({"soc": [0.5, 0.5, 0.8]})

    result = remove_duplicate_rows(dataframe)

    assert len(result) == 2
