"""Train a simple baseline model once the target column is known."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score

from src.data.load_data import load_dataset


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing for numeric and categorical columns."""
    numeric_columns = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = features.select_dtypes(exclude=["number"]).columns.tolist()

    return ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), numeric_columns),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a baseline predictive maintenance model.")
    parser.add_argument("--data", required=True, help="Path to dataset file.")
    parser.add_argument("--target", required=True, help="Target column name.")
    parser.add_argument("--output", default="models/baseline_model.joblib", help="Model output path.")
    args = parser.parse_args()

    dataframe = load_dataset(args.data)
    if args.target not in dataframe.columns:
        raise ValueError(f"Target column '{args.target}' not found in dataset.")

    features = dataframe.drop(columns=[args.target])
    target = dataframe[args.target]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target if target.nunique() > 1 else None,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(features)),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(classification_report(y_test, predictions))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    print(f"Saved model to {output_path}")


if __name__ == "__main__":
    main()
