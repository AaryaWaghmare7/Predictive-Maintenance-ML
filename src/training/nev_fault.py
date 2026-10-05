"""Fixed Experiment 2 candidates; selection never receives the test dataset."""

from __future__ import annotations

import warnings

import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.data.nev_fault import NEV_FEATURES, get_nev_xy
from src.evaluation.metrics import multiclass_classification_metrics
from src.preprocessing.cleaning import build_numeric_preprocessor

RANDOM_STATE = 42
CV_FOLDS = 5


def build_nev_candidates() -> dict[str, Pipeline]:
    """Unfitted fixed configurations, with no test-driven tuning or resampling.

    Reuse the existing median/scaling preprocessor for Logistic Regression.
    Trees need no scaling, but use a named-column median imputer in Pipeline.
    Every learned transformation is fitted independently inside each CV fold.
    """
    estimators = {
        "logistic_regression": LogisticRegression(
            class_weight="balanced", solver="lbfgs", max_iter=2000, random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.1, max_leaf_nodes=31,
            l2_regularization=1.0, early_stopping=False, random_state=RANDOM_STATE
        ),
    }
    candidates = {}
    for name, estimator in estimators.items():
        preprocessor = build_numeric_preprocessor(list(NEV_FEATURES)) if name == "logistic_regression" else ColumnTransformer(
            [("numeric", SimpleImputer(strategy="median"), list(NEV_FEATURES))], remainder="drop"
        )
        candidates[name] = Pipeline([("preprocessor", preprocessor), ("classifier", estimator)])
    return candidates


def select_nev_model(training: pd.DataFrame) -> dict[str, object]:
    """Choose by mean five-fold training-only macro F1, before test evaluation."""
    x_train, y_train = get_nev_xy(training)
    if y_train.value_counts().min() < CV_FOLDS or set(y_train) != {0, 1, 2, 3}:
        raise ValueError("Training requires all four classes and at least five rows per class")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy", "roc_auc_ovr": "roc_auc_ovr",
        "precision_macro": make_scorer(precision_score, average="macro", zero_division=0),
        "recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
        "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
    }
    summaries = {}
    with warnings.catch_warnings():
        # A silently non-converged baseline is not an acceptable final result.
        warnings.simplefilter("error", ConvergenceWarning)
        for name, pipeline in build_nev_candidates().items():
            scores = cross_validate(pipeline, x_train, y_train, cv=cv, scoring=scoring, error_score="raise")
            summaries[name] = {
                metric: {"mean": float(scores[f"test_{metric}"].mean()),
                         "std": float(scores[f"test_{metric}"].std()),
                         "folds": scores[f"test_{metric}"].tolist()}
                for metric in scoring
            }
    selected = max(summaries, key=lambda name: summaries[name]["f1_macro"]["mean"])
    tree_names = ("random_forest", "hist_gradient_boosting")
    best_tree = max(tree_names, key=lambda name: summaries[name]["f1_macro"]["mean"])
    return {"selected_model": selected, "best_tree": best_tree, "cv": summaries}


def explain_nev_tree(training: pd.DataFrame, tree_name: str) -> dict[str, object]:
    """Permutation importance on one training-only validation fold, never test.

    This is descriptive model dependence, not causality. A single-fold
    importance ranking is uncertain, especially for correlated features.
    """
    if tree_name not in {"random_forest", "hist_gradient_boosting"}:
        raise ValueError("Interpretability requires one of the approved tree candidates")
    x, y = get_nev_xy(training)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    train_indices, validation_indices = next(cv.split(x, y))
    estimator = clone(build_nev_candidates()[tree_name])
    estimator.fit(x.iloc[train_indices], y.iloc[train_indices])
    importance = permutation_importance(
        estimator, x.iloc[validation_indices], y.iloc[validation_indices],
        scoring="f1_macro", n_repeats=10, random_state=RANDOM_STATE, n_jobs=1,
    )
    result = {"model": tree_name, "validation_rows": len(validation_indices),
              "permutation_macro_f1_drop": {
                  feature: {"mean": float(mean), "std": float(std)}
                  for feature, mean, std in zip(NEV_FEATURES, importance.importances_mean, importance.importances_std)
              }}
    classifier = estimator.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        result["impurity_importance"] = dict(zip(NEV_FEATURES, classifier.feature_importances_.tolist()))
    return result


def fit_and_evaluate_nev(
    training: pd.DataFrame, testing: pd.DataFrame, selection: dict[str, object]
) -> tuple[dict[str, Pipeline], dict[str, dict]]:
    """Fit fixed candidates on supplied training, then evaluate supplied test.

    The precommitted CV selection is not changed by test-set scores. Call this
    once per approved experiment, not repeatedly to optimize test metrics.
    """
    x_train, y_train = get_nev_xy(training)
    x_test, y_test = get_nev_xy(testing)
    models = build_nev_candidates()
    if selection["selected_model"] not in models:
        raise ValueError("Invalid pre-test model selection")
    results = {}
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        for name, pipeline in models.items():
            pipeline.fit(x_train, y_train)
            results[name] = multiclass_classification_metrics(
                y_test, pipeline.predict(x_test), pipeline.predict_proba(x_test), pipeline.classes_
            )
    return models, results
