"""Model paths and trusted saved-NEV-pipeline loading; never train on load."""

from __future__ import annotations

from pathlib import Path
import warnings

import joblib
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.config.paths import MODEL_DIR
from src.data.nev_fault import NEV_CLASSES, NEV_FEATURES

NEV_PIPELINE_PATH = MODEL_DIR / "experiment_2" / "nev_fault_pipeline.joblib"


def get_model_path(model_name: str) -> Path:
    """Return where a model artifact should be saved or loaded."""
    return MODEL_DIR / model_name


def load_nev_fault_pipeline(path: str | Path = NEV_PIPELINE_PATH) -> Pipeline:
    """Load a trusted local artifact without fitting or silently replacing it.

    Joblib can execute code while deserializing. This path is controlled by
    application startup, never supplied by an API request or CSV upload.
    Incompatible sklearn versions are rejected rather than silently accepted.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Saved NEV pipeline is missing; obtain the trusted local artifact")
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        pipeline = joblib.load(path)
    if not isinstance(pipeline, Pipeline):
        raise ValueError("Saved NEV artifact must be a complete sklearn Pipeline")
    if not isinstance(pipeline.named_steps.get("classifier"), RandomForestClassifier):
        raise ValueError("Expected the approved Random Forest fault classifier")
    if list(getattr(pipeline, "feature_names_in_", [])) != list(NEV_FEATURES):
        raise ValueError("Saved pipeline does not match the approved seven-feature ordering")
    if list(getattr(pipeline, "classes_", [])) != list(NEV_CLASSES):
        raise ValueError("Saved pipeline does not match the four documented fault classes")
    if not callable(getattr(pipeline, "predict_proba", None)):
        raise ValueError("Saved pipeline must support class probabilities")
    return pipeline
