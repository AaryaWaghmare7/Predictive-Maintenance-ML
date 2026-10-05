"""Local inference for the saved Experiment 2 pipeline; no API or website."""

import pandas as pd
from sklearn.pipeline import Pipeline

from src.data.nev_fault import NEV_CLASSES, NEV_FEATURES, validate_nev_frame


def predict_nev_faults(pipeline: Pipeline, sensors: pd.DataFrame) -> pd.DataFrame:
    """Accept normalized named inputs, and return codes/names/probabilities.

    Inputs must already use the dataset's (currently undocumented) normalized
    scale, not raw physical readings. Reordered columns are safe; missing,
    extra, target, nonfinite and out-of-range inputs are rejected, not guessed.
    """
    validate_nev_frame(sensors, require_target=False)
    if list(pipeline.classes_) != list(NEV_CLASSES):
        raise ValueError("Inference pipeline must contain all four NEV classes")
    inputs = sensors.loc[:, list(NEV_FEATURES)]
    predictions = pipeline.predict(inputs).astype(int)
    probabilities = pipeline.predict_proba(inputs)
    output = pd.DataFrame({"fault_code": predictions,
                           "fault_name": [NEV_CLASSES[code] for code in predictions]}, index=sensors.index)
    for index, code in enumerate(pipeline.classes_):
        output[f"probability_{int(code)}"] = probabilities[:, index]
    return output
