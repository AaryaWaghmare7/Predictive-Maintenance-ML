"""Check the existing trusted pipeline on synthetic normalized inputs; no fit."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.nev_fault import NEV_CLASSES, NEV_FEATURES, file_sha256
from src.prediction.nev_fault import predict_nev_faults
from src.storage.model_registry import NEV_PIPELINE_PATH, load_nev_fault_pipeline


def smoke_test_saved_model() -> pd.DataFrame:
    """Validate class mapping/probabilities/order without raw data or training.

    These four observations are synthetic examples, not evaluation labels.
    Missing artifacts raise an error. They are never regenerated automatically.
    """
    fingerprint = file_sha256(NEV_PIPELINE_PATH)
    pipeline = load_nev_fault_pipeline()
    sensors = pd.DataFrame([
        [.88, .6, .5, .3, .18, .5, .5],
        [.85, .5, .2, .4, .75, .5, .5],
        [.6, .15, .4, .3, .2, .5, .5],
        [.3, .3, .45, .8, .1, .5, .5],
    ], columns=NEV_FEATURES)
    output = predict_nev_faults(pipeline, sensors)
    pd.testing.assert_frame_equal(output, predict_nev_faults(pipeline, sensors[sensors.columns[::-1]]))
    probabilities = output.filter(like="probability_")
    assert probabilities.ge(0).all().all() and probabilities.le(1).all().all()
    assert probabilities.sum(axis=1).sub(1).abs().lt(1e-12).all()
    assert output.fault_name.tolist() == [NEV_CLASSES[code] for code in output.fault_code]
    assert file_sha256(NEV_PIPELINE_PATH) == fingerprint
    return output


if __name__ == "__main__":
    print("Synthetic normalized observations; not new evaluation metrics:")
    print(smoke_test_saved_model().to_string(index=False))
    print("Saved-model loading, classes, probabilities and column reordering verified; no training performed.")
