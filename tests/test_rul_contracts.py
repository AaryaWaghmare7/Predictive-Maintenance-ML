"""Future RUL schema tests, not RUL model training or numerical predictions."""

import pytest
from pydantic import ValidationError

from src.rul.contracts import RULPrediction


def test_rul_contract_has_no_placeholder_remaining_life():
    with pytest.raises(ValidationError):
        RULPrediction(component="motor", machine_id="unit-1", unit="operating_hours", model_version="validated-version")
    assert RULPrediction.model_fields["estimated_rul"].is_required()


@pytest.mark.parametrize("invalid", [-1, float("nan"), float("inf"), True, "12"])
def test_rul_contract_rejects_invalid_values(invalid):
    with pytest.raises(ValidationError):
        RULPrediction(component="motor", machine_id="unit-1", estimated_rul=invalid,
                      unit="operating_hours", model_version="unit-test-schema")
