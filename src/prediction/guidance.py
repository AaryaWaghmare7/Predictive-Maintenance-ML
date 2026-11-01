"""Versioned educational guidance, explicitly separate from fault/risk estimates."""

import json
from pathlib import Path

KNOWLEDGE = json.loads(Path(__file__).with_name("fault_guidance.json").read_text(encoding="utf-8"))


def fault_guidance(code, readings):
    rule = KNOWLEDGE["classes"][str(code)]
    return {"version": KNOWLEDGE["version"], "source": KNOWLEDGE["source"],
        "observed_normalized_readings": {f: float(readings[f]) for f in rule["sensors"]},
        "sensor_relevance": "These are inspection context, not feature attribution or proof of a cause. Physical limits cannot be inferred from undocumented normalization.",
        "possible_causes": rule["possible_causes"], "inspection_steps": rule["checks"],
        "maintenance_priority": {"category": rule["priority"],
            "basis": "A non-Normal dataset label requests review; a Normal label retains routine review. This is an administrative inspection rule, not urgency, safety clearance or calibrated risk.",
            "inputs": {"predicted_class": code}, "rule_version": KNOWLEDGE["version"],
            "time_horizon": None, "missing_information": ["OEM limits", "validated component history", "calibrated failure model"]},
        "failure_risk": {"status": "unavailable", "reason": "No validated future-event model, horizon or physical thresholds"},
        "rul": {"status": "unavailable", "reason": "No verified component-life dataset or model"},
        "limitations": KNOWLEDGE["limitation"], "safety": KNOWLEDGE["safety"]}
