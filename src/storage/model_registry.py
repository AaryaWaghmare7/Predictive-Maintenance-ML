"""Helpers for model storage and versioning.

Later, this file can track saved model filenames, versions, metrics, and
training dates. For now it only defines the intended storage location.
"""

from __future__ import annotations

from pathlib import Path

from src.config.paths import MODEL_DIR


def get_model_path(model_name: str) -> Path:
    """Return where a model artifact should be saved or loaded."""
    return MODEL_DIR / model_name
