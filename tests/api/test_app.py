"""Tests for the future API placeholder."""

from __future__ import annotations

from src.api.app import health_check


def test_health_check() -> None:
    response = health_check()

    assert response["status"] == "ok"
