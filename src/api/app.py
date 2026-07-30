"""Future API application placeholder.

This file is reserved for a future web/API deployment, for example using
FastAPI. We are not adding API behavior yet because the ML pipeline is
not ready.
"""

from __future__ import annotations


def health_check() -> dict[str, str]:
    """Return a simple status response for future API testing."""
    return {"status": "ok", "message": "EV Predictive Maintenance AI API placeholder"}
