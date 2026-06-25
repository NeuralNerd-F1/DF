"""Shared enums carry exactly the states the spec/constitution define."""

from __future__ import annotations

from app.contracts.enums import ConfidenceState, Status


def test_status_values() -> None:
    assert {s.name for s in Status} == {"OK", "DEGRADED", "FAILED"}
    assert isinstance(Status.OK, str)
    assert Status.OK == "OK"


def test_confidence_state_values() -> None:
    """The five mandatory uncertainty states (constitution #4 + spec §5.1/§5.2)."""
    assert {c.name for c in ConfidenceState} == {
        "HIGH",
        "MEDIUM",
        "LOW_CONFIDENCE",
        "AMBIGUOUS",
        "OUT_OF_DISTRIBUTION",
    }


def test_confidence_state_is_str_enum() -> None:
    assert isinstance(ConfidenceState.LOW_CONFIDENCE, str)
    assert ConfidenceState.LOW_CONFIDENCE == "LOW_CONFIDENCE"
