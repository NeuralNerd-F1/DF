"""F16 frontend↔API contract drift guard.

The frontend generates its TS types + zod validators from ``frontend/openapi.snapshot.json`` (an
offline dump of ``create_app().openapi()``). If a backend rename drops a field the screens read, it
must fail HERE (pytest), not silently in the browser (F16 spec §4.0-c / §8). This test asserts the
LIVE schema still carries the fields the Command Centre and Driver Profile consume.
"""

from __future__ import annotations

from pathlib import Path

from app.api.main import create_app
from app.contracts.config import load_config
from app.contracts.enums import ConfidenceState

_CONFIG = Path(__file__).resolve().parents[2] / "config.yaml"


def _schemas() -> dict:
    app = create_app(load_config(str(_CONFIG)))
    return app.openapi()["components"]["schemas"]


def test_driver_summary_carries_the_command_centre_fields() -> None:
    props = _schemas()["DriverSummary"]["properties"]
    for field in ("id", "code", "team", "radar_values"):  # team = the F16 seam
        assert field in props, f"DriverSummary lost {field!r} — breaks the Command Centre"


def test_fingerprint_response_fields() -> None:
    props = _schemas()["FingerprintResponse"]["properties"]
    for field in ("vector_5d", "shap_radar", "nl_description", "stability"):
        assert field in props, f"FingerprintResponse lost {field!r} — breaks the Driver Profile"


def test_setup_parameter_fields() -> None:
    props = _schemas()["SetupParameterSchema"]["properties"]
    for field in ("name", "value", "value_range", "rating", "confidence", "shap_available"):
        assert field in props, f"SetupParameterSchema lost {field!r} — breaks the setup panel"


def test_stability_fields() -> None:
    props = _schemas()["StabilityResponse"]["properties"]
    for field in ("stability_score", "pairwise_js", "seasons_available", "partial"):
        assert field in props, f"StabilityResponse lost {field!r}"


def test_confidence_state_values_match_the_frontend_union() -> None:
    # The frontend colours + gates off exactly these five states (lib/api/schema.d.ts).
    assert {s.value for s in ConfidenceState} == {
        "HIGH",
        "MEDIUM",
        "LOW_CONFIDENCE",
        "AMBIGUOUS",
        "OUT_OF_DISTRIBUTION",
    }
