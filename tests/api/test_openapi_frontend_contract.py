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
    # team = the F16 seam; full_name/signature/profile = the Driver Overview card sources
    for field in ("id", "code", "team", "full_name", "radar_values", "signature", "profile"):
        assert field in props, f"DriverSummary lost {field!r} — breaks the Driver Overview"


def test_driver_profile_and_signal_schemas_carry_the_overview_fields() -> None:
    schemas = _schemas()
    for field in ("laps_analysed", "seasons", "lap_time_consistency_s", "identification_f1"):
        assert field in schemas["DriverProfileSchema"]["properties"], (
            f"DriverProfileSchema lost {field!r} — breaks the overview stats row"
        )
    for field in ("feature", "attribution", "percentile"):
        assert field in schemas["DriverSignalSchema"]["properties"], (
            f"DriverSignalSchema lost {field!r} — breaks the key signals"
        )


def test_fingerprint_response_fields() -> None:
    props = _schemas()["FingerprintResponse"]["properties"]
    # shap_signed = the profile screen's confirms/argues panel source
    for field in ("vector_5d", "shap_radar", "shap_signed", "nl_description", "stability"):
        assert field in props, f"FingerprintResponse lost {field!r} — breaks the Driver Profile"


def test_setup_parameter_fields() -> None:
    props = _schemas()["SetupParameterSchema"]["properties"]
    for field in (
        "name", "value", "value_range", "rating", "confidence", "shap_available",
        "grid_mean", "basis_features",
    ):
        assert field in props, f"SetupParameterSchema lost {field!r} — breaks the setup panel"


def test_tyre_wear_schema_carries_the_screen_fields() -> None:
    schemas = _schemas()
    for field in (
        "n_stints", "median_stint_len", "deg_slope_s_per_lap", "total_delta_s",
        "fuel_slope_s_per_lap", "cliff_age", "curve",
    ):
        assert field in schemas["TyreCompoundStatsSchema"]["properties"], (
            f"TyreCompoundStatsSchema lost {field!r} — breaks the Tyre Wear screen"
        )
    for field in ("age", "delta_s", "q25", "q75", "n"):
        assert field in schemas["TyreCurvePointSchema"]["properties"], (
            f"TyreCurvePointSchema lost {field!r} — breaks the degradation chart"
        )


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


# --- F17 (Screen 3 Ingestion/Prediction + Screen 4 Comparison) ------------------------------------
# New screens read these schemas (aliased in lib/api/types.ts, guarded in lib/api/validators.ts).
# A backend rename must fail HERE, not in the browser.


def test_predict_response_fields() -> None:
    props = _schemas()["PredictResponse"]["properties"]
    for field in (
        "predicted_driver",
        "confidence",
        "confidence_state",
        "top_3",
        "shap_values",
        "shap_available",
        "model_version",
        "confidence_breakdown",
    ):
        assert field in props, f"PredictResponse lost {field!r} — breaks Screen 3 prediction"


def test_confidence_breakdown_fields() -> None:
    props = _schemas()["ConfidenceBreakdownSchema"]["properties"]
    for field in ("feature_quality", "distribution", "model_confidence", "overall"):
        assert field in props, f"ConfidenceBreakdownSchema lost {field!r} — breaks the breakdown"


def test_unknown_fingerprint_response_fields() -> None:
    props = _schemas()["UnknownFingerprintResponse"]["properties"]
    for field in (
        "fingerprint_id",
        "fingerprint_vector",
        "nearest_neighbours",
        "risk_flags",
        "data_quality",
        "confidence_state",
        "warnings",
    ):
        assert field in props, f"UnknownFingerprintResponse lost {field!r} — Unknown Driver tab"


def test_compare_response_fields() -> None:
    props = _schemas()["CompareResponse"]["properties"]
    for field in (
        "driver_1_radar",
        "driver_2_radar",
        "delta_shap",
        "most_distinctive_features",
        "compatibility_delta",
    ):
        assert field in props, f"CompareResponse lost {field!r} — breaks Screen 4 comparison"


def test_ingest_status_response_fields() -> None:
    # /ingest/status is include_in_schema=False (outside the presented surface), so its DTO is
    # not in the OpenAPI components; the frontend poll contract is asserted on the model itself.
    from app.api.schemas import IngestStatusResponse

    props = IngestStatusResponse.model_json_schema()["properties"]
    for field in ("status", "progress", "error_detail", "result"):
        assert field in props, f"IngestStatusResponse lost {field!r} — FastF1 pipeline poll"


def test_upload_response_fields() -> None:
    # /ingest/upload is a 503 stub (F05), but its schema ships so the report table is a 1-line swap.
    props = _schemas()["UploadResponse"]["properties"]
    for field in ("upload_id", "laps_accepted", "laps_rejected", "cleaning_report", "warnings"):
        assert field in props, f"UploadResponse lost {field!r} — breaks the Manual Upload swap-in"


# --- F18 (Screen 5 Style Space Map) ---------------------------------------------------------------


def test_style_space_driver_fields() -> None:
    props = _schemas()["StyleSpaceDriver"]["properties"]
    for field in ("code", "coords_2d", "coords_5d"):
        assert field in props, f"StyleSpaceDriver lost {field!r} — breaks the Style Space scatter"


def test_style_space_response_fields() -> None:
    props = _schemas()["StyleSpaceResponse"]["properties"]
    for field in ("drivers", "unknown_drivers"):
        assert field in props, f"StyleSpaceResponse lost {field!r} — breaks Screen 5"


# --- F19 (Screen 6 Historical Race Replay Monitor) ------------------------------------------------
# Screen 6 reads ReplayLap/ReplayResponse (aliased in lib/api/types.ts, guarded by zReplay). The
# replay redesign widened the DTO with τ + the DegradationAlert (closing div #72–#74) and added
# the /sessions catalog (closing div #77).


def test_replay_lap_fields() -> None:
    props = _schemas()["ReplayLap"]["properties"]
    for field in (
        "lap_number", "feature_vector", "deviation_score", "alert_state",
        "fingerprint_snapshot", "threshold", "alert",
    ):
        assert field in props, f"ReplayLap lost {field!r} — breaks the Replay Monitor"


def test_replay_alert_fields() -> None:
    props = _schemas()["ReplayAlert"]["properties"]
    for field in (
        "alert_lap", "first_crossing_lap", "laps_since_threshold_crossed", "deviation_score",
        "threshold", "alert_confidence", "feature_attribution", "top_features",
        "optimal_window_remaining_laps",
    ):
        assert field in props, f"ReplayAlert lost {field!r} — breaks the alert banner"


def test_replay_response_fields() -> None:
    props = _schemas()["ReplayResponse"]["properties"]
    assert "laps" in props, "ReplayResponse lost 'laps' — breaks Screen 6"


def test_session_summary_fields() -> None:
    schemas = _schemas()
    for field in ("id", "circuit", "circuit_type", "season", "round", "session_type", "drivers"):
        assert field in schemas["SessionSummary"]["properties"], (
            f"SessionSummary lost {field!r} — breaks the session picker"
        )
    for field in ("code", "n_laps"):
        assert field in schemas["SessionDriverSchema"]["properties"], (
            f"SessionDriverSchema lost {field!r} — breaks the driver picker"
        )
