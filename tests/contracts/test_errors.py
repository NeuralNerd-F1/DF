"""The ErrorCode enum is the canonical registry — assert it is complete and clean."""

from __future__ import annotations

from app.contracts.errors import ErrorCode

# The full canonical set, seeded from the master spec (§3-§8). Grouped by owning
# feature for readability. This test is the drift guard: adding or removing a member
# without updating this set fails the build.
EXPECTED_CODES: set[str] = {
    # Sentinel + foundation (F00)
    "OK",
    "CONFIG_001",
    "DATA_QUALITY_LOG_UNWRITABLE",
    # Ingestion (F01)
    "DATA_001",
    "STALE_DATA",
    "DATA_SOURCE_UNAVAILABLE",
    "COMPOUND_UNKNOWN",
    "COMPOUND_FALLBACK",
    "DERIVED_AGE",
    "CACHE_CORRUPT",
    "TRACK_STATUS_UNAVAILABLE",
    # Lap filter chain (F02)
    "LAP_001",
    "LAP_002",
    "LAP_003",
    "LAP_INTERP",
    "LAP_004",
    "LAP_005",
    "LAP_006",
    "LAP_PARTIAL",
    "COMBO_001",
    "DATA_INSUFFICIENT",
    "ORPHAN_SAMPLES",
    "VALIDATION_ERROR",
    # Feature extraction (F03)
    "FEATURE_001",
    "LAP_NULL_EXCESS",
    "IMPUTED",
    # Corner decomposition (F09)
    "CORNER_001",
    # Manual upload (F05)
    "UPLOAD_001",
    "UPLOAD_002",
    "UPLOAD_003",
    "UPLOAD_004",
    "RESAMPLED",
    # Classifier (F06)
    "LOW_CONFIDENCE",
    "AMBIGUOUS",
    "MODEL_NOT_FOUND",
    "INVALID_INPUT",
    "PERF_WARN",
    # Classifier training (F06a)
    "TRAIN_LEAKAGE",
    "MODEL_TRAINING_FAILED",
    # Style space (F08)
    "SPARSE_PROJECTION",
    "OUT_OF_DISTRIBUTION",
    "UMAP_FALLBACK",
    "PROJECTION_FALLBACK",
    # Degradation detector (F10)
    "MONITOR_INACTIVE",
    "BASELINE_INSUFFICIENT",
    "FALLBACK_THRESHOLD",
    "DEGRADED_QUALITY",
    "DISTANCE_NAN",
    "MONITOR_ERROR",
    # SHAP (F07)
    "SHAP_UNAVAILABLE",
    "SHAP_CONSISTENCY_WARN",
    # Personalisation (F11)
    "CONSTRAINT_CLIPPED",
    "CONSISTENCY_WARNING",
    "CIRCUIT_OVERRIDE_MONACO",
    "WIDE_CONFIDENCE",
    # Normalisation (F04)
    "NORMALISATION_FALLBACK",
    # Cache (F14)
    "CACHE_MISS",
    "CACHE_UNAVAILABLE",
    "CACHE_DEGRADED",
    # Database / platform (F12, F13)
    "DB_UNAVAILABLE",
    "HEALTH_FAIL",
}


def test_registry_is_exactly_the_canonical_set() -> None:
    """No code is missing and none has crept in unannounced."""
    actual = {c.name for c in ErrorCode}
    assert actual == EXPECTED_CODES


def test_codes_named_by_written_specs_are_present() -> None:
    """Codes referenced by the already-written F01/F06 specs must exist."""
    for code in (
        "DATA_001",
        "STALE_DATA",
        "DATA_SOURCE_UNAVAILABLE",
        "COMPOUND_UNKNOWN",
        "DERIVED_AGE",
        "LOW_CONFIDENCE",
        "AMBIGUOUS",
        "MODEL_NOT_FOUND",
        "INVALID_INPUT",
        "PERF_WARN",
    ):
        assert code in ErrorCode.__members__


def test_name_equals_value() -> None:
    """Each member's value is its own code string (clean serialisation)."""
    for c in ErrorCode:
        assert c.value == c.name


def test_values_are_unique() -> None:
    values = [c.value for c in ErrorCode]
    assert len(values) == len(set(values))


def test_is_str_enum() -> None:
    """Members are strings so they drop into JSON / the data_quality_log directly."""
    assert isinstance(ErrorCode.LAP_003, str)
    assert ErrorCode.LAP_003 == "LAP_003"
