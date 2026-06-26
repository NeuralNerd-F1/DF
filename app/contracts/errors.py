"""The canonical APEX error-code registry.

The project constitution requires that error codes are an ``Enum`` here and that
their strings are never hardcoded elsewhere. This enum is seeded *complete* from the
master specification so that every downstream feature references an existing member
rather than inventing a string. Each member is tagged with the feature that owns its
behaviour and the master-spec section that defines it; new codes are appended here as
features are built, keeping this the single source of truth.

Members serialise to their string value (``ErrorCode.LAP_003.value == "LAP_003"``),
so they drop straight into the ``data_quality_log`` and API payloads.
"""

from __future__ import annotations

from enum import StrEnum


class ErrorCode(StrEnum):
    """Every named error code, flag, fallback, and warning in the APEX spec."""

    # --- Sentinel -----------------------------------------------------------
    OK = "OK"  # clean lap / no error (master spec §8.4: laps.error_code = 'OK')

    # --- Foundation (F00) ---------------------------------------------------
    CONFIG_001 = "CONFIG_001"  # config.yaml missing or a required key absent/invalid
    DATA_QUALITY_LOG_UNWRITABLE = "DATA_QUALITY_LOG_UNWRITABLE"  # log sink unwritable

    # --- Data sources / ingestion (F01) — §3.1, §8.2, §8.3 ------------------
    DATA_001 = "DATA_001"  # FastF1 session unavailable; skip, never substitute
    STALE_DATA = "STALE_DATA"  # served from cache while source down (flag)
    DATA_SOURCE_UNAVAILABLE = "DATA_SOURCE_UNAVAILABLE"  # source down + no cache
    COMPOUND_UNKNOWN = "COMPOUND_UNKNOWN"  # OpenF1 + Ergast both down; compound UNKNOWN
    COMPOUND_FALLBACK = "COMPOUND_FALLBACK"  # OpenF1 down; compound sourced from Ergast (flag)
    DERIVED_AGE = "DERIVED_AGE"  # tyre age derived from pit flags (tag)
    CACHE_CORRUPT = "CACHE_CORRUPT"  # local Parquet cache unreadable/unwritable (F01 integrity)

    # --- Lap filter chain (F02) — §3.2, §3.3 --------------------------------
    LAP_001 = "LAP_001"  # not green-flag (SC / VSC / red)
    LAP_002 = "LAP_002"  # outlap / inlap (pit in/out event)
    LAP_003 = "LAP_003"  # >10% consecutive NaN in a telemetry channel
    LAP_INTERP = "LAP_INTERP"  # 1-5 NaN gap interpolated (accept + INTERPOLATED tag)
    LAP_004 = "LAP_004"  # effective sampling rate < 20 Hz
    LAP_005 = "LAP_005"  # speed anomaly (stationary samples > 1%)
    LAP_006 = "LAP_006"  # lap time outside 115% of driver session median
    LAP_PARTIAL = "LAP_PARTIAL"  # tyre age UNKNOWN -> F27/F28 NULL, lap still accepted
    COMBO_001 = "COMBO_001"  # < 30 clean laps post-filter -> exclude combination
    DATA_INSUFFICIENT = "DATA_INSUFFICIENT"  # < 30 clean laps coverage (§3.2)

    # --- Feature extraction (F03) — §4 --------------------------------------
    FEATURE_001 = "FEATURE_001"  # feature out-of-range / failed -> NULL + logged
    LAP_NULL_EXCESS = "LAP_NULL_EXCESS"  # > 6 NULL features -> reject vector (§4.5)

    # --- Corner decomposition (F09) — §4.4 ----------------------------------
    CORNER_001 = "CORNER_001"  # curvature / X-Y unavailable -> skip decomposition

    # --- Manual upload (F05) — §3.4 -----------------------------------------
    UPLOAD_001 = "UPLOAD_001"  # missing required columns
    UPLOAD_002 = "UPLOAD_002"  # speed likely mph not km/h (warning)
    UPLOAD_003 = "UPLOAD_003"  # > 5% corrupt rows
    UPLOAD_004 = "UPLOAD_004"  # empty file
    RESAMPLED = "RESAMPLED"  # inconsistent sampling resampled to 50 Hz (flag)

    # --- Classifier (F06) — §5.1 --------------------------------------------
    LOW_CONFIDENCE = "LOW_CONFIDENCE"  # max prob < confidence gate
    AMBIGUOUS = "AMBIGUOUS"  # Shannon entropy > threshold (likely non-F1 input)
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"  # champion model file absent at startup
    INVALID_INPUT = "INVALID_INPUT"  # > 6 NULLs -> do not call model
    PERF_WARN = "PERF_WARN"  # prediction latency over warn threshold

    # --- Style space (F08) — §5.2 -------------------------------------------
    SPARSE_PROJECTION = "SPARSE_PROJECTION"  # unknown driver < 30 laps
    OUT_OF_DISTRIBUTION = "OUT_OF_DISTRIBUTION"  # Mahalanobis > 3 SD from training
    UMAP_FALLBACK = "UMAP_FALLBACK"  # UMAP model corrupt / missing -> PCA fallback
    PROJECTION_FALLBACK = "PROJECTION_FALLBACK"  # transform() failed -> NN fallback

    # --- Degradation detector (F10) — §5.3 ----------------------------------
    MONITOR_INACTIVE = "MONITOR_INACTIVE"  # stint < 8 laps
    BASELINE_INSUFFICIENT = "BASELINE_INSUFFICIENT"  # < 3 clean baseline laps
    FALLBACK_THRESHOLD = "FALLBACK_THRESHOLD"  # global driver variation used (new circuit)
    DEGRADED_QUALITY = "DEGRADED_QUALITY"  # all baseline laps interpolated -> cap confidence
    DISTANCE_NAN = "DISTANCE_NAN"  # cosine distance NaN; skip lap in rolling mean
    MONITOR_ERROR = "MONITOR_ERROR"  # 3 consecutive NaN -> halt monitoring for stint

    # --- SHAP explainability (F07) — §6.2 -----------------------------------
    SHAP_UNAVAILABLE = "SHAP_UNAVAILABLE"  # SHAP raised -> prediction returned without SHAP
    SHAP_CONSISTENCY_WARN = "SHAP_CONSISTENCY_WARN"  # SHAP sum far from expected value

    # --- Personalisation (F11) — §7.2 ---------------------------------------
    CONSTRAINT_CLIPPED = "CONSTRAINT_CLIPPED"  # value clipped to FIA legal bound
    CONSISTENCY_WARNING = "CONSISTENCY_WARNING"  # front wing vs brake bias inconsistent
    CIRCUIT_OVERRIDE_MONACO = "CIRCUIT_OVERRIDE_MONACO"  # Monaco max-downforce override
    WIDE_CONFIDENCE = "WIDE_CONFIDENCE"  # regression confidence < 0.5 -> return a range

    # --- Normalisation (F04) — §8.2 -----------------------------------------
    NORMALISATION_FALLBACK = "NORMALISATION_FALLBACK"  # circuit params missing -> global params

    # --- Cache (F14) — §8.2, §8.3 -------------------------------------------
    CACHE_MISS = "CACHE_MISS"  # key absent / Redis down -> graceful miss
    CACHE_UNAVAILABLE = "CACHE_UNAVAILABLE"  # Redis unavailable (logged)
    CACHE_DEGRADED = "CACHE_DEGRADED"  # breaker open; writes silently dropped

    # --- Database / platform (F12, F13) — §8.3, §8.5 ------------------------
    DB_UNAVAILABLE = "DB_UNAVAILABLE"  # Postgres breaker open -> API 503
    HEALTH_FAIL = "HEALTH_FAIL"  # critical component down at startup -> 503
