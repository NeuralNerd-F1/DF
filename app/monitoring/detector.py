"""F10 — Model 3 (tyre degradation detector): within-stint statistical deviation.

This is **not a separate ML model** (master spec §5.3): it is a statistical deviation
detector operating on the fingerprint vectors produced by Models 1/2. Within a race stint
it measures the **cosine distance** between each lap's F04-normalised **then F10
train-only-standardised** 26-feature vector (F27/F28 excluded from the cosine core) and a
within-stint **baseline fingerprint** (the first ``baseline_laps`` clean laps), smooths it
with a ``rolling_window``-lap rolling mean, adds a weighted **live-window F27/F28 slope
component**, and fires a degradation alert when the composite exceeds a per-driver /
per-circuit-type threshold ``τ = μ + alert_threshold_sd·σ`` calibrated from **training data
only**. The goal is an alert **2–5 laps before** the behavioural signal becomes a lap-time
signal (§2.3) — the system's RQ3 instrument and dissertation Contribution 2.

Architecture (mirrors F04/F06/F08 — one writer, read-only serving, no silent failure):

* :func:`calibrate_stint_variation` is the ONLY fit site. It quarantines held-out + non-train
  rows **before any reduction**, a fail-loud :func:`_assert_no_leakage` tripwire raises
  :class:`DetectorCalibrationError` (``TRAIN_LEAKAGE``) on any survivor (**no artefact on a
  poisoned run**), fits the train-only 26-dim feature standardiser (divergence #7 — the F08-#6
  twin), pools per-``(driver, circuit_type)`` deviation stats + F27/F28 slope stats + pit-window
  medians, and writes ``artefacts/stint_variation.json`` deterministically (sorted keys, no
  timestamp, embedded ``content_hash`` → byte-identical on re-run).
* :func:`load_detector` reads + verifies the artefact (schema, hash, empty-thresholds,
  missing/wrong-width scaler, structural-knob mismatch → **raise** ``MODEL_NOT_FOUND``;
  crossing-knob mismatch → serve with ``pit_window_valid=False``).
* :func:`monitor_stint`, :func:`health` and :func:`lead_time_report` perform **no ``.fit(``**
  and **never raise** — every fault is a typed result + a ``data_quality_log`` row.

Every threshold / filename is read from ``config`` (``monitoring:`` block + referenced
``features.min_stint_laps`` / ``validate.degradation_features`` / ``normalise.min_std``); there
are no magic numbers here. **No new error code** — the six §5.3 codes are pre-seeded in
``errors.py`` and F10 reuses ``LAP_001`` / ``INVALID_INPUT`` / ``MODEL_NOT_FOUND`` / … per the
F08 precedent.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import mlflow
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.contracts.config import AppConfig, MonitoringConfig
from app.contracts.enums import Status
from app.contracts.errors import ErrorCode
from app.contracts.logging import DataQualityLogError, log_data_quality
from app.contracts.results import BaseResult
from app.ml.stylespace import ProjectionResult, StyleSpaceHandle, project_unknown
from app.pipeline.features import (
    FEATURE_IDS,
    DegradationResult,
    StintLap,
    compute_degradation_slopes,
)
from app.pipeline.normalise import (
    NormalisationError,
    NormalisationParams,
    apply_normalisation,
    load_normalisation_params,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

# --- Structural constants (NOT tunable thresholds — those live in config) -----------------
SCHEMA_VERSION = "1.0"  # bumped only when the on-disk artefact shape changes
_VERSION_HASH_LEN = 16
_AGG_DDOF = 0  # population std (single sample -> 0.0, never NaN) — matches F04/F08 convention
_MIN_STD = 1.0e-8  # standardiser / σ floor when config.normalise is absent (else normalise.min_std)
_GLOBAL_SLOPE_KEY = "__global__"  # reserved slope-stats tier for the all-circuit-type pool
_SLOPE_STINT_SENTINEL = 0  # monitor_stint holds exactly one stint -> fixed stint_id for F03b
_RAW_F01 = "F01"  # throttle_mean — F03b's F27 regressor (RAW; F04 does not touch it)
_RAW_F06 = "F06"  # brake_peak_mean — F03b's F28 regressor (RAW)
_RAW_F27_KEY = "F27"  # tyre-age throttle slope (attribution key + slope-stats key)
_RAW_F28_KEY = "F28"  # tyre-age brake slope

# Threshold-source labels (StintMonitorResult.threshold_source; never bare strings elsewhere).
_SRC_CELL = "driver_circuit_type"
_SRC_DRIVER = "driver_global"
_SRC_GLOBAL = "global"

# Wide-matrix column names (mirrors classifier/stylespace _REQUIRED_COLUMNS convention).
COL_DRIVER = "driver"
COL_CIRCUIT = "circuit"
COL_CIRCUIT_TYPE = "circuit_type"
COL_SEASON = "season"
COL_SESSION = "session"
COL_STINT = "stint"
COL_LAP = "lap_number"
COL_SPLIT = "split"
COL_TRACK_STATUS = "track_status"
COL_IS_CLEAN = "is_clean"
COL_HAS_INTERP = "has_interpolation"
COL_TYRE_AGE = "tyre_age"
COL_COMPOUND = "compound"
COL_PIT_LAP = "pit_lap"

# Calibration matrix superset (§3). Serving needs only the per-lap subset below.
_REQUIRED_COLUMNS = (
    COL_DRIVER,
    COL_CIRCUIT,
    COL_CIRCUIT_TYPE,
    COL_SEASON,
    COL_SESSION,
    COL_STINT,
    COL_LAP,
    COL_SPLIT,
    COL_TRACK_STATUS,
    COL_IS_CLEAN,
    COL_HAS_INTERP,
    COL_TYRE_AGE,
    *FEATURE_IDS,
)
_SERVING_REQUIRED = (
    COL_LAP,
    COL_TYRE_AGE,
    COL_TRACK_STATUS,
    COL_IS_CLEAN,
    COL_HAS_INTERP,
    *FEATURE_IDS,
)
_STINT_GROUP_KEYS = (COL_DRIVER, COL_SEASON, COL_SESSION, COL_STINT)

# The five structural knobs whose value the Pass-B μ/σ depend on: a mismatch invalidates the
# artefact (load refuses). The two crossing knobs are applied at serve, so a mismatch only
# suppresses the pit-window statistic (pit_window_valid=False). (§4.3)
_STRUCTURAL_KNOBS = (
    "rolling_window",
    "baseline_laps",
    "baseline_window_extended",
    "slope_weight",
    "min_stint_laps",
)
_CROSSING_KNOBS = ("alert_threshold_sd", "alert_confirm_laps")

# data_quality_log detail tags (mirrors normalise.py / stylespace.py REASON_* convention).
REASON_SLOPE_STINT = "SLOPE_STINT_NO_TYRE_AGE"
REASON_SLOPE_STATS = "SLOPE_STATS_MISSING_ALL_TIERS"
REASON_ZERO_COMPOSITE = "ZERO_COMPOSITE_SCORES"
REASON_FALLBACK = "THRESHOLD_FALLBACK"
REASON_BASELINE_SKIP = "BASELINE_CANDIDATE_INELIGIBLE"
REASON_MID_SKIP = "MID_STINT_INELIGIBLE"
REASON_DISTANCE_NAN = "COSINE_DISTANCE_NAN"
REASON_HALT = "CONSECUTIVE_NAN_HALT"
REASON_BASELINE_INSUFF = "BASELINE_INSUFFICIENT"
REASON_INACTIVE = "STINT_TOO_SHORT"
REASON_DEGRADED_QUALITY = "ALL_BASELINE_LAPS_INTERPOLATED"
REASON_INVALID = "STINT_INPUT_MALFORMED"
REASON_BACKSTOP = "UNFORESEEN_SERVING_FAULT"
REASON_WATCH = "SINGLE_CROSSING_WATCH"
REASON_MLFLOW = "MLFLOW_LOGGING_DEGRADED"
REASON_NO_STINTS = "NO_USABLE_STINTS"

_ENTITY_LAP = "lap"
_ENTITY_STINT = "stint"
_ENTITY_BUILD = "training_run"
_ENTITY_MODEL = "model"


# ----------------------------------------------------------------------------------
# Feature-owned enums (per enums.py docstring: "the degradation alert band" lives here)
# ----------------------------------------------------------------------------------


class StintMonitorState(StrEnum):
    """Per-lap ``alert_state`` (§8.5) + stint-level ``monitor_state``."""

    INACTIVE = "INACTIVE"  # stint < activation floor — nothing scored
    BASELINE = "BASELINE"  # lap selected into the baseline fingerprint
    SKIPPED = "SKIPPED"  # lap not scored: ineligible (non-green / unclean) — per-lap only
    PENDING = "PENDING"  # scored, but rolling window not yet filled (no composite score)
    NOMINAL = "NOMINAL"  # composite score at/below threshold
    WATCH = "WATCH"  # threshold crossed, not yet confirmed ("logged but not alerted")
    ALERT = "ALERT"  # crossing confirmed alert_confirm_laps consecutive laps
    HALTED = "HALTED"  # MONITOR_ERROR — monitoring halted for this stint


class AlertConfidence(StrEnum):
    """§5.3 "confidence of alert (LOW / MEDIUM / HIGH)"."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


# ----------------------------------------------------------------------------------
# Exceptions (only the two build/load entry points raise; serving never does)
# ----------------------------------------------------------------------------------


class DetectorCalibrationError(Exception):
    """Raised by :func:`calibrate_stint_variation` on a build-breaking fault (no artefact written).

    Carries ``TRAIN_LEAKAGE`` (a held-out / non-train row reached the fit) or
    ``MODEL_TRAINING_FAILED`` (missing columns / empty pool / no usable stints). Fail-loud (#1)."""

    def __init__(self, message: str, *, error_code: ErrorCode) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code


class DetectorLoadError(Exception):
    """Raised by :func:`load_detector` when the artefact cannot serve. Carries ``MODEL_NOT_FOUND``
    — the service refuses to start (mirrors F06b/F08 load errors)."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = ErrorCode.MODEL_NOT_FOUND


# ----------------------------------------------------------------------------------
# Typed outputs (all Pydantic, extra="forbid"; BaseResult where a status is carried)
# ----------------------------------------------------------------------------------


class LapDeviation(BaseModel):
    """One row per stint lap (timeline chart). Baseline, skipped and scored laps alike — no lap is
    ever silently dropped (constitution rule 1)."""

    model_config = ConfigDict(extra="forbid")

    lap_number: int
    state: StintMonitorState
    deviation_raw: float | None = None  # 1 − cos; None for BASELINE / SKIPPED / NaN laps
    deviation_rolling: float | None = None  # rolling mean; None until the window fills
    deviation_score: float | None = None  # rolling + slope component (the τ comparand)
    slope_component: float = 0.0  # slope_weight · slope_z (0.0 when inactive)
    slope_component_active: bool = False
    threshold: float | None = None  # τ for this stint (constant per stint)
    flags: list[ErrorCode] = Field(default_factory=list)


class DegradationAlert(BaseModel):
    """§5.3 alert payload (mapped to ``degradation_alerts`` by F12 — see spec §2)."""

    model_config = ConfigDict(extra="forbid")

    alert_lap: int  # lap the confirmation completed on → degradation_alerts.lap_number
    first_crossing_lap: int  # start of the confirmed run (lead_time_actual seam)
    laps_since_threshold_crossed: int
    deviation_score: float  # → degradation_alerts.deviation_score
    threshold: float
    alert_confidence: AlertConfidence  # capped LOW under DEGRADED_QUALITY
    feature_attribution: dict[str, float]  # 28 keys; sums to deviation_score
    top_features: list[str]  # attribution_top_k names (Screen 6 banner; payload-only)
    optimal_window_remaining_laps: int | None = None  # §12.1 pit-window recommendation


class StintMonitorResult(BaseResult):
    """Never-raises serving result for one stint prefix."""

    driver: str = ""
    circuit: str = ""
    circuit_type: str | None = None
    monitor_state: StintMonitorState = StintMonitorState.INACTIVE
    laps: list[LapDeviation] = Field(default_factory=list)
    alert: DegradationAlert | None = None
    baseline_lap_numbers: list[int] = Field(default_factory=list)
    baseline_extended: bool = False
    degraded_quality: bool = False
    threshold_source: str = ""
    pit_window_valid: bool = True
    fingerprint_snapshot: ProjectionResult | None = None
    fingerprint_available: bool = False
    n_laps_scored: int = 0
    consecutive_nan: int = 0


class StintVariationFitResult(BaseResult):
    """Calibration output (mirrors F08 ``StyleSpaceBuildResult``)."""

    artefact_path: str = ""
    content_hash: str = ""
    n_stints_used: int = 0
    n_entries: int = 0  # (driver, circuit_type) cells written
    n_driver_global: int = 0
    coverage: float = 0.0  # cells with n_stints ≥ min_calibration_stints
    mlflow_run_id: str | None = None


class StintLeadTime(BaseModel):
    """One evaluated stint (Week-11 alert dump row, §11.3)."""

    model_config = ConfigDict(extra="forbid")

    driver: str
    circuit: str
    circuit_type: str | None
    compound: str  # F01 token; "UNKNOWN" excluded from stratified views
    n_laps: int
    pit_lap: int  # ground truth (explicit column or final-lap proxy)
    alert_lap: int | None  # None = no alert (miss)
    lead_time_laps: int | None  # pit_lap − alert_lap; negative = alert after pit
    safety_car_affected: bool  # any non-green lap in the stint (FN characterisation)


class LeadTimeReport(BaseResult):
    """§11.3 evaluation machinery — read-only, never raises (formally run in F20)."""

    n_stints: int = 0
    mean_lead_time_laps: float | None = None
    tp_rate: float | None = None
    fp_rate: float | None = None
    per_stint: list[StintLeadTime] = Field(default_factory=list)
    per_compound: dict[str, dict[str, float]] = Field(default_factory=dict)
    early_alerts_by_driver_circuit: dict[str, int] = Field(default_factory=dict)
    sensitivity: dict[str, dict[str, dict[str, float]]] = Field(default_factory=dict)
    optimal_threshold_per_circuit_type: dict[str, float] = Field(default_factory=dict)


class MonitorHealthResult(BaseResult):
    """Readiness for F13's ``GET /health`` (mirrors F06b / F08)."""

    ready: bool = False
    model_version: str = ""
    reason: ErrorCode | None = None


# ----------------------------------------------------------------------------------
# On-disk artefact model (Pydantic → schema-mismatch detection for free, like F04)
# ----------------------------------------------------------------------------------


class _FeatureStat(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mean: float
    std: float


class _CellStat(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mean: float
    std: float
    n_stints: int
    median_crossing_to_pit: float | None = None


class _CellEntry(_CellStat):
    driver: str
    circuit_type: str


class _DriverEntry(_CellStat):
    driver: str


class _CalibrationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    alert_threshold_sd: float
    alert_confirm_laps: int
    rolling_window: int
    baseline_laps: int
    baseline_window_extended: int
    slope_weight: float
    min_stint_laps: int


class _StintVariationArtefact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str
    config: _CalibrationConfig
    cosine_features: list[str]
    feature_scaler: dict[str, _FeatureStat]
    entries: list[_CellEntry]
    driver_global: list[_DriverEntry]
    global_: _CellStat
    slope_stats: dict[str, dict[str, _FeatureStat]]
    content_hash: str


# ----------------------------------------------------------------------------------
# Loaded serving bundle (immutable; mirrors StyleSpaceHandle)
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Cell:
    """One resolved threshold cell: μ/σ of intra-stint variation + the pit-window median."""

    mean: float
    std: float
    n_stints: int
    median: float | None


@dataclass(frozen=True)
class MonitorHandle:
    """The loaded, immutable serving bundle (built once at startup)."""

    entries: dict[tuple[str, str], _Cell]  # (driver, circuit_type) → cell
    driver_global: dict[str, _Cell]  # driver → cell (all circuit types)
    global_cell: _Cell  # all-drivers pool
    slope_stats: dict[str, dict[str, tuple[float, float]]]  # circuit_type|__global__ → {F27,F28}
    cosine_features: tuple[str, ...]  # the 26 cosine dims (FEATURE_IDS − degradation_features)
    feature_mean: dict[str, float]  # train-only standardiser (§4.0-e)
    feature_std: dict[str, float]
    calibration_config: _CalibrationConfig
    pit_window_valid: bool  # False when crossing knobs differ from calibration (§4.3)
    normalisation_params: NormalisationParams
    model_version: str
    min_std: float
    ready: bool = True


# ----------------------------------------------------------------------------------
# data-quality ledger (never propagates a sink failure — mirrors F04/F06/F08)
# ----------------------------------------------------------------------------------


@dataclass
class _DQLedger:
    log_path: Path
    rows: int = 0
    failed: int = 0

    def record(
        self, *, code: ErrorCode, entity_type: str, entity_id: str, detail: str, fallback_used: bool
    ) -> None:
        try:
            log_data_quality(
                error_code=code,
                entity_type=entity_type,
                entity_id=entity_id,
                error_detail=detail,
                fallback_used=fallback_used,
                log_path=self.log_path,
            )
            self.rows += 1
        except DataQualityLogError:
            self.failed += 1


@dataclass
class _DQEvent:
    """A deferred data-quality row produced by the pure scoring core; recorded by the serving
    caller's ledger (so the core stays pure and calibration does not spam serving rows)."""

    code: ErrorCode
    entity_type: str
    entity_id: str
    detail: str
    fallback_used: bool


def _dedup(flags: Iterable[ErrorCode]) -> list[ErrorCode]:
    """Order-preserving de-duplication (per-lap + stint-level codes can overlap)."""
    return list(dict.fromkeys(flags))


def _resolve_min_std(config: AppConfig) -> float:
    """The standardiser / σ floor: ``config.normalise.min_std`` when present, else :data:`_MIN_STD`.
    Never a bare literal, so the magic-number scan stays clean."""
    if config.normalise is not None:
        return float(config.normalise.min_std)
    return _MIN_STD


def _cosine_feature_ids(config: AppConfig) -> tuple[str, ...]:
    """The 26 cosine-core dims: ``FEATURE_IDS`` minus ``validate.degradation_features`` (F27/F28).
    F27/F28 enter only via the live-window slope component (§4.0-b)."""
    if config.lap_validation is None:
        raise ValueError("config.validate block absent; cannot resolve degradation_features")
    excluded = set(config.lap_validation.degradation_features)
    return tuple(f for f in FEATURE_IDS if f not in excluded)


# ----------------------------------------------------------------------------------
# Deterministic serialisation (mirrors F04's canonical-JSON + content-hash — copied
# locally per the house pattern: there is no shared hashing utility)
# ----------------------------------------------------------------------------------


def _canonical_body(artefact: _StintVariationArtefact) -> str:
    """Deterministic JSON of the artefact WITHOUT the hash field — the bytes the hash covers.
    Sorted keys + shortest-round-trip floats → byte-stable across runs (reproducibility, #3)."""
    body = artefact.model_dump(exclude={"content_hash"})
    return json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash_artefact(artefact: _StintVariationArtefact) -> str:
    return sha256(_canonical_body(artefact).encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------------
# Pure per-lap helpers (unit-testable in isolation)
# ----------------------------------------------------------------------------------


def _finite_or_none(x: object) -> float | None:
    """Coerce to a finite float, else None (NULL / NaN / Inf / non-numeric all → None)."""
    if x is None:
        return None
    try:
        xf = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return xf if math.isfinite(xf) else None


def _tyre_age_int(x: object) -> int | None:
    """Integer tyre age, or None for UNKNOWN (consistent with F02 ``LAP_PARTIAL``)."""
    v = _finite_or_none(x)
    return int(round(v)) if v is not None else None


def _is_green(track_status: object, green_code: str) -> bool:
    """Green iff every token of the status string equals the green code (F02's rule,
    ``validate.py`` "green iff every token == this"). A null status is NOT green here — callers
    handle null separately (it defers to ``is_clean``)."""
    if track_status is None:
        return False
    s = str(track_status)
    return len(s) > 0 and all(ch == green_code for ch in s)


def _is_eligible(is_clean: object, track_status: object, green_code: str) -> bool:
    """A lap is eligible iff ``is_clean`` AND (status is null OR green). A known non-green token
    makes it ineligible regardless of ``is_clean`` (§3 / §4.2-3)."""
    if not bool(is_clean):
        return False
    if track_status is None:
        return True
    return _is_green(track_status, green_code)


def _cosine_distance(x: np.ndarray, b: np.ndarray) -> tuple[float, np.ndarray | None]:
    """``(1 − cos(x, b), per-feature ½(x̂ᵢ − b̂ᵢ)² terms)``. The exact additive decomposition
    (divergence #11) uses the unit-vector identity ``1 − cos = ½‖x̂ − b̂‖²`` — valid because x̂/b̂
    are unit-normalised here. A zero-norm / non-finite vector → ``(NaN, None)`` (the DISTANCE_NAN
    path); the terms sum to the distance to float tolerance (the attribution invariant)."""
    xn = float(np.linalg.norm(x))
    bn = float(np.linalg.norm(b))
    if not (math.isfinite(xn) and math.isfinite(bn)) or xn == 0.0 or bn == 0.0:
        return math.nan, None
    cos = float(np.dot(x, b) / (xn * bn))
    d = 1.0 - cos
    if not math.isfinite(d):
        return math.nan, None
    xhat = x / xn
    bhat = b / bn
    terms = 0.5 * (xhat - bhat) ** 2
    return d, terms.astype(float)


# ----------------------------------------------------------------------------------
# Slope component helpers (F03b reused, never reimplemented — §4.2-8 adapter)
# ----------------------------------------------------------------------------------


def _stint_laps(records: Sequence[Mapping[str, Any]]) -> list[StintLap]:
    """Frame → ``StintLap`` adapter (§4.2-8): RAW F01/F06 + integer-coerced tyre age."""
    return [
        StintLap(
            lap_number=int(r[COL_LAP]),
            tyre_age=_tyre_age_int(r.get(COL_TYRE_AGE)),
            throttle_mean=_finite_or_none(r.get(_RAW_F01)),
            brake_peak_mean=_finite_or_none(r.get(_RAW_F06)),
        )
        for r in records
    ]


def _slope_ready(prefix: Sequence[StintLap], min_stint_laps: int) -> bool:
    """True iff ``compute_degradation_slopes`` would return BOTH F27 and F28 non-NULL for this
    prefix — replicating F03b's usability rule (``features.py`` ``_stint_slope``) so F03b is
    NEVER invoked on input it would NULL (zero ``FEATURE_001`` spam; §4.1-6 / §4.2-8)."""
    for attr in ("throttle_mean", "brake_peak_mean"):
        ages: list[float] = []
        for lap in prefix:
            if lap.tyre_age is None:
                continue
            val = getattr(lap, attr)
            if val is None:
                continue
            ages.append(float(lap.tyre_age))
        if len(ages) < min_stint_laps:
            return False
        if float(np.ptp(np.asarray(ages))) == 0.0:  # tyre age constant → slope undefined
            return False
    return True


def _slope_z(
    result: DegradationResult, slope_lookup: dict[str, tuple[float, float]], min_std: float
) -> float | None:
    """``(|z(f27)| + |z(f28)|) / 2`` from the resolved per-circuit-type (fallback global) slope
    stats. Returns None if either slope is NULL (should not happen behind :func:`_slope_ready`)."""
    if result.f27 is None or result.f28 is None:
        return None
    zs: list[float] = []
    for fid, val in ((_RAW_F27_KEY, result.f27), (_RAW_F28_KEY, result.f28)):
        mean, std = slope_lookup[fid]
        zs.append(abs((val - mean) / max(std, min_std)))
    return float(sum(zs) / 2.0)


# ----------------------------------------------------------------------------------
# The pure scoring core (§4.2 steps 3–8) — shared by monitor_stint AND calibration Pass B/C
# ----------------------------------------------------------------------------------


@dataclass
class _LapCore:
    lap_number: int
    state: StintMonitorState
    deviation_raw: float | None = None
    deviation_rolling: float | None = None
    deviation_score: float | None = None
    slope_component: float = 0.0
    slope_component_active: bool = False
    flags: list[ErrorCode] = field(default_factory=list)
    cosine_attribution: np.ndarray | None = None  # rolling-mean per-feature terms (26,), scored


@dataclass
class _StintCore:
    outcome: str  # "ACTIVE" | "BASELINE_INSUFFICIENT" | "MONITOR_ERROR"
    laps: list[_LapCore] = field(default_factory=list)
    baseline_lap_numbers: list[int] = field(default_factory=list)
    baseline_extended: bool = False
    degraded_quality: bool = False
    n_valid_distances: int = 0
    n_composite: int = 0
    consecutive_nan: int = 0
    halted_lap: int | None = None
    dq_events: list[_DQEvent] = field(default_factory=list)


def _score_stint_core(
    records: Sequence[Mapping[str, Any]],
    *,
    circuit: str,
    circuit_type: str | None,
    cosine_features: tuple[str, ...],
    feature_mean: Mapping[str, float],
    feature_std: Mapping[str, float],
    slope_stats: Mapping[str, dict[str, tuple[float, float]]],
    norm_params: NormalisationParams,
    config: AppConfig,
    min_std: float,
    log: Path,
    entity_prefix: str,
) -> _StintCore:
    """Baseline → cosine → rolling → slope-component over one activated stint (already length- and
    input-gated by the caller). Pure: writes NO data_quality_log rows (returns ``_DQEvent``s the
    serving caller records); the ONLY sink it touches is F04/F03b's own read-only logging on the
    shared ``log`` path, and F03b is pre-gated so it never logs (§4.1-3 train/serve parity)."""
    assert config.monitoring is not None
    mon = config.monitoring
    green_code = _resolve_green_code(config)
    baseline_laps = mon.baseline_laps
    rolling_window = mon.rolling_window
    slope_weight = mon.slope_weight
    min_stint_laps = (
        config.features.min_stint_laps if config.features is not None else baseline_laps
    )

    core = _StintCore(outcome="ACTIVE")

    # --- Baseline selection (§4.2-3): earliest `baseline_laps` eligible in the first-3 window,
    #     extending to the first `baseline_window_extended` when short.
    def _eligible(rec: Mapping[str, Any]) -> bool:
        return _is_eligible(rec.get(COL_IS_CLEAN), rec.get(COL_TRACK_STATUS), green_code)

    first_window = records[:baseline_laps]
    elig_first = [r for r in first_window if _eligible(r)]
    if len(elig_first) >= baseline_laps:
        baseline_recs = elig_first[:baseline_laps]
        core.baseline_extended = False
    else:
        ext_window = records[: mon.baseline_window_extended]
        elig_ext = [r for r in ext_window if _eligible(r)]
        if len(elig_ext) < baseline_laps:
            core.outcome = "BASELINE_INSUFFICIENT"
            core.dq_events.append(
                _DQEvent(
                    code=ErrorCode.BASELINE_INSUFFICIENT,
                    entity_type=_ENTITY_STINT,
                    entity_id=entity_prefix,
                    detail=(
                        f"{REASON_BASELINE_INSUFF}: {len(elig_ext)} eligible laps in the first "
                        f"{mon.baseline_window_extended} < {baseline_laps} required"
                    ),
                    fallback_used=False,
                )
            )
            # Still emit one row per lap so no lap is silently dropped; every INELIGIBLE candidate
            # is DQ-logged with LAP_001 (§4.2-3 is unconditional — same as the success path).
            for rec in records:
                lap_no = int(rec[COL_LAP])
                if _eligible(rec):
                    core.laps.append(_LapCore(lap_number=lap_no, state=StintMonitorState.SKIPPED))
                else:
                    core.laps.append(
                        _LapCore(
                            lap_number=lap_no, state=StintMonitorState.SKIPPED,
                            flags=[ErrorCode.LAP_001],
                        )
                    )
                    core.dq_events.append(
                        _DQEvent(
                            code=ErrorCode.LAP_001, entity_type=_ENTITY_LAP,
                            entity_id=f"{entity_prefix}/lap{lap_no}",
                            detail=f"{REASON_BASELINE_SKIP}: ineligible baseline candidate",
                            fallback_used=False,
                        )
                    )
            return core
        baseline_recs = elig_ext[:baseline_laps]
        core.baseline_extended = True

    baseline_nums = {int(r[COL_LAP]) for r in baseline_recs}
    core.baseline_lap_numbers = sorted(baseline_nums)

    # --- Baseline quality (§4.2-4): all-interpolated → DEGRADED_QUALITY (confidence capped LOW).
    if baseline_recs and all(bool(r.get(COL_HAS_INTERP)) for r in baseline_recs):
        core.degraded_quality = True
        core.dq_events.append(
            _DQEvent(
                code=ErrorCode.DEGRADED_QUALITY,
                entity_type=_ENTITY_STINT,
                entity_id=entity_prefix,
                detail=f"{REASON_DEGRADED_QUALITY}: alert confidence capped at LOW",
                fallback_used=False,
            )
        )

    # --- Standardised baseline fingerprint b (per-feature mean of the baseline standardised
    #     vectors), computed once through the shared normalise+standardise choke point (§4.2-5/6).
    baseline_vectors = [
        _standardise_lap(rec, cosine_features, feature_mean, feature_std, norm_params, circuit,
                         config, log, min_std)[0]
        for rec in baseline_recs
    ]
    # Per-dim mean over the FINITE baseline values (NaN-safe, mirrors F08 ``aggregate_driver``): a
    # NULL/NaN feature in ONE baseline lap must not poison that dim of ``b`` for the whole stint.
    # A dim that is NaN across ALL baseline laps stays NaN → honest DISTANCE_NAN, never a silent 0.
    b_mat = np.vstack(baseline_vectors)
    b_finite = np.isfinite(b_mat)
    b_counts = b_finite.sum(axis=0)
    b_vec = np.where(
        b_counts > 0, np.where(b_finite, b_mat, 0.0).sum(axis=0) / np.maximum(b_counts, 1), np.nan
    )

    # --- Slope pre-gate (§4.2-8): decide ONCE whether this stint can contribute a slope at all,
    #     and whether the slope stats exist at any tier — either miss → cosine-only + one DQ row.
    all_stintlaps = _stint_laps(records)
    slope_lookup = slope_stats.get(circuit_type or "") or slope_stats.get(_GLOBAL_SLOPE_KEY)
    slope_possible = _slope_ready(all_stintlaps, min_stint_laps)
    # LAP_PARTIAL (F02's "tyre age UNKNOWN -> F27/F28 NULL") is the operative code — NOT F03b's
    # FEATURE_001: F03b is never invoked here, so a no-tyre-age stint yields ZERO FEATURE_001 rows
    # and exactly one F10 LAP_PARTIAL row (§7-11).
    if slope_lookup is None:
        core.dq_events.append(
            _DQEvent(
                code=ErrorCode.LAP_PARTIAL,
                entity_type=_ENTITY_STINT,
                entity_id=entity_prefix,
                detail=f"{REASON_SLOPE_STATS}: cosine-only scoring (no slope stats at any tier)",
                fallback_used=True,
            )
        )
    elif not slope_possible:
        core.dq_events.append(
            _DQEvent(
                code=ErrorCode.LAP_PARTIAL,
                entity_type=_ENTITY_STINT,
                entity_id=entity_prefix,
                detail=f"{REASON_SLOPE_STINT}: cosine-only scoring (stint cannot produce a slope)",
                fallback_used=True,
            )
        )
    slope_active_stint = slope_lookup is not None and slope_possible

    # --- Main lap loop: baseline / skipped / scored (rolling + slope + attribution). The alert
    #     state machine (WATCH/ALERT) runs in the caller once τ is resolved.
    valid_distances: list[float] = []
    valid_terms: list[np.ndarray] = []
    baseline_window = set(int(r[COL_LAP]) for r in records[: mon.baseline_window_extended])
    for idx, rec in enumerate(records):
        lap_no = int(rec[COL_LAP])
        if lap_no in baseline_nums:
            core.laps.append(_LapCore(lap_number=lap_no, state=StintMonitorState.BASELINE))
            continue
        if not _eligible(rec):
            core.laps.append(
                _LapCore(
                    lap_number=lap_no, state=StintMonitorState.SKIPPED, flags=[ErrorCode.LAP_001]
                )
            )
            reason = REASON_BASELINE_SKIP if lap_no in baseline_window else REASON_MID_SKIP
            core.dq_events.append(
                _DQEvent(
                    code=ErrorCode.LAP_001,
                    entity_type=_ENTITY_LAP,
                    entity_id=f"{entity_prefix}/lap{lap_no}",
                    detail=f"{reason}: ineligible (non-green / unclean)",
                    fallback_used=False,
                )
            )
            continue

        x_vec = _standardise_lap(
            rec, cosine_features, feature_mean, feature_std, norm_params, circuit, config, log,
            min_std,
        )[0]
        d_k, terms = _cosine_distance(x_vec, b_vec)
        if not math.isfinite(d_k) or terms is None:
            core.consecutive_nan += 1
            lap_core = _LapCore(
                lap_number=lap_no, state=StintMonitorState.PENDING, flags=[ErrorCode.DISTANCE_NAN]
            )
            core.dq_events.append(
                _DQEvent(
                    code=ErrorCode.DISTANCE_NAN,
                    entity_type=_ENTITY_LAP,
                    entity_id=f"{entity_prefix}/lap{lap_no}",
                    detail=f"{REASON_DISTANCE_NAN}: skipped in rolling mean",
                    fallback_used=False,
                )
            )
            if core.consecutive_nan >= config.monitoring.max_consecutive_nan:
                lap_core.state = StintMonitorState.HALTED
                core.laps.append(lap_core)
                core.halted_lap = lap_no
                core.outcome = "MONITOR_ERROR"
                core.dq_events.append(
                    _DQEvent(
                        code=ErrorCode.MONITOR_ERROR,
                        entity_type=_ENTITY_STINT,
                        entity_id=entity_prefix,
                        detail=(
                            f"{REASON_HALT}: {core.consecutive_nan} consecutive NaN distances; "
                            "monitoring halted for this stint"
                        ),
                        fallback_used=False,
                    )
                )
                return core
            core.laps.append(lap_core)
            continue

        # Valid distance: reset the NaN run, push into the rolling buffer.
        core.consecutive_nan = 0
        core.n_valid_distances += 1
        valid_distances.append(d_k)
        valid_terms.append(terms)
        lap_core = _LapCore(
            lap_number=lap_no, state=StintMonitorState.PENDING, deviation_raw=d_k
        )
        if len(valid_distances) >= rolling_window:
            window_d = valid_distances[-rolling_window:]
            window_terms = valid_terms[-rolling_window:]
            r_k = float(np.mean(window_d))
            cosine_attr = np.mean(np.vstack(window_terms), axis=0).astype(float)
            slope_component = 0.0
            slope_active = False
            if slope_active_stint:
                prefix = all_stintlaps[: idx + 1]
                if _slope_ready(prefix, min_stint_laps):
                    slope_res = compute_degradation_slopes(
                        prefix, stint_id=_SLOPE_STINT_SENTINEL, config=config, log_path=log
                    )
                    assert slope_lookup is not None
                    z = _slope_z(slope_res, slope_lookup, min_std)
                    if z is not None:
                        slope_component = slope_weight * z
                        slope_active = True
            lap_core.deviation_rolling = r_k
            lap_core.slope_component = slope_component
            lap_core.slope_component_active = slope_active
            lap_core.deviation_score = r_k + slope_component
            lap_core.cosine_attribution = cosine_attr
            lap_core.state = StintMonitorState.NOMINAL  # threshold layer refines to WATCH/ALERT
            core.n_composite += 1
        core.laps.append(lap_core)

    return core


def _resolve_green_code(config: AppConfig) -> str:
    if config.lap_validation is None:
        raise ValueError("config.validate block absent; cannot resolve green_track_status_code")
    return config.lap_validation.green_track_status_code


def _standardise_lap(
    rec: Mapping[str, Any],
    cosine_features: tuple[str, ...],
    feature_mean: Mapping[str, float],
    feature_std: Mapping[str, float],
    norm_params: NormalisationParams,
    circuit: str,
    config: AppConfig,
    log: Path,
    min_std: float,
) -> tuple[np.ndarray, bool]:
    """The single normalise+standardise choke point (§4.2-5, shared by baseline + scoring + Pass B).
    F04 ``apply_normalisation`` first (read-only, never re-fits), then the train-only per-feature
    standardiser over the 26 cosine dims (``std`` floored at the config-resolved ``min_std`` — the
    SAME floor the fit used, one choke point). Returns ``(26-vector, all-finite?)`` — a NULL/NaN dim
    becomes ``NaN`` so the cosine distance is NaN (DISTANCE_NAN), never a silent zero."""
    vector: dict[str, float | None] = {fid: _finite_or_none(rec.get(fid)) for fid in FEATURE_IDS}
    res = apply_normalisation(
        vector, circuit=circuit, params=norm_params, config=config, log_path=log
    )
    out = np.empty(len(cosine_features), dtype=float)
    all_finite = True
    for i, fid in enumerate(cosine_features):
        nv = _finite_or_none(res.normalised_vector.get(fid))
        if nv is None:
            out[i] = math.nan
            all_finite = False
        else:
            out[i] = (nv - feature_mean[fid]) / max(feature_std[fid], min_std)
    return out, all_finite


def _zero_composite_code(core: _StintCore) -> ErrorCode:
    """The code for the single stint-level "nothing scored" row (§6 table: "accumulated per-lap
    flags"). Mirror the dominant per-lap flag — DISTANCE_NAN (NaN-starved) or LAP_001 (all
    safety-car) — falling back to MONITOR_INACTIVE ("monitoring produced no score")."""
    seen = {code for lap in core.laps for code in lap.flags}
    if ErrorCode.DISTANCE_NAN in seen:
        return ErrorCode.DISTANCE_NAN
    if ErrorCode.LAP_001 in seen:
        return ErrorCode.LAP_001
    return ErrorCode.MONITOR_INACTIVE


# ----------------------------------------------------------------------------------
# Alert state machine (§4.2-11) — runs on the core's scored laps
# ----------------------------------------------------------------------------------


@dataclass
class _AlertOutcome:
    alert_lap: int | None = None
    first_crossing_lap: int | None = None


def _run_alert_machine(
    laps: list[_LapCore], *, tau: float, confirm_laps: int
) -> _AlertOutcome:
    """Strict-``>`` crossing → WATCH on the first of a run; ``confirm_laps`` consecutive scored
    crossings → ALERT. A sub-threshold scored lap resets the run; a SKIPPED lap FREEZES it
    (neither extends nor resets). Mutates each scored lap's ``state`` to WATCH/ALERT (§4.2-11).
    A single unconfirmed crossing leaves the lap in the WATCH state — the persisted, auditable
    "logged but not alerted" record (§5.3); no separate error-coded DQ row (divergence #12)."""
    out = _AlertOutcome()
    run = 0
    run_start: int | None = None
    for lap in laps:
        if lap.state is StintMonitorState.SKIPPED:
            continue  # freeze — the confirmation counter neither extends nor resets across skips
        if lap.deviation_score is None:
            continue  # BASELINE / PENDING / NaN — not part of the run (pre-window only)
        if lap.deviation_score > tau:
            run += 1
            if run == 1:
                run_start = lap.lap_number
            if run >= confirm_laps:
                lap.state = StintMonitorState.ALERT
                if run == confirm_laps:  # fire ONCE, at the confirming lap (not every lap after)
                    out.alert_lap = lap.lap_number
                    out.first_crossing_lap = run_start
            else:
                lap.state = StintMonitorState.WATCH
        else:
            run = 0
            run_start = None
            lap.state = StintMonitorState.NOMINAL
    return out


# ----------------------------------------------------------------------------------
# Threshold resolution (§4.2-9) — fallback tiers with the min_calibration_stints gate
# ----------------------------------------------------------------------------------


def _resolve_cell(
    handle: MonitorHandle, driver: str, circuit_type: str | None, min_calibration_stints: int
) -> tuple[_Cell, str, bool]:
    """``(cell, threshold_source, is_fallback)`` via ``(driver, circuit_type)`` [gated on
    ``min_calibration_stints``] → ``(driver, global)`` → ``global``. Every fallback hop is flagged
    FALLBACK_THRESHOLD by the caller."""
    if circuit_type is not None:
        cell = handle.entries.get((driver, circuit_type))
        if cell is not None and cell.n_stints >= min_calibration_stints:
            return cell, _SRC_CELL, False
    driver_cell = handle.driver_global.get(driver)
    if driver_cell is not None:
        return driver_cell, _SRC_DRIVER, True
    return handle.global_cell, _SRC_GLOBAL, True


# ----------------------------------------------------------------------------------
# Serving — monitor_stint (read-only, never raises)
# ----------------------------------------------------------------------------------


def monitor_stint(
    stint: pd.DataFrame,
    *,
    driver: str,
    circuit: str,
    circuit_type: str | None,
    handle: MonitorHandle,
    config: AppConfig,
    style_handle: StyleSpaceHandle | None = None,
    log_path: str | Path | None = None,
) -> StintMonitorResult:
    """Monitor one stint prefix for tyre degradation. **Never raises** — every fault (malformed
    input, too-short stint, unusable baseline, telemetry NaNs, unforeseen internal error) is a
    typed :class:`StintMonitorResult` + a ``data_quality_log`` row. Pure function of the prefix
    it is given (stateless; F14 replays by re-calling with a growing prefix, §4.0-d)."""
    log = Path(log_path) if log_path is not None else Path(config.paths.data_quality_log)
    ledger = _DQLedger(log_path=log)
    # Build the result shell from the string kwargs ONLY (never touch `handle` here) so a None /
    # malformed handle cannot escape the never-raises contract before the try (Finding 2, F08
    # precedent: all handle-dependent access lives inside the guarded body).
    base = StintMonitorResult(driver=driver, circuit=circuit, circuit_type=circuit_type)
    try:
        if config.monitoring is None:
            raise ValueError("config.monitoring block absent")
        if handle is None or not getattr(handle, "ready", False):
            raise ValueError("detector handle is None or not ready")
        base.pit_window_valid = handle.pit_window_valid
        mon = config.monitoring
        min_stint_laps = (
            config.features.min_stint_laps if config.features is not None else mon.baseline_laps
        )
        entity_prefix = f"{driver}/{circuit}"

        # 1. Input gate — required columns, non-empty, strictly increasing lap_number (§4.2-1).
        gate = _input_gate(stint, _SERVING_REQUIRED)
        if gate is not None:
            ledger.record(
                code=ErrorCode.INVALID_INPUT, entity_type=_ENTITY_STINT, entity_id=entity_prefix,
                detail=f"{REASON_INVALID}: {gate}", fallback_used=False,
            )
            return _finalise(base, ledger, status=Status.FAILED, error_code=ErrorCode.INVALID_INPUT,
                             monitor_state=StintMonitorState.INACTIVE)
        records = cast("list[dict[str, Any]]", stint.to_dict("records"))

        # 2. Fingerprint snapshot (attempted regardless of activation — §4.2-2/10).
        snapshot = _fingerprint_snapshot(stint, circuit=circuit, style_handle=style_handle,
                                         config=config, log=log)
        base.fingerprint_snapshot = snapshot
        base.fingerprint_available = snapshot is not None and snapshot.coords_5d is not None

        # 3. Activation gate (§4.2-2): too-short stint → MONITOR_INACTIVE (fingerprint still shown).
        if len(records) < min_stint_laps:
            ledger.record(
                code=ErrorCode.MONITOR_INACTIVE, entity_type=_ENTITY_STINT, entity_id=entity_prefix,
                detail=f"{REASON_INACTIVE}: {len(records)} laps < {min_stint_laps}",
                fallback_used=False,
            )
            return _finalise(base, ledger, status=Status.DEGRADED,
                             error_code=ErrorCode.MONITOR_INACTIVE,
                             monitor_state=StintMonitorState.INACTIVE)

        # 4. Score the stint (baseline → cosine → rolling → slope) via the shared core.
        core = _score_stint_core(
            records, circuit=circuit, circuit_type=circuit_type,
            cosine_features=handle.cosine_features, feature_mean=handle.feature_mean,
            feature_std=handle.feature_std, slope_stats=handle.slope_stats,
            norm_params=handle.normalisation_params, config=config, min_std=handle.min_std,
            log=log, entity_prefix=entity_prefix,
        )
        for ev in core.dq_events:
            ledger.record(code=ev.code, entity_type=ev.entity_type, entity_id=ev.entity_id,
                          detail=ev.detail, fallback_used=ev.fallback_used)
        base.baseline_lap_numbers = core.baseline_lap_numbers
        base.baseline_extended = core.baseline_extended
        base.degraded_quality = core.degraded_quality
        base.n_laps_scored = core.n_valid_distances
        base.consecutive_nan = core.consecutive_nan

        flags: list[ErrorCode] = []
        # Unknown circuit (single circuit per stint) → F04 global params → NORMALISATION_FALLBACK.
        if circuit not in handle.normalisation_params.per_circuit:
            flags.append(ErrorCode.NORMALISATION_FALLBACK)
        if core.degraded_quality:
            flags.append(ErrorCode.DEGRADED_QUALITY)

        # 4a. Unusable baseline → BASELINE_INSUFFICIENT (FAILED) — no threshold needed.
        if core.outcome == "BASELINE_INSUFFICIENT":
            base.laps = _to_lap_deviations(core.laps, tau=None)
            return _finalise(base, ledger, status=Status.FAILED,
                             error_code=ErrorCode.BASELINE_INSUFFICIENT,
                             monitor_state=StintMonitorState.SKIPPED, extra_flags=flags)

        # 5. Threshold cell (§4.2-9) — resolve after baseline succeeds so τ is on every lap row.
        cell, source, is_fallback = _resolve_cell(
            handle, driver, circuit_type, mon.min_calibration_stints
        )
        base.threshold_source = source
        tau = cell.mean + mon.alert_threshold_sd * cell.std
        if is_fallback:
            flags.append(ErrorCode.FALLBACK_THRESHOLD)
            ledger.record(
                code=ErrorCode.FALLBACK_THRESHOLD, entity_type=_ENTITY_STINT,
                entity_id=entity_prefix,
                detail=f"{REASON_FALLBACK}: threshold_source={source}", fallback_used=True,
            )

        # 6. Alert state machine (only over confirmed scored laps). A single unconfirmed crossing
        #    leaves the lap in the WATCH state — the persisted, auditable "logged but not alerted"
        #    record (§5.3); no separate error-coded DQ row is minted for it (no code applies, and
        #    F10 adds none — divergence #12).
        alert_out = _run_alert_machine(core.laps, tau=tau, confirm_laps=mon.alert_confirm_laps)
        base.laps = _to_lap_deviations(core.laps, tau=tau)
        for lap in core.laps:  # per-lap flags bubble up to the stint-level flags list
            flags.extend(lap.flags)

        # 6a. Halt (max_consecutive_nan) → MONITOR_ERROR (honest partial series).
        if core.outcome == "MONITOR_ERROR":
            return _finalise(base, ledger, status=Status.FAILED, error_code=ErrorCode.MONITOR_ERROR,
                             monitor_state=StintMonitorState.HALTED, extra_flags=flags)

        # 7. Build the alert object (if one fired).
        if alert_out.alert_lap is not None:
            base.alert = _build_alert(
                core, alert_out, cell=cell, tau=tau, mon=mon,
                cosine_features=handle.cosine_features,
                degraded_quality=core.degraded_quality, pit_window_valid=handle.pit_window_valid,
            )

        # 8. Zero-composite outcome (§4.2-7): activated but nothing scored → DEGRADED + PENDING +
        #    one stint-level row whose code mirrors the accumulated per-lap flags (§6 table).
        if core.n_composite == 0:
            zero_code = _zero_composite_code(core)
            ledger.record(
                code=zero_code, entity_type=_ENTITY_STINT, entity_id=entity_prefix,
                detail=f"{REASON_ZERO_COMPOSITE}: activated stint produced no composite score",
                fallback_used=False,
            )
            return _finalise(base, ledger, status=Status.DEGRADED, error_code=None,
                             monitor_state=StintMonitorState.PENDING, extra_flags=flags)

        last_state = base.laps[-1].state if base.laps else StintMonitorState.NOMINAL
        degraded = bool(flags)
        return _finalise(
            base, ledger, status=Status.DEGRADED if degraded else Status.OK, error_code=None,
            monitor_state=last_state, extra_flags=flags,
        )
    except Exception as exc:  # noqa: BLE001 — the never-raises backstop (§4.0-c)
        try:
            ledger.record(
                code=ErrorCode.MONITOR_ERROR, entity_type=_ENTITY_STINT,
                entity_id=f"{driver}/{circuit}",
                detail=f"{REASON_BACKSTOP}: {type(exc).__name__}: {exc}", fallback_used=False,
            )
        except Exception:  # noqa: BLE001 — even the logger must not break never-raises
            pass
        return _finalise(base, ledger, status=Status.FAILED, error_code=ErrorCode.MONITOR_ERROR,
                         monitor_state=StintMonitorState.HALTED)


def _input_gate(stint: pd.DataFrame, required: tuple[str, ...]) -> str | None:
    """Return a failure detail string, or None when the frame passes the input gate (§4.2-1)."""
    if not isinstance(stint, pd.DataFrame) or stint.empty:
        return "empty or non-DataFrame stint"
    missing = sorted(set(required) - set(stint.columns))
    if missing:
        return f"missing required columns: {missing}"
    laps = stint[COL_LAP].tolist()
    ints: list[int] = []
    for v in laps:
        iv = _finite_or_none(v)
        if iv is None:
            return "non-numeric lap_number"
        ints.append(int(iv))
    if any(b <= a for a, b in zip(ints, ints[1:], strict=False)):
        return "lap_number not strictly increasing (or duplicate laps)"
    return None


def _fingerprint_snapshot(
    stint: pd.DataFrame, *, circuit: str, style_handle: StyleSpaceHandle | None, config: AppConfig,
    log: Path,
) -> ProjectionResult | None:
    """Project the RAW stint prefix through F08 (never blocks scoring — F07 precedent). Any
    exception (should be impossible — F08 never raises) → None (defence in depth, §4.2-10)."""
    if style_handle is None:
        return None
    try:
        raw = stint.reindex(columns=list(FEATURE_IDS))
        return project_unknown(raw, circuit=circuit, handle=style_handle, config=config,
                               log_path=log)
    except Exception:  # noqa: BLE001 — a fingerprint fault never blocks a degradation score
        return None


def _to_lap_deviations(laps: list[_LapCore], *, tau: float | None) -> list[LapDeviation]:
    return [
        LapDeviation(
            lap_number=lc.lap_number, state=lc.state, deviation_raw=lc.deviation_raw,
            deviation_rolling=lc.deviation_rolling, deviation_score=lc.deviation_score,
            slope_component=lc.slope_component, slope_component_active=lc.slope_component_active,
            threshold=tau, flags=_dedup(lc.flags),
        )
        for lc in laps
    ]


def _build_alert(
    core: _StintCore, alert_out: _AlertOutcome, *, cell: _Cell, tau: float, mon: MonitoringConfig,
    cosine_features: tuple[str, ...], degraded_quality: bool, pit_window_valid: bool,
) -> DegradationAlert:
    """Assemble the §5.3 alert payload from the confirming lap (exact 28-key attribution summing
    to ``deviation_score``; confidence bands from the resolved cell's μ/σ, capped LOW under
    DEGRADED_QUALITY)."""
    alert_lap = cast("int", alert_out.alert_lap)
    first_crossing = cast("int", alert_out.first_crossing_lap)
    lap_core = next(lc for lc in core.laps if lc.lap_number == alert_lap)
    score = cast("float", lap_core.deviation_score)

    attribution: dict[str, float] = {}
    if lap_core.cosine_attribution is not None:
        attribution = {
            fid: float(v)
            for fid, v in zip(cosine_features, lap_core.cosine_attribution, strict=True)
        }
    half = lap_core.slope_component / 2.0
    attribution[_RAW_F27_KEY] = half
    attribution[_RAW_F28_KEY] = half

    if degraded_quality:
        confidence = AlertConfidence.LOW
    elif score >= cell.mean + mon.confidence_high_sd * cell.std:
        confidence = AlertConfidence.HIGH
    elif score >= cell.mean + mon.confidence_medium_sd * cell.std:
        confidence = AlertConfidence.MEDIUM
    else:
        confidence = AlertConfidence.LOW

    laps_since = alert_lap - first_crossing
    optimal_window: int | None = None
    if pit_window_valid and cell.median is not None:
        optimal_window = max(0, int(round(cell.median)) - laps_since)

    top = sorted(attribution, key=lambda kk: attribution[kk], reverse=True)[: mon.attribution_top_k]
    return DegradationAlert(
        alert_lap=alert_lap, first_crossing_lap=first_crossing,
        laps_since_threshold_crossed=laps_since, deviation_score=score, threshold=tau,
        alert_confidence=confidence, feature_attribution=attribution, top_features=top,
        optimal_window_remaining_laps=optimal_window,
    )


def _finalise(
    result: StintMonitorResult, ledger: _DQLedger, *, status: Status,
    error_code: ErrorCode | None, monitor_state: StintMonitorState,
    extra_flags: list[ErrorCode] | None = None,
) -> StintMonitorResult:
    """Stamp status/state/flags + the DQ-sink-unwritable signal, then return (single exit point)."""
    result.status = status
    result.error_code = error_code
    result.monitor_state = monitor_state
    flags = list(extra_flags or [])
    if ledger.failed:
        flags.append(ErrorCode.DATA_QUALITY_LOG_UNWRITABLE)
    result.flags = _dedup(flags)
    result.logged = ledger.rows > 0
    return result


# ----------------------------------------------------------------------------------
# Calibration — calibrate_stint_variation (the ONLY fit site) — fail-loud
# ----------------------------------------------------------------------------------


def _assert_no_leakage(train: pd.DataFrame, config: AppConfig) -> None:
    """Fail loud if any held-out circuit survives into the training set (non-negotiable #2).
    Called AFTER the exclusion filter as a tripwire; raises ``DetectorCalibrationError`` — no
    artefact is written on a poisoned run (copied per module, mirroring F04/F06a/F08)."""
    held_out = set(config.held_out_circuits)
    present = set(train[COL_CIRCUIT].unique()) if len(train) else set()
    leaked = sorted(held_out & present)
    if leaked:
        raise DetectorCalibrationError(
            f"held-out circuits present in training input after filter: {leaked} "
            "(would leak into the fit — aborting; no artefact written)",
            error_code=ErrorCode.TRAIN_LEAKAGE,
        )


def calibrate_stint_variation(
    matrix: pd.DataFrame,
    *,
    config: AppConfig,
    artefact_dir: str | Path | None = None,
    log_path: str | Path | None = None,
) -> StintVariationFitResult:
    """Fit the intra-stint variation stats + feature standardiser + slope stats + pit-window
    medians on the TRAINING split ONLY and write ``stint_variation.json``. Raises
    :class:`DetectorCalibrationError` on missing config/columns, the leakage tripwire
    (``TRAIN_LEAKAGE``), or no usable stints (``MODEL_TRAINING_FAILED``) — no artefact on those
    paths. Deterministic (no RNG, no wall clock) → byte-identical artefact on re-run."""
    if config.monitoring is None or config.normalise is None or config.features is None:
        raise DetectorCalibrationError(
            "config.monitoring / normalise / features block absent; cannot calibrate",
            error_code=ErrorCode.MODEL_TRAINING_FAILED,
        )
    mon = config.monitoring
    art_dir = Path(artefact_dir) if artefact_dir is not None else Path(config.paths.artefacts_dir)
    art_dir.mkdir(parents=True, exist_ok=True)
    log = Path(log_path) if log_path is not None else Path(config.paths.data_quality_log)
    ledger = _DQLedger(log_path=log)
    min_std = _resolve_min_std(config)
    min_stint_laps = config.features.min_stint_laps
    cosine_features = _cosine_feature_ids(config)

    def _fail(detail: str, code: ErrorCode) -> DetectorCalibrationError:
        ledger.record(code=code, entity_type=_ENTITY_BUILD, entity_id=mon.mlflow_experiment,
                      detail=detail, fallback_used=False)
        return DetectorCalibrationError(detail, error_code=code)

    # 1. Columns.
    missing = sorted(set(_REQUIRED_COLUMNS) - set(matrix.columns))
    if missing:
        raise _fail(f"calibration matrix missing required columns: {missing}",
                    ErrorCode.MODEL_TRAINING_FAILED)

    # 2. Quarantine held-out + non-train BEFORE any reduction, then the fail-loud tripwire.
    train_mask = (matrix[COL_SPLIT] == "train") & (
        ~matrix[COL_CIRCUIT].isin(config.held_out_circuits)
    )
    train = cast("pd.DataFrame", matrix.loc[train_mask].copy())
    try:
        _assert_no_leakage(train, config)
    except DetectorCalibrationError as exc:
        ledger.record(code=exc.error_code, entity_type=_ENTITY_BUILD,
                      entity_id=mon.mlflow_experiment, detail=str(exc), fallback_used=False)
        raise
    if len(train) == 0:
        raise _fail("no training rows after excluding held-out / non-train split",
                    ErrorCode.MODEL_TRAINING_FAILED)

    # 3. Load F04 params (read-only; the pipeline normalises RAW rows exactly once, §4.1-3).
    try:
        norm_params = load_normalisation_params(art_dir / config.normalise.artefact_filename)
    except NormalisationError as exc:
        raise _fail(f"normalisation artefact unavailable: {exc}",
                    ErrorCode.MODEL_TRAINING_FAILED) from exc

    # 4. Fit the train-only feature standardiser (the ONLY .fit in F10; §4.0-e). A throwaway F04
    #    transform of the quarantined rows, per-feature mean/std over the 26 cosine dims.
    feature_mean, feature_std = _fit_feature_scaler(
        train, cosine_features, norm_params, config, log, min_std
    )

    # 5. Group into stints; keep those with >= min_stint_laps rows.
    kept = [
        g.sort_values(COL_LAP)
        for _, g in train.groupby(list(_STINT_GROUP_KEYS), dropna=False)
        if len(g) >= min_stint_laps
    ]
    if not kept:
        raise _fail(f"{REASON_NO_STINTS}: no stint has >= {min_stint_laps} training laps",
                    ErrorCode.MODEL_TRAINING_FAILED)

    # 6. Pass A — pooled live-window slope stats per circuit_type (+ global).
    slope_stats = _calibrate_slope_stats(kept, config=config, min_stint_laps=min_stint_laps,
                                         min_std=min_std, log=log)

    # 7. Pass B — deviation stats (μ/σ/n_stints) per (driver, circuit_type) / driver / global.
    dev_pool = _DeviationPool()
    for stint_df in kept:
        records = cast("list[dict[str, Any]]", stint_df.to_dict("records"))
        driver = str(records[0][COL_DRIVER])
        ct = _circuit_type_of(records[0])
        circ = str(records[0][COL_CIRCUIT])
        core = _score_stint_core(
            records, circuit=circ, circuit_type=ct, cosine_features=cosine_features,
            feature_mean=feature_mean, feature_std=feature_std, slope_stats=slope_stats,
            norm_params=norm_params, config=config, min_std=min_std, log=log,
            entity_prefix=f"{driver}/{circ}",
        )
        scores = [lc.deviation_score for lc in core.laps if lc.deviation_score is not None]
        if scores:
            dev_pool.add(driver, ct, scores)

    # 8. Pass C — pit-window medians (score each stint at exactly the τ §4.2-9 would resolve).
    pit_pool = _PitWindowPool()
    prelim = _build_handle_from_pool(
        dev_pool, slope_stats, cosine_features, feature_mean, feature_std, norm_params,
        mon, min_std, pit_window_valid=True,
    )
    for stint_df in kept:
        records = cast("list[dict[str, Any]]", stint_df.to_dict("records"))
        driver = str(records[0][COL_DRIVER])
        ct = _circuit_type_of(records[0])
        circ = str(records[0][COL_CIRCUIT])
        cell, _src, _fb = _resolve_cell(prelim, driver, ct, mon.min_calibration_stints)
        tau = cell.mean + mon.alert_threshold_sd * cell.std
        core = _score_stint_core(
            records, circuit=circ, circuit_type=ct, cosine_features=cosine_features,
            feature_mean=feature_mean, feature_std=feature_std, slope_stats=slope_stats,
            norm_params=norm_params, config=config, min_std=min_std, log=log,
            entity_prefix=f"{driver}/{circ}",
        )
        alert_out = _run_alert_machine(core.laps, tau=tau, confirm_laps=mon.alert_confirm_laps)
        if alert_out.first_crossing_lap is not None:
            last_lap = int(records[-1][COL_LAP])
            gap = float(last_lap - alert_out.first_crossing_lap)
            pit_pool.add(driver, ct, gap)

    # 9. Build + write the artefact deterministically.
    artefact = _assemble_artefact(
        dev_pool, pit_pool, slope_stats, cosine_features, feature_mean, feature_std, mon,
        min_stint_laps, min_std,
    )
    artefact.content_hash = _hash_artefact(artefact)
    path = art_dir / mon.variation_artefact_filename
    text = json.dumps(artefact.model_dump(), sort_keys=True, indent=2, allow_nan=False) + "\n"
    # A filesystem write fault (permission denied / disk full / read-only dir) is a typed build
    # failure with no artefact, never a raw OSError escaping the DetectorCalibrationError contract
    # (§6 "write fault" row). mkdir(exist_ok) is idempotent — a stale/partial file is overwritten.
    try:
        art_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise _fail(f"artefact write fault: {exc}", ErrorCode.MODEL_TRAINING_FAILED) from exc

    n_entries = len(artefact.entries)
    n_usable = sum(1 for e in artefact.entries if e.n_stints >= mon.min_calibration_stints)
    coverage = (n_usable / n_entries) if n_entries else 0.0

    # 10. MLflow (own experiment) — a fault degrades, never aborts the write (F08 mechanism).
    mlflow_run_id, mlflow_degraded = _log_mlflow(
        config, art_dir=art_dir, path=path, ledger=ledger, n_stints=len(kept), n_entries=n_entries,
        coverage=coverage,
    )
    # Week-11 deliverable: lead-time report over 2022 val stints logged to the run (§4.4).
    _log_week11_report(matrix, prelim, config, art_dir=art_dir, mlflow_run_id=mlflow_run_id,
                       log=log)

    flags = [ErrorCode.DATA_QUALITY_LOG_UNWRITABLE] if ledger.failed else []
    return StintVariationFitResult(
        status=Status.DEGRADED if mlflow_degraded else Status.OK,
        error_code=None, flags=flags, logged=ledger.rows > 0,
        artefact_path=str(path), content_hash=artefact.content_hash, n_stints_used=len(kept),
        n_entries=n_entries, n_driver_global=len(artefact.driver_global), coverage=coverage,
        mlflow_run_id=mlflow_run_id,
    )


def _circuit_type_of(rec: Mapping[str, Any]) -> str | None:
    ct = rec.get(COL_CIRCUIT_TYPE)
    if ct is None or (isinstance(ct, float) and math.isnan(ct)):
        return None
    return str(ct)


def _fit_feature_scaler(
    train: pd.DataFrame, cosine_features: tuple[str, ...], norm_params: NormalisationParams,
    config: AppConfig, log: Path, min_std: float,
) -> tuple[dict[str, float], dict[str, float]]:
    """The ONLY ``.fit`` in F10 (§4.0-e): F04-normalise the quarantined training rows once, then
    per-feature mean/std over the 26 cosine dims (std floored at ``normalise.min_std``)."""
    records = cast("list[dict[str, Any]]", train.to_dict("records"))
    cols: dict[str, list[float]] = {fid: [] for fid in cosine_features}
    for rec in records:
        circ = str(rec[COL_CIRCUIT])
        vector: dict[str, float | None] = {
            fid: _finite_or_none(rec.get(fid)) for fid in FEATURE_IDS
        }
        res = apply_normalisation(vector, circuit=circ, params=norm_params, config=config,
                                  log_path=log)
        for fid in cosine_features:
            nv = _finite_or_none(res.normalised_vector.get(fid))
            if nv is not None:
                cols[fid].append(nv)
    feature_mean: dict[str, float] = {}
    feature_std: dict[str, float] = {}
    for fid in cosine_features:
        arr = np.asarray(cols[fid], dtype=float)
        feature_mean[fid] = float(np.mean(arr)) if arr.size else 0.0
        std = float(np.std(arr, ddof=_AGG_DDOF)) if arr.size else min_std
        feature_std[fid] = max(std, min_std)
    return feature_mean, feature_std


def _calibrate_slope_stats(
    kept: list[pd.DataFrame], *, config: AppConfig, min_stint_laps: int, min_std: float, log: Path,
) -> dict[str, dict[str, tuple[float, float]]]:
    """Pass A: pool per-prefix live-window slopes per circuit_type (+ global). F03b is invoked
    ONLY on prefixes it would not NULL (pre-gated) — zero ``FEATURE_001`` spam (§4.1-6)."""
    pooled: dict[str, dict[str, list[float]]] = {}

    def _add(ct: str, f27: float, f28: float) -> None:
        d = pooled.setdefault(ct, {_RAW_F27_KEY: [], _RAW_F28_KEY: []})
        d[_RAW_F27_KEY].append(f27)
        d[_RAW_F28_KEY].append(f28)

    for stint_df in kept:
        records = cast("list[dict[str, Any]]", stint_df.to_dict("records"))
        ct = _circuit_type_of(records[0]) or ""
        stintlaps = _stint_laps(records)
        for k in range(len(stintlaps)):
            prefix = stintlaps[: k + 1]
            if not _slope_ready(prefix, min_stint_laps):
                continue
            res = compute_degradation_slopes(prefix, stint_id=_SLOPE_STINT_SENTINEL, config=config,
                                             log_path=log)
            if res.f27 is None or res.f28 is None:
                continue
            _add(ct, res.f27, res.f28)
            _add(_GLOBAL_SLOPE_KEY, res.f27, res.f28)

    stats: dict[str, dict[str, tuple[float, float]]] = {}
    for ct, feats in pooled.items():
        stats[ct] = {
            fid: (
                float(np.mean(feats[fid])),
                max(float(np.std(feats[fid], ddof=_AGG_DDOF)), min_std),
            )
            for fid in (_RAW_F27_KEY, _RAW_F28_KEY)
        }
    return stats


@dataclass
class _DeviationPool:
    """Pools composite scores into (driver, circuit_type) / driver / global tiers with per-tier
    contributing-stint counts."""

    cell_scores: dict[tuple[str, str], list[float]] = field(default_factory=dict)
    cell_stints: dict[tuple[str, str], int] = field(default_factory=dict)
    driver_scores: dict[str, list[float]] = field(default_factory=dict)
    driver_stints: dict[str, int] = field(default_factory=dict)
    global_scores: list[float] = field(default_factory=list)
    global_stints: int = 0

    def add(self, driver: str, circuit_type: str | None, scores: list[float]) -> None:
        if circuit_type is not None:
            key = (driver, circuit_type)
            self.cell_scores.setdefault(key, []).extend(scores)
            self.cell_stints[key] = self.cell_stints.get(key, 0) + 1
        self.driver_scores.setdefault(driver, []).extend(scores)
        self.driver_stints[driver] = self.driver_stints.get(driver, 0) + 1
        self.global_scores.extend(scores)
        self.global_stints += 1


@dataclass
class _PitWindowPool:
    cell_gaps: dict[tuple[str, str], list[float]] = field(default_factory=dict)
    driver_gaps: dict[str, list[float]] = field(default_factory=dict)
    global_gaps: list[float] = field(default_factory=list)

    def add(self, driver: str, circuit_type: str | None, gap: float) -> None:
        if circuit_type is not None:
            self.cell_gaps.setdefault((driver, circuit_type), []).append(gap)
        self.driver_gaps.setdefault(driver, []).append(gap)
        self.global_gaps.append(gap)


def _stats(scores: list[float], min_std: float) -> tuple[float, float]:
    arr = np.asarray(scores, dtype=float)
    return float(np.mean(arr)), max(float(np.std(arr, ddof=_AGG_DDOF)), min_std)


def _median_or_none(gaps: list[float] | None) -> float | None:
    return float(np.median(np.asarray(gaps, dtype=float))) if gaps else None


def _build_handle_from_pool(
    dev: _DeviationPool, slope_stats: dict[str, dict[str, tuple[float, float]]],
    cosine_features: tuple[str, ...], feature_mean: dict[str, float], feature_std: dict[str, float],
    norm_params: NormalisationParams, mon: MonitoringConfig, min_std: float, *,
    pit_window_valid: bool,
) -> MonitorHandle:
    """A transient in-memory handle over the Pass-B tables so Pass C resolves τ through the exact
    same :func:`_resolve_cell` path serving uses (no pit-window medians yet)."""
    entries = {
        key: _Cell(*_stats(scores, min_std), n_stints=dev.cell_stints[key], median=None)
        for key, scores in dev.cell_scores.items()
    }
    driver_global = {
        d: _Cell(*_stats(scores, min_std), n_stints=dev.driver_stints[d], median=None)
        for d, scores in dev.driver_scores.items()
    }
    global_cell = (
        _Cell(*_stats(dev.global_scores, min_std), n_stints=dev.global_stints, median=None)
        if dev.global_scores
        else _Cell(0.0, min_std, 0, None)
    )
    return MonitorHandle(
        entries=entries, driver_global=driver_global, global_cell=global_cell,
        slope_stats=slope_stats, cosine_features=cosine_features, feature_mean=feature_mean,
        feature_std=feature_std,
        calibration_config=_calibration_config_of(mon), pit_window_valid=pit_window_valid,
        normalisation_params=norm_params, model_version="", min_std=min_std,
    )


def _calibration_config_of(mon: MonitoringConfig) -> _CalibrationConfig:
    return _CalibrationConfig(
        alert_threshold_sd=mon.alert_threshold_sd, alert_confirm_laps=mon.alert_confirm_laps,
        rolling_window=mon.rolling_window, baseline_laps=mon.baseline_laps,
        baseline_window_extended=mon.baseline_window_extended, slope_weight=mon.slope_weight,
        min_stint_laps=0,  # overwritten by the caller (features.min_stint_laps)
    )


def _assemble_artefact(
    dev: _DeviationPool, pit: _PitWindowPool,
    slope_stats: dict[str, dict[str, tuple[float, float]]], cosine_features: tuple[str, ...],
    feature_mean: dict[str, float], feature_std: dict[str, float], mon: MonitoringConfig,
    min_stint_laps: int, min_std: float,
) -> _StintVariationArtefact:
    entries = []
    for (driver, ct), scores in sorted(dev.cell_scores.items()):
        mean, std = _stats(scores, min_std)
        entries.append(
            _CellEntry(
                driver=driver, circuit_type=ct, mean=mean, std=std,
                n_stints=dev.cell_stints[(driver, ct)],
                median_crossing_to_pit=_median_or_none(pit.cell_gaps.get((driver, ct))),
            )
        )
    driver_global = []
    for driver, scores in sorted(dev.driver_scores.items()):
        mean, std = _stats(scores, min_std)
        driver_global.append(
            _DriverEntry(
                driver=driver, mean=mean, std=std, n_stints=dev.driver_stints[driver],
                median_crossing_to_pit=_median_or_none(pit.driver_gaps.get(driver)),
            )
        )
    global_mean, global_std = (
        _stats(dev.global_scores, min_std) if dev.global_scores else (0.0, min_std)
    )
    cfg = _calibration_config_of(mon)
    cfg.min_stint_laps = min_stint_laps
    return _StintVariationArtefact(
        schema_version=SCHEMA_VERSION, config=cfg, cosine_features=list(cosine_features),
        feature_scaler={fid: _FeatureStat(mean=feature_mean[fid], std=feature_std[fid])
                        for fid in cosine_features},
        entries=entries, driver_global=driver_global,
        global_=_CellStat(mean=global_mean, std=global_std, n_stints=dev.global_stints,
                          median_crossing_to_pit=_median_or_none(pit.global_gaps)),
        slope_stats={
            ct: {fid: _FeatureStat(mean=m, std=s) for fid, (m, s) in feats.items()}
            for ct, feats in slope_stats.items()
        },
        content_hash="",
    )


def _log_mlflow(
    config: AppConfig, *, art_dir: Path, path: Path, ledger: _DQLedger, n_stints: int,
    n_entries: int, coverage: float,
) -> tuple[str | None, bool]:
    assert config.monitoring is not None
    mon = config.monitoring
    run_id: str | None = None
    degraded = False
    try:
        mlflow.set_tracking_uri(mon.mlflow_tracking_uri)
        art_loc = (art_dir / "mlartifacts").resolve().as_uri()
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(mon.mlflow_experiment)
        if exp is None:
            client.create_experiment(mon.mlflow_experiment, artifact_location=art_loc)
        mlflow.set_experiment(mon.mlflow_experiment)
        with mlflow.start_run(run_name="degradation-calibration") as run:
            mlflow.log_params(
                {
                    "baseline_laps": mon.baseline_laps, "rolling_window": mon.rolling_window,
                    "alert_threshold_sd": mon.alert_threshold_sd,
                    "alert_confirm_laps": mon.alert_confirm_laps, "slope_weight": mon.slope_weight,
                    "min_calibration_stints": mon.min_calibration_stints,
                }
            )
            mlflow.log_metric("n_stints_used", float(n_stints))
            mlflow.log_metric("n_entries", float(n_entries))
            mlflow.log_metric("coverage", float(coverage))
            mlflow.log_artifact(str(path))
            run_id = run.info.run_id
    except Exception as exc:  # noqa: BLE001 — MLflow fault degrades, does not abort the write
        degraded = True
        ledger.record(code=ErrorCode.MODEL_TRAINING_FAILED, entity_type=_ENTITY_BUILD,
                      entity_id=mon.mlflow_experiment, detail=f"{REASON_MLFLOW}: {exc}",
                      fallback_used=True)
    return run_id, degraded


def _log_week11_report(
    matrix: pd.DataFrame, handle_no_pit: MonitorHandle, config: AppConfig, *, art_dir: Path,
    mlflow_run_id: str | None, log: Path,
) -> None:
    """§4.4 Week-11 deliverable: run ``lead_time_report`` over the ``split=="val"`` validation
    stints (2022 by APEX's split convention — the ``val`` split IS the 2022 validation set, so no
    magic season literal is introduced), training circuits only (no held-out contact), and log it
    as an MLflow artefact of the calibration run. Best-effort — never raises, never blocks."""
    if COL_SPLIT not in matrix.columns:
        return
    try:
        val = cast(
            "pd.DataFrame",
            matrix.loc[
                (matrix[COL_SPLIT] == "val")
                & (~matrix[COL_CIRCUIT].isin(config.held_out_circuits))
            ].copy(),
        )
        if val.empty:
            return
        report = lead_time_report(val, handle=handle_no_pit, config=config, log_path=log)
        report_path = art_dir / "week11_lead_time_report.json"
        report_path.write_text(json.dumps(report.model_dump(), indent=2, default=str),
                               encoding="utf-8")
        if mlflow_run_id is not None:
            with mlflow.start_run(run_id=mlflow_run_id):
                mlflow.log_artifact(str(report_path))
    except Exception:  # noqa: BLE001 — a Week-11 reporting fault never breaks calibration
        return


# ----------------------------------------------------------------------------------
# Load + health (§4.3) — fail-loud only when the artefact cannot serve
# ----------------------------------------------------------------------------------


def load_detector(
    *,
    config: AppConfig,
    artefact_dir: str | Path | None = None,
    log_path: str | Path | None = None,
) -> MonitorHandle:
    """Load the serving bundle. Missing / corrupt / hash- or schema-mismatched artefact, an empty
    threshold table, a missing/wrong-width ``feature_scaler``, or a **structural-knob** mismatch →
    DQ-log then **raise** :class:`DetectorLoadError` (``MODEL_NOT_FOUND``). A **crossing-knob**
    mismatch loads with ``pit_window_valid=False`` + one DQ row; an empty ``slope_stats`` table
    serves cosine-only (degrade, don't die)."""
    if config.monitoring is None or config.normalise is None or config.features is None:
        raise DetectorLoadError(
            "config.monitoring / normalise / features block absent; cannot load"
        )
    mon = config.monitoring
    art_dir = Path(artefact_dir) if artefact_dir is not None else Path(config.paths.artefacts_dir)
    log = Path(log_path) if log_path is not None else Path(config.paths.data_quality_log)
    ledger = _DQLedger(log_path=log)

    path = art_dir / mon.variation_artefact_filename
    if not path.is_file():
        raise _load_fail(ledger, f"stint_variation artefact not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _load_fail(ledger, f"stint_variation artefact unreadable/corrupt: {exc}") from exc
    if not isinstance(raw, dict):
        raise _load_fail(ledger, "stint_variation artefact must be a JSON object")
    try:
        artefact = _StintVariationArtefact(**raw)
    except ValidationError as exc:
        raise _load_fail(ledger, f"stint_variation artefact schema mismatch: {exc}") from exc
    if artefact.schema_version != SCHEMA_VERSION:
        raise _load_fail(
            ledger, f"schema_version {artefact.schema_version!r} != expected {SCHEMA_VERSION!r}"
        )
    if _hash_artefact(artefact) != artefact.content_hash:
        raise _load_fail(ledger, "content_hash mismatch (corrupt/tampered)")

    # Empty threshold table → no usable tier → refuse.
    if not artefact.entries and not artefact.driver_global and artefact.global_.n_stints == 0:
        raise _load_fail(ledger, "empty threshold table (no entries at any tier)")

    # feature_scaler must exactly cover the current cosine feature set (never a silent unscaled
    # serve; §4.0-e). The set is derived from validate.degradation_features.
    cosine_features = _cosine_feature_ids(config)
    if set(artefact.feature_scaler) != set(cosine_features) or set(artefact.cosine_features) != set(
        cosine_features
    ):
        raise _load_fail(
            ledger,
            f"feature_scaler/cosine_features width mismatch: artefact has "
            f"{len(artefact.feature_scaler)} dims, config expects {len(cosine_features)}",
        )

    # Structural-knob mismatch → μ/σ stale → refuse; crossing-knob mismatch → pit_window suppressed.
    live = _calibration_config_of(mon)
    live.min_stint_laps = config.features.min_stint_laps
    structural_bad = [k for k in _STRUCTURAL_KNOBS
                      if getattr(artefact.config, k) != getattr(live, k)]
    if structural_bad:
        raise _load_fail(
            ledger, f"structural-knob mismatch {structural_bad} (artefact vs live) — recalibrate"
        )
    crossing_bad = [k for k in _CROSSING_KNOBS if getattr(artefact.config, k) != getattr(live, k)]
    pit_window_valid = not crossing_bad
    if crossing_bad:
        ledger.record(
            code=ErrorCode.FALLBACK_THRESHOLD, entity_type=_ENTITY_MODEL, entity_id="startup",
            detail=(
                f"crossing-knob mismatch {crossing_bad}: pit-window estimate unavailable until "
                "recalibration"
            ),
            fallback_used=True,
        )

    if not artefact.slope_stats:
        ledger.record(
            code=ErrorCode.LAP_PARTIAL, entity_type=_ENTITY_MODEL, entity_id="startup",
            detail=f"{REASON_SLOPE_STATS}: serving cosine-only (empty slope_stats)",
            fallback_used=True,
        )

    try:
        norm_params = load_normalisation_params(art_dir / config.normalise.artefact_filename)
    except NormalisationError as exc:
        raise _load_fail(ledger, f"normalisation artefact unavailable: {exc}") from exc

    entries = {
        (e.driver, e.circuit_type): _Cell(e.mean, e.std, e.n_stints, e.median_crossing_to_pit)
        for e in artefact.entries
    }
    driver_global = {
        e.driver: _Cell(e.mean, e.std, e.n_stints, e.median_crossing_to_pit)
        for e in artefact.driver_global
    }
    g = artefact.global_
    global_cell = _Cell(g.mean, g.std, g.n_stints, g.median_crossing_to_pit)
    slope_stats = {
        ct: {fid: (fs.mean, fs.std) for fid, fs in feats.items()}
        for ct, feats in artefact.slope_stats.items()
    }
    model_version = f"sha256:{artefact.content_hash[:_VERSION_HASH_LEN]}"
    return MonitorHandle(
        entries=entries, driver_global=driver_global, global_cell=global_cell,
        slope_stats=slope_stats, cosine_features=cosine_features,
        feature_mean={fid: fs.mean for fid, fs in artefact.feature_scaler.items()},
        feature_std={fid: fs.std for fid, fs in artefact.feature_scaler.items()},
        calibration_config=artefact.config, pit_window_valid=pit_window_valid,
        normalisation_params=norm_params, model_version=model_version,
        min_std=_resolve_min_std(config),
    )


def _load_fail(ledger: _DQLedger, detail: str) -> DetectorLoadError:
    """Log ``MODEL_NOT_FOUND`` and build the fail-loud load error (artefact cannot serve)."""
    ledger.record(code=ErrorCode.MODEL_NOT_FOUND, entity_type=_ENTITY_MODEL, entity_id="startup",
                  detail=detail, fallback_used=False)
    return DetectorLoadError(detail)


def health(handle: MonitorHandle | None) -> MonitorHealthResult:
    """Readiness for F13's ``GET /health``. Not-ready → ``HEALTH_FAIL`` (→ 503) with reason
    ``MODEL_NOT_FOUND``; ready → OK + ``model_version`` (mirrors F06b/F08)."""
    if handle is None or not handle.ready:
        return MonitorHealthResult(
            status=Status.FAILED, error_code=ErrorCode.HEALTH_FAIL, ready=False,
            reason=ErrorCode.MODEL_NOT_FOUND,
        )
    return MonitorHealthResult(status=Status.OK, ready=True, model_version=handle.model_version)


# ----------------------------------------------------------------------------------
# Evaluation machinery — lead_time_report (shipped here, formally run in F20) — never raises
# ----------------------------------------------------------------------------------


def lead_time_report(
    matrix: pd.DataFrame,
    *,
    handle: MonitorHandle,
    config: AppConfig,
    log_path: str | Path | None = None,
) -> LeadTimeReport:
    """§11.3 evaluation over the evaluation matrix (pit lap = explicit ``pit_lap`` column or the
    final-lap proxy): mean lead time, per-stint TP/FP, FN characterisation, compound
    stratification (``"UNKNOWN"`` excluded from strata only), early-alert map, the threshold
    sensitivity sweep at ``sensitivity_grid_sd`` per circuit type, and the arg-best threshold per
    circuit type. Read-only (no ``.fit``); malformed input → typed ``INVALID_INPUT``; no raise."""
    log = Path(log_path) if log_path is not None else Path(config.paths.data_quality_log)
    ledger = _DQLedger(log_path=log)
    try:
        if config.monitoring is None:
            return LeadTimeReport(status=Status.FAILED, error_code=ErrorCode.INVALID_INPUT)
        mon = config.monitoring
        eval_required = (*_REQUIRED_COLUMNS, COL_COMPOUND)
        if not isinstance(matrix, pd.DataFrame) or matrix.empty:
            return _lead_time_invalid(ledger, "empty or non-DataFrame evaluation matrix")
        missing = sorted(set(eval_required) - set(matrix.columns))
        if missing:
            return _lead_time_invalid(ledger, f"missing required columns: {missing}")

        stints = [
            g.sort_values(COL_LAP)
            for _, g in matrix.groupby(list(_STINT_GROUP_KEYS), dropna=False)
        ]
        per_stint: list[StintLeadTime] = []
        for stint_df in stints:
            per_stint.append(
                _evaluate_stint(stint_df, handle=handle, config=config,
                                alert_sd=mon.alert_threshold_sd, confirm=mon.alert_confirm_laps,
                                log=log)
            )

        report = _aggregate_lead_times(per_stint)
        report.early_alerts_by_driver_circuit = _early_alerts(per_stint)
        report.per_compound = _per_compound(per_stint)
        report.sensitivity, report.optimal_threshold_per_circuit_type = _sensitivity_sweep(
            stints, handle=handle, config=config, grid=mon.sensitivity_grid_sd,
            confirm=mon.alert_confirm_laps, log=log,
        )
        report.status = Status.OK
        report.logged = ledger.rows > 0
        if ledger.failed:
            report.flags = [ErrorCode.DATA_QUALITY_LOG_UNWRITABLE]
        return report
    except Exception as exc:  # noqa: BLE001 — never-raises backstop
        return _lead_time_invalid(ledger, f"{REASON_BACKSTOP}: {type(exc).__name__}: {exc}")


def _lead_time_invalid(ledger: _DQLedger, detail: str) -> LeadTimeReport:
    try:
        ledger.record(code=ErrorCode.INVALID_INPUT, entity_type=_ENTITY_BUILD,
                      entity_id="lead_time_report", detail=detail, fallback_used=False)
    except Exception:  # noqa: BLE001
        pass
    flags = [ErrorCode.DATA_QUALITY_LOG_UNWRITABLE] if ledger.failed else []
    return LeadTimeReport(status=Status.FAILED, error_code=ErrorCode.INVALID_INPUT, flags=flags,
                          logged=ledger.rows > 0)


def _alert_lap_for(
    stint_df: pd.DataFrame, *, handle: MonitorHandle, config: AppConfig, alert_sd: float,
    confirm: int, log: Path,
) -> int | None:
    """Score one stint and return the confirmed ``alert_lap`` (or None). Reuses the serving core
    with an explicit τ multiplier so the sensitivity sweep can vary it without mutating config."""
    records = cast("list[dict[str, Any]]", stint_df.to_dict("records"))
    driver = str(records[0][COL_DRIVER])
    ct = _circuit_type_of(records[0])
    circ = str(records[0][COL_CIRCUIT])
    assert config.features is not None
    if len(records) < config.features.min_stint_laps:
        return None
    cell, _src, _fb = _resolve_cell(handle, driver, ct,
                                    cast("Any", config.monitoring).min_calibration_stints)
    tau = cell.mean + alert_sd * cell.std
    core = _score_stint_core(
        records, circuit=circ, circuit_type=ct, cosine_features=handle.cosine_features,
        feature_mean=handle.feature_mean, feature_std=handle.feature_std,
        slope_stats=handle.slope_stats, norm_params=handle.normalisation_params, config=config,
        min_std=handle.min_std, log=log, entity_prefix=f"{driver}/{circ}",
    )
    if core.outcome != "ACTIVE":
        return None
    alert_out = _run_alert_machine(core.laps, tau=tau, confirm_laps=confirm)
    return alert_out.alert_lap


def _evaluate_stint(
    stint_df: pd.DataFrame, *, handle: MonitorHandle, config: AppConfig, alert_sd: float,
    confirm: int, log: Path,
) -> StintLeadTime:
    records = cast("list[dict[str, Any]]", stint_df.to_dict("records"))
    driver = str(records[0][COL_DRIVER])
    circ = str(records[0][COL_CIRCUIT])
    ct = _circuit_type_of(records[0])
    compound = str(records[0].get(COL_COMPOUND, "UNKNOWN"))
    green = _resolve_green_code(config)
    laps = [int(r[COL_LAP]) for r in records]
    if COL_PIT_LAP in stint_df.columns:
        pit_val = _finite_or_none(records[0].get(COL_PIT_LAP))
        pit_lap = int(pit_val) if pit_val is not None else max(laps)
    else:
        pit_lap = max(laps)
    sc_affected = any(
        not _is_green(r.get(COL_TRACK_STATUS), green) and r.get(COL_TRACK_STATUS) is not None
        for r in records
    )
    alert_lap = _alert_lap_for(stint_df, handle=handle, config=config, alert_sd=alert_sd,
                               confirm=confirm, log=log)
    lead = (pit_lap - alert_lap) if alert_lap is not None else None
    return StintLeadTime(
        driver=driver, circuit=circ, circuit_type=ct, compound=compound, n_laps=len(records),
        pit_lap=pit_lap, alert_lap=alert_lap, lead_time_laps=lead, safety_car_affected=sc_affected,
    )


def _is_tp(s: StintLeadTime) -> bool:
    return s.lead_time_laps is not None and 0 < s.lead_time_laps <= 5


def _is_early(s: StintLeadTime) -> bool:
    return s.lead_time_laps is not None and s.lead_time_laps > 5


def _rates(stints: list[StintLeadTime]) -> tuple[float | None, float | None, float | None]:
    if not stints:
        return None, None, None
    n = len(stints)
    tp = sum(1 for s in stints if _is_tp(s)) / n
    fp = sum(1 for s in stints if _is_early(s)) / n
    leads = [float(s.lead_time_laps) for s in stints if s.lead_time_laps is not None]
    mean_lead = float(np.mean(leads)) if leads else None
    return tp, fp, mean_lead


def _aggregate_lead_times(per_stint: list[StintLeadTime]) -> LeadTimeReport:
    tp, fp, mean_lead = _rates(per_stint)
    return LeadTimeReport(n_stints=len(per_stint), mean_lead_time_laps=mean_lead, tp_rate=tp,
                          fp_rate=fp, per_stint=per_stint)


def _early_alerts(per_stint: list[StintLeadTime]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for s in per_stint:
        if _is_early(s):
            key = f"{s.driver}/{s.circuit}"
            counts[key] = counts.get(key, 0) + 1
    return counts


def _per_compound(per_stint: list[StintLeadTime]) -> dict[str, dict[str, float]]:
    by: dict[str, list[StintLeadTime]] = {}
    for s in per_stint:
        if s.compound == "UNKNOWN":
            continue  # excluded from the stratified view only (still present in totals)
        by.setdefault(s.compound, []).append(s)
    out: dict[str, dict[str, float]] = {}
    for compound, group in by.items():
        tp, fp, mean_lead = _rates(group)
        out[compound] = {
            "tp_rate": tp or 0.0, "fp_rate": fp or 0.0,
            "mean_lead": mean_lead if mean_lead is not None else 0.0,
        }
    return out


def _sensitivity_sweep(
    stints: list[pd.DataFrame], *, handle: MonitorHandle, config: AppConfig, grid: list[float],
    confirm: int, log: Path,
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, float]]:
    """TP/FP per grid point, stratified per circuit type; plus the arg-best grid point per circuit
    type (max TP, tie-break min FP)."""
    sensitivity: dict[str, dict[str, dict[str, float]]] = {}
    per_ct_best: dict[str, tuple[float, float, float]] = {}  # ct → (grid_sd, tp, fp)
    for grid_sd in grid:
        by_ct: dict[str, list[StintLeadTime]] = {}
        for stint_df in stints:
            evaluated = _evaluate_stint(stint_df, handle=handle, config=config, alert_sd=grid_sd,
                                        confirm=confirm, log=log)
            ct_key = evaluated.circuit_type if evaluated.circuit_type is not None else _SRC_GLOBAL
            by_ct.setdefault(ct_key, []).append(evaluated)
        grid_key = f"{grid_sd}"
        sensitivity[grid_key] = {}
        for ct_key, group in by_ct.items():
            tp, fp, _mean = _rates(group)
            tp = tp or 0.0
            fp = fp or 0.0
            sensitivity[grid_key][ct_key] = {"tp_rate": tp, "fp_rate": fp}
            best = per_ct_best.get(ct_key)
            if best is None or tp > best[1] or (tp == best[1] and fp < best[2]):
                per_ct_best[ct_key] = (grid_sd, tp, fp)
    optimal = {ct: best[0] for ct, best in per_ct_best.items()}
    return sensitivity, optimal
