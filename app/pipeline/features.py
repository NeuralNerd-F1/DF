"""F03a — scalar-channel feature extraction + the shared 28-key vector assembly.

This module owns two things (master spec §4):

1. The 22 **scalar-channel** lap features computable directly from validated telemetry —
   F01–F18 (throttle, braking, speed, gear, engine) and F22–F24, F26 (interaction /
   lap-phase). Each has a defined computation, a valid range (``config.features``), and an
   out-of-range / NULL rule. A feature that is out of range or cannot be computed returns
   **NULL** (logged ``FEATURE_001``); the lap is *not* rejected here.

2. The infrastructure the whole feature matrix plugs into: the canonical 28-key vector
   contract, the :class:`FeatureComputationError` type + the catch-mark-NULL-log behaviour,
   and the §4.5 6-NULL tolerance gate (``<= null_ceiling`` NULLs → impute from a train-only
   median table + ``IMPUTED`` flag; ``> null_ceiling`` → reject ``LAP_NULL_EXCESS``). F03b
   later fills F19–F21, F25, F27–F28 into the *same* vector via :func:`assemble_feature_vector`.

The two public entry points (:func:`compute_scalar_features`, :func:`assemble_feature_vector`)
**never raise** for a data fault — every failure is caught, coded, logged, and returned in a
typed result object (non-negotiable #1). No normalisation is performed in this module: F03a
emits *raw* features; circuit-aware Z-scoring of F09/F11/F13/F14 is F04's train-only job.

**Time base.** F01 persists no per-sample clock (see ``validate.py`` LAP_004), so the sample
period used by the rate/duration features (F03, F07, F08, F16, F24) is derived from
``lap_time / n_samples`` — the same proxy F02 uses — unless a real per-sample time column is
present. If neither is available, those features return NULL rather than fabricate a clock.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import Field

from app.contracts.config import (
    GEOMETRY_FEATURE_IDS,
    SCALAR_FEATURE_IDS,
    AppConfig,
    FeaturesConfig,
)
from app.contracts.enums import Status
from app.contracts.errors import ErrorCode
from app.contracts.logging import log_data_quality
from app.contracts.results import BaseResult

# The 28 canonical feature IDs (master spec §4.2). F03a computes the 22 scalar IDs; the
# other six (F19–F21, F25, F27, F28) are F03b's — they appear in the assembled vector as
# keys with NULL/imputed values, never missing.
FEATURE_IDS: tuple[str, ...] = tuple(f"F{n:02d}" for n in range(1, 29))

# Reasons recorded in a NULL/cap diagnostic (and the data_quality_log detail).
REASON_OUT_OF_RANGE = "OUT_OF_RANGE"
REASON_NULL_TRIGGER = "NULL_TRIGGER"
REASON_OVERLAP_CAPPED = "OVERLAP_CAPPED"
REASON_UNEXPECTED = "UNEXPECTED"

# Required telemetry channels in the validated wide frame (post-F02).
_CHANNELS = ("Speed", "Throttle", "Brake", "RPM", "Gear")
# Candidate per-sample time columns, in preference order (real clock beats the proxy).
_TIME_COLUMNS = ("time", "Time", "SessionTime")


class FeatureComputationError(Exception):
    """Raised by a single feature function when its value is out of range or its
    precondition (documented §4.2 NULL trigger) is unmet.

    Carries ``ErrorCode.FEATURE_001`` and a ``reason`` so the driver can NULL the feature
    and log *why* — the §4.1 contract: a feature never silently returns a bad value.
    """

    error_code = ErrorCode.FEATURE_001

    def __init__(self, feature_id: str, reason: str, detail: str = "") -> None:
        super().__init__(f"{feature_id}: {reason} {detail}".strip())
        self.feature_id = feature_id
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class FeatureMedians:
    """Train-only imputation medians (§4.5): driver × circuit-type, with a global fallback.

    Computed exclusively from training laps and **injected** — F03a never derives a median
    from the lap(s) under evaluation (non-negotiable #2). ``median_for`` prefers the
    driver×circuit-type median and falls back to the global training median.
    """

    global_medians: dict[str, float]
    driver_circuit_medians: dict[tuple[str, str], dict[str, float]] = field(default_factory=dict)

    def median_for(self, feature_id: str, driver: str, circuit_type: str | None) -> float | None:
        if circuit_type is not None:
            dc = self.driver_circuit_medians.get((driver, circuit_type))
            if dc is not None and feature_id in dc:
                return dc[feature_id]
        return self.global_medians.get(feature_id)


class ScalarFeatureResult(BaseResult):
    """Per-lap output of :func:`compute_scalar_features` — the 22 scalar values + diagnostics.

    ``status`` is ``OK`` when all 22 computed in range, else ``DEGRADED`` (some NULL/capped).
    Lap-level ``error_code`` stays ``None`` — a NULL feature is not a lap failure.
    """

    lap_number: int
    features: dict[str, float | None] = Field(default_factory=dict)
    null_features: list[str] = Field(default_factory=list)
    capped_features: list[str] = Field(default_factory=list)
    detail: dict[str, dict[str, object]] = Field(default_factory=dict)


class FeatureResult(BaseResult):
    """Per-lap assembled 28-key vector — the contract F04/the DB consume, and F03b plugs into.

    ``null_count`` is the NULL count **before** imputation. On accept (``<= null_ceiling``
    NULL), ``feature_vector`` has those NULLs imputed and ``imputed_features`` lists them; on
    reject (``> null_ceiling``), ``status=FAILED``, ``error_code=LAP_NULL_EXCESS``, and the
    vector is left un-imputed. If a NULL cannot be imputed (no train median at all), the lap
    is also ``FAILED`` (``error_code=FEATURE_001``) rather than returned with a surviving NULL.
    """

    lap_number: int
    feature_vector: dict[str, float | None] = Field(default_factory=dict)
    null_count: int = 0
    imputed_features: list[str] = Field(default_factory=list)
    corner_type: str | None = None


class GeometryFeatureResult(BaseResult):
    """Per-lap output of :func:`compute_geometry_features` (F03b) — the four X/Y-derived
    cornering values + diagnostics. ``status`` is ``OK`` only when all four computed in range,
    else ``DEGRADED``. ``error_code`` carries ``CORNER_001`` when X/Y was unavailable (the
    shared NULL path); lap-level result is never ``FAILED`` — F03b never rejects a lap."""

    lap_number: int
    features: dict[str, float | None] = Field(default_factory=dict)  # exactly {F19,F20,F21,F25}
    null_features: list[str] = Field(default_factory=list)
    detail: dict[str, dict[str, object]] = Field(default_factory=dict)


class DegradationResult(BaseResult):
    """Per-**stint** output of :func:`compute_degradation_slopes` (F03b) — the F27/F28 tyre-age
    slopes, broadcast to every lap of the stint via ``applies_to``. ``f27``/``f28`` are NULL
    (with ``null_reason``) when the stint has < ``min_stint_laps`` usable laps or tyre age is
    UNKNOWN. Unconstrained — slopes are never range-gated."""

    stint_id: int
    f27: float | None = None
    f28: float | None = None
    applies_to: list[int] = Field(default_factory=list)  # lap numbers the slopes apply to
    null_reason: str | None = None


@dataclass(frozen=True)
class StintLap:
    """One lap's degradation inputs (F03b): tyre age + the per-lap F01/F06 values F27/F28
    regress against. Collected by the orchestrator from F03a's per-lap results — F03b does not
    recompute them. ``tyre_age=None`` marks UNKNOWN (consistent with F02's ``LAP_PARTIAL``)."""

    lap_number: int
    tyre_age: int | None
    throttle_mean: float | None  # F01
    brake_peak_mean: float | None  # F06


# ----------------------------------------------------------------------------------
# Data-quality ledger (one row per NULL / capped / imputed feature and per rejected lap)
# ----------------------------------------------------------------------------------


@dataclass
class _DQLedger:
    """Writes one ``data_quality_log`` row per NULL/cap/impute/rejection (mirrors F02)."""

    log_path: Path
    rows: int = 0

    def record(
        self, *, code: ErrorCode, entity_id: str, detail: str, fallback_used: bool
    ) -> None:
        log_data_quality(
            error_code=code,
            entity_type="lap",
            entity_id=entity_id,
            error_detail=detail,
            fallback_used=fallback_used,
            log_path=self.log_path,
        )
        self.rows += 1


# ----------------------------------------------------------------------------------
# Channel adapter + per-lap derived context + vectorised helpers
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Channels:
    """One lap's telemetry as float numpy arrays (NaN-padded)."""

    speed: np.ndarray
    throttle: np.ndarray
    brake: np.ndarray
    rpm: np.ndarray
    gear: np.ndarray


@dataclass(frozen=True)
class _LapContext:
    """Per-lap context + arrays derived once and shared across the feature functions.

    Deriving the braking-event runs, the corner minima, and the sample period a single time
    here (rather than re-running them inside each feature) keeps the hot path O(n) per lap.
    ``dt_ms`` / ``time`` are ``None`` when no real or proxy clock is available — the
    time-dependent features then NULL instead of computing on a fabricated clock.
    """

    session_max_rpm: float | None
    lap_distance_m: float | None
    time: np.ndarray | None
    dt_ms: float | None
    braking_events: np.ndarray  # (N, 2) [start, stop) runs where Brake > threshold
    corners: np.ndarray  # speed-minimum indices (prominence-filtered)


def _numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def _extract_channels(lap: pd.DataFrame) -> _Channels:
    """Pull the five required channels as float arrays (absent channel -> all-NaN)."""
    n = len(lap)
    arr = {
        ch: _numeric(lap[ch]) if ch in lap.columns else np.full(n, np.nan)
        for ch in _CHANNELS
    }
    return _Channels(
        speed=arr["Speed"], throttle=arr["Throttle"], brake=arr["Brake"],
        rpm=arr["RPM"], gear=arr["Gear"],
    )


def _first_finite_scalar(lap: pd.DataFrame, column: str) -> float | None:
    if column not in lap.columns:
        return None
    vals = _numeric(lap[column])
    finite = vals[np.isfinite(vals)]
    return float(finite[0]) if finite.size else None


def _resolve_time_base(
    lap: pd.DataFrame, lap_time_s: float | None, n: int, cfg: FeaturesConfig
) -> tuple[np.ndarray | None, float | None]:
    """Establish the per-sample clock: real time column if present, else the
    ``lap_time / n_samples`` proxy (the documented F02 method), else ``(None, None)``."""
    for col in _TIME_COLUMNS:
        if col in lap.columns:
            t = _numeric(lap[col])
            d = np.diff(t)
            d = d[np.isfinite(d) & (d > 0)]
            if d.size:
                return t, float(np.median(d)) * 1000.0
    lt = lap_time_s if lap_time_s is not None else _first_finite_scalar(lap, "lap_time")
    if lt is not None and lt > 0 and n > 0:
        dt_s = lt / n
        return np.arange(n, dtype=float) * dt_s, dt_s * 1000.0
    return None, None


def _finite(a: np.ndarray) -> np.ndarray:
    """Drop NaN/inf from a 1-D array."""
    return a[np.isfinite(a)]


def _runs(mask: np.ndarray) -> np.ndarray:
    """Contiguous ``True`` runs of a boolean mask as an ``(N, 2)`` array of ``[start, stop)``."""
    if mask.size == 0:
        return np.empty((0, 2), dtype=int)
    m = mask.astype(np.int8)
    edges = np.diff(m, prepend=np.int8(0), append=np.int8(0))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return np.stack([starts, stops], axis=1)


def _corner_indices(speed: np.ndarray, cfg: FeaturesConfig) -> np.ndarray:
    """Sample indices of speed local minima with prominence >= ``corner_min_prominence_kmh``.

    Scalar-derived corner detection (NOT X/Y — that is F03b's corner-*type* decomposition).
    NaN samples are handled implicitly: a comparison against NaN is ``False`` (so a
    NaN-adjacent sample is never flagged a minimum) and a window spanning a NaN yields a NaN
    peak, which fails the prominence test — corners bordering a dropout are conservatively
    skipped rather than fabricated.
    """
    s = speed
    if s.size < 3:
        return np.empty(0, dtype=int)
    is_min = (s[1:-1] < s[:-2]) & (s[1:-1] <= s[2:])
    minima = np.flatnonzero(is_min) + 1
    if minima.size == 0:
        return np.empty(0, dtype=int)
    kept: list[int] = []
    for i, m in enumerate(minima):  # iterates detected minima (few), not samples
        left_start = int(minima[i - 1]) if i > 0 else 0
        right_stop = int(minima[i + 1]) + 1 if i + 1 < minima.size else s.size
        left_peak = float(np.max(s[left_start : m + 1]))
        right_peak = float(np.max(s[m:right_stop]))
        if min(left_peak, right_peak) - float(s[m]) >= cfg.corner_min_prominence_kmh:
            kept.append(int(m))
    return np.asarray(kept, dtype=int)


def _checked(feature_id: str, value: float, cfg: FeaturesConfig) -> float:
    """Range-gate one feature value (§4.2): in range → return; outside/NaN → raise."""
    lo, hi = cfg.valid_ranges[feature_id]
    if not (lo <= value <= hi):  # NaN compares False → raised as out-of-range
        raise FeatureComputationError(
            feature_id, REASON_OUT_OF_RANGE, f"value={value} not in [{lo}, {hi}]"
        )
    return float(value)


def _require(feature_id: str, condition: bool, detail: str) -> None:
    """Raise the NULL-trigger path when a feature's precondition is unmet."""
    if not condition:
        raise FeatureComputationError(feature_id, REASON_NULL_TRIGGER, detail)


def _require_dt(feature_id: str, ctx: _LapContext) -> float:
    """Return the sample period in ms, or NULL-trigger when no clock is available."""
    _require(feature_id, ctx.dt_ms is not None, "time base unavailable (no clock / lap_time)")
    return ctx.dt_ms  # type: ignore[return-value]


# ----------------------------------------------------------------------------------
# The 22 scalar feature functions (each vectorised; raises on NULL trigger / out of range)
# ----------------------------------------------------------------------------------


def _f01(ch: _Channels, ctx: _LapContext, cfg: FeaturesConfig) -> float:
    t = _finite(ch.throttle)
    _require("F01", t.size > 0, "all throttle samples NaN")
    return _checked("F01", float(np.mean(t)), cfg)


def _f02(ch: _Channels, ctx: _LapContext, cfg: FeaturesConfig) -> float:
    t = _finite(ch.throttle)
    _require("F02", t.size > 0, "all throttle samples NaN")
    std = float(np.std(t))
    _require("F02", std > 0.0, "constant throttle (std=0) — sensor fault")
    return _checked("F02", std, cfg)


def _f03(ch: _Channels, ctx: _LapContext, cfg: FeaturesConfig) -> float:
    dt_ms = _require_dt("F03", ctx)
    d = np.diff(ch.throttle)
    rising = d[np.isfinite(d) & (d > 0)]
    _require("F03", rising.size > 0, "no throttle-increasing events")
    rate = float(np.mean(rising)) * (cfg.sample_period_ms / dt_ms)
    return _checked("F03", rate, cfg)


def _f04(ch: _Channels, ctx: _LapContext, cfg: FeaturesConfig) -> float:
    t = _finite(ch.throttle)
    _require("F04", t.size > 0, "all throttle samples NaN")
    return _checked("F04", float(np.mean(t >= cfg.throttle_full_pct)), cfg)


def _f05(ch: _Channels, ctx: _LapContext, cfg: FeaturesConfig) -> float:
    t = _finite(ch.throttle)
    _require("F05", t.size > 0, "all throttle samples NaN")
    return _checked("F05", float(np.mean(t < cfg.throttle_lift_pct)), cfg)


def _f06(ch: _Channels, ctx: _LapContext, cfg: FeaturesConfig) -> float:
    events = ctx.braking_events
    _require(
        "F06",
        len(events) >= cfg.min_braking_events,
        f"{len(events)} braking events < {cfg.min_braking_events}",
    )
    peaks = np.array([float(np.nanmax(ch.brake[s:e])) for s, e in events])
    top = np.sort(peaks)[-cfg.brake_peak_top_n :]
    return _checked("F06", float(np.mean(top)), cfg)


def _f07(ch: _Channels, ctx: _LapContext, cfg: FeaturesConfig) -> float:
    dt_ms = _require_dt("F07", ctx)
    events = ctx.braking_events
    _require("F07", len(events) > 0, "no braking events detected")
    durations_ms = (events[:, 1] - events[:, 0]) * dt_ms
    return _checked("F07", float(np.mean(durations_ms)), cfg)


def _f08(ch: _Channels, ctx: _LapContext, cfg: FeaturesConfig) -> float:
    dt_ms = _require_dt("F08", ctx)
    d = np.diff(ch.brake)
    releasing = -d[np.isfinite(d) & (d < 0)]
    _require("F08", releasing.size > 0, "no brake-release events")
    rate = float(np.mean(releasing)) * (cfg.sample_period_ms / dt_ms)
    return _checked("F08", rate, cfg)


def _f09(ch: _Channels, ctx: _LapContext, cfg: FeaturesConfig) -> float:
    corners = ctx.corners
    _require("F09", corners.size > 0, "corner detection failed")
    starts = ctx.braking_events[:, 0] if ctx.braking_events.size else np.empty(0, dtype=int)
    _require("F09", starts.size > 0, "no braking events to anchor corner entry")
    entry: list[float] = []
    for c in corners:  # iterates detected corners (few), not samples
        prior = starts[starts <= c]
        if prior.size:
            entry.append(float(ch.speed[prior[-1]]))
    _require("F09", len(entry) > 0, "no corner has a preceding braking event")
    return _checked("F09", float(np.mean(entry)), cfg)


def _f10_overlap_raw(ch: _Channels, cfg: FeaturesConfig) -> float:
    """Raw overlap fraction (cap/log decision is made by the driver, not here)."""
    both = (ch.throttle > cfg.overlap_throttle_pct) & (ch.brake > cfg.overlap_brake_pct)
    return float(np.mean(both))


def _f11(ch: _Channels, ctx: _LapContext, cfg: FeaturesConfig) -> float:
    corners = ctx.corners
    _require("F11", corners.size > 0, "corner detection failed")
    return _checked("F11", float(np.mean(ch.speed[corners])), cfg)


def _f12(ch: _Channels, ctx: _LapContext, cfg: FeaturesConfig) -> float:
    s = _finite(ch.speed)
    _require("F12", s.size > 0, "all speed samples NaN")
    _require("F12", float(np.ptp(s)) > 0.0, "all speed samples identical — sensor fault")
    return _checked("F12", float(np.var(s)), cfg)


def _f13(ch: _Channels, ctx: _LapContext, cfg: FeaturesConfig) -> float:
    s = _finite(ch.speed)
    _require("F13", s.size > 0, "all speed samples NaN")
    return _checked("F13", float(np.percentile(s, cfg.speed_p_low)), cfg)


def _f14(ch: _Channels, ctx: _LapContext, cfg: FeaturesConfig) -> float:
    s = _finite(ch.speed)
    _require("F14", s.size > 0, "all speed samples NaN")
    return _checked("F14", float(np.percentile(s, cfg.speed_p_high)), cfg)


def _f15(ch: _Channels, ctx: _LapContext, cfg: FeaturesConfig) -> float:
    _require(
        "F15",
        ctx.lap_distance_m is not None and ctx.lap_distance_m > 0,
        "lap distance cannot be computed",
    )
    gear = _finite(ch.gear)
    _require("F15", gear.size > 0, "all gear samples NaN")
    changes = int(np.count_nonzero(np.diff(gear) != 0))
    km = float(ctx.lap_distance_m) / 1000.0  # type: ignore[arg-type]
    return _checked("F15", changes / km, cfg)


def _f16(ch: _Channels, ctx: _LapContext, cfg: FeaturesConfig) -> float:
    dt_ms = _require_dt("F16", ctx)
    gear = _finite(ch.gear)
    _require("F16", gear.size > 0, "all gear samples NaN")
    boundaries = np.flatnonzero(np.diff(gear) != 0) + 1
    n_changes = int(boundaries.size)
    _require(
        "F16",
        n_changes >= cfg.min_gear_changes,
        f"{n_changes} gear changes < {cfg.min_gear_changes}",
    )
    edges = np.concatenate(([0], boundaries, [gear.size]))
    seg_lengths = np.diff(edges)
    return _checked("F16", float(np.mean(seg_lengths)) * dt_ms, cfg)


def _f17(ch: _Channels, ctx: _LapContext, cfg: FeaturesConfig) -> float:
    _require(
        "F17",
        ctx.session_max_rpm is not None and ctx.session_max_rpm > 0,
        "session max RPM not deterministic",
    )
    r = _finite(ch.rpm)
    _require("F17", r.size > 0, "all RPM samples NaN")
    threshold = (cfg.redline_pct / 100.0) * float(ctx.session_max_rpm)  # type: ignore[arg-type]
    return _checked("F17", float(np.mean(r > threshold)), cfg)


def _f18(ch: _Channels, ctx: _LapContext, cfg: FeaturesConfig) -> float:
    r = _finite(ch.rpm)
    _require("F18", r.size > 0, "all RPM samples NaN")
    std = float(np.std(r))
    _require("F18", std > 0.0, "constant RPM — sensor fault")
    return _checked("F18", std, cfg)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    am, bm = a[mask], b[mask]
    # A constant input makes correlation undefined (-> NaN -> out-of-range -> NULL); that is
    # the intended path, so short-circuit rather than emit numpy's divide-by-zero warning.
    if np.ptp(am) == 0 or np.ptp(bm) == 0:
        return float("nan")
    return float(np.corrcoef(am, bm)[0, 1])


def _f22(ch: _Channels, ctx: _LapContext, cfg: FeaturesConfig) -> float:
    return _checked("F22", _pearson(ch.throttle, ch.speed), cfg)


def _f23(ch: _Channels, ctx: _LapContext, cfg: FeaturesConfig) -> float:
    _require("F23", ctx.braking_events.size > 0, "no braking events")
    decel = -np.diff(ch.speed)  # deceleration over [i, i+1]
    # Pair each interval's deceleration with the brake pressure at its START sample (the
    # pressure that caused it), so brake and deceleration are aligned, not lagged.
    return _checked("F23", _pearson(ch.brake[:-1], decel), cfg)


def _f24(ch: _Channels, ctx: _LapContext, cfg: FeaturesConfig) -> float:
    _require_dt("F24", ctx)  # needs a clock for the gap timing
    time = ctx.time
    events = ctx.braking_events
    _require("F24", len(events) > 0, "no braking events")
    applied = ch.throttle > cfg.throttle_lift_pct
    gaps_ms: list[float] = []
    for _s, stop in events:  # iterates braking events (few), not samples
        after = np.flatnonzero(applied[stop:])
        if after.size:
            idx = stop + int(after[0])
            gaps_ms.append((float(time[idx]) - float(time[stop - 1])) * 1000.0)  # type: ignore[index]
    _require(
        "F24",
        len(gaps_ms) >= cfg.min_transitions,
        f"{len(gaps_ms)} transitions < {cfg.min_transitions}",
    )
    return _checked("F24", float(np.mean(gaps_ms)), cfg)


def _f26(ch: _Channels, ctx: _LapContext, cfg: FeaturesConfig) -> float:
    t = ch.throttle
    half = t.size // 2
    _require("F26", half > 0 and (t.size - half) > 0, "lap too short to split into phases")
    first = _finite(t[:half])
    second = _finite(t[half:])
    _require("F26", first.size > 0 and second.size > 0, "a lap phase has no valid throttle")
    return _checked("F26", float(np.mean(first)) - float(np.mean(second)), cfg)


# fid -> compute function. F10 is handled specially by the driver (cap, not raise).
_FEATURE_FUNCS = {
    "F01": _f01, "F02": _f02, "F03": _f03, "F04": _f04, "F05": _f05,
    "F06": _f06, "F07": _f07, "F08": _f08, "F09": _f09, "F11": _f11,
    "F12": _f12, "F13": _f13, "F14": _f14, "F15": _f15, "F16": _f16,
    "F17": _f17, "F18": _f18, "F22": _f22, "F23": _f23, "F24": _f24,
    "F26": _f26,
}


# ----------------------------------------------------------------------------------
# Public entry points
# ----------------------------------------------------------------------------------


def compute_scalar_features(
    lap: pd.DataFrame,
    *,
    lap_number: int,
    session_max_rpm: float | None,
    lap_distance_m: float | None,
    config: AppConfig,
    lap_time_s: float | None = None,
    log_path: str | Path | None = None,
    entity_id: str | None = None,
) -> ScalarFeatureResult:
    """Compute the 22 scalar features for one validated lap.

    Each feature is computed independently: a :class:`FeatureComputationError` (out of range
    or NULL trigger) or any unexpected error is caught, the feature is set NULL, and one
    ``FEATURE_001`` row is logged with the reason. F10 over the cap is clamped + logged (not
    NULL). This function never raises for a data fault.
    """
    path = Path(log_path) if log_path is not None else Path(config.paths.data_quality_log)
    ledger = _DQLedger(log_path=path)
    eid = entity_id or f"lap{lap_number}"

    fcfg = config.features
    if fcfg is None:
        # Config block absent — typed failure, no bare raise (#1). All features NULL.
        ledger.record(
            code=ErrorCode.CONFIG_001,
            entity_id=eid,
            detail="config.features block absent; cannot compute features",
            fallback_used=False,
        )
        return ScalarFeatureResult(
            lap_number=lap_number,
            features={fid: None for fid in SCALAR_FEATURE_IDS},
            null_features=list(SCALAR_FEATURE_IDS),
            status=Status.DEGRADED,
            error_code=ErrorCode.CONFIG_001,
            logged=True,
        )

    ch = _extract_channels(lap)
    time, dt_ms = _resolve_time_base(lap, lap_time_s, len(lap), fcfg)
    ctx = _LapContext(
        session_max_rpm=session_max_rpm,
        lap_distance_m=lap_distance_m,
        time=time,
        dt_ms=dt_ms,
        braking_events=_runs(ch.brake > fcfg.braking_event_threshold_bar),
        corners=_corner_indices(ch.speed, fcfg),
    )

    features: dict[str, float | None] = {}
    null_features: list[str] = []
    capped_features: list[str] = []
    detail: dict[str, dict[str, object]] = {}

    for fid in SCALAR_FEATURE_IDS:
        try:
            if fid == "F10":
                _require(
                    "F10",
                    bool(np.isfinite(ch.throttle).any() and np.isfinite(ch.brake).any()),
                    "no finite throttle/brake samples",
                )
                raw = _f10_overlap_raw(ch, fcfg)
                if np.isfinite(raw) and raw > fcfg.overlap_cap:
                    # Clamp to the cap, then range-gate it like any value so a misconfigured
                    # cap above F10's range cannot silently emit an out-of-range number.
                    features[fid] = _checked("F10", float(fcfg.overlap_cap), fcfg)
                    capped_features.append(fid)
                    detail[fid] = {"reason": REASON_OVERLAP_CAPPED, "raw": raw}
                    ledger.record(
                        code=ErrorCode.FEATURE_001,
                        entity_id=eid,
                        detail=(
                            f"F10 overlap {raw:.4f} > cap {fcfg.overlap_cap}; "
                            "clamped (sensor fault)"
                        ),
                        fallback_used=True,
                    )
                    continue
                features[fid] = _checked("F10", raw, fcfg)
                continue
            features[fid] = _FEATURE_FUNCS[fid](ch, ctx, fcfg)
        except FeatureComputationError as exc:
            features[fid] = None
            null_features.append(fid)
            detail[fid] = {"reason": exc.reason, "detail": exc.detail}
            ledger.record(
                code=ErrorCode.FEATURE_001,
                entity_id=eid,
                detail=f"{fid} NULL ({exc.reason}): {exc.detail}",
                fallback_used=False,
            )
        except Exception as exc:  # noqa: BLE001 — last-resort: NULL, never propagate (#1)
            features[fid] = None
            null_features.append(fid)
            detail[fid] = {"reason": REASON_UNEXPECTED, "error": repr(exc)}
            ledger.record(
                code=ErrorCode.FEATURE_001,
                entity_id=eid,
                detail=f"{fid} NULL (UNEXPECTED): {exc!r}",
                fallback_used=False,
            )

    status = Status.OK if not null_features and not capped_features else Status.DEGRADED
    return ScalarFeatureResult(
        lap_number=lap_number,
        features=features,
        null_features=null_features,
        capped_features=capped_features,
        detail=detail,
        status=status,
        logged=ledger.rows > 0,
    )


def assemble_feature_vector(
    values: dict[str, float | None],
    *,
    lap_number: int,
    driver: str,
    circuit_type: str | None,
    medians: FeatureMedians,
    config: AppConfig,
    forced_null: list[str] | None = None,
    flags: list[ErrorCode] | None = None,
    corner_type: str | None = None,
    log_path: str | Path | None = None,
    entity_id: str | None = None,
) -> FeatureResult:
    """Assemble the canonical 28-key vector and apply the §4.5 NULL-tolerance gate.

    ``values`` is the union of every contributor's output (F03a's 22 + F03b's six); any
    missing key is treated as NULL. ``forced_null`` (F02's per-lap ``null_features``, e.g.
    F27/F28 under ``LAP_PARTIAL``) is applied before counting. ``> null_ceiling`` NULLs →
    reject ``LAP_NULL_EXCESS`` (no imputation); ``<= null_ceiling`` → impute each NULL from
    the train-only ``medians`` table and tag ``IMPUTED``. A NULL with no train median at all
    leaves the vector incomplete, so the lap is failed (``FEATURE_001``) rather than returned
    as accepted with a surviving NULL. Never raises for a data fault.
    """
    path = Path(log_path) if log_path is not None else Path(config.paths.data_quality_log)
    ledger = _DQLedger(log_path=path)
    eid = entity_id or f"lap{lap_number}"
    out_flags = list(flags) if flags else []
    null_ceiling = config.thresholds.null_ceiling

    vector: dict[str, float | None] = {fid: values.get(fid) for fid in FEATURE_IDS}
    for fid in forced_null or []:
        vector[fid] = None
    null_ids = [fid for fid in FEATURE_IDS if vector[fid] is None]
    null_count = len(null_ids)

    if null_count > null_ceiling:
        ledger.record(
            code=ErrorCode.LAP_NULL_EXCESS,
            entity_id=eid,
            detail=f"{null_count} NULL features > ceiling {null_ceiling}: {null_ids}",
            fallback_used=False,
        )
        return FeatureResult(
            lap_number=lap_number,
            feature_vector=vector,
            null_count=null_count,
            imputed_features=[],
            corner_type=corner_type,
            status=Status.FAILED,
            error_code=ErrorCode.LAP_NULL_EXCESS,
            flags=out_flags,
            logged=True,
        )

    imputed: list[str] = []
    unimputable: list[str] = []
    for fid in null_ids:
        median = medians.median_for(fid, driver, circuit_type)
        if median is None:
            unimputable.append(fid)
            ledger.record(
                code=ErrorCode.FEATURE_001,
                entity_id=eid,
                detail=f"{fid} NULL and no train median available; cannot impute",
                fallback_used=False,
            )
            continue
        vector[fid] = float(median)
        imputed.append(fid)
        ledger.record(
            code=ErrorCode.IMPUTED,
            entity_id=eid,
            detail=f"{fid} imputed from train median ({driver}/{circuit_type})={median}",
            fallback_used=True,
        )

    if unimputable:
        # The vector cannot be made whole, so it must not be returned as accepted (#1) — a
        # surviving NULL would poison F04 normalisation / the classifier downstream.
        return FeatureResult(
            lap_number=lap_number,
            feature_vector=vector,
            null_count=null_count,
            imputed_features=imputed,
            corner_type=corner_type,
            status=Status.FAILED,
            error_code=ErrorCode.FEATURE_001,
            flags=out_flags,
            logged=True,
        )

    if imputed:
        out_flags.append(ErrorCode.IMPUTED)
        status = Status.DEGRADED
    else:
        status = Status.OK if null_count == 0 else Status.DEGRADED

    return FeatureResult(
        lap_number=lap_number,
        feature_vector=vector,
        null_count=null_count,
        imputed_features=imputed,
        corner_type=corner_type,
        status=status,
        error_code=None,
        flags=out_flags,
        logged=ledger.rows > 0,
    )


# ==================================================================================
# F03b — geometry (F19–F21, F25) + degradation slopes (F27, F28)
#
# Plugs its six values into assemble_feature_vector (above) unchanged. The four cornering
# features share one X/Y-availability precondition: X/Y missing -> all four NULL + one
# CORNER_001 row, lap continues (§4.4). F27/F28 are stint-level (not lap-level) and
# unconstrained. Reuses F03a's FeatureComputationError / _checked / _require / _runs /
# _corner_indices / _DQLedger / _finite / _resolve_time_base.
# ==================================================================================

REASON_XY_UNAVAILABLE = "X_Y_UNAVAILABLE"


@dataclass(frozen=True)
class _GeometryContext:
    """Per-lap arrays derived once for the four cornering features."""

    signed_g: np.ndarray  # signed lateral G per sample (sign from the curvature cross-product)
    speed: np.ndarray
    brake: np.ndarray
    corners: np.ndarray  # apex (speed-minimum) indices
    duration_s: float | None  # lap wall-time, for F21's per-second rate


def _xy_available(lap: pd.DataFrame, cfg: FeaturesConfig) -> bool:
    """True iff the X/Y position trace exists with enough finite samples to derive curvature.

    Requires at least 3 finite paired samples AND a finite fraction >= ``min_xy_finite_fraction``
    — a near-total dropout (a handful of finite points in a mostly-NaN trace) is treated as
    unavailable (-> CORNER_001), not fabricated into a cornering value."""
    if "X" not in lap.columns or "Y" not in lap.columns:
        return False
    n = len(lap)
    if n == 0:
        return False
    x = _numeric(lap["X"])
    y = _numeric(lap["Y"])
    finite = int(np.count_nonzero(np.isfinite(x) & np.isfinite(y)))
    return finite >= 3 and (finite / n) >= cfg.min_xy_finite_fraction


def _signed_lateral_g(
    x: np.ndarray, y: np.ndarray, speed: np.ndarray, cfg: FeaturesConfig
) -> np.ndarray:
    """Signed lateral G per sample from the X/Y trace + speed.

    Curvature κ = |x'y'' − y'x''| / (x'² + y'²)^1.5 (gradients via ``np.gradient``); lateral
    acceleration a = v²·κ (v = Speed ÷ 3.6 m/s); G = a / ``gravity_m_s2``. The sign carries the
    turn direction (sign of the cross-product) so F21 can count left/right reversals.
    """
    x1, y1 = np.gradient(x), np.gradient(y)
    x2, y2 = np.gradient(x1), np.gradient(y1)
    cross = x1 * y2 - y1 * x2
    denom = (x1**2 + y1**2) ** 1.5
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = np.where(denom > 0, np.abs(cross) / denom, 0.0)
    v = speed / 3.6  # km/h -> m/s
    g = (v**2 * kappa) / cfg.gravity_m_s2
    return np.sign(cross) * g


def _lateral_g_peaks(absg: np.ndarray, cfg: FeaturesConfig) -> np.ndarray:
    """Values of |lateral G| local maxima whose prominence >= ``lateral_g_peak_min_prominence``.

    A NaN window yields a NaN valley, so peaks bordering a dropout are conservatively dropped.
    """
    a = absg
    if a.size < 3:
        return np.empty(0, dtype=float)
    is_peak = (a[1:-1] > a[:-2]) & (a[1:-1] >= a[2:])
    idx = np.flatnonzero(is_peak) + 1
    if idx.size == 0:
        return np.empty(0, dtype=float)
    kept: list[float] = []
    for i, m in enumerate(idx):  # iterates detected peaks (few), not samples
        left_start = int(idx[i - 1]) if i > 0 else 0
        right_stop = int(idx[i + 1]) + 1 if i + 1 < idx.size else a.size
        left_valley = float(np.min(a[left_start : m + 1]))
        right_valley = float(np.min(a[m:right_stop]))
        if float(a[m]) - max(left_valley, right_valley) >= cfg.lateral_g_peak_min_prominence:
            kept.append(float(a[m]))
    return np.asarray(kept, dtype=float)


def _f19(gctx: _GeometryContext, cfg: FeaturesConfig) -> float:
    absg = _finite(np.abs(gctx.signed_g))
    _require("F19", absg.size > 0, "lateral G not computable")
    return _checked("F19", float(np.mean(absg)), cfg)


def _f20(gctx: _GeometryContext, cfg: FeaturesConfig) -> float:
    # _finite() first (like F19/F21) so a single NaN sample cannot poison a neighbour's
    # prominence valley and spuriously drop a genuine peak.
    peaks = _lateral_g_peaks(_finite(np.abs(gctx.signed_g)), cfg)
    _require(
        "F20",
        peaks.size >= cfg.min_lateral_g_peaks,
        f"{peaks.size} distinct lateral-G peaks < {cfg.min_lateral_g_peaks}",
    )
    top = np.sort(peaks)[-cfg.lateral_g_peak_top_n :]
    return _checked("F20", float(np.mean(top)), cfg)


def _f21(gctx: _GeometryContext, cfg: FeaturesConfig) -> float:
    _require("F21", gctx.duration_s is not None and gctx.duration_s > 0, "lap duration unavailable")
    g = _finite(gctx.signed_g)
    _require("F21", g.size > 0, "lateral G not computable")
    # Sign of lateral G, ignoring |G| below the deadband (noise around straight-line driving).
    sign = np.where(np.abs(g) >= cfg.steering_g_deadband, np.sign(g), 0.0)
    nz = sign[sign != 0]
    reversals = int(np.count_nonzero(np.diff(nz) != 0)) if nz.size > 1 else 0
    return _checked("F21", reversals / float(gctx.duration_s), cfg)  # type: ignore[arg-type]


def _f25(gctx: _GeometryContext, cfg: FeaturesConfig) -> float:
    corners = gctx.corners
    _require("F25", corners.size > 0, "corner segmentation failed")
    speed, brake = gctx.speed, gctx.brake
    brake_on = (brake > cfg.braking_event_threshold_bar).astype(float)
    n = speed.size
    ratios: list[float] = []
    for i, c in enumerate(corners):  # iterates detected corners (few), not samples
        left_bound = int(corners[i - 1]) if i > 0 else 0
        right_bound = int(corners[i + 1]) if i + 1 < corners.size else n - 1
        # Corner window = the speed maxima bracketing the apex; split at the apex.
        left_max = (
            left_bound + int(np.argmax(speed[left_bound : c + 1])) if c > left_bound else left_bound
        )
        right_max = (
            c + int(np.argmax(speed[c : right_bound + 1])) if right_bound > c else right_bound
        )
        # Exit window is [apex, right_max) — exclusive of right_max so the shared inter-corner
        # speed-maximum index is not double-counted across two adjacent corners.
        entry = float(np.sum(brake_on[left_max:c]))
        exit_ = float(np.sum(brake_on[c:right_max]))
        if entry + exit_ > 0:
            ratios.append(entry / (entry + exit_))  # fraction of in-corner braking before apex
    _require("F25", len(ratios) > 0, "no corner has braking to segment")
    return _checked("F25", float(np.mean(ratios)), cfg)


_GEOM_FUNCS = {"F19": _f19, "F20": _f20, "F21": _f21, "F25": _f25}


def compute_geometry_features(
    lap: pd.DataFrame,
    *,
    lap_number: int,
    config: AppConfig,
    log_path: str | Path | None = None,
    entity_id: str | None = None,
) -> GeometryFeatureResult:
    """Compute the four X/Y-derived cornering features (F19, F20, F21, F25) for one lap.

    If the X/Y trace is unavailable, all four are NULL, one ``CORNER_001`` row is logged, and
    the lap **continues** (§4.4) — F03b never rejects a lap; the 6-NULL ceiling (F03a) decides
    rejection. With X/Y present, each feature is computed independently: a
    ``FeatureComputationError`` (out of range — no clamp — or NULL trigger) or any unexpected
    error is caught, the feature NULL'd, and one ``FEATURE_001`` row logged. Never raises.
    """
    path = Path(log_path) if log_path is not None else Path(config.paths.data_quality_log)
    ledger = _DQLedger(log_path=path)
    eid = entity_id or f"lap{lap_number}"

    fcfg = config.features
    if fcfg is None:
        ledger.record(
            code=ErrorCode.CONFIG_001,
            entity_id=eid,
            detail="config.features block absent; cannot compute geometry features",
            fallback_used=False,
        )
        return GeometryFeatureResult(
            lap_number=lap_number,
            features={fid: None for fid in GEOMETRY_FEATURE_IDS},
            null_features=list(GEOMETRY_FEATURE_IDS),
            status=Status.DEGRADED,
            error_code=ErrorCode.CONFIG_001,
            logged=True,
        )

    if not _xy_available(lap, fcfg):
        # §4.4: corner decomposition skipped; the four cornering features NULL, lap continues.
        ledger.record(
            code=ErrorCode.CORNER_001,
            entity_id=eid,
            detail="X/Y position trace unavailable; cornering features F19,F20,F21,F25 NULL",
            fallback_used=False,
        )
        return GeometryFeatureResult(
            lap_number=lap_number,
            features={fid: None for fid in GEOMETRY_FEATURE_IDS},
            null_features=list(GEOMETRY_FEATURE_IDS),
            detail={fid: {"reason": REASON_XY_UNAVAILABLE} for fid in GEOMETRY_FEATURE_IDS},
            status=Status.DEGRADED,
            error_code=ErrorCode.CORNER_001,
            logged=True,
        )

    features: dict[str, float | None] = {}
    null_features: list[str] = []
    detail: dict[str, dict[str, object]] = {}

    try:
        ch = _extract_channels(lap)
        x = _numeric(lap["X"])
        y = _numeric(lap["Y"])
        time, dt_ms = _resolve_time_base(lap, None, len(lap), fcfg)
        if time is not None and time.size >= 2 and np.isfinite(time).all():
            duration_s: float | None = float(time[-1] - time[0])
        elif dt_ms is not None:
            duration_s = dt_ms / 1000.0 * (len(lap) - 1)
        else:
            duration_s = None
        gctx = _GeometryContext(
            signed_g=_signed_lateral_g(x, y, ch.speed, fcfg),
            speed=ch.speed,
            brake=ch.brake,
            corners=_corner_indices(ch.speed, fcfg),
            duration_s=duration_s,
        )
    except Exception as exc:  # noqa: BLE001 — derived-context build must not propagate (#1)
        for fid in GEOMETRY_FEATURE_IDS:
            ledger.record(
                code=ErrorCode.FEATURE_001,
                entity_id=eid,
                detail=f"{fid} NULL (UNEXPECTED building geometry context): {exc!r}",
                fallback_used=False,
            )
        return GeometryFeatureResult(
            lap_number=lap_number,
            features={fid: None for fid in GEOMETRY_FEATURE_IDS},
            null_features=list(GEOMETRY_FEATURE_IDS),
            detail={
                fid: {"reason": REASON_UNEXPECTED, "error": repr(exc)}
                for fid in GEOMETRY_FEATURE_IDS
            },
            status=Status.DEGRADED,
            logged=True,
        )

    for fid in GEOMETRY_FEATURE_IDS:
        try:
            features[fid] = _GEOM_FUNCS[fid](gctx, fcfg)
        except FeatureComputationError as exc:
            features[fid] = None
            null_features.append(fid)
            detail[fid] = {"reason": exc.reason, "detail": exc.detail}
            ledger.record(
                code=ErrorCode.FEATURE_001,
                entity_id=eid,
                detail=f"{fid} NULL ({exc.reason}): {exc.detail}",
                fallback_used=False,
            )
        except Exception as exc:  # noqa: BLE001 — last-resort: NULL, never propagate (#1)
            features[fid] = None
            null_features.append(fid)
            detail[fid] = {"reason": REASON_UNEXPECTED, "error": repr(exc)}
            ledger.record(
                code=ErrorCode.FEATURE_001,
                entity_id=eid,
                detail=f"{fid} NULL (UNEXPECTED): {exc!r}",
                fallback_used=False,
            )

    status = Status.OK if not null_features else Status.DEGRADED
    return GeometryFeatureResult(
        lap_number=lap_number,
        features=features,
        null_features=null_features,
        detail=detail,
        status=status,
        logged=ledger.rows > 0,
    )


def _stint_slope(
    stint: list[StintLap], attr: str, cfg: FeaturesConfig
) -> tuple[float | None, str | None]:
    """Degree-1 polyfit slope of ``attr`` (F01 or F06) vs tyre age across a stint's usable laps.

    A *usable* lap has a known tyre age (not UNKNOWN) and a non-NULL feature value. Returns
    ``(None, reason)`` when there are no known-tyre-age laps, fewer than ``min_stint_laps``
    usable laps, or the tyre ages are constant (slope undefined); else ``(slope, None)``.
    Unconstrained — the slope is never range-gated.
    """
    known_age = [lap for lap in stint if lap.tyre_age is not None]
    if not known_age:
        return None, "tyre age UNKNOWN"
    ages: list[float] = []
    vals: list[float] = []
    for lap in known_age:
        value = getattr(lap, attr)
        if value is None:
            continue  # known tyre age but this feature is NULL for the lap; excluded
        ages.append(float(lap.tyre_age))  # type: ignore[arg-type]
        vals.append(float(value))
    if len(vals) < cfg.min_stint_laps:
        return None, f"{len(vals)} usable laps < {cfg.min_stint_laps}"
    age_arr = np.asarray(ages)
    if float(np.ptp(age_arr)) == 0.0:
        return None, "tyre age constant across stint; slope undefined"
    return float(np.polyfit(age_arr, np.asarray(vals), 1)[0]), None


def compute_degradation_slopes(
    stint: list[StintLap],
    *,
    stint_id: int,
    config: AppConfig,
    log_path: str | Path | None = None,
) -> DegradationResult:
    """Compute the stint-level tyre-age slopes F27 (throttle_mean) and F28 (brake_peak_mean).

    Computes only when the stint has >= ``min_stint_laps`` usable laps with known tyre age,
    else NULL (one ``FEATURE_001`` row per lap of the stint, per NULL'd slope). The slopes are
    broadcast to every lap of the stint via ``applies_to``. Never raises for a data fault.
    """
    path = Path(log_path) if log_path is not None else Path(config.paths.data_quality_log)
    ledger = _DQLedger(log_path=path)
    applies = [lap.lap_number for lap in stint]

    if not stint:
        # Degenerate input — log at stint level so the NULL slopes are never unaudited (#1).
        ledger.record(
            code=ErrorCode.FEATURE_001,
            entity_id=f"stint{stint_id}",
            detail="empty stint; F27/F28 NULL",
            fallback_used=False,
        )
        return DegradationResult(
            stint_id=stint_id, applies_to=[], null_reason="empty stint",
            status=Status.DEGRADED, logged=True,
        )

    fcfg = config.features
    if fcfg is None:
        for lap_no in applies:
            ledger.record(
                code=ErrorCode.CONFIG_001,
                entity_id=f"lap{lap_no}",
                detail="config.features block absent; cannot compute degradation slopes",
                fallback_used=False,
            )
        return DegradationResult(
            stint_id=stint_id,
            applies_to=applies,
            null_reason="config.features absent",
            status=Status.DEGRADED,
            error_code=ErrorCode.CONFIG_001,
            logged=bool(applies),
        )

    try:
        f27, r27 = _stint_slope(stint, "throttle_mean", fcfg)
        f28, r28 = _stint_slope(stint, "brake_peak_mean", fcfg)
    except Exception as exc:  # noqa: BLE001 — last-resort: NULL both, never propagate (#1)
        f27, f28 = None, None
        r27 = r28 = f"unexpected: {exc!r}"

    for lap_no in applies:
        if f27 is None:
            ledger.record(
                code=ErrorCode.FEATURE_001,
                entity_id=f"lap{lap_no}",
                detail=f"F27 NULL: {r27}",
                fallback_used=False,
            )
        if f28 is None:
            ledger.record(
                code=ErrorCode.FEATURE_001,
                entity_id=f"lap{lap_no}",
                detail=f"F28 NULL: {r28}",
                fallback_used=False,
            )

    null_reason = "; ".join(r for r in (r27, r28) if r) or None
    status = Status.OK if f27 is not None and f28 is not None else Status.DEGRADED
    return DegradationResult(
        stint_id=stint_id,
        f27=f27,
        f28=f28,
        applies_to=applies,
        null_reason=null_reason,
        status=status,
        logged=ledger.rows > 0,
    )
