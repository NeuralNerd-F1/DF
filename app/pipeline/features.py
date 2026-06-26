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

from app.contracts.config import SCALAR_FEATURE_IDS, AppConfig, FeaturesConfig
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
