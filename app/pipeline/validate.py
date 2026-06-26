"""F02 — lap filter chain / validation contract (master spec §3.2, §3.3).

The data-quality gate between F01's cached telemetry and F03 feature extraction.
Every lap is run through nine filters *in order*; the **first failing filter rejects
the lap** with its specific ``ErrorCode``, while two filters tag-and-accept
(``LAP_INTERP``, ``LAP_PARTIAL``). After all laps are filtered, a driver-circuit
combination with fewer than ``thresholds.min_clean_laps`` clean laps is excluded.

This module honours the constitution's first non-negotiable — *no component fails
silently*. The validator NEVER raises an unhandled exception (§8.2): every rejection,
tag, and exclusion is caught, coded, written as one row to the ``data_quality_log``,
and returned inside a typed result object. Nothing is dropped without an audit row.

F01 → F02 handoff
-----------------
F01 persists one wide Parquet per session (one row per telemetry sample), with the
per-lap context columns left-joined on ``lap``: ``track_status`` (raw FastF1
``Laps.TrackStatus`` token — *this* module owns the green-flag interpretation),
``pit_in`` / ``pit_out``, ``lap_time``, ``tyre_age``. :func:`validate_session` reads
that frame directly and groups it by ``lap`` via an internal adapter — there is no
restructuring of F01's output.

Thresholds are read from ``config.yaml``: the F02-specific ones via
``config.lap_validation`` (a :class:`ValidateConfig`), and the three shared bounds —
``sample_rate_floor_hz`` (20 Hz), ``laptime_bound_pct`` (115 %), ``min_clean_laps``
(30) — via ``config.thresholds``. There is no numeric literal for any threshold here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from app.contracts.config import AppConfig, ValidateConfig
from app.contracts.enums import LapStatus, Status
from app.contracts.errors import ErrorCode
from app.contracts.logging import DataQualityLogError, log_data_quality
from app.contracts.results import BaseResult

# ----------------------------------------------------------------------------------
# Result objects
# ----------------------------------------------------------------------------------


class LapValidationResult(BaseModel):
    """Outcome of running one lap through the filter chain.

    Standalone (not a :class:`BaseResult`): a lap's axis is the binary
    ``ACCEPTED`` / ``REJECTED`` of :class:`LapStatus`, which the OK/DEGRADED/FAILED
    ``Status`` trichotomy would misrepresent. It still carries ``logged`` so the
    no-silent-failure audit discipline holds at the per-lap level.
    """

    model_config = ConfigDict(extra="forbid")

    lap_number: int
    status: LapStatus
    error_code: ErrorCode = ErrorCode.OK  # OK when the lap is clean / accepted
    flags: list[ErrorCode] = Field(default_factory=list)  # accept-time tags
    null_features: list[str] = Field(default_factory=list)  # e.g. ["F27","F28"] on LAP_PARTIAL
    detail: dict[str, object] = Field(default_factory=dict)  # channel, dropout_pct, rate, ...
    logged: bool = False

    @property
    def accepted(self) -> bool:
        """True when the lap enters the feature matrix."""
        return self.status is LapStatus.ACCEPTED


class ValidationResult(BaseResult):
    """Per driver-circuit combination outcome (§8.2 Validator).

    ``status`` is ``OK`` normally and ``FAILED`` when the combination is excluded;
    in the excluded case ``error_code`` is ``COMBO_001`` (the filter-chain action
    code) and the combination's ``data_quality_log`` row carries ``DATA_INSUFFICIENT``
    (§3.2). Clean laps are still returned even when the combination is excluded.
    """

    season: int
    circuit: str
    driver: str
    clean_laps: list[LapValidationResult] = Field(default_factory=list)
    rejected_laps: list[LapValidationResult] = Field(default_factory=list)
    clean_lap_count: int = 0
    combination_excluded: bool = False


# ----------------------------------------------------------------------------------
# data_quality_log ledger (per-lap / per-combination)
# ----------------------------------------------------------------------------------


@dataclass
class _DQLedger:
    """Writes one ``data_quality_log`` row per rejection / tag / exclusion.

    Unlike F01's session-scoped ledger, F02 logs at two granularities (``"lap"`` and
    ``"combination"``), so ``entity_type``/``entity_id`` are passed per call.
    """

    log_path: Path
    rows: int = 0

    def record(
        self,
        *,
        code: ErrorCode,
        entity_type: str,
        entity_id: str,
        detail: str,
        fallback_used: bool,
    ) -> None:
        log_data_quality(
            error_code=code,
            entity_type=entity_type,
            entity_id=entity_id,
            error_detail=detail,
            fallback_used=fallback_used,
            log_path=self.log_path,
        )
        self.rows += 1


# ----------------------------------------------------------------------------------
# Lap adapter (resolves the wide-frame -> per-lap divergence)
# ----------------------------------------------------------------------------------


@dataclass
class _LapView:
    """A single lap's telemetry + per-lap context, grouped out of the wide frame."""

    lap_number: int
    channels: dict[str, np.ndarray]  # required-channel sample arrays (float, NaN-padded)
    track_status: object  # raw FastF1 token (or null marker)
    pit_in: bool
    pit_out: bool
    lap_time: object  # seconds (or null marker)
    tyre_age: object  # value (or null marker -> LAP_PARTIAL)
    n_samples: int


_FALSY_STRINGS = {"false", "0", "0.0", "no", "f", "n"}


def _is_missing(value: object) -> bool:
    """True for a null-ish per-lap scalar (NA / None / NaN / empty / sentinel)."""
    if value is None:
        return True
    try:
        if bool(pd.isna(value)):
            return True
    except (TypeError, ValueError):
        # pd.isna is ambiguous on exotic/array-like types -> fall through to the
        # string-sentinel check below. Not a swallowed data path (non-negotiable #1).
        pass
    return isinstance(value, str) and value.strip().lower() in {"", "nan", "none", "<na>"}


def _as_bool(value: object) -> bool:
    """Coerce a per-lap flag scalar to bool; null markers and falsy strings are False.

    Robust to a flag arriving as an object/string column (e.g. ``"False"`` / ``"0"``
    from a re-serialised cache), which ``bool()`` would otherwise read as truthy and
    mis-classify a green lap as an in/out lap (LAP_002).
    """
    if _is_missing(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() not in _FALSY_STRINGS
    return bool(value)


def _channel_array(sub: pd.DataFrame, channel: str, n: int) -> np.ndarray:
    """Numeric sample array for ``channel``; an absent channel is treated as all-NaN."""
    if channel not in sub.columns:
        return np.full(n, np.nan, dtype=float)
    return pd.to_numeric(sub[channel], errors="coerce").to_numpy(dtype=float)


def _first_scalar(sub: pd.DataFrame, column: str) -> object:
    """First value of a per-lap context column (constant per lap), or None if absent."""
    if column not in sub.columns or len(sub) == 0:
        return None
    return sub[column].iloc[0]


def _group_session(df: pd.DataFrame, vcfg: ValidateConfig) -> list[_LapView]:
    """Group the wide F01 frame into one :class:`_LapView` per lap (sorted by lap)."""
    if "lap" not in df.columns:
        return []
    views: list[_LapView] = []
    for lap_number, sub in df.groupby("lap", sort=True):
        n = len(sub)
        channels = {ch: _channel_array(sub, ch, n) for ch in vcfg.required_channels}
        views.append(
            _LapView(
                lap_number=int(lap_number),  # type: ignore[arg-type]
                channels=channels,
                track_status=_first_scalar(sub, "track_status"),
                pit_in=_as_bool(_first_scalar(sub, "pit_in")),
                pit_out=_as_bool(_first_scalar(sub, "pit_out")),
                lap_time=_first_scalar(sub, "lap_time"),
                tyre_age=_first_scalar(sub, "tyre_age"),
                n_samples=n,
            )
        )
    return views


def _racing_median_laptime(views: list[_LapView], vcfg: ValidateConfig) -> float | None:
    """Median lap time over *racing* laps — green-flag and non-pit — as the LAP_006
    reference. In/out laps and SC/VSC laps are slow and would inflate the median,
    making the 115% plausibility bound too lenient, so they are excluded here even
    though the per-lap filters reject them later (master §3.3: "the session's median
    lap time for that driver"). Reuses the grouped views — no second groupby pass.
    """
    times: list[float] = []
    for view in views:
        if _is_missing(view.lap_time):
            continue
        tokens = _track_tokens(view.track_status)
        if not _is_green(tokens, vcfg) or view.pit_in or view.pit_out:
            continue
        times.append(float(view.lap_time))  # type: ignore[arg-type]
    if not times:
        return None
    return float(np.median(times))


def _is_green(tokens: str | None, vcfg: ValidateConfig) -> bool:
    """True iff every track-status token is the configured green code (and present)."""
    if not tokens:
        return False
    return all(ch == vcfg.green_track_status_code for ch in tokens)


# ----------------------------------------------------------------------------------
# Telemetry helpers (vectorised — no per-sample Python loops)
# ----------------------------------------------------------------------------------


def _nan_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return ``(start, length)`` for each maximal run of True in a boolean mask."""
    if not mask.any():
        return []
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(padded))
    starts = edges[0::2]
    ends = edges[1::2]
    return [(int(s), int(e - s)) for s, e in zip(starts, ends, strict=True)]


def _max_consecutive_nan(arr: np.ndarray) -> int:
    """Longest run of consecutive NaN samples in ``arr``."""
    return max((length for _, length in _nan_runs(np.isnan(arr))), default=0)


def _track_tokens(value: object) -> str | None:
    """Normalise a raw track-status scalar to its token string, or None if missing.

    FastF1 ``TrackStatus`` tokens are single digits (possibly concatenated, e.g. "12").
    An integer-valued numeric — whether a real ``float`` (``1.0``) or an upstream
    stringified float (``"1.0"``) — collapses to its integer form so a genuinely-green
    lap is not mis-rejected as LAP_001.
    """
    if _is_missing(value):
        return None
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float):
        return str(int(value)) if float(value).is_integer() else str(value)
    s = str(value).strip()
    try:
        f = float(s)
    except ValueError:
        return s
    return str(int(f)) if f.is_integer() else s


def _interpolate_small_gaps(arr: np.ndarray, max_gap: int) -> tuple[np.ndarray, int]:
    """Linear-interpolate interior NaN runs of length ``1..max_gap``.

    A run is only filled if it has finite neighbours on both sides — an edge gap
    (touching the start/end of the lap) has no two endpoints to interpolate between
    and is left as NaN. Returns ``(filled_copy, largest_gap_filled)``; the original
    array is untouched. Runs longer than ``max_gap`` are left for downstream NULL
    handling (LAP_003 already rejected anything > the consecutive-NaN bound).
    """
    mask = np.isnan(arr)
    if not mask.any():
        return arr, 0
    out = arr.copy()
    largest = 0
    n = arr.size
    for start, length in _nan_runs(mask):
        if length > max_gap:
            continue
        left, right = start - 1, start + length
        if left < 0 or right >= n or np.isnan(out[left]) or np.isnan(out[right]):
            continue
        out[start:right] = np.linspace(out[left], out[right], length + 2)[1:-1]
        largest = max(largest, length)
    return out, largest


# ----------------------------------------------------------------------------------
# Per-lap filter chain
# ----------------------------------------------------------------------------------


def _validate_lap(
    view: _LapView,
    *,
    session_median: float | None,
    cfg: AppConfig,
    vcfg: ValidateConfig,
    ledger: _DQLedger,
    entity_prefix: str,
) -> LapValidationResult:
    """Run one lap through the 9 filters in order; first failure rejects."""
    entity_id = f"{entity_prefix}/lap{view.lap_number}"
    n = view.n_samples

    def reject(code: ErrorCode, detail: dict[str, object], message: str) -> LapValidationResult:
        ledger.record(
            code=code, entity_type="lap", entity_id=entity_id, detail=message, fallback_used=False
        )
        return LapValidationResult(
            lap_number=view.lap_number,
            status=LapStatus.REJECTED,
            error_code=code,
            detail=detail,
            logged=True,
        )

    # Filter 1 — LAP_001: green flag. Missing/unknown status is REJECTED, never green.
    tokens = _track_tokens(view.track_status)
    if tokens is None or tokens == "":
        return reject(
            ErrorCode.LAP_001, {"reason": "track status unavailable"}, "track status unavailable"
        )
    if not _is_green(tokens, vcfg):
        return reject(
            ErrorCode.LAP_001,
            {"reason": "not green-flag", "track_status": tokens},
            f"not green-flag (track_status={tokens})",
        )

    # Filter 2 — LAP_002: outlap / inlap (pit in/out event).
    if view.pit_in or view.pit_out:
        return reject(
            ErrorCode.LAP_002,
            {"reason": "in/out lap", "pit_in": view.pit_in, "pit_out": view.pit_out},
            "outlap/inlap (pit event)",
        )

    # Filter 3 — LAP_003: telemetry completeness (major). Reject > N% consecutive NaN.
    if n == 0:
        return reject(ErrorCode.LAP_003, {"reason": "no telemetry samples"}, "empty lap: 0 samples")
    for channel in vcfg.required_channels:
        run = _max_consecutive_nan(view.channels[channel])
        if run * 100.0 > vcfg.max_consecutive_nan_pct * n:
            dropout_pct = round(run / n * 100.0, 4)
            return reject(
                ErrorCode.LAP_003,
                {"channel": channel, "dropout_pct": dropout_pct},
                f"{channel} consecutive-NaN {dropout_pct}% > {vcfg.max_consecutive_nan_pct}%",
            )

    # Filter 4 — LAP_INTERP: fill 1..max_gap NaN runs. The fill happens now so F02's
    # own later filters (LAP_004 rate, LAP_005 stationary) see the repaired signal. It
    # is local to this view and NOT persisted to the returned result (F02 emits verdicts,
    # not telemetry); the LAP_INTERP flag tells F03 the lap had small gaps, and F03
    # re-applies the same config-driven interpolation when it reads the lap. The tag is
    # deferred to accept time because LAP_INTERP is, per §3.3, an accept action.
    interpolated: list[str] = []
    max_gap_filled = 0
    for channel in vcfg.required_channels:
        filled, largest = _interpolate_small_gaps(
            view.channels[channel], vcfg.interp_gap_max_samples
        )
        if largest > 0:
            view.channels[channel] = filled
            interpolated.append(channel)
            max_gap_filled = max(max_gap_filled, largest)

    # Filter 5 — LAP_004: effective sampling rate after interpolation. Proxy is
    # samples / lap_time (F01 persists no per-sample clock); lap_time is required.
    if _is_missing(view.lap_time) or float(view.lap_time) <= 0:  # type: ignore[arg-type]
        return reject(
            ErrorCode.LAP_004,
            {"reason": "lap_time unavailable", "sampling_rate_hz": None},
            "lap_time unavailable; cannot compute sampling rate",
        )
    effective_hz = n / float(view.lap_time)  # type: ignore[arg-type]
    if effective_hz < cfg.thresholds.sample_rate_floor_hz:
        return reject(
            ErrorCode.LAP_004,
            {"sampling_rate_hz": round(effective_hz, 4)},
            f"{effective_hz:.4f} Hz < {cfg.thresholds.sample_rate_floor_hz} Hz",
        )

    # Filter 6 — LAP_005: speed/stationary anomaly. Stationary >= max fraction rejects.
    speed = view.channels["Speed"]
    stationary_count = int(np.sum(speed <= vcfg.min_moving_speed_kmh))  # NaN <= x is False
    if stationary_count / n >= vcfg.max_stationary_fraction:
        stationary_fraction = round(stationary_count / n, 6)
        return reject(
            ErrorCode.LAP_005,
            {"stationary_fraction": stationary_fraction},
            f"stationary fraction {stationary_fraction} >= {vcfg.max_stationary_fraction}",
        )

    # Filter 7 — LAP_006: lap-time plausibility. Strictly > 115% of session median rejects.
    if session_median is not None:
        bound = cfg.thresholds.laptime_bound_pct / 100.0
        if float(view.lap_time) > bound * session_median:  # type: ignore[arg-type]
            return reject(
                ErrorCode.LAP_006,
                {"lap_time_s": float(view.lap_time), "session_median_s": session_median},  # type: ignore[arg-type]
                f"lap_time {float(view.lap_time)}s > {cfg.thresholds.laptime_bound_pct}% of "  # type: ignore[arg-type]
                f"median {session_median}s",
            )

    # Filter 8 — LAP_PARTIAL: tyre age UNKNOWN -> accept, F27/F28 NULL for this lap.
    null_features: list[str] = []
    if _is_missing(view.tyre_age):
        null_features = list(vcfg.degradation_features)

    # --- Lap ACCEPTED. Emit accept-time tags, one data_quality_log row each. -------
    flags: list[ErrorCode] = []
    detail: dict[str, object] = {}
    if interpolated:
        flags.append(ErrorCode.LAP_INTERP)
        detail["interpolated_channels"] = interpolated
        detail["max_gap_filled"] = max_gap_filled
        ledger.record(
            code=ErrorCode.LAP_INTERP,
            entity_type="lap",
            entity_id=entity_id,
            detail=f"interpolated {interpolated} (max gap {max_gap_filled} samples)",
            fallback_used=True,
        )
    if null_features:
        flags.append(ErrorCode.LAP_PARTIAL)
        ledger.record(
            code=ErrorCode.LAP_PARTIAL,
            entity_type="lap",
            entity_id=entity_id,
            detail=f"tyre age unknown; {null_features} NULL for this lap",
            fallback_used=True,
        )

    return LapValidationResult(
        lap_number=view.lap_number,
        status=LapStatus.ACCEPTED,
        error_code=ErrorCode.OK,
        flags=flags,
        null_features=null_features,
        detail=detail,
        logged=bool(flags),
    )


# ----------------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------------


def validate_session(
    telemetry: pd.DataFrame,
    *,
    season: int,
    circuit: str,
    driver: str,
    config: AppConfig,
    log_path: str | Path | None = None,
) -> ValidationResult:
    """Validate every lap of one driver-circuit session against the 9 filters.

    Args:
        telemetry: F01's wide cached frame (one row per sample, per-lap context joined).
        season / circuit / driver: identity of the combination being validated.
        config: loaded :class:`AppConfig` (reads ``lap_validation`` + ``thresholds``).
        log_path: ``data_quality_log`` sink; defaults to ``config.paths.data_quality_log``.

    Returns:
        A :class:`ValidationResult`. Never raises for a data fault — every failure is
        coded and logged (§8.2).
    """
    path = Path(log_path) if log_path is not None else Path(config.paths.data_quality_log)
    ledger = _DQLedger(log_path=path)
    entity_prefix = f"{season}/{circuit}/{driver}"

    vcfg = config.lap_validation
    if vcfg is None:
        # Config block absent — typed failure, no bare raise (#1).
        ledger.record(
            code=ErrorCode.CONFIG_001,
            entity_type="combination",
            entity_id=entity_prefix,
            detail="config.validate block absent; cannot validate laps",
            fallback_used=False,
        )
        return ValidationResult(
            season=season,
            circuit=circuit,
            driver=driver,
            status=Status.FAILED,
            error_code=ErrorCode.CONFIG_001,
            logged=True,
        )

    # Structural guard: a frame with no `lap` column is a malformed F01 handoff, not a
    # genuinely empty session — surface it distinctly (VALIDATION_ERROR) so an operator
    # is not misled by a DATA_INSUFFICIENT "driver lacked laps" reading.
    if "lap" not in telemetry.columns:
        ledger.record(
            code=ErrorCode.VALIDATION_ERROR,
            entity_type="combination",
            entity_id=entity_prefix,
            detail="input frame has no 'lap' column; cannot group laps",
            fallback_used=False,
        )
        return ValidationResult(
            season=season, circuit=circuit, driver=driver,
            status=Status.FAILED, error_code=ErrorCode.VALIDATION_ERROR, logged=True,
        )

    # Never drop a row silently (#1): groupby("lap") discards samples whose `lap` key is
    # null, so account for them with one audit row before they vanish.
    orphan_mask = telemetry["lap"].isna()
    n_orphan = int(orphan_mask.sum())
    if n_orphan:
        ledger.record(
            code=ErrorCode.ORPHAN_SAMPLES,
            entity_type="session",
            entity_id=entity_prefix,
            detail=f"{n_orphan} telemetry samples have no lap key; dropped before validation",
            fallback_used=False,
        )
        telemetry = cast("pd.DataFrame", telemetry.loc[~orphan_mask])

    views = _group_session(telemetry, vcfg)
    session_median = _racing_median_laptime(views, vcfg)
    clean: list[LapValidationResult] = []
    rejected: list[LapValidationResult] = []

    for view in views:
        try:
            result = _validate_lap(
                view,
                session_median=session_median,
                cfg=config,
                vcfg=vcfg,
                ledger=ledger,
                entity_prefix=entity_prefix,
            )
        except DataQualityLogError:
            # The audit sink itself is unwritable (DATA_QUALITY_LOG_UNWRITABLE). Coding-
            # and-logging is impossible when logging is down, and validating laps with no
            # audit trail would itself be a silent failure — so this infra fault is
            # surfaced per the logging contract, not swallowed.
            raise
        except Exception as exc:  # noqa: BLE001 — never-raise for DATA faults (§8.2)
            # Backstop for an unexpected validator/internal error (the filters above are
            # total, so this should not fire). Coded VALIDATION_ERROR — NOT a per-lap data
            # code like LAP_003 — so the DQ log distinguishes a code bug from genuine bad
            # telemetry. The lap is rejected (coded + logged), never dropped; session
            # continues to the next lap (no cascade failure).
            ledger.record(
                code=ErrorCode.VALIDATION_ERROR,
                entity_type="lap",
                entity_id=f"{entity_prefix}/lap{view.lap_number}",
                detail=f"unexpected validation error: {exc!r}",
                fallback_used=False,
            )
            result = LapValidationResult(
                lap_number=view.lap_number,
                status=LapStatus.REJECTED,
                error_code=ErrorCode.VALIDATION_ERROR,
                detail={"reason": "unexpected validation error", "error": repr(exc)},
                logged=True,
            )
        (clean if result.accepted else rejected).append(result)

    clean_count = len(clean)
    excluded = clean_count < config.thresholds.min_clean_laps
    if excluded:
        ledger.record(
            code=ErrorCode.DATA_INSUFFICIENT,
            entity_type="combination",
            entity_id=entity_prefix,
            detail=f"only {clean_count} clean laps (< {config.thresholds.min_clean_laps})",
            fallback_used=False,
        )

    return ValidationResult(
        season=season,
        circuit=circuit,
        driver=driver,
        status=Status.FAILED if excluded else Status.OK,
        error_code=ErrorCode.COMBO_001 if excluded else None,
        clean_laps=clean,
        rejected_laps=rejected,
        clean_lap_count=clean_count,
        combination_excluded=excluded,
        logged=ledger.rows > 0,
    )
