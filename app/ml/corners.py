"""F09 — Corner-type decomposition + sub-fingerprints (ablation): master §4.4.

Segments each lap's corners into three categories by **apex radius** — **slow** (< 100 m),
**medium** (100–200 m), **high-speed** (> 200 m) — reusing F03b's **curvature** primitive
(κ = |x'y'' − y'x''| / (x'² + y'²)^1.5) and F03a's `_corner_indices` + the F25 speed-maxima corner
windows, so a "corner" means the same thing everywhere. Each corner type yields a 5-feature
**sub-vector** (apex speed, lateral G, throttle, brake, steering-reversal rate); aggregated per
`(driver, corner_type)` these are the Screen-2 **sub-fingerprints**. The headline research
deliverable is the **ablation** (§13/§15): does adding the corner sub-features to the lap vector
**improve the classifier's driver separation**? — answered by a self-contained F06a-CV comparison of
**base (28) vs expanded (28 + 15)** matrices (it does NOT retrain the shipped champion).

Failsafe (§4.4): when the X/Y trace is unavailable the decomposition is **skipped**, `CORNER_001` is
logged, and the lap **continues** (its F01–F26 are F03's, untouched) — F03b's exact behaviour. F09
adds **no new error code**. F09 is **F04-free** (the corner sub-features are new derived quantities,
not F04's 28 lap features; XGBoost is scale-invariant and F19 min-max-scales the radar at render).
`decompose_lap` / `run_ablation` / `health` **never raise**; `build_sub_fingerprints` /
`load_sub_fingerprints` raise typed build/load errors. Every threshold/filename is from ``config``.
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
import xgboost as xgb
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from app.contracts.config import AppConfig, CornersConfig
from app.contracts.enums import Status
from app.contracts.errors import ErrorCode
from app.contracts.logging import DataQualityLogError, log_data_quality
from app.contracts.results import BaseResult
from app.ml.classifier import _base_xgb_params
from app.pipeline.features import (
    FEATURE_IDS,
    _corner_indices,
    _extract_channels,
    _numeric,
    _resolve_time_base,
    _signed_lateral_g,
    _xy_available,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

# --- Structural constants (NOT tunable thresholds — those live in config) ------------------
SCHEMA_VERSION = "1.0"
_VERSION_HASH_LEN = 16

# §4.1 (divergence #19) — the corner-scoped sub-features carrying a driver's cornering signature.
CORNER_SUB_FEATURES: tuple[str, ...] = (
    "apex_speed_mean",
    "lateral_g_mean",
    "throttle_mean",
    "brake_mean",
    "steering_reversal_rate",
)

COL_DRIVER = "driver"
COL_CIRCUIT = "circuit"
COL_SEASON = "season"
COL_SPLIT = "split"

# data_quality_log detail tags.
REASON_XY_UNAVAILABLE = "X_Y_UNAVAILABLE"
REASON_SUBFEATURE_NULL = "CORNER_SUBFEATURE_NULL"
REASON_THIN_CELL = "CELL_BELOW_MIN_CORNERS"
REASON_NO_CELLS = "NO_USABLE_SUBFINGERPRINT_CELLS"
REASON_BACKSTOP = "UNFORESEEN_FAULT"
REASON_INVALID = "ABLATION_INPUT_MALFORMED"
REASON_MLFLOW = "MLFLOW_LOGGING_DEGRADED"

_ENTITY_LAP = "lap"
_ENTITY_DRIVER = "driver"
_ENTITY_BUILD = "training_run"
_ENTITY_MODEL = "model"


# ----------------------------------------------------------------------------------
# Feature-owned enum (per enums.py docstring — feature-specific enums live with the feature)
# ----------------------------------------------------------------------------------


class CornerType(StrEnum):
    """§4.4 — the three radius bands (also the master §8.4 ``lap_features.corner_type`` ENUM)."""

    SLOW = "SLOW"  # apex radius < corners.radius_slow_max (100 m)
    MEDIUM = "MEDIUM"  # radius_slow_max ≤ radius < corners.radius_medium_max (200 m)
    HIGH_SPEED = "HIGH_SPEED"  # radius ≥ radius_medium_max


# The 15 flattened corner sub-feature column names (ablation "expanded" matrix, §4.3) + the 3
# per-type corner-count columns (sub-fingerprint weighting, §4.2).
def _sub_col(ct: CornerType, feat: str) -> str:
    return f"{ct.value}_{feat}"


def _n_col(ct: CornerType) -> str:
    return f"{ct.value}_n_corners"


_CORNER_COLUMNS: tuple[str, ...] = tuple(
    _sub_col(ct, f) for ct in CornerType for f in CORNER_SUB_FEATURES
)
_N_COLUMNS: tuple[str, ...] = tuple(_n_col(ct) for ct in CornerType)


# ----------------------------------------------------------------------------------
# Exceptions (only build/load raise; decompose / ablation / health never do)
# ----------------------------------------------------------------------------------


class SubFingerprintBuildError(Exception):
    """Raised by :func:`build_sub_fingerprints` on a build-breaking fault (no artefact written).
    Carries ``TRAIN_LEAKAGE`` or ``MODEL_TRAINING_FAILED``. Fail-loud (#1)."""

    def __init__(self, message: str, *, error_code: ErrorCode) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code


class CornerLoadError(Exception):
    """Raised by :func:`load_sub_fingerprints` when the artefact cannot serve. Carries
    ``MODEL_NOT_FOUND`` — the service refuses to start (F06b/F08/F10/F11 precedent)."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = ErrorCode.MODEL_NOT_FOUND


# ----------------------------------------------------------------------------------
# Typed outputs (all Pydantic, extra="forbid"; BaseResult where a status is carried)
# ----------------------------------------------------------------------------------


class CornerSubVector(BaseModel):
    """One corner type's sub-features for one lap. ``features`` is exactly ``CORNER_SUB_FEATURES``;
    all-None when ``n_corners == 0`` or the feature could not be computed."""

    model_config = ConfigDict(extra="forbid")

    corner_type: CornerType
    n_corners: int
    features: dict[str, float | None]


class CornerDecompositionResult(BaseResult):
    """Per-lap decomposition — never raises. Always carries all three ``CornerType`` sub-vectors."""

    lap_number: int
    sub_vectors: dict[str, CornerSubVector] = Field(default_factory=dict)
    xy_available: bool = False
    n_corners_total: int = 0


class SubFingerprint(BaseModel):
    """Per-``(driver, corner_type)`` aggregate (the Screen-2 radar source)."""

    model_config = ConfigDict(extra="forbid")

    driver: str
    corner_type: CornerType
    features: dict[str, float | None]
    n_corners: int
    n_laps: int


class SubFingerprintFitResult(BaseResult):
    """Build output (mirrors F08/F10/F11 build results)."""

    artefact_path: str = ""
    content_hash: str = ""
    n_drivers: int = 0
    n_entries: int = 0  # (driver, corner_type) cells written
    coverage: dict[str, float] = Field(default_factory=dict)  # corner_type → fraction of drivers
    mlflow_run_id: str | None = None


class AblationReport(BaseResult):
    """§4.4-c ablation — read-only, never raises (formally run in F20)."""

    base_macro_f1: float | None = None
    expanded_macro_f1: float | None = None
    delta: float | None = None  # expanded − base
    per_driver_delta: dict[str, float] = Field(default_factory=dict)
    finding: str = ""  # "IMPROVES" | "NEUTRAL" | "DEGRADES"
    n_base_features: int = 0
    n_expanded_features: int = 0
    mlflow_run_id: str | None = None


class CornerHealthResult(BaseResult):
    """Readiness for F13's ``GET /health`` (mirrors F06b/F08/F10/F11)."""

    ready: bool = False
    model_version: str = ""
    reason: ErrorCode | None = None


# ----------------------------------------------------------------------------------
# On-disk artefact model (Pydantic → schema-mismatch detection for free, like F04/F10)
# ----------------------------------------------------------------------------------


class _CellEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    driver: str
    corner_type: str
    features: dict[str, float | None]
    n_corners: int
    n_laps: int


class _SubFingerprintArtefact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str
    sub_features: list[str]
    cells: list[_CellEntry]
    content_hash: str


@dataclass(frozen=True)
class CornerHandle:
    """The loaded, immutable serving bundle (built once at startup). F04-free (§4.0-f)."""

    cells: dict[tuple[str, str], SubFingerprint]  # (driver, corner_type) → sub-fingerprint
    sub_features: tuple[str, ...]
    model_version: str
    ready: bool = True


# ----------------------------------------------------------------------------------
# data-quality ledger (never propagates a sink failure — mirrors F04/F06/F08/F10/F11)
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


def _dedup(flags: Iterable[ErrorCode]) -> list[ErrorCode]:
    return list(dict.fromkeys(flags))


# ----------------------------------------------------------------------------------
# Pure geometry helpers — curvature + corner windows (reuse F03b's exact formula / segmentation)
# ----------------------------------------------------------------------------------


def _curvature(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-sample curvature κ = |x'y'' − y'x''| / (x'² + y'²)^1.5 — the SAME formula F03b's
    :func:`_signed_lateral_g` uses (gradients via ``np.gradient``). The corner **radius = 1/κ**."""
    x1, y1 = np.gradient(x), np.gradient(y)
    x2, y2 = np.gradient(x1), np.gradient(y1)
    cross = x1 * y2 - y1 * x2
    denom = (x1**2 + y1**2) ** 1.5
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0, np.abs(cross) / denom, 0.0)


def _corner_windows(speed: np.ndarray, corners: np.ndarray) -> list[tuple[int, int, int]]:
    """``(apex, left_max, right_max)`` per corner — **replicates F25's segmentation exactly**
    (``features.py:920-929``): the speed maxima bracketing the apex, half-open ``[left_max,
    right_max)`` so the shared inter-corner maximum is never double-counted. A parity test pins this
    against F03b (F09 never re-invents the corner boundaries)."""
    n = int(speed.size)
    out: list[tuple[int, int, int]] = []
    for i, c in enumerate(corners):
        c = int(c)
        left_bound = int(corners[i - 1]) if i > 0 else 0
        right_bound = int(corners[i + 1]) if i + 1 < corners.size else n - 1
        left_max = (
            left_bound + int(np.argmax(speed[left_bound : c + 1])) if c > left_bound else left_bound
        )
        right_max = (
            c + int(np.argmax(speed[c : right_bound + 1])) if right_bound > c else right_bound
        )
        out.append((c, left_max, right_max))
    return out


def _apex_radius(kappa: np.ndarray, apex: int, lm: int, rm: int) -> float | None:
    """The corner's characteristic **radius = 1/κ at the apex** — the speed-minimum sample is the
    corner's defining tightest point, and a mid-window central-difference κ is robust to the
    boundary/window transition artefacts that dominate ``max(κ)`` (divergence #18). Falls back to
    max over the window only if the apex κ is non-finite; ``None`` when neither is a real curvature
    (a straight mis-flagged as an apex)."""
    k_apex = float(kappa[apex]) if 0 <= apex < kappa.size and math.isfinite(kappa[apex]) else 0.0
    if k_apex > 0.0:
        return 1.0 / k_apex
    window = kappa[lm:rm]
    k_max = float(np.nanmax(window)) if window.size and np.isfinite(window).any() else 0.0
    return 1.0 / k_max if k_max > 0.0 else None


def _classify(radius: float, cfg: CornersConfig) -> CornerType:
    """§4.4 — apex radius → corner type (slow < radius_slow_max ≤ medium < radius_medium_max)."""
    if radius < cfg.radius_slow_max:
        return CornerType.SLOW
    if radius < cfg.radius_medium_max:
        return CornerType.MEDIUM
    return CornerType.HIGH_SPEED


def _lap_duration_s(lap: pd.DataFrame, n: int, fcfg: Any) -> float | None:
    """Lap wall-time (for the per-second steering-reversal rate) — F03b's exact derivation."""
    time, dt_ms = _resolve_time_base(lap, None, n, fcfg)
    if time is not None and time.size >= 2 and bool(np.isfinite(time).all()):
        return float(time[-1] - time[0])
    if dt_ms is not None:
        return dt_ms / 1000.0 * (n - 1)
    return None


@dataclass
class _Accum:
    """Accumulates one corner type's sub-features across a lap's corners of that type."""

    apex_speeds: list[float] = field(default_factory=list)
    g_sum: float = 0.0
    g_n: int = 0
    thr_sum: float = 0.0
    thr_n: int = 0
    brk_sum: float = 0.0
    brk_n: int = 0
    reversals: int = 0
    n_samples: int = 0

    @property
    def n_corners(self) -> int:
        return len(self.apex_speeds)

    def add(
        self, apex: int, lm: int, rm: int, speed: np.ndarray, signed_g: np.ndarray,
        throttle: np.ndarray, brake: np.ndarray, deadband: float,
    ) -> None:
        self.apex_speeds.append(float(speed[apex]))  # NaN apex handled at finalize (nan-safe mean)
        absg = np.abs(signed_g[lm:rm])
        thr = throttle[lm:rm]
        brk = brake[lm:rm]
        for arr, (s_attr, n_attr) in (
            (absg, ("g_sum", "g_n")), (thr, ("thr_sum", "thr_n")), (brk, ("brk_sum", "brk_n"))
        ):
            finite = arr[np.isfinite(arr)]
            setattr(self, s_attr, getattr(self, s_attr) + float(finite.sum()))
            setattr(self, n_attr, getattr(self, n_attr) + int(finite.size))
        # Sign-reversals of signed lateral G (ignoring |G| < deadband) — F21's rule, per window.
        sign = np.where(np.abs(signed_g[lm:rm]) >= deadband, np.sign(signed_g[lm:rm]), 0.0)
        nz = sign[sign != 0]
        self.reversals += int(np.count_nonzero(np.diff(nz) != 0)) if nz.size > 1 else 0
        self.n_samples += rm - lm

    def finalize(
        self, duration_s: float | None, n_lap: int
    ) -> tuple[dict[str, float | None], bool]:
        finite_apex = [s for s in self.apex_speeds if math.isfinite(s)]
        apex_speed_mean = float(np.mean(finite_apex)) if finite_apex else None
        lateral_g_mean = self.g_sum / self.g_n if self.g_n > 0 else None
        throttle_mean = self.thr_sum / self.thr_n if self.thr_n > 0 else None
        brake_mean = self.brk_sum / self.brk_n if self.brk_n > 0 else None
        # Time over this type's windows: sample count × per-sample dt (per-sample rate if no clock).
        if duration_s is not None and n_lap > 1 and self.n_samples > 0:
            window_time = self.n_samples * (duration_s / (n_lap - 1))
        else:
            window_time = float(self.n_samples)
        rate = self.reversals / window_time if window_time > 0 else None
        feats: dict[str, float | None] = {
            "apex_speed_mean": apex_speed_mean,
            "lateral_g_mean": lateral_g_mean,
            "throttle_mean": throttle_mean,
            "brake_mean": brake_mean,
            "steering_reversal_rate": rate,
        }
        return feats, any(v is None for v in feats.values())


# ----------------------------------------------------------------------------------
# Per-lap decomposition — decompose_lap (never raises)
# ----------------------------------------------------------------------------------


def decompose_lap(
    lap: pd.DataFrame,
    *,
    lap_number: int,
    config: AppConfig,
    log_path: str | Path | None = None,
) -> CornerDecompositionResult:
    """Decompose one lap into slow/medium/high-speed corner sub-vectors (§4.1). **Never raises** —
    X/Y unavailable → `CORNER_001` skip (all three sub-vectors NULL, the lap continues); any
    unforeseen fault → `CORNER_001` result, never a propagated exception (§4.0-d/e)."""
    log = Path(log_path) if log_path is not None else Path(config.paths.data_quality_log)
    ledger = _DQLedger(log_path=log)
    eid = f"lap{lap_number}"
    empty = {ct.value: CornerSubVector(corner_type=ct, n_corners=0,
                                       features={f: None for f in CORNER_SUB_FEATURES})
             for ct in CornerType}
    base = CornerDecompositionResult(lap_number=lap_number, sub_vectors=dict(empty))
    try:
        if config.features is None or config.corners is None:
            raise ValueError("config.features / corners block absent")
        fcfg = config.features
        ccfg = config.corners
        if not _xy_available(lap, fcfg):
            ledger.record(code=ErrorCode.CORNER_001, entity_type=_ENTITY_LAP, entity_id=eid,
                          detail=f"{REASON_XY_UNAVAILABLE}: corner decomposition skipped",
                          fallback_used=False)
            return _finalise(base, ledger, status=Status.DEGRADED,
                             error_code=ErrorCode.CORNER_001, xy_available=False)

        ch = _extract_channels(lap)
        x, y = _numeric(lap["X"]), _numeric(lap["Y"])
        speed, throttle, brake = ch.speed, ch.throttle, ch.brake
        kappa = _curvature(x, y)
        signed_g = _signed_lateral_g(x, y, speed, fcfg)
        duration_s = _lap_duration_s(lap, int(speed.size), fcfg)
        deadband = fcfg.steering_g_deadband
        corners = _corner_indices(speed, fcfg)

        acc: dict[CornerType, _Accum] = {ct: _Accum() for ct in CornerType}
        for apex, lm, rm in _corner_windows(speed, corners):
            if rm <= lm:
                continue
            radius = _apex_radius(kappa, apex, lm, rm)
            if radius is None:  # a straight mis-flagged as an apex — not a corner
                continue
            acc[_classify(radius, ccfg)].add(apex, lm, rm, speed, signed_g, throttle, brake,
                                             deadband)

        sub_vectors: dict[str, CornerSubVector] = {}
        n_total = 0
        degraded = False
        for ct in CornerType:
            a = acc[ct]
            if a.n_corners == 0:
                sub_vectors[ct.value] = CornerSubVector(
                    corner_type=ct, n_corners=0, features={f: None for f in CORNER_SUB_FEATURES}
                )
                degraded = True  # a lap rarely has all three corner types
                continue
            feats, feat_null = a.finalize(duration_s, int(speed.size))
            if feat_null:
                degraded = True
                nulls = [k for k, v in feats.items() if v is None]  # name the uncomputable features
                ledger.record(code=ErrorCode.FEATURE_001, entity_type=_ENTITY_LAP, entity_id=eid,
                              detail=f"{REASON_SUBFEATURE_NULL}: {ct.value}: {nulls}",
                              fallback_used=False)
            sub_vectors[ct.value] = CornerSubVector(corner_type=ct, n_corners=a.n_corners,
                                                    features=feats)
            n_total += a.n_corners

        base.sub_vectors = sub_vectors
        return _finalise(base, ledger, status=Status.DEGRADED if degraded else Status.OK,
                         error_code=None, xy_available=True, n_corners_total=n_total)
    except Exception as exc:  # noqa: BLE001 — the never-raises backstop (§4.0-e)
        try:
            ledger.record(code=ErrorCode.CORNER_001, entity_type=_ENTITY_LAP, entity_id=eid,
                          detail=f"{REASON_BACKSTOP}: {type(exc).__name__}: {exc}",
                          fallback_used=False)
        except Exception:  # noqa: BLE001 — even the logger must not break never-raises
            pass
        return _finalise(base, ledger, status=Status.DEGRADED, error_code=ErrorCode.CORNER_001,
                         xy_available=False)


def _finalise(
    result: CornerDecompositionResult, ledger: _DQLedger, *, status: Status,
    error_code: ErrorCode | None, xy_available: bool, n_corners_total: int = 0,
) -> CornerDecompositionResult:
    result.status = status
    result.error_code = error_code
    result.xy_available = xy_available
    result.n_corners_total = n_corners_total
    result.flags = [ErrorCode.DATA_QUALITY_LOG_UNWRITABLE] if ledger.failed else []
    result.logged = ledger.rows > 0
    return result


def decomposition_row(result: CornerDecompositionResult) -> dict[str, float | None]:
    """Flatten a decomposition to the per-lap corner columns (the 15 sub-features + 3 counts) — the
    building block a caller uses to assemble the sub-fingerprint / ablation matrices."""
    row: dict[str, float | None] = {}
    for ct in CornerType:
        sv = result.sub_vectors.get(ct.value)
        for feat in CORNER_SUB_FEATURES:
            row[_sub_col(ct, feat)] = sv.features.get(feat) if sv is not None else None
        row[_n_col(ct)] = float(sv.n_corners) if sv is not None else 0.0
    return row


# ----------------------------------------------------------------------------------
# Deterministic serialisation (canonical JSON + content hash — copied locally, F04/F10 pattern)
# ----------------------------------------------------------------------------------


def _canonical_body(artefact: _SubFingerprintArtefact) -> str:
    body = artefact.model_dump(exclude={"content_hash"})
    return json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash_artefact(artefact: _SubFingerprintArtefact) -> str:
    return sha256(_canonical_body(artefact).encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------------
# Sub-fingerprints — build_sub_fingerprints (the ONE aggregate writer) — fail-loud
# ----------------------------------------------------------------------------------


def _assert_no_leakage(train: pd.DataFrame, config: AppConfig) -> None:
    """Fail loud if any held-out circuit survives into the training set (non-negotiable #2).
    Raises ``SubFingerprintBuildError`` — no artefact on a poisoned run (F04/F06a/F08/F10/F11)."""
    held_out = set(config.held_out_circuits)
    present = set(train[COL_CIRCUIT].unique()) if len(train) else set()
    leaked = sorted(held_out & present)
    if leaked:
        raise SubFingerprintBuildError(
            f"held-out circuits present in training input after filter: {leaked} "
            "(would leak into the aggregate — aborting; no artefact written)",
            error_code=ErrorCode.TRAIN_LEAKAGE,
        )


def build_sub_fingerprints(
    frame: pd.DataFrame,
    *,
    config: AppConfig,
    artefact_dir: str | Path | None = None,
    log_path: str | Path | None = None,
) -> SubFingerprintFitResult:
    """Aggregate the per-lap corner sub-features (the flattened `decomposition_row` columns) into
    per-`(driver, corner_type)` sub-fingerprints on the TRAINING split ONLY and write
    ``corner_subfingerprints.json``. Raises :class:`SubFingerprintBuildError` on missing config/
    columns, the leakage tripwire (``TRAIN_LEAKAGE``), or no usable cells
    (``MODEL_TRAINING_FAILED``) — no artefact on those paths. Deterministic → byte-identical."""
    if config.corners is None:
        raise SubFingerprintBuildError("config.corners block absent; cannot build",
                                       error_code=ErrorCode.MODEL_TRAINING_FAILED)
    ccfg = config.corners
    art_dir = Path(artefact_dir) if artefact_dir is not None else Path(config.paths.artefacts_dir)
    art_dir.mkdir(parents=True, exist_ok=True)
    log = Path(log_path) if log_path is not None else Path(config.paths.data_quality_log)
    ledger = _DQLedger(log_path=log)

    def _fail(detail: str, code: ErrorCode) -> SubFingerprintBuildError:
        ledger.record(code=code, entity_type=_ENTITY_BUILD, entity_id=ccfg.mlflow_experiment,
                      detail=detail, fallback_used=False)
        return SubFingerprintBuildError(detail, error_code=code)

    required = {COL_DRIVER, COL_CIRCUIT, COL_SPLIT, *_CORNER_COLUMNS, *_N_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise _fail(f"sub-fingerprint frame missing required columns: {missing}",
                    ErrorCode.MODEL_TRAINING_FAILED)

    train_mask = (frame[COL_SPLIT] == "train") & (
        ~frame[COL_CIRCUIT].isin(config.held_out_circuits)
    )
    train = cast("pd.DataFrame", frame.loc[train_mask].copy())
    try:
        _assert_no_leakage(train, config)
    except SubFingerprintBuildError as exc:
        ledger.record(code=exc.error_code, entity_type=_ENTITY_BUILD,
                      entity_id=ccfg.mlflow_experiment, detail=str(exc), fallback_used=False)
        raise
    if len(train) == 0:
        raise _fail("no training rows after excluding held-out / non-train split",
                    ErrorCode.MODEL_TRAINING_FAILED)

    cells: list[_CellEntry] = []
    drivers = sorted(train[COL_DRIVER].unique().tolist())
    per_type_drivers: dict[str, int] = {ct.value: 0 for ct in CornerType}
    n_dropped = 0  # thin (driver, corner_type) cells → surfaced as DEGRADED on the typed result
    for driver in drivers:
        d = cast("pd.DataFrame", train.loc[train[COL_DRIVER] == driver])
        for ct in CornerType:
            n_corners = int(d[_n_col(ct)].fillna(0).sum())
            n_laps = int((d[_n_col(ct)].fillna(0) > 0).sum())
            if n_corners < ccfg.min_corners_per_type:
                n_dropped += 1
                ledger.record(
                    code=ErrorCode.DATA_INSUFFICIENT, entity_type=_ENTITY_DRIVER,
                    entity_id=f"{driver}/{ct.value}",
                    detail=f"{REASON_THIN_CELL}: {n_corners} < {ccfg.min_corners_per_type}",
                    fallback_used=False,
                )
                continue
            feats = _weighted_means(d, ct)
            cells.append(_CellEntry(driver=str(driver), corner_type=ct.value, features=feats,
                                    n_corners=n_corners, n_laps=n_laps))
            per_type_drivers[ct.value] += 1
    if not cells:
        raise _fail(f"{REASON_NO_CELLS}: no (driver, corner_type) cell had >= "
                    f"{ccfg.min_corners_per_type} corners", ErrorCode.MODEL_TRAINING_FAILED)

    artefact = _SubFingerprintArtefact(
        schema_version=SCHEMA_VERSION, sub_features=list(CORNER_SUB_FEATURES),
        cells=sorted(cells, key=lambda c: (c.driver, c.corner_type)), content_hash="",
    )
    artefact.content_hash = _hash_artefact(artefact)
    path = art_dir / ccfg.subfingerprint_artefact_filename
    text = json.dumps(artefact.model_dump(), sort_keys=True, indent=2, allow_nan=False) + "\n"
    try:
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise _fail(f"artefact write fault: {exc}", ErrorCode.MODEL_TRAINING_FAILED) from exc

    n_drivers = len(drivers)
    coverage = {ct: (per_type_drivers[ct] / n_drivers if n_drivers else 0.0)
                for ct in per_type_drivers}
    mlflow_run_id, mlflow_degraded = _log_mlflow(config, art_dir=art_dir, path=path, ledger=ledger,
                                                 n_drivers=n_drivers, n_entries=len(cells),
                                                 coverage=coverage)
    # A dropped thin cell is a genuine coverage loss — surface it on the typed result (status +
    # flag), not only in the DQ log (§6 row 4: "Cell dropped; status=DEGRADED"). H-1.
    flags = _dedup(
        ([ErrorCode.DATA_INSUFFICIENT] if n_dropped else [])
        + ([ErrorCode.DATA_QUALITY_LOG_UNWRITABLE] if ledger.failed else [])
    )
    return SubFingerprintFitResult(
        status=Status.DEGRADED if (mlflow_degraded or n_dropped) else Status.OK, error_code=None,
        flags=flags, logged=ledger.rows > 0, artefact_path=str(path),
        content_hash=artefact.content_hash, n_drivers=n_drivers, n_entries=len(cells),
        coverage=coverage, mlflow_run_id=mlflow_run_id,
    )


def _weighted_means(d: pd.DataFrame, ct: CornerType) -> dict[str, float | None]:
    """Corner-count-weighted mean of each sub-feature over a driver's laps of this corner type
    (laps with 0 corners of the type or a NULL sub-feature contribute nothing)."""
    w_all = d[_n_col(ct)].fillna(0.0).to_numpy(dtype=float)
    out: dict[str, float | None] = {}
    for feat in CORNER_SUB_FEATURES:
        v = d[_sub_col(ct, feat)].to_numpy(dtype=float)
        m = np.isfinite(v) & (w_all > 0)
        wsum = float(w_all[m].sum())
        out[feat] = float((v[m] * w_all[m]).sum() / wsum) if wsum > 0 else None
    return out


def _log_mlflow(
    config: AppConfig, *, art_dir: Path, path: Path, ledger: _DQLedger, n_drivers: int,
    n_entries: int, coverage: dict[str, float],
) -> tuple[str | None, bool]:
    assert config.corners is not None
    ccfg = config.corners
    run_id: str | None = None
    degraded = False
    try:
        mlflow.set_tracking_uri(ccfg.mlflow_tracking_uri)
        art_loc = (art_dir / "mlartifacts").resolve().as_uri()
        client = mlflow.tracking.MlflowClient()
        if client.get_experiment_by_name(ccfg.mlflow_experiment) is None:
            client.create_experiment(ccfg.mlflow_experiment, artifact_location=art_loc)
        mlflow.set_experiment(ccfg.mlflow_experiment)
        with mlflow.start_run(run_name="subfingerprint-build") as run:
            mlflow.log_metric("n_drivers", float(n_drivers))
            mlflow.log_metric("n_entries", float(n_entries))
            for ct, frac in coverage.items():
                mlflow.log_metric(f"coverage_{ct}", float(frac))
            mlflow.log_artifact(str(path))
            run_id = run.info.run_id
    except Exception as exc:  # noqa: BLE001 — MLflow fault degrades, never aborts the write
        degraded = True
        ledger.record(code=ErrorCode.MODEL_TRAINING_FAILED, entity_type=_ENTITY_BUILD,
                      entity_id=ccfg.mlflow_experiment, detail=f"{REASON_MLFLOW}: {exc}",
                      fallback_used=True)
    return run_id, degraded


# ----------------------------------------------------------------------------------
# Load + health (§4.4)
# ----------------------------------------------------------------------------------


def load_sub_fingerprints(
    *, config: AppConfig, artefact_dir: str | Path | None = None,
    log_path: str | Path | None = None,
) -> CornerHandle:
    """Load the serving bundle. Missing / corrupt / hash- or schema-mismatch / empty-cell table →
    DQ-log then **raise** :class:`CornerLoadError` (``MODEL_NOT_FOUND``)."""
    if config.corners is None:
        raise CornerLoadError("config.corners block absent; cannot load")
    ccfg = config.corners
    art_dir = Path(artefact_dir) if artefact_dir is not None else Path(config.paths.artefacts_dir)
    log = Path(log_path) if log_path is not None else Path(config.paths.data_quality_log)
    ledger = _DQLedger(log_path=log)

    path = art_dir / ccfg.subfingerprint_artefact_filename
    if not path.is_file():
        raise _load_fail(ledger, f"sub-fingerprint artefact not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _load_fail(ledger, f"sub-fingerprint artefact unreadable/corrupt: {exc}") from exc
    if not isinstance(raw, dict):
        raise _load_fail(ledger, "sub-fingerprint artefact must be a JSON object")
    try:
        artefact = _SubFingerprintArtefact(**raw)
    except ValidationError as exc:
        raise _load_fail(ledger, f"sub-fingerprint artefact schema mismatch: {exc}") from exc
    if artefact.schema_version != SCHEMA_VERSION:
        raise _load_fail(ledger,
                         f"schema_version {artefact.schema_version!r} != {SCHEMA_VERSION!r}")
    if _hash_artefact(artefact) != artefact.content_hash:
        raise _load_fail(ledger, "content_hash mismatch (corrupt/tampered)")
    if not artefact.cells:
        raise _load_fail(ledger, "empty sub-fingerprint table (no cells)")

    cells = {
        (e.driver, e.corner_type): SubFingerprint(
            driver=e.driver, corner_type=CornerType(e.corner_type), features=e.features,
            n_corners=e.n_corners, n_laps=e.n_laps,
        )
        for e in artefact.cells
    }
    model_version = f"sha256:{artefact.content_hash[:_VERSION_HASH_LEN]}"
    return CornerHandle(cells=cells, sub_features=tuple(artefact.sub_features),
                        model_version=model_version)


def _load_fail(ledger: _DQLedger, detail: str) -> CornerLoadError:
    ledger.record(code=ErrorCode.MODEL_NOT_FOUND, entity_type=_ENTITY_MODEL, entity_id="startup",
                  detail=detail, fallback_used=False)
    return CornerLoadError(detail)


def health(handle: CornerHandle | None) -> CornerHealthResult:
    """Readiness for F13's ``GET /health``. Not-ready → ``HEALTH_FAIL`` (→ 503) with reason
    ``MODEL_NOT_FOUND``; ready → OK + ``model_version`` (F06b/F08/F10/F11 precedent)."""
    if handle is None or not handle.ready:
        return CornerHealthResult(status=Status.FAILED, error_code=ErrorCode.HEALTH_FAIL,
                                  ready=False, reason=ErrorCode.MODEL_NOT_FOUND)
    return CornerHealthResult(status=Status.OK, ready=True, model_version=handle.model_version)


# ----------------------------------------------------------------------------------
# Ablation — run_ablation (§4.3, read-only, never raises)
# ----------------------------------------------------------------------------------


def run_ablation(
    frame: pd.DataFrame, *, config: AppConfig, log_path: str | Path | None = None,
) -> AblationReport:
    """§4.4-c — does corner-type information improve the classifier? Scores **base** (28
    ``FEATURE_IDS``) vs **expanded** (base ‖ the 15 corner sub-features) with the F06a CV
    (``StratifiedGroupKFold``-by-circuit, seed 42) and reports the macro-F1 delta + finding.
    Read-only (no `.fit` outside the CV; the champion is untouched); malformed input → typed
    ``INVALID_INPUT``; **never raises**. Held-out circuits enter only via F20's invocation."""
    log = Path(log_path) if log_path is not None else Path(config.paths.data_quality_log)
    ledger = _DQLedger(log_path=log)
    try:
        if config.corners is None or config.classifier is None:
            return _ablation_invalid(ledger, "config.corners / classifier block absent")
        ccfg = config.corners
        base_cols = list(FEATURE_IDS)
        required = {COL_DRIVER, COL_CIRCUIT, COL_SPLIT, *base_cols, *_CORNER_COLUMNS}
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return _ablation_invalid(ledger, "empty or non-DataFrame ablation matrix")
        missing = sorted(required - set(frame.columns))
        if missing:
            return _ablation_invalid(ledger, f"missing required columns: {missing}")

        train_mask = (frame[COL_SPLIT] == "train") & (
            ~frame[COL_CIRCUIT].isin(config.held_out_circuits)
        )
        train = cast("pd.DataFrame", frame.loc[train_mask].copy())
        _assert_no_leakage(train, config)  # tripwire (caught by the backstop → INVALID_INPUT)
        drivers = sorted(train[COL_DRIVER].unique().tolist())
        n_circuits = int(train[COL_CIRCUIT].nunique())
        if len(drivers) < 2 or n_circuits < 2:
            return _ablation_invalid(ledger, "insufficient training data for the ablation CV")

        label_index = {d: i for i, d in enumerate(drivers)}
        y = train[COL_DRIVER].map(label_index).to_numpy(dtype=int)
        groups = train[COL_CIRCUIT].to_numpy()
        x_base = train[base_cols].to_numpy(dtype=float)
        x_exp = train[[*base_cols, *_CORNER_COLUMNS]].to_numpy(dtype=float)
        params = _base_xgb_params(seed=config.seeds.xgboost, n_classes=len(drivers))
        # StratifiedGroupKFold groups by CIRCUIT — clamp folds to the circuit count (F06a protocol).
        cv = StratifiedGroupKFold(n_splits=min(ccfg.cv_folds, n_circuits), shuffle=True,
                                  random_state=config.seeds.xgboost)

        base_macro, base_per = _cv_f1(params, x_base, y, groups, cv, n_classes=len(drivers))
        exp_macro, exp_per = _cv_f1(params, x_exp, y, groups, cv, n_classes=len(drivers))
        delta = exp_macro - base_macro
        finding = ("IMPROVES" if delta > ccfg.ablation_margin
                   else "DEGRADES" if delta < -ccfg.ablation_margin else "NEUTRAL")
        per_driver = {d: float(exp_per[i] - base_per[i]) for d, i in label_index.items()}

        run_id, mlflow_degraded = _log_ablation_mlflow(config, base_macro, exp_macro, delta,
                                                       finding, ledger)
        return AblationReport(
            status=Status.DEGRADED if mlflow_degraded else Status.OK, error_code=None,
            flags=[ErrorCode.DATA_QUALITY_LOG_UNWRITABLE] if ledger.failed else [],
            logged=ledger.rows > 0, base_macro_f1=base_macro, expanded_macro_f1=exp_macro,
            delta=delta, per_driver_delta=per_driver, finding=finding,
            n_base_features=len(base_cols),
            n_expanded_features=len(base_cols) + len(_CORNER_COLUMNS),
            mlflow_run_id=run_id,
        )
    except Exception as exc:  # noqa: BLE001 — never-raises backstop
        return _ablation_invalid(ledger, f"{REASON_BACKSTOP}: {type(exc).__name__}: {exc}")


def _cv_f1(
    params: dict[str, Any], x: np.ndarray, y: np.ndarray, groups: np.ndarray,
    cv: StratifiedGroupKFold, *, n_classes: int,
) -> tuple[float, np.ndarray]:
    """Mean CV macro-F1 + per-class mean F1 (for the per-driver delta). Reuses the F06a
    ``XGBClassifier`` + ``StratifiedGroupKFold`` protocol; XGBoost consumes NaN natively (a lap with
    no corners of a type leaves those columns NaN — no imputation needed)."""
    macros: list[float] = []
    per = np.zeros(n_classes, dtype=float)
    folds = 0
    for tr, va in cv.split(x, y, groups):
        model = xgb.XGBClassifier(**params)
        model.fit(x[tr], y[tr])
        pred = model.predict(x[va])
        macros.append(float(f1_score(y[va], pred, average="macro")))
        per += np.asarray(
            f1_score(y[va], pred, labels=list(range(n_classes)), average=None, zero_division=0),
            dtype=float,
        )
        folds += 1
    return (float(np.mean(macros)) if macros else math.nan), (per / folds if folds else per)


def _log_ablation_mlflow(
    config: AppConfig, base: float, expanded: float, delta: float, finding: str, ledger: _DQLedger,
) -> tuple[str | None, bool]:
    assert config.corners is not None
    ccfg = config.corners
    run_id: str | None = None
    degraded = False
    try:
        mlflow.set_tracking_uri(ccfg.mlflow_tracking_uri)
        client = mlflow.tracking.MlflowClient()
        if client.get_experiment_by_name(ccfg.mlflow_experiment) is None:
            client.create_experiment(ccfg.mlflow_experiment)
        mlflow.set_experiment(ccfg.mlflow_experiment)
        with mlflow.start_run(run_name="corner-ablation") as run:
            mlflow.log_metric("base_macro_f1", float(base))
            mlflow.log_metric("expanded_macro_f1", float(expanded))
            mlflow.log_metric("delta", float(delta))
            mlflow.set_tag("finding", finding)
            run_id = run.info.run_id
    except Exception as exc:  # noqa: BLE001 — MLflow fault degrades, never aborts the report
        degraded = True
        ledger.record(code=ErrorCode.MODEL_TRAINING_FAILED, entity_type=_ENTITY_BUILD,
                      entity_id=ccfg.mlflow_experiment, detail=f"{REASON_MLFLOW}: {exc}",
                      fallback_used=True)
    return run_id, degraded


def _ablation_invalid(ledger: _DQLedger, detail: str) -> AblationReport:
    try:
        ledger.record(code=ErrorCode.INVALID_INPUT, entity_type=_ENTITY_BUILD,
                      entity_id="ablation", detail=f"{REASON_INVALID}: {detail}",
                      fallback_used=False)
    except Exception:  # noqa: BLE001
        pass
    flags = [ErrorCode.DATA_QUALITY_LOG_UNWRITABLE] if ledger.failed else []
    return AblationReport(status=Status.FAILED, error_code=ErrorCode.INVALID_INPUT, flags=flags,
                          logged=ledger.rows > 0)


# Column-name accessors (public for callers / tests building the matrices).
def corner_sub_columns() -> tuple[str, ...]:
    """The 15 flattened corner sub-feature column names (ablation expanded matrix)."""
    return _CORNER_COLUMNS


def corner_count_columns() -> tuple[str, ...]:
    """The 3 per-corner-type count column names (sub-fingerprint weighting)."""
    return _N_COLUMNS
