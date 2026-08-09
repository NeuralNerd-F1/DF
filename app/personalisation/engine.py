"""F11 — Model 4 (car personalisation engine): fingerprint → 10-parameter setup recommendation.

The engine (master §7) translates a driver's **5D style fingerprint** (from F08) into a
**10-parameter car-setup recommendation** through three layers every recommendation must pass:

* **Layer 1 — learned mapping** (§7.1): one **XGBoost regressor per setup parameter**, input =
  the F08 5D fingerprint ‖ a **drop-first circuit one-hot** (8 dims for 4 circuit types). The
  regression **targets are approximated from telemetry, never measured** (§15.2, divergence #13):
  each target is a documented equal-weight min-max map of the §7.3 fingerprint features onto the
  parameter's FIA range, computed on the **fastest-quartile clean-lap aggregates** (divergence #14).
* **Layer 2 — constraint engine** (§7.2): FIA-bound clip (`CONSTRAINT_CLIPPED`), the Monaco
  max-downforce override (`CIRCUIT_OVERRIDE_MONACO`), the front-wing-vs-brake-bias physical
  consistency flag (`CONSISTENCY_WARNING`, flag-only), and the confidence floor
  (`WIDE_CONFIDENCE` → a ±`wide_range_pct` range instead of a point).
* **Layer 3 — confidence scoring** (§7.2, divergence #17): a per-parameter reliability
  `r_p = max(0, 1 − CV-nRMSE)` (GroupKFold-by-**driver**, no driver spans folds), attenuated by an
  unknown driver's OOD-ness; **confidence is never NULL** (§8.4).

It also ships a SHAP decomposition per recommendation (§6.1, the F07 TreeExplainer pattern copied
per regressor — a SHAP fault never blocks a recommendation) and the §7.4 unknown-driver onboarding
package. Architecture mirrors F06a/F08/F10: `train_personalisation` is the ONE fit site
(quarantine → `_assert_no_leakage` tripwire → fit → content-hashed joblib bundle → MLflow
degrade-don't-die); `load_personalisation` refuses to serve a broken bundle (`MODEL_NOT_FOUND`);
`recommend_setup` / `onboard_unknown` / `health` **never raise**. **No new error code** — the four
§7 codes are pre-seeded in ``errors.py``. Every threshold/bound/filename comes from ``config``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import mlflow
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from joblib import dump as joblib_dump
from joblib import load as joblib_load
from pydantic import BaseModel, ConfigDict, Field
from sklearn.model_selection import GroupKFold

from app.contracts.config import _PERSONALISATION_PARAM_KEYS, AppConfig, PersonalisationConfig
from app.contracts.enums import Status
from app.contracts.errors import ErrorCode
from app.contracts.logging import DataQualityLogError, log_data_quality
from app.contracts.results import BaseResult
from app.ml.stylespace import ProjectionResult, StyleSpaceHandle, project_unknown
from app.pipeline.features import FEATURE_IDS
from app.pipeline.normalise import (
    NormalisationError,
    apply_normalisation,
    load_normalisation_params,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

# --- Structural constants (NOT tunable thresholds — those live in config) ------------------
SCHEMA_VERSION = "1.0"
_VERSION_HASH_LEN = 16
_N_FP_DIMS = 5  # F08 5D fingerprint (umap_n_components_5d) — the regressor's driver-level input

# §15.2 — the mandatory disclaimer on EVERY recommendation ("approximated, not measured").
APPROX_DISCLAIMER = (
    "Setup values are approximated from telemetry signatures, not measured from car setup data. "
    "They provide a starting direction, not an engineering specification."
)

# Ordinal encoding for the categorical suspension parameters (divergence #15).
_CATEGORY_LABELS: tuple[str, ...] = ("Soft", "Medium", "Stiff")

# Wide-matrix column names.
COL_DRIVER = "driver"
COL_CIRCUIT = "circuit"
COL_CIRCUIT_TYPE = "circuit_type"
COL_SEASON = "season"
COL_SPLIT = "split"
COL_LAP_TIME = "lap_time_s"  # §7.1 "best-performing" selector (fastest-quartile clean laps)

_REQUIRED_COLUMNS = (
    COL_DRIVER,
    COL_CIRCUIT,
    COL_CIRCUIT_TYPE,
    COL_SEASON,
    COL_SPLIT,
    COL_LAP_TIME,
    *FEATURE_IDS,
)

# data_quality_log detail tags (mirrors the REASON_* convention).
REASON_THIN_CELL = "CELL_BELOW_MIN_CLEAN_LAPS"
REASON_NO_COORD = "DRIVER_5D_COORD_MISSING"
REASON_UNKNOWN_DRIVER = "UNKNOWN_DRIVER_USE_ONBOARDING"
REASON_UNKNOWN_CIRCUIT_TYPE = "UNKNOWN_CIRCUIT_TYPE"
REASON_CLIP = "FIA_BOUND_CLIP"
REASON_MONACO = "MONACO_MAX_DOWNFORCE_OVERRIDE"
REASON_CONSISTENCY = "FRONT_WING_BRAKE_BIAS_INCONSISTENT"
REASON_WIDE = "CONFIDENCE_FLOOR_RANGE"
REASON_SHAP = "SHAP_RAISED"
REASON_BACKSTOP = "UNFORESEEN_SERVING_FAULT"
REASON_MLFLOW = "MLFLOW_LOGGING_DEGRADED"
REASON_NO_CELLS = "NO_USABLE_TRAINING_CELLS"
REASON_EXPLAINER = "EXPLAINER_BUILD_FAILED"
REASON_INVALID = "SERVE_INPUT_MALFORMED"
REASON_GRID_REFERENCE = "GRID_REFERENCE_UNAVAILABLE"
_FALLBACK_LOG = "logs/data_quality_log.jsonl"  # last-resort DQ sink when config.paths is unreadable

_ENTITY_CELL = "cell"
_ENTITY_SETUP = "setup"
_ENTITY_BUILD = "training_run"
_ENTITY_MODEL = "model"


# ----------------------------------------------------------------------------------
# Feature-owned enums (per enums.py docstring — feature-specific enums live with the feature)
# ----------------------------------------------------------------------------------


class SetupConfidence(StrEnum):
    """Per-parameter rating (§7.4 "HIGH / MEDIUM / LOW / WIDE")."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    WIDE = "WIDE"


class SetupOutputKind(StrEnum):
    """§7.3 output-format column."""

    DEGREES = "DEGREES"
    PERCENT = "PERCENT"
    INDEX = "INDEX"
    INTEGER_LAPS = "INTEGER_LAPS"
    CATEGORY = "CATEGORY"  # Soft/Medium/Stiff (suspension) — ordinal-regressed (divergence #15)


@dataclass(frozen=True)
class _ParamSpec:
    """One of the 10 setup parameters (§7.3): name, output kind, fingerprint basis, unit."""

    name: str
    kind: SetupOutputKind
    basis: tuple[str, ...]  # §7.3 fingerprint features the target heuristic reads
    unit: str


# §4.1 — the 10-parameter setup vector, canonical order (matches _PERSONALISATION_PARAM_KEYS).
_SETUP_PARAMETERS: tuple[_ParamSpec, ...] = (
    _ParamSpec("front_wing_angle", SetupOutputKind.DEGREES, ("F19", "F14"), "degrees"),
    _ParamSpec("rear_wing_angle", SetupOutputKind.DEGREES, ("F04", "F14"), "degrees"),
    _ParamSpec("brake_bias", SetupOutputKind.PERCENT, ("F06", "F08", "F10"), "% front"),
    _ParamSpec("differential_ramp_entry", SetupOutputKind.INDEX, ("F25", "F09"), "index"),
    _ParamSpec("differential_ramp_exit", SetupOutputKind.INDEX, ("F03", "F22"), "index"),
    _ParamSpec("throttle_map_curve", SetupOutputKind.INDEX, ("F01", "F02", "F03", "F05"), "index"),
    _ParamSpec("suspension_stiffness_front", SetupOutputKind.CATEGORY, ("F21", "F19"), ""),
    _ParamSpec("suspension_stiffness_rear", SetupOutputKind.CATEGORY, ("F12", "F20"), ""),
    _ParamSpec("tyre_warmup_laps", SetupOutputKind.INTEGER_LAPS, ("F27",), "laps"),
    _ParamSpec("brake_duct_aperture", SetupOutputKind.PERCENT, ("F06", "F07", "F08"), "% open"),
)
_PARAM_NAMES: tuple[str, ...] = tuple(p.name for p in _SETUP_PARAMETERS)
assert _PARAM_NAMES == _PERSONALISATION_PARAM_KEYS, "param order must match the config key list"
_FRONT_WING = "front_wing_angle"
_REAR_WING = "rear_wing_angle"
_BRAKE_BIAS = "brake_bias"


# ----------------------------------------------------------------------------------
# Exceptions (only the two build/load entry points raise; serving never does)
# ----------------------------------------------------------------------------------


class PersonalisationTrainingError(Exception):
    """Raised by :func:`train_personalisation` on a build-breaking fault (no artefact written).
    Carries ``TRAIN_LEAKAGE`` or ``MODEL_TRAINING_FAILED``. Fail-loud (#1)."""

    def __init__(self, message: str, *, error_code: ErrorCode) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code


class PersonalisationLoadError(Exception):
    """Raised by :func:`load_personalisation` when the bundle cannot serve. Carries
    ``MODEL_NOT_FOUND`` — the service refuses to start (F06b/F08/F10 precedent)."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = ErrorCode.MODEL_NOT_FOUND


# ----------------------------------------------------------------------------------
# Typed outputs (all Pydantic, extra="forbid"; BaseResult where a status is carried)
# ----------------------------------------------------------------------------------


class SetupParameter(BaseModel):
    """One of the 10 setup parameters (§7.3). Confidence is NEVER NULL (§8.4)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    kind: SetupOutputKind
    value: float | str | None = None  # point estimate (None only when WIDE range-only)
    value_range: tuple[float, float] | None = None  # populated iff rating == WIDE
    category: str | None = None  # Soft/Medium/Stiff when kind == CATEGORY
    confidence: float  # [0, 1] per-parameter reliability — never NULL
    rating: SetupConfidence
    unit: str = ""
    fia_min: float | None = None
    fia_max: float | None = None
    constraint_flags: list[ErrorCode] = Field(default_factory=list)
    shap_top_features: list[str] = Field(default_factory=list)
    shap_available: bool = False
    # The grid reference (system input #2 — "setup tendencies"): the mean of the fitted model's own
    # recommendations across every known driver in the SAME circuit context. Lets the UI state a
    # direction ("above the grid average") as a fact about the model's outputs — never a claim
    # about a real team's setup. None for CATEGORY kinds (a mean over category ordinals has no
    # honest reading) and when the reference could not be computed (degraded, logged).
    grid_mean: float | None = None
    # The §7.3 fingerprint features the TARGET HEURISTIC was built from — a design-time constant
    # (identical for every driver), served so the UI can say what this parameter is derived from
    # without pretending the per-driver SHAP (over opaque UMAP dims) says it.
    basis_features: list[str] = Field(default_factory=list)


class SetupRecommendation(BaseResult):
    """Never-raises serving result for one (driver, circuit)."""

    driver: str = ""
    circuit: str = ""
    circuit_type: str | None = None
    parameters: list[SetupParameter] = Field(default_factory=list)  # exactly 10, canonical order
    consistency_warning: bool = False
    monaco_override: bool = False
    threshold_source: str = ""
    approximation_disclaimer: str = APPROX_DISCLAIMER
    model_version: str = ""


class Neighbour(BaseModel):
    """One nearest known driver by cosine (mirrors F08 field-for-field, re-declared locally)."""

    model_config = ConfigDict(extra="forbid")

    driver: str
    cosine_similarity: float
    distance: float


class OnboardingPackage(BaseResult):
    """§7.4 unknown-driver onboarding — never raises."""

    setup: SetupRecommendation | None = None
    style_match: list[Neighbour] = Field(default_factory=list)
    fingerprint_snapshot: ProjectionResult | None = None
    fingerprint_available: bool = False
    risk_flags: list[str] = Field(default_factory=list)
    n_laps_used: int = 0
    n_laps_rejected: int = 0
    imputed_features: list[str] = Field(default_factory=list)


class PersonalisationFitResult(BaseResult):
    """Training output (mirrors F08/F10 build results)."""

    artefact_path: str = ""
    content_hash: str = ""
    n_train_cells: int = 0
    parameters_fitted: int = 0
    reliability: dict[str, float] = Field(default_factory=dict)
    n_reliable: int = 0  # parameters with reliability ≥ confidence_medium (§14 finding)
    mlflow_run_id: str | None = None


class PersonalisationHealthResult(BaseResult):
    """Readiness for F13's ``GET /health`` (mirrors F06b/F08/F10)."""

    ready: bool = False
    model_version: str = ""
    reason: ErrorCode | None = None


# ----------------------------------------------------------------------------------
# Loaded serving bundle (immutable; mirrors StyleSpaceHandle / MonitorHandle)
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class PersonalisationHandle:
    """The loaded, immutable serving bundle (built once at startup)."""

    regressors: dict[str, Any]  # param name → fitted XGBRegressor
    reliability: dict[str, float]  # param name → CV reliability r_p ∈ [0, 1]
    explainers: dict[str, Any]  # param name → shap.TreeExplainer | None (best-effort; F07)
    fia_bounds: dict[str, tuple[float, float]]
    circuit_types: tuple[str, ...]  # the drop-first one-hot vocabulary
    input_dim: int  # 5 + (len(circuit_types) − 1)
    model_version: str
    ready: bool = True


# ----------------------------------------------------------------------------------
# data-quality ledger (never propagates a sink failure — mirrors F04/F06/F08/F10)
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


def _safe_log_path(config: AppConfig, log_path: str | Path | None) -> Path:
    """Resolve the DQ sink defensively so a ``None``/malformed config cannot make a never-raises
    serving entry point raise before its ``try`` (Finding: pre-try escape)."""
    if log_path is not None:
        return Path(log_path)
    try:
        return Path(config.paths.data_quality_log)
    except Exception:  # noqa: BLE001 — a bad config must not break the never-raises contract
        return Path(_FALLBACK_LOG)


# ----------------------------------------------------------------------------------
# Pure helpers (unit-testable in isolation)
# ----------------------------------------------------------------------------------


def _finite_or_none(x: object) -> float | None:
    if x is None:
        return None
    try:
        xf = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return xf if math.isfinite(xf) else None


def _circuit_onehot(circuit_type: str | None, circuit_types: Sequence[str]) -> np.ndarray:
    """Drop-first one-hot (§4.0-f, divergence #16): the first circuit type is the reference (all
    zeros); an unknown type → all zeros too (served, degraded — the caller flags FALLBACK).
    Returns a ``len(circuit_types) − 1`` vector."""
    n = max(len(circuit_types) - 1, 0)
    out = np.zeros(n, dtype=float)
    if circuit_type is not None and circuit_type in circuit_types:
        idx = list(circuit_types).index(circuit_type)
        if idx >= 1:  # idx 0 is the dropped reference
            out[idx - 1] = 1.0
    return out


def _build_input(
    coord5: np.ndarray, circuit_type: str | None, circuit_types: Sequence[str]
) -> np.ndarray:
    """``[5D fingerprint ‖ drop-first circuit one-hot]`` — the regressor's input row (§4.2-1)."""
    return np.concatenate([np.asarray(coord5, dtype=float).ravel()[:_N_FP_DIMS],
                           _circuit_onehot(circuit_type, circuit_types)]).astype(float)


def _input_labels(circuit_types: Sequence[str]) -> list[str]:
    """Human labels for the regressor's input dims (SHAP over the model's OWN inputs, §4.2-5): the
    5 fingerprint axes then the drop-first circuit dummies."""
    return [f"fp{i + 1}" for i in range(_N_FP_DIMS)] + [
        f"circuit:{ct}" for ct in list(circuit_types)[1:]
    ]


def _round_by_kind(value: float, kind: SetupOutputKind) -> float:
    """INDEX / INTEGER_LAPS / CATEGORY are integer-valued (§7.3 output formats); DEGREES / PERCENT
    stay continuous. Applied AFTER the FIA clip so the rounded value stays in-bounds."""
    if kind in (SetupOutputKind.INDEX, SetupOutputKind.INTEGER_LAPS, SetupOutputKind.CATEGORY):
        return float(round(value))
    return value


def _category_of(ordinal: float) -> str:
    idx = int(min(max(round(ordinal), 0), len(_CATEGORY_LABELS) - 1))
    return _CATEGORY_LABELS[idx]


def _rating_of(confidence: float, cfg: PersonalisationConfig) -> SetupConfidence:
    """HIGH / MEDIUM / LOW / WIDE bands (§4.0-c). ``WIDE`` iff below the confidence floor."""
    if confidence < cfg.confidence_floor:
        return SetupConfidence.WIDE
    if confidence >= cfg.confidence_high:
        return SetupConfidence.HIGH
    if confidence >= cfg.confidence_medium:
        return SetupConfidence.MEDIUM
    return SetupConfidence.LOW


def _ood_attenuation(ood_max_pc_sd: float, cfg: PersonalisationConfig) -> float:
    """Shrink confidence toward ``ood_floor`` as an unknown driver's fingerprint OOD-ness exceeds
    ``ood_sd_threshold`` (§4.0-c). A known / in-distribution driver passes 0.0 → attenuation 1.0."""
    if ood_max_pc_sd <= cfg.ood_sd_threshold:
        return 1.0
    excess = ood_max_pc_sd - cfg.ood_sd_threshold
    # Linear decay over one further threshold-width, floored at ood_floor.
    frac = min(1.0, excess / max(cfg.ood_sd_threshold, 1e-9))
    return float(cfg.ood_floor + (1.0 - cfg.ood_floor) * (1.0 - frac))


# ----------------------------------------------------------------------------------
# Setup-target heuristic (§4.0-b, §4.1) — the honest telemetry approximation (divergence #13)
# ----------------------------------------------------------------------------------


def _minmax_stats(agg_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature min/max over the training cell aggregates (columns = FEATURE_IDS order)."""
    return agg_matrix.min(axis=0), agg_matrix.max(axis=0)


def _target_for(
    agg_row: np.ndarray, spec: _ParamSpec, fmin: np.ndarray, fmax: np.ndarray,
    fia_bounds: Mapping[str, tuple[float, float]], min_std: float,
) -> float:
    """One cell's approximated setup target for one parameter (§4.1): equal-weight mean of the
    min-max-normalised basis features, affinely mapped onto the FIA range, clipped."""
    idx = [FEATURE_IDS.index(f) for f in spec.basis]
    norm_vals = [
        (agg_row[i] - fmin[i]) / max(fmax[i] - fmin[i], min_std) for i in idx
    ]
    frac = float(np.mean(norm_vals)) if norm_vals else 0.0
    lo, hi = fia_bounds[spec.name]
    return float(min(max(lo + (hi - lo) * frac, lo), hi))


# ----------------------------------------------------------------------------------
# Aggregate builder (§4.0-a) — fastest-quartile clean-lap aggregate per (driver, circuit)
# ----------------------------------------------------------------------------------


@dataclass
class _Cell:
    driver: str
    circuit: str
    circuit_type: str | None
    aggregate: np.ndarray  # F04-normalised 28-dim mean over the fastest-quartile clean laps


def _build_cells(
    train: pd.DataFrame, norm_params: Any, config: AppConfig, min_clean_laps: int, top_q: float,
    log: Path, ledger: _DQLedger,
) -> tuple[list[_Cell], int]:
    """One aggregate per ``(driver, circuit)`` cell over its fastest-quartile clean laps
    (``lap_time_s ≤ quantile``); cells with < ``min_clean_laps`` laps are dropped + logged. Returns
    ``(cells, n_thin_dropped)`` so the caller can surface ``DATA_INSUFFICIENT`` on the result."""
    cells: list[_Cell] = []
    n_thin = 0
    for key, grp in train.groupby([COL_DRIVER, COL_CIRCUIT], dropna=False):
        driver, circuit = cast("tuple[Any, Any]", key)
        laps = grp
        if len(laps) < min_clean_laps:
            n_thin += 1
            ledger.record(
                code=ErrorCode.DATA_INSUFFICIENT, entity_type=_ENTITY_CELL,
                entity_id=f"{driver}/{circuit}",
                detail=f"{REASON_THIN_CELL}: {len(laps)} laps < {min_clean_laps}; dropped",
                fallback_used=False,
            )
            continue
        cutoff = float(laps[COL_LAP_TIME].quantile(top_q))
        fastest = cast("pd.DataFrame", laps.loc[laps[COL_LAP_TIME] <= cutoff])
        records = cast("list[dict[str, Any]]", fastest.to_dict("records"))
        vectors: list[np.ndarray] = []
        for rec in records:
            vec = {fid: _finite_or_none(rec.get(fid)) for fid in FEATURE_IDS}
            res = apply_normalisation(vec, circuit=str(circuit), params=norm_params, config=config,
                                      log_path=log)
            vectors.append(
                np.array(
                    [_norm_or_nan(res.normalised_vector.get(fid)) for fid in FEATURE_IDS],
                    dtype=float,
                )
            )
        agg = np.nanmean(np.vstack(vectors), axis=0)
        agg = np.where(np.isfinite(agg), agg, 0.0)  # an all-NULL feature → 0.0 (never NaN)
        ct = _circuit_type_of(records[0])
        cells.append(_Cell(str(driver), str(circuit), ct, agg))
    return cells, n_thin


def _norm_or_nan(x: object) -> float:
    v = _finite_or_none(x)
    return v if v is not None else math.nan


def _circuit_type_of(rec: Mapping[str, Any]) -> str | None:
    ct = rec.get(COL_CIRCUIT_TYPE)
    if ct is None or (isinstance(ct, float) and math.isnan(ct)):
        return None
    return str(ct)


# ----------------------------------------------------------------------------------
# Training — train_personalisation (the ONLY fit site) — fail-loud
# ----------------------------------------------------------------------------------


def _assert_no_leakage(train: pd.DataFrame, config: AppConfig) -> None:
    """Fail loud if any held-out circuit survives into the training set (non-negotiable #2).
    Called AFTER the exclusion filter as a tripwire; raises ``PersonalisationTrainingError`` — no
    artefact is written on a poisoned run (copied per module, F04/F06a/F08/F10 pattern)."""
    held_out = set(config.held_out_circuits)
    present = set(train[COL_CIRCUIT].unique()) if len(train) else set()
    leaked = sorted(held_out & present)
    if leaked:
        raise PersonalisationTrainingError(
            f"held-out circuits present in training input after filter: {leaked} "
            "(would leak into the fit — aborting; no artefact written)",
            error_code=ErrorCode.TRAIN_LEAKAGE,
        )


def _base_xgb_params(seed: int) -> dict[str, Any]:
    """Fixed XGBRegressor params. ``n_jobs=1`` pins single-threaded training (float-order
    reproducibility, non-negotiable #3), mirroring F06a's classifier params."""
    return {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": seed,
        "n_jobs": 1,
        "n_estimators": 200,
        "max_depth": 3,
        "learning_rate": 0.1,
    }


def _reliability(
    x: np.ndarray, groups: np.ndarray, agg_matrix: np.ndarray, spec: _ParamSpec, *,
    params: dict[str, Any], n_splits: int, fia_bounds: Mapping[str, tuple[float, float]],
    min_std: float,
) -> float:
    """Per-parameter reliability ``r_p = max(0, 1 − nRMSE)`` via GroupKFold-by-driver (no driver
    spans folds; divergence #17). The setup **targets are recomputed per CV fold from the TRAIN
    fold's min/max only** — a validation fold's labels are never defined using its own extrema, so
    ``r_p`` is an honest generalisation estimate (the §14 finding rests on it). ``nRMSE`` = CV-RMSE
    ÷ the parameter's FIA range."""
    n_groups = len(np.unique(groups))
    folds = min(n_splits, n_groups)
    if folds < 2:
        return 0.0  # cannot cross-validate reliability with < 2 driver groups
    lo, hi = fia_bounds[spec.name]
    cv = GroupKFold(n_splits=folds)
    errors: list[float] = []
    for tr, va in cv.split(x, groups=groups):
        fmin_f, fmax_f = agg_matrix[tr].min(axis=0), agg_matrix[tr].max(axis=0)
        y_tr = np.array(
            [_target_for(agg_matrix[i], spec, fmin_f, fmax_f, fia_bounds, min_std) for i in tr]
        )
        y_va = np.array(
            [_target_for(agg_matrix[i], spec, fmin_f, fmax_f, fia_bounds, min_std) for i in va]
        )
        model = xgb.XGBRegressor(**params)
        model.fit(x[tr], y_tr)
        pred = np.asarray(model.predict(x[va]), dtype=float)
        errors.append(float(np.sqrt(np.mean((pred - y_va) ** 2))))
    nrmse = float(np.mean(errors)) / max(hi - lo, min_std)
    return float(min(max(1.0 - nrmse, 0.0), 1.0))


def train_personalisation(
    matrix: pd.DataFrame,
    *,
    style_handle: StyleSpaceHandle,
    config: AppConfig,
    artefact_dir: str | Path | None = None,
    log_path: str | Path | None = None,
) -> PersonalisationFitResult:
    """Fit the 10 setup regressors + per-parameter reliability + SHAP explainers on the TRAINING
    split ONLY and write ``personalisation.pkl``. Raises :class:`PersonalisationTrainingError` on
    missing config/columns, the leakage tripwire (``TRAIN_LEAKAGE``), or no usable cells
    (``MODEL_TRAINING_FAILED``) — no artefact on those paths. Deterministic (seeds fixed)."""
    if config.personalisation is None or config.normalise is None:
        raise PersonalisationTrainingError(
            "config.personalisation / normalise block absent; cannot train",
            error_code=ErrorCode.MODEL_TRAINING_FAILED,
        )
    pcfg = config.personalisation
    circuit_types = tuple(config.circuit_types)
    art_dir = Path(artefact_dir) if artefact_dir is not None else Path(config.paths.artefacts_dir)
    art_dir.mkdir(parents=True, exist_ok=True)
    log = Path(log_path) if log_path is not None else Path(config.paths.data_quality_log)
    ledger = _DQLedger(log_path=log)
    min_std = float(config.normalise.min_std)
    min_clean = config.thresholds.min_clean_laps
    seed = config.seeds.xgboost

    def _fail(detail: str, code: ErrorCode) -> PersonalisationTrainingError:
        ledger.record(code=code, entity_type=_ENTITY_BUILD, entity_id=pcfg.mlflow_experiment,
                      detail=detail, fallback_used=False)
        return PersonalisationTrainingError(detail, error_code=code)

    missing = sorted(set(_REQUIRED_COLUMNS) - set(matrix.columns))
    if missing:
        raise _fail(f"training matrix missing required columns: {missing}",
                    ErrorCode.MODEL_TRAINING_FAILED)

    # Quarantine held-out + non-train BEFORE any reduction, then the fail-loud tripwire.
    train_mask = (matrix[COL_SPLIT] == "train") & (
        ~matrix[COL_CIRCUIT].isin(config.held_out_circuits)
    )
    train = cast("pd.DataFrame", matrix.loc[train_mask].copy())
    try:
        _assert_no_leakage(train, config)
    except PersonalisationTrainingError as exc:
        ledger.record(code=exc.error_code, entity_type=_ENTITY_BUILD,
                      entity_id=pcfg.mlflow_experiment, detail=str(exc), fallback_used=False)
        raise
    if len(train) == 0:
        raise _fail("no training rows after excluding held-out / non-train split",
                    ErrorCode.MODEL_TRAINING_FAILED)

    try:
        norm_params = load_normalisation_params(art_dir / config.normalise.artefact_filename)
    except NormalisationError as exc:
        raise _fail(f"normalisation artefact unavailable: {exc}",
                    ErrorCode.MODEL_TRAINING_FAILED) from exc

    cells, n_thin = _build_cells(train, norm_params, config, min_clean, pcfg.top_quartile_pct, log,
                                 ledger)
    # Keep only cells whose driver has a 5D fingerprint in the style space.
    kept: list[_Cell] = []
    n_dropped = n_thin
    for cell in cells:
        if cell.driver not in style_handle.known_coords_5d:
            n_dropped += 1
            ledger.record(code=ErrorCode.DATA_INSUFFICIENT, entity_type=_ENTITY_CELL,
                          entity_id=f"{cell.driver}/{cell.circuit}",
                          detail=f"{REASON_NO_COORD}: no 5D fingerprint; cell dropped",
                          fallback_used=False)
            continue
        kept.append(cell)
    if not kept:
        raise _fail(f"{REASON_NO_CELLS}: no (driver, circuit) cell had >= {min_clean} clean laps "
                    "and a 5D fingerprint", ErrorCode.MODEL_TRAINING_FAILED)

    # Build X (5D ‖ one-hot) + the 10 approximated targets (min-max over the kept aggregates).
    agg_matrix = np.vstack([c.aggregate for c in kept])
    fmin, fmax = _minmax_stats(agg_matrix)
    x = np.vstack([
        _build_input(style_handle.known_coords_5d[c.driver], c.circuit_type, circuit_types)
        for c in kept
    ])
    groups = np.array([c.driver for c in kept])
    input_dim = x.shape[1]

    regressors: dict[str, Any] = {}
    reliability: dict[str, float] = {}
    explainers: dict[str, Any] = {}
    params = _base_xgb_params(seed)
    for spec in _SETUP_PARAMETERS:
        # Champion targets use the GLOBAL kept-cell min/max (the champion is fit on all data by
        # design — no held-out); reliability recomputes them fold-locally (§4.0-c / _reliability).
        y = np.array([
            _target_for(c.aggregate, spec, fmin, fmax, pcfg.fia_bounds, min_std) for c in kept
        ])
        reliability[spec.name] = _reliability(
            x, groups, agg_matrix, spec, params=params, n_splits=pcfg.cv_folds,
            fia_bounds=pcfg.fia_bounds, min_std=min_std,
        )
        model = xgb.XGBRegressor(**params)
        model.fit(x, y)
        regressors[spec.name] = model
        explainers[spec.name] = _build_explainer(model, ledger, log)

    bundle = {
        "schema_version": SCHEMA_VERSION,
        "param_order": list(_PARAM_NAMES),
        "regressors": regressors,
        "reliability": reliability,
        "fia_bounds": {k: list(v) for k, v in pcfg.fia_bounds.items()},
        "circuit_types": list(circuit_types),
        "input_dim": input_dim,
    }
    path = art_dir / pcfg.model_artefact_filename
    sha_path = Path(f"{path}.sha256")
    try:
        joblib_dump(bundle, path)
        content_hash = sha256(path.read_bytes()).hexdigest()
        sha_path.write_text(content_hash, encoding="utf-8")  # integrity sidecar (F08 precedent)
    except OSError as exc:
        raise _fail(f"artefact write fault: {exc}", ErrorCode.MODEL_TRAINING_FAILED) from exc

    n_reliable = sum(1 for r in reliability.values() if r >= pcfg.confidence_medium)
    mlflow_run_id, mlflow_degraded = _log_mlflow(config, art_dir=art_dir, path=path, ledger=ledger,
                                                 n_cells=len(kept), reliability=reliability,
                                                 n_reliable=n_reliable)

    # A dropped (thin / no-fingerprint) cell degrades the result AND is surfaced on flags (§6 row).
    degraded = mlflow_degraded or n_dropped > 0
    flags = [ErrorCode.DATA_QUALITY_LOG_UNWRITABLE] if ledger.failed else []
    if n_dropped > 0:
        flags.append(ErrorCode.DATA_INSUFFICIENT)
    return PersonalisationFitResult(
        status=Status.DEGRADED if degraded else Status.OK, error_code=None, flags=flags,
        logged=ledger.rows > 0, artefact_path=str(path), content_hash=content_hash,
        n_train_cells=len(kept), parameters_fitted=len(regressors), reliability=reliability,
        n_reliable=n_reliable, mlflow_run_id=mlflow_run_id,
    )


def _build_explainer(model: Any, ledger: _DQLedger, log: Path) -> Any:
    """Build an exact `TreeExplainer` over one regressor (F07 pattern, copied per regressor). A
    build fault → None + a ``SHAP_UNAVAILABLE`` row; serving degrades (`shap_available=False`),
    never blocks (§6.2 precedent)."""
    try:
        return shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    except Exception as exc:  # noqa: BLE001 — a SHAP build fault never blocks a recommendation
        ledger.record(code=ErrorCode.SHAP_UNAVAILABLE, entity_type=_ENTITY_MODEL,
                      entity_id="startup",
                      detail=f"{REASON_EXPLAINER}: {type(exc).__name__}: {exc}",
                      fallback_used=True)
        return None


def _log_mlflow(
    config: AppConfig, *, art_dir: Path, path: Path, ledger: _DQLedger, n_cells: int,
    reliability: dict[str, float], n_reliable: int,
) -> tuple[str | None, bool]:
    """MLflow logging (own experiment) — degrade-don't-die (F10 mechanism); NO model registry."""
    assert config.personalisation is not None
    pcfg = config.personalisation
    run_id: str | None = None
    degraded = False
    try:
        mlflow.set_tracking_uri(pcfg.mlflow_tracking_uri)
        art_loc = (art_dir / "mlartifacts").resolve().as_uri()
        client = mlflow.tracking.MlflowClient()
        if client.get_experiment_by_name(pcfg.mlflow_experiment) is None:
            client.create_experiment(pcfg.mlflow_experiment, artifact_location=art_loc)
        mlflow.set_experiment(pcfg.mlflow_experiment)
        with mlflow.start_run(run_name="personalisation-train") as run:
            mlflow.log_params({
                "top_quartile_pct": pcfg.top_quartile_pct, "cv_folds": pcfg.cv_folds,
                "confidence_floor": pcfg.confidence_floor, "n_parameters": len(_SETUP_PARAMETERS),
            })
            mlflow.log_metric("n_train_cells", float(n_cells))
            mlflow.log_metric("n_reliable", float(n_reliable))
            for name, r in reliability.items():
                mlflow.log_metric(f"reliability_{name}", float(r))
            mlflow.log_artifact(str(path))
            run_id = run.info.run_id
    except Exception as exc:  # noqa: BLE001 — MLflow fault degrades, never aborts the write
        degraded = True
        ledger.record(code=ErrorCode.MODEL_TRAINING_FAILED, entity_type=_ENTITY_BUILD,
                      entity_id=pcfg.mlflow_experiment, detail=f"{REASON_MLFLOW}: {exc}",
                      fallback_used=True)
    return run_id, degraded


# ----------------------------------------------------------------------------------
# Load + health (§4.5)
# ----------------------------------------------------------------------------------


def load_personalisation(
    *,
    config: AppConfig,
    artefact_dir: str | Path | None = None,
    log_path: str | Path | None = None,
) -> PersonalisationHandle:
    """Load the serving bundle. Missing / corrupt / wrong-regressor-count / missing-reliability /
    FIA-or-vocab-mismatch → DQ-log then **raise** :class:`PersonalisationLoadError`
    (``MODEL_NOT_FOUND``). Explainers are rebuilt best-effort (a build fault degrades that
    parameter's SHAP, never refuses — F07 precedent)."""
    if config.personalisation is None:
        raise PersonalisationLoadError("config.personalisation block absent; cannot load")
    pcfg = config.personalisation
    circuit_types = tuple(config.circuit_types)
    art_dir = Path(artefact_dir) if artefact_dir is not None else Path(config.paths.artefacts_dir)
    log = Path(log_path) if log_path is not None else Path(config.paths.data_quality_log)
    ledger = _DQLedger(log_path=log)

    path = art_dir / pcfg.model_artefact_filename
    if not path.is_file():
        raise _load_fail(ledger, f"personalisation artefact not found: {path}")
    raw_bytes = path.read_bytes()
    # Integrity: the pkl bytes must match their .sha256 sidecar (detects a tampered-but-valid
    # pickle; F08 precedent). A missing sidecar or a mismatch refuses to serve.
    sha_path = Path(f"{path}.sha256")
    if not sha_path.is_file():
        raise _load_fail(ledger, f"integrity sidecar not found: {sha_path.name}")
    if sha256(raw_bytes).hexdigest() != sha_path.read_text(encoding="utf-8").strip():
        raise _load_fail(ledger, "content-hash mismatch (corrupt/tampered)")
    try:
        bundle = joblib_load(path)
    except Exception as exc:  # noqa: BLE001 — corrupt/unpicklable bundle = no usable model
        raise _load_fail(ledger, f"personalisation artefact unreadable/corrupt: {exc}") from exc
    if not isinstance(bundle, dict) or bundle.get("schema_version") != SCHEMA_VERSION:
        raise _load_fail(ledger, "personalisation artefact schema mismatch")

    regressors = bundle.get("regressors", {})
    reliability = bundle.get("reliability", {})
    if set(regressors) != set(_PARAM_NAMES):
        raise _load_fail(ledger, f"regressor set mismatch: expected the 10 parameters, got "
                                 f"{sorted(regressors)}")
    if set(reliability) != set(_PARAM_NAMES):
        raise _load_fail(ledger, "reliability table missing one or more parameters")

    live_bounds: dict[str, tuple[float, float]] = dict(pcfg.fia_bounds)
    art_bounds = {k: tuple(v) for k, v in bundle.get("fia_bounds", {}).items()}
    if art_bounds != {k: tuple(v) for k, v in live_bounds.items()}:
        raise _load_fail(ledger, "fia_bounds mismatch (artefact vs live) — recalibrate")
    if tuple(bundle.get("circuit_types", [])) != circuit_types:
        raise _load_fail(ledger, "circuit_types vocab mismatch (artefact vs live) — recalibrate")

    explainers = {name: _build_explainer(regressors[name], ledger, log) for name in _PARAM_NAMES}
    model_version = f"sha256:{sha256(raw_bytes).hexdigest()[:_VERSION_HASH_LEN]}"
    return PersonalisationHandle(
        regressors=regressors, reliability={k: float(v) for k, v in reliability.items()},
        explainers=explainers, fia_bounds=live_bounds, circuit_types=circuit_types,
        input_dim=int(bundle.get("input_dim", _N_FP_DIMS + max(len(circuit_types) - 1, 0))),
        model_version=model_version,
    )


def _load_fail(ledger: _DQLedger, detail: str) -> PersonalisationLoadError:
    ledger.record(code=ErrorCode.MODEL_NOT_FOUND, entity_type=_ENTITY_MODEL, entity_id="startup",
                  detail=detail, fallback_used=False)
    return PersonalisationLoadError(detail)


def health(handle: PersonalisationHandle | None) -> PersonalisationHealthResult:
    """Readiness for F13's ``GET /health``. Not-ready → ``HEALTH_FAIL`` (→ 503) with reason
    ``MODEL_NOT_FOUND``; ready → OK + ``model_version`` (F06b/F08/F10 precedent)."""
    if handle is None or not handle.ready:
        return PersonalisationHealthResult(
            status=Status.FAILED, error_code=ErrorCode.HEALTH_FAIL, ready=False,
            reason=ErrorCode.MODEL_NOT_FOUND,
        )
    return PersonalisationHealthResult(status=Status.OK, ready=True,
                                       model_version=handle.model_version)


# ----------------------------------------------------------------------------------
# Serving — recommend_setup / onboard_unknown (read-only, never raise)
# ----------------------------------------------------------------------------------


def recommend_setup(
    *,
    driver: str,
    circuit: str,
    circuit_type: str | None,
    handle: PersonalisationHandle,
    style_handle: StyleSpaceHandle,
    config: AppConfig,
    log_path: str | Path | None = None,
) -> SetupRecommendation:
    """Recommend a car setup for a **known** driver × circuit. **Never raises** — every fault
    (unknown driver, unknown circuit type, internal error) is a typed :class:`SetupRecommendation`
    (all-`WIDE` on a backstop) + a ``data_quality_log`` row. Nothing that could raise (config/handle
    access, model construction) runs before the ``try`` — a ``None``/malformed config or handle is a
    backstop result, never a propagated exception."""
    ledger = _DQLedger(log_path=_safe_log_path(config, log_path))
    base = SetupRecommendation()  # defaults only — cannot raise (Finding: no pre-try escape)
    try:
        if config.personalisation is None:
            raise ValueError("config.personalisation block absent")
        if handle is None or not getattr(handle, "ready", False):
            raise ValueError("personalisation handle is None or not ready")
        base.driver, base.circuit, base.circuit_type = str(driver), str(circuit), circuit_type
        base.model_version = handle.model_version
        base.threshold_source = "driver_circuit_type"
        if not driver or not circuit:  # malformed serve input → INVALID_INPUT (§6)
            ledger.record(code=ErrorCode.INVALID_INPUT, entity_type=_ENTITY_SETUP,
                          entity_id=f"{driver}/{circuit}",
                          detail=f"{REASON_INVALID}: empty driver/circuit", fallback_used=False)
            base.status = Status.FAILED
            base.error_code = ErrorCode.INVALID_INPUT
            base.logged = ledger.rows > 0
            return base
        if driver not in style_handle.known_coords_5d:
            ledger.record(code=ErrorCode.INVALID_INPUT, entity_type=_ENTITY_SETUP,
                          entity_id=f"{driver}/{circuit}",
                          detail=f"{REASON_UNKNOWN_DRIVER}: driver not in the style space",
                          fallback_used=False)
            base.status = Status.FAILED
            base.error_code = ErrorCode.INVALID_INPUT
            base.logged = ledger.rows > 0
            return base
        coord5 = np.asarray(style_handle.known_coords_5d[driver], dtype=float)
        grid_means = _grid_reference(
            style_handle.known_coords_5d, circuit=circuit, circuit_type=circuit_type,
            handle=handle, config=config.personalisation, ledger=ledger,
            entity_id=f"{driver}/{circuit}",
        )
        rec = _recommend_from_coord(
            coord5, circuit=circuit, circuit_type=circuit_type, handle=handle,
            config=config.personalisation, ledger=ledger, ood_max_pc_sd=0.0,
            entity_id=f"{driver}/{circuit}", grid_means=grid_means,
        )
        rec.driver, rec.circuit, rec.circuit_type = driver, circuit, circuit_type
        rec.model_version = handle.model_version
        rec.threshold_source = "driver_circuit_type"
        rec.logged = ledger.rows > 0
        if ledger.failed:
            rec.flags = _dedup([*rec.flags, ErrorCode.DATA_QUALITY_LOG_UNWRITABLE])
        return rec
    except Exception as exc:  # noqa: BLE001 — the never-raises backstop
        return _backstop_recommendation(base, ledger, config, handle, exc)


def onboard_unknown(
    laps: pd.DataFrame,
    *,
    circuit: str,
    circuit_type: str | None,
    handle: PersonalisationHandle,
    style_handle: StyleSpaceHandle,
    config: AppConfig,
    log_path: str | Path | None = None,
) -> OnboardingPackage:
    """§7.4 onboarding for an uploaded unknown driver. **Never raises** — projects via F08's
    ``project_unknown`` (which never raises); a ``coords_5d=None`` fallback yields a WIDE-heavy
    setup sheet, never a crash."""
    log = _safe_log_path(config, log_path)
    ledger = _DQLedger(log_path=log)
    pkg = OnboardingPackage()
    try:
        if config.personalisation is None:
            raise ValueError("config.personalisation block absent")
        if handle is None or not getattr(handle, "ready", False):
            raise ValueError("personalisation handle is None or not ready")
        snapshot = project_unknown(laps.reindex(columns=list(FEATURE_IDS)), circuit=circuit,
                                   handle=style_handle, config=config, log_path=log)
        pkg.fingerprint_snapshot = snapshot
        pkg.fingerprint_available = snapshot.coords_5d is not None
        pkg.style_match = [
            Neighbour(driver=n.driver, cosine_similarity=n.cosine_similarity, distance=n.distance)
            for n in snapshot.nearest_neighbours
        ]
        pkg.risk_flags = list(snapshot.risk_flags)
        pkg.n_laps_used = snapshot.n_laps_used
        pkg.n_laps_rejected = snapshot.laps_rejected

        if snapshot.coords_5d is None:
            pkg.setup = _wide_only_recommendation(
                circuit, circuit_type, handle, config.personalisation
            )
        else:
            coord5 = np.asarray(snapshot.coords_5d, dtype=float)
            grid_means = _grid_reference(
                style_handle.known_coords_5d, circuit=circuit, circuit_type=circuit_type,
                handle=handle, config=config.personalisation, ledger=ledger,
                entity_id=f"unknown/{circuit}",
            )
            pkg.setup = _recommend_from_coord(
                coord5, circuit=circuit, circuit_type=circuit_type, handle=handle,
                config=config.personalisation, ledger=ledger, ood_max_pc_sd=snapshot.ood_max_pc_sd,
                entity_id=f"unknown/{circuit}", grid_means=grid_means,
            )
            pkg.setup.circuit, pkg.setup.circuit_type = circuit, circuit_type
            pkg.setup.model_version = handle.model_version
            pkg.setup.threshold_source = "unknown_driver"
        pkg.status = Status.DEGRADED if snapshot.status is not Status.OK else Status.OK
        pkg.logged = ledger.rows > 0
        if ledger.failed:
            pkg.flags = [ErrorCode.DATA_QUALITY_LOG_UNWRITABLE]
        return pkg
    except Exception as exc:  # noqa: BLE001 — the never-raises backstop
        try:
            ledger.record(code=ErrorCode.INFERENCE_ERROR, entity_type=_ENTITY_SETUP,
                          entity_id=f"unknown/{circuit}",
                          detail=f"{REASON_BACKSTOP}: {type(exc).__name__}: {exc}",
                          fallback_used=False)
        except Exception:  # noqa: BLE001
            pass
        pkg.status = Status.FAILED
        pkg.error_code = ErrorCode.INFERENCE_ERROR
        pkg.setup = _wide_sheet_from_bounds(circuit, circuit_type, handle, config)  # always 10
        pkg.logged = ledger.rows > 0
        return pkg


# ----------------------------------------------------------------------------------
# The shared recommendation core (Layers 1–2–3 + SHAP) — pure given a 5D coord
# ----------------------------------------------------------------------------------


def _grid_reference(
    known_coords_5d: dict[str, Any], *, circuit: str, circuit_type: str | None,
    handle: PersonalisationHandle, config: PersonalisationConfig, ledger: _DQLedger,
    entity_id: str,
) -> dict[str, float]:
    """The serve-time grid reference (system input #2): per numeric parameter, the mean of the
    fitted regressor's predictions over EVERY known driver in the same circuit context, put through
    the same Monaco override + FIA clip as a served value (rounding skipped — a mean is not a
    setting). Predict-only — never a fit. A fault degrades to ``{}`` (tendencies simply absent)
    with a logged row; it never blocks the recommendation itself."""
    try:
        if not known_coords_5d:
            return {}
        rows = [
            _build_input(np.asarray(c, dtype=float), circuit_type, handle.circuit_types)
            for c in known_coords_5d.values()
        ]
        grid_x = np.stack(rows)
        is_monaco = circuit == config.monaco_circuit
        out: dict[str, float] = {}
        for spec in _SETUP_PARAMETERS:
            if spec.kind is SetupOutputKind.CATEGORY:
                continue  # a mean over category ordinals has no honest reading
            lo, hi = handle.fia_bounds[spec.name]
            if is_monaco and spec.name in (_FRONT_WING, _REAR_WING):
                out[spec.name] = float(hi)  # the override applies to every driver equally
                continue
            preds = np.asarray(handle.regressors[spec.name].predict(grid_x), dtype=float)
            out[spec.name] = float(np.clip(preds, lo, hi).mean())
        return out
    except Exception as exc:  # noqa: BLE001 — a reference stat never blocks the recommendation
        ledger.record(code=ErrorCode.INFERENCE_ERROR, entity_type=_ENTITY_SETUP,
                      entity_id=entity_id,
                      detail=f"{REASON_GRID_REFERENCE}: {type(exc).__name__}: {exc}",
                      fallback_used=True)
        return {}


def _recommend_from_coord(
    coord5: np.ndarray, *, circuit: str, circuit_type: str | None, handle: PersonalisationHandle,
    config: PersonalisationConfig, ledger: _DQLedger, ood_max_pc_sd: float, entity_id: str,
    grid_means: dict[str, float] | None = None,
) -> SetupRecommendation:
    """Run Layers 1–3 + SHAP over one 5D coordinate → the 10 `SetupParameter`s (always 10, never a
    silent partial). Shared by `recommend_setup` (attenuation 1.0) and `onboard_unknown`."""
    x = _build_input(coord5, circuit_type, handle.circuit_types).reshape(1, -1)
    input_labels = _input_labels(handle.circuit_types)
    attenuation = _ood_attenuation(ood_max_pc_sd, config)
    is_monaco = circuit == config.monaco_circuit
    flags: list[ErrorCode] = []
    # A None OR not-in-vocab circuit type → all-zero (generic) one-hot; flag the fallback either way
    # so a caller with no circuit-type info is never silently mapped to the reference circuit.
    if circuit_type is None or circuit_type not in handle.circuit_types:
        flags.append(ErrorCode.FALLBACK_THRESHOLD)
        ledger.record(code=ErrorCode.FALLBACK_THRESHOLD, entity_type=_ENTITY_SETUP,
                      entity_id=entity_id,
                      detail=f"{REASON_UNKNOWN_CIRCUIT_TYPE}: {circuit_type!r}; generic mapping",
                      fallback_used=True)

    params: list[SetupParameter] = []
    predicted: dict[str, float] = {}
    for spec in _SETUP_PARAMETERS:
        lo, hi = handle.fia_bounds[spec.name]
        raw = float(handle.regressors[spec.name].predict(x)[0])
        monaco_here = is_monaco and spec.name in (_FRONT_WING, _REAR_WING)
        value = hi if monaco_here else raw  # §4.2-4a Monaco max-downforce override wins
        clipped = min(max(value, lo), hi)
        param_flags: list[ErrorCode] = []
        if not monaco_here and clipped != value:
            param_flags.append(ErrorCode.CONSTRAINT_CLIPPED)
            ledger.record(code=ErrorCode.CONSTRAINT_CLIPPED, entity_type=_ENTITY_SETUP,
                          entity_id=f"{entity_id}/{spec.name}",
                          detail=f"{REASON_CLIP}: {value:.4f} -> {clipped:.4f} [{lo}, {hi}]",
                          fallback_used=False)
        clipped = _round_by_kind(clipped, spec.kind)
        predicted[spec.name] = clipped

        # A Monaco-overridden wing is a hard FIA rule → maximally certain (confidence 1.0 / HIGH),
        # keeping the numeric confidence and rating consistent for the UI gauge.
        confidence = 1.0 if monaco_here else float(
            min(max(handle.reliability[spec.name] * attenuation, 0.0), 1.0)
        )
        rating = SetupConfidence.HIGH if monaco_here else _rating_of(confidence, config)
        param = _assemble_parameter(spec, clipped, lo, hi, confidence, rating, config, monaco_here,
                                    param_flags, ledger, entity_id)
        param.grid_mean = (grid_means or {}).get(spec.name)
        param.basis_features = list(spec.basis)
        params.append(param)

    # §4.2-4d physical consistency (checked AFTER clipping; flag only, never a value change).
    consistency = (
        predicted[_FRONT_WING] > config.consistency_front_wing_deg
        and predicted[_BRAKE_BIAS] < config.consistency_brake_bias_pct
    )
    if consistency:
        flags.append(ErrorCode.CONSISTENCY_WARNING)
        ledger.record(code=ErrorCode.CONSISTENCY_WARNING, entity_type=_ENTITY_SETUP,
                      entity_id=entity_id,
                      detail=f"{REASON_CONSISTENCY}: front_wing={predicted[_FRONT_WING]:.2f} > "
                             f"{config.consistency_front_wing_deg}, brake_bias="
                             f"{predicted[_BRAKE_BIAS]:.2f} < {config.consistency_brake_bias_pct}",
                      fallback_used=False)

    # SHAP per parameter (over the model's own 8 inputs — §4.2-5; a fault degrades, never blocks).
    for param, spec in zip(params, _SETUP_PARAMETERS, strict=True):
        _attach_shap(param, handle.explainers[spec.name], x, input_labels, config.shap_top_k,
                     ledger, entity_id)

    monaco_override = is_monaco
    if monaco_override:
        flags.append(ErrorCode.CIRCUIT_OVERRIDE_MONACO)
        ledger.record(code=ErrorCode.CIRCUIT_OVERRIDE_MONACO, entity_type=_ENTITY_SETUP,
                      entity_id=entity_id,
                      detail=f"{REASON_MONACO}: front+rear wing forced to max legal downforce",
                      fallback_used=False)
    for p in params:
        flags.extend(p.constraint_flags)
    degraded = bool(flags) or attenuation < 1.0
    return SetupRecommendation(
        status=Status.DEGRADED if degraded else Status.OK, error_code=None, flags=_dedup(flags),
        circuit=circuit, circuit_type=circuit_type, parameters=params,
        consistency_warning=consistency, monaco_override=monaco_override,
    )


def _assemble_parameter(
    spec: _ParamSpec, value: float, lo: float, hi: float, confidence: float,
    rating: SetupConfidence, config: PersonalisationConfig, monaco_here: bool,
    param_flags: list[ErrorCode], ledger: _DQLedger, entity_id: str,
) -> SetupParameter:
    """Build one `SetupParameter`, applying the confidence floor → WIDE range (§4.2-4c). A Monaco
    override is a hard certainty — it is never converted to a WIDE range."""
    category = _category_of(value) if spec.kind is SetupOutputKind.CATEGORY else None
    if monaco_here:  # a hard override wins — point value, never a range
        return SetupParameter(
            name=spec.name, kind=spec.kind, value=value, category=category, confidence=confidence,
            rating=rating, unit=spec.unit, fia_min=lo, fia_max=hi,
            constraint_flags=[*param_flags, ErrorCode.CIRCUIT_OVERRIDE_MONACO],
        )
    if rating is SetupConfidence.WIDE:
        span_lo = min(max(value * (1.0 - config.wide_range_pct), lo), hi)
        span_hi = min(max(value * (1.0 + config.wide_range_pct), lo), hi)
        ledger.record(code=ErrorCode.WIDE_CONFIDENCE, entity_type=_ENTITY_SETUP,
                      entity_id=f"{entity_id}/{spec.name}",
                      detail=f"{REASON_WIDE}: {confidence:.3f} < {config.confidence_floor}",
                      fallback_used=False)
        return SetupParameter(
            name=spec.name, kind=spec.kind, value=None,
            value_range=(round(span_lo, 6), round(span_hi, 6)), category=None,
            confidence=confidence, rating=rating, unit=spec.unit, fia_min=lo, fia_max=hi,
            constraint_flags=[*param_flags, ErrorCode.WIDE_CONFIDENCE],
        )
    return SetupParameter(
        name=spec.name, kind=spec.kind, value=value, category=category, confidence=confidence,
        rating=rating, unit=spec.unit, fia_min=lo, fia_max=hi, constraint_flags=list(param_flags),
    )


def _attach_shap(
    param: SetupParameter, explainer: Any, x: np.ndarray, input_labels: list[str], k: int,
    ledger: _DQLedger, entity_id: str,
) -> None:
    """Attach the top-k SHAP drivers (over the regressor's 8 inputs, §4.2-5). A fault →
    ``shap_available=False`` + a ``SHAP_UNAVAILABLE`` row; the parameter is still returned."""
    if explainer is None:
        return
    try:
        values = np.asarray(explainer(x).values, dtype=float)
        row = values[0] if values.ndim == 2 else values  # regressor → (1, n) or (n,)
        order = np.argsort(np.abs(row))[::-1][:k]
        param.shap_top_features = [input_labels[i] for i in order]
        param.shap_available = True
    except Exception as exc:  # noqa: BLE001 — a SHAP fault never blocks a recommendation
        ledger.record(code=ErrorCode.SHAP_UNAVAILABLE, entity_type=_ENTITY_SETUP,
                      entity_id=f"{entity_id}/{param.name}",
                      detail=f"{REASON_SHAP}: {type(exc).__name__}: {exc}", fallback_used=True)
        param.shap_available = False


def _wide_only_recommendation(
    circuit: str, circuit_type: str | None, handle: PersonalisationHandle,
    config: PersonalisationConfig,
) -> SetupRecommendation:
    """A fully-WIDE sheet used when no 5D coordinate is available (sparse unknown driver, §4.3, or
    the backstop). Every parameter is a range at its FIA midpoint; confidence 0.0 (never NULL)."""
    params: list[SetupParameter] = []
    for spec in _SETUP_PARAMETERS:
        lo, hi = handle.fia_bounds[spec.name]
        mid = _round_by_kind((lo + hi) / 2.0, spec.kind)
        span_lo = min(max(mid * (1.0 - config.wide_range_pct), lo), hi)
        span_hi = min(max(mid * (1.0 + config.wide_range_pct), lo), hi)
        params.append(SetupParameter(
            name=spec.name, kind=spec.kind, value=None,
            value_range=(round(span_lo, 6), round(span_hi, 6)), confidence=0.0,
            rating=SetupConfidence.WIDE, unit=spec.unit, fia_min=lo, fia_max=hi,
            constraint_flags=[ErrorCode.WIDE_CONFIDENCE],
        ))
    return SetupRecommendation(
        status=Status.DEGRADED, error_code=None, flags=[ErrorCode.WIDE_CONFIDENCE], circuit=circuit,
        circuit_type=circuit_type, parameters=params, model_version=handle.model_version,
    )


def _resolve_fia_bounds(
    handle: PersonalisationHandle | None, config: AppConfig,
) -> dict[str, tuple[float, float]] | None:
    """Prefer the loaded handle's bounds (available even when the config block is the fault source);
    fall back to the config block. Lets the backstop honour "always 10 parameters" on any fault."""
    if handle is not None and getattr(handle, "fia_bounds", None):
        return handle.fia_bounds
    pers = getattr(config, "personalisation", None)  # getattr guards a None/malformed config
    if pers is not None:
        return dict(pers.fia_bounds)
    return None


def _full_bounds_wide_params(bounds: dict[str, tuple[float, float]]) -> list[SetupParameter]:
    """Ten all-`WIDE` parameters spanning the full FIA band (backstop / no-config path). Confidence
    is 0.0 (never NULL), value is a range, never a point."""
    return [
        SetupParameter(
            name=spec.name, kind=spec.kind, value=None,
            value_range=(round(bounds[spec.name][0], 6), round(bounds[spec.name][1], 6)),
            confidence=0.0, rating=SetupConfidence.WIDE, unit=spec.unit,
            fia_min=bounds[spec.name][0], fia_max=bounds[spec.name][1],
            constraint_flags=[ErrorCode.WIDE_CONFIDENCE],
        )
        for spec in _SETUP_PARAMETERS
    ]


def _wide_sheet_from_bounds(
    circuit: str, circuit_type: str | None, handle: PersonalisationHandle | None, config: AppConfig,
) -> SetupRecommendation | None:
    """A full-`WIDE` sheet for the onboarding backstop, sourcing bounds from the handle/config."""
    bounds = _resolve_fia_bounds(handle, config)
    if bounds is None:
        return None
    return SetupRecommendation(
        status=Status.FAILED, error_code=ErrorCode.INFERENCE_ERROR,
        flags=[ErrorCode.WIDE_CONFIDENCE], circuit=circuit, circuit_type=circuit_type,
        parameters=_full_bounds_wide_params(bounds),
        model_version=handle.model_version if handle is not None else "",
    )


def _backstop_recommendation(
    base: SetupRecommendation, ledger: _DQLedger, config: AppConfig,
    handle: PersonalisationHandle | None, exc: Exception,
) -> SetupRecommendation:
    """The never-raises backstop: a typed all-`WIDE` `INFERENCE_ERROR` result with **all 10
    parameters** (bounds resolved from the handle so the "always 10" contract holds even when the
    config block is the fault source), confidence never NULL, never a propagated exception."""
    try:
        ledger.record(code=ErrorCode.INFERENCE_ERROR, entity_type=_ENTITY_SETUP,
                      entity_id=f"{base.driver}/{base.circuit}",
                      detail=f"{REASON_BACKSTOP}: {type(exc).__name__}: {exc}", fallback_used=False)
    except Exception:  # noqa: BLE001 — even the logger must not break never-raises
        pass
    bounds = _resolve_fia_bounds(handle, config)
    if bounds is not None:
        base.parameters = _full_bounds_wide_params(bounds)
    base.status = Status.FAILED
    base.error_code = ErrorCode.INFERENCE_ERROR
    base.logged = ledger.rows > 0
    return base
