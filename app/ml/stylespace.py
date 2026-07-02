"""F08 — Model 2 (driver style space): UMAP · PCA fallback · unknown-driver projection.

Projects each driver's aggregate 28-feature distribution into one continuous style space — a
**5D fingerprint** (consumed by F10/F11) and a **2D visualisation** (Screen 5). The space is fit on
**normalised training laps only**; an **unknown driver projects in by a forward pass**
(``umap.transform`` — no retrain). Every projection carries an explicit confidence gate (sparse /
out-of-distribution) and a **fail-loud fallback chain** (UMAP → PCA → standardised-aggregate
cosine NN). Like
F07, F08 introduces **no new error code** — the four §5.2 codes and the ``OUT_OF_DISTRIBUTION``
``ConfidenceState`` are pre-seeded.

The 56-dim aggregate mixes z-scored (~unit-scale) and raw-physical-unit columns spanning ~4 orders
of magnitude, so both fit sites first rebalance it with a **train-only aggregate StandardScaler**
(persisted in the PCA bundle, applied read-only at every forward pass; a bundle without it — or a
UMAP fitted against a different scaler — never serves). A documented master gap-fill (spec §9.7).

There are exactly **two fit sites** — :func:`build_style_space` (the production model) and the
dissertation-only :func:`stability_report` — both train-only and leakage-guarded.
The five **serving** functions (:func:`load_style_space`, :func:`project_unknown`,
:func:`get_style_space`, :func:`health`, :func:`teammate_proximity`) perform **no ``.fit(``** and
**never raise** (every fault is a coded result). Normalisation is reused via F04's read-only
:func:`apply_normalisation` — **never refit** here. No silent failure (#1); every threshold
is read from ``config`` (no magic numbers); the OOD/risk standardisation floor is
``config.normalise.min_std`` (fallback :data:`_MIN_STD`).
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import mlflow
import numpy as np
import pandas as pd
import umap
from joblib import dump as joblib_dump
from joblib import load as joblib_load
from pydantic import BaseModel, ConfigDict, Field
from scipy.spatial import procrustes
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from app.contracts.config import AppConfig
from app.contracts.enums import ConfidenceState, Status
from app.contracts.errors import ErrorCode
from app.contracts.logging import DataQualityLogError, log_data_quality
from app.contracts.results import BaseResult
from app.pipeline.features import FEATURE_IDS
from app.pipeline.normalise import (
    NormalisationError,
    NormalisationParams,
    apply_normalisation,
    load_normalisation_params,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

# Required non-feature columns in the wide training matrix (mirrors classifier._REQUIRED_COLUMNS).
COL_DRIVER = "driver"
COL_CIRCUIT = "circuit"
COL_SEASON = "season"
COL_SPLIT = "split"
_REQUIRED_COLUMNS = (COL_DRIVER, COL_CIRCUIT, COL_SEASON, COL_SPLIT, *FEATURE_IDS)

# Structural constants (not tunable thresholds — those live in config; excluded from the literal
# scan). ``_AGGREGATE_DDOF = 0`` matches F04's population-std convention so a single lap yields
# std=0.0 (finite, not NaN). ``_MIN_STD`` is the standardisation floor used only when
# ``config.normalise`` is absent (normally ``config.normalise.min_std`` supplies it).
_AGGREGATE_DDOF = 0
_MIN_STD = 1.0e-8
_MIN_UMAP_NEIGHBORS = 2  # UMAP structural minimum (the config validator also pins n_neighbors >= 2)
_N_FEATURES = 2 * len(FEATURE_IDS)  # 56 = 28 means + 28 stds
_VERSION_HASH_LEN = 16
_ENGINE_UMAP = "umap"
_ENGINE_PCA = "pca"
_ENGINE_RAW_NN = "raw_nn"

# data_quality_log detail tags (mirrors normalise.py / inference.py REASON_* convention).
REASON_SPARSE = "SPARSE_UNKNOWN_DRIVER"
REASON_OOD = "OUT_OF_DISTRIBUTION"
REASON_UMAP_CORRUPT = "UMAP_ARTEFACT_CORRUPT"
REASON_UMAP_ABSENT = "UMAP_ARTEFACT_ABSENT"
REASON_TRANSFORM_RAISED = "TRANSFORM_RAISED"
REASON_NON_FINITE_AGGREGATE = "NON_FINITE_AGGREGATE"
REASON_ZERO_LAPS = "ZERO_CLEAN_LAPS"
REASON_PCA_ABSENT = "PCA_FALLBACK_ABSENT"
REASON_PCA_CORRUPT = "PCA_FALLBACK_CORRUPT"
REASON_PARAMS_UNAVAILABLE = "NORMALISATION_UNAVAILABLE"
REASON_SCALER_ABSENT = "AGGREGATE_SCALER_ABSENT"
REASON_BUNDLE_SKEW = "CROSS_BUNDLE_SCALER_MISMATCH"
REASON_MLFLOW = "MLFLOW_LOGGING_DEGRADED"
_ENTITY_UNKNOWN = "unknown_driver"
_ENTITY_MODEL = "model"
_ENTITY_BUILD = "training_run"


# ----------------------------------------------------------------------------------
# Exceptions (only the two fit-adjacent entry points raise; serving never does)
# ----------------------------------------------------------------------------------


class StyleSpaceBuildError(Exception):
    """Raised by :func:`build_style_space` on a build-breaking fault (no artefact on these paths).

    Carries ``TRAIN_LEAKAGE`` (held-out circuit reached the fit) or ``MODEL_TRAINING_FAILED``
    (missing columns / empty pool / too few drivers / a UMAP·PCA fit exception). Fail-loud (#1)."""

    def __init__(self, message: str, *, error_code: ErrorCode) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code


class StyleSpaceLoadError(Exception):
    """Raised by :func:`load_style_space` only when NO projection engine is usable (PCA and/or
    normalisation missing/corrupt). Carries ``MODEL_NOT_FOUND`` — the service refuses to start
    (mirrors F06b ``load_model``). A corrupt/missing UMAP alone is NOT this — it degrades to PCA."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = ErrorCode.MODEL_NOT_FOUND


# ----------------------------------------------------------------------------------
# Typed outputs
# ----------------------------------------------------------------------------------


class Neighbour(BaseModel):
    """One nearest known driver by cosine similarity."""

    model_config = ConfigDict(extra="forbid")

    driver: str
    cosine_similarity: float
    distance: float  # 1 - cosine_similarity


class StabilityReport(BaseResult):
    """Dissertation evidence (§5.2/§11.2): UMAP Procrustes stability over a fixed seed sequence.
    ``passes`` is a research finding, not a system error — never raises, never blocks."""

    procrustes_variance: float | None = None
    per_run_disparity: list[float] = Field(default_factory=list)
    n_runs: int = 0
    n_drivers: int = 0
    passes: bool = False


class TeammateProximityResult(BaseResult):
    """Dissertation evidence (§5.2/§11.2): mean intra- vs inter-team 5D distance. Never raises."""

    mean_intra_team: float | None = None
    mean_inter_team: float | None = None
    passes: bool = False
    teams_evaluated: int = 0


class StyleSpaceBuildResult(BaseResult):
    """Offline build summary. Emits driver-level coords AND per-``(driver, season)`` coords (the
    latter via a forward-pass ``transform`` of per-season aggregates — not a new fit)."""

    umap_artefact_path: str = ""
    umap_sha256: str = ""
    pca_artefact_path: str = ""
    drivers: list[str] = Field(default_factory=list)
    coords_5d: dict[str, list[float]] = Field(default_factory=dict)
    coords_2d: dict[str, list[float]] = Field(default_factory=dict)
    season_coords_5d: dict[str, dict[str, list[float]]] = Field(default_factory=dict)
    season_coords_2d: dict[str, dict[str, list[float]]] = Field(default_factory=dict)
    n_drivers: int = 0
    n_features: int = 0
    effective_n_neighbors: int = 0
    stability: StabilityReport | None = None
    proximity: TeammateProximityResult | None = None
    mlflow_run_id: str | None = None


class ProjectionResult(BaseResult):
    """One unknown-driver forward pass. ``confidence_state`` is ALWAYS populated (decoupled from
    ``status``); ``coords`` are ``None`` only on the ``PROJECTION_FALLBACK`` raw-NN path."""

    coords_5d: list[float] | None = None
    coords_2d: list[float] | None = None
    nearest_neighbours: list[Neighbour] = Field(default_factory=list)
    confidence_state: ConfidenceState = ConfidenceState.HIGH
    sparse: bool = False
    ood: bool = False
    ood_max_pc_sd: float = 0.0
    projection_engine: str = _ENGINE_UMAP
    risk_flags: list[str] = Field(default_factory=list)
    n_laps_used: int = 0
    laps_rejected: int = 0


class StyleSpaceResult(BaseResult):
    """All known drivers' coords for ``GET /stylespace`` — driver-level AND per-season."""

    drivers: list[str] = Field(default_factory=list)
    coords_5d: dict[str, list[float]] = Field(default_factory=dict)
    coords_2d: dict[str, list[float]] = Field(default_factory=dict)
    season_coords_5d: dict[str, dict[str, list[float]]] = Field(default_factory=dict)
    season_coords_2d: dict[str, dict[str, list[float]]] = Field(default_factory=dict)
    engine: str = _ENGINE_UMAP


class StyleSpaceHealthResult(BaseResult):
    """Readiness for F13's ``GET /health`` (mirrors F06b ``HealthResult``)."""

    ready: bool = False
    model_version: str = ""
    engine: str = ""
    reason: ErrorCode | None = None


@dataclass(frozen=True)
class StyleSpaceHandle:
    """The loaded, immutable serving bundle (built once at startup; mirrors F06b handle)."""

    umap_5d: Any | None
    umap_2d: Any | None
    pca: Any
    scaler: Any  # train-only aggregate standardiser (applied read-only at every forward pass)
    drivers: list[str]
    known_aggregate: np.ndarray  # n_drivers × 56 (raw-NN fallback + provenance)
    known_coords_5d: dict[str, np.ndarray]
    known_coords_2d: dict[str, np.ndarray]
    season_coords_5d: dict[str, dict[str, list[float]]]
    season_coords_2d: dict[str, dict[str, list[float]]]
    pc_std: np.ndarray  # per-PC training spread (OOD standardiser)
    feature_mean: np.ndarray  # per-dim mean of the 56-d aggregate matrix (risk_flags)
    feature_std: np.ndarray
    normalisation_params: NormalisationParams
    engine: str  # "umap" | "pca"
    model_version: str
    min_std: float
    ready: bool = True


# ----------------------------------------------------------------------------------
# data-quality ledger (never propagates a sink failure — mirrors F04/F06)
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


def _dedup(flags: list[ErrorCode]) -> list[ErrorCode]:
    """Order-preserving de-duplication (F04's carried flags + this layer's own can overlap)."""
    return list(dict.fromkeys(flags))


def _resolve_min_std(config: AppConfig) -> float:
    """The standardisation floor: ``config.normalise.min_std`` when present, else :data:`_MIN_STD`.
    Never a bare literal, so the magic-number scan stays clean."""
    if config.normalise is not None:
        return float(config.normalise.min_std)
    return _MIN_STD


def _scaler_sha(scaler: Any) -> str:
    """Deterministic fingerprint of the aggregate scaler's fitted params. Written into the UMAP
    bundle at build and compared at load, so a UMAP fitted against a DIFFERENT scaler (partial
    rebuild / pre-scaler artefact) is detected as cross-bundle skew and degraded — its geometry
    is meaningless in this scaled space and must never serve silently."""
    mean = np.asarray(scaler.mean_, dtype=float)
    scale = np.asarray(scaler.scale_, dtype=float)
    return sha256(mean.tobytes() + scale.tobytes()).hexdigest()


# ----------------------------------------------------------------------------------
# Pure helpers (unit-testable in isolation)
# ----------------------------------------------------------------------------------


def aggregate_driver(feature_rows: pd.DataFrame) -> np.ndarray:
    """Aggregate a set of clean lap feature rows to the 56-dim ``[mean(F01..F28), std(F01..F28)]``.

    Population std (``ddof=0``) so a single lap yields ``std=0.0`` (finite). NULLs are ignored per
    feature; an **all-NULL** feature is explicitly ``mean=std=0.0`` (never NaN). Fully vectorised —
    no per-row Python loop, and no ``nanmean``/``nanstd`` (which warn + NaN on all-NaN slices).
    """
    vals = feature_rows.reindex(columns=list(FEATURE_IDS)).to_numpy(dtype=float)
    if vals.ndim == 1:
        vals = vals.reshape(1, -1)
    finite = np.isfinite(vals)
    counts = finite.sum(axis=0).astype(float)
    safe = np.where(finite, vals, 0.0)
    zeros = np.zeros(len(FEATURE_IDS), dtype=float)
    means = np.divide(safe.sum(axis=0), counts, out=zeros.copy(), where=counts > 0)
    mean_sq = np.divide(
        np.where(finite, vals * vals, 0.0).sum(axis=0), counts, out=zeros.copy(), where=counts > 0
    )
    var = np.maximum(mean_sq - means * means, 0.0)  # population variance (ddof=_AGGREGATE_DDOF)
    stds = np.where(counts > 0, np.sqrt(var), 0.0)
    return np.concatenate([means, stds]).astype(float)


def nearest_neighbours(
    query: np.ndarray, known_coords: Mapping[str, np.ndarray], k: int
) -> list[Neighbour]:
    """Top-``k`` known drivers by cosine similarity to ``query``. A zero-norm vector (all-zero
    coord) → similarity ``0.0`` (never a div-by-zero / NaN neighbour)."""
    q = np.asarray(query, dtype=float)
    qn = float(np.linalg.norm(q))
    sims: list[tuple[str, float]] = []
    for driver, coord in known_coords.items():
        c = np.asarray(coord, dtype=float)
        cn = float(np.linalg.norm(c))
        sim = float(np.dot(q, c) / (qn * cn)) if qn > 0.0 and cn > 0.0 else 0.0
        sims.append((driver, sim))
    sims.sort(key=lambda kv: kv[1], reverse=True)
    return [
        Neighbour(driver=d, cosine_similarity=s, distance=1.0 - s) for d, s in sims[: max(k, 0)]
    ]


def ood_check(
    aggregate_56: np.ndarray,
    pca: Any,
    pc_std: np.ndarray,
    threshold: float,
    min_std: float,
    *,
    scaler: Any,
) -> tuple[bool, float]:
    """Max over principal components of ``|pc_score_i / pc_std_i|``; ``> threshold`` → out of
    distribution. The RAW aggregate is standardised by the train-only ``scaler`` (fitted at build;
    read-only here) before the PC transform — the PCA basis and ``pc_std`` live in the scaled
    space, so an unscaled pass would yield meaningless z-scores. ``pc_std`` is floored at
    ``min_std`` to avoid div-by-zero (F04's idea)."""
    x = scaler.transform(np.asarray(aggregate_56, dtype=float).reshape(1, -1))
    z = pca.transform(x)[0] / np.maximum(np.asarray(pc_std, dtype=float), min_std)
    max_abs = float(np.max(np.abs(z))) if z.size else 0.0
    return max_abs > threshold, max_abs


def _project_aggregate(
    agg: np.ndarray,
    *,
    scaler: Any,
    umap_5d: Any | None,
    umap_2d: Any | None,
    pca: Any,
    engine: str,
    n5: int,
    n2: int,
) -> tuple[list[float], list[float]]:
    """Forward-pass one RAW 56-dim aggregate to ``(coords_5d, coords_2d)`` on the active engine.

    The aggregate is first standardised by the train-only ``scaler`` (fitted at build; read-only
    here) — every engine was fit in the scaled space. The SINGLE forward-pass path shared by
    ``build_style_space`` (per-season emission), ``load_style_space`` (known/season coord
    reconstruction) and ``project_unknown`` — so scaling is applied identically everywhere and
    per-season coords are demonstrably a ``transform``, never a new fit.
    """
    x = scaler.transform(np.asarray(agg, dtype=float).reshape(1, -1))
    if engine == _ENGINE_UMAP:
        if umap_5d is None or umap_2d is None:  # explicit (not assert — survives `python -O`)
            raise ValueError("engine=='umap' but UMAP models are not loaded")
        c5 = np.asarray(umap_5d.transform(x)[0], dtype=float)
        c2 = np.asarray(umap_2d.transform(x)[0], dtype=float)
    else:  # PCA engine — first n5/n2 components of the single PCA transform
        p = np.asarray(pca.transform(x)[0], dtype=float)
        c5, c2 = p[:n5], p[:n2]
    return [float(v) for v in c5], [float(v) for v in c2]


def _risk_flags(
    agg: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    threshold: float,
    min_std: float,
) -> list[str]:
    """Feature IDs whose unknown-aggregate mean lies ``> threshold`` SD outside the known-driver
    per-feature distribution (§7.4). ``feature_std`` floored at ``min_std`` so a near-constant
    feature neither spuriously flags (``inf``) nor silently drops (``0/0 = NaN``)."""
    n = len(FEATURE_IDS)
    means = np.asarray(agg, dtype=float)[:n]
    fmean = np.asarray(feature_mean, dtype=float)[:n]
    fstd = np.maximum(np.asarray(feature_std, dtype=float)[:n], min_std)
    z = np.abs(means - fmean) / fstd
    return [FEATURE_IDS[i] for i in range(n) if z[i] > threshold]


# ----------------------------------------------------------------------------------
# Leakage guard + normalisation (train-only; reuse F04's readers, never refit)
# ----------------------------------------------------------------------------------


def _assert_no_leakage(train: pd.DataFrame, config: AppConfig) -> None:
    """Fail loud if any held-out circuit reached the fit pool (non-negotiable #2).

    Local to F08 (preserving independence from ``classifier.py``); mirrors F04/F06a. Raises
    :class:`StyleSpaceBuildError` (``TRAIN_LEAKAGE``) — no artefact is written on a poisoned run.
    """
    held_out = set(config.held_out_circuits)
    present = set(train[COL_CIRCUIT].unique()) if len(train) else set()
    leaked = sorted(held_out & present)
    if leaked:
        raise StyleSpaceBuildError(
            f"held-out circuits present in the style-space fit pool: {leaked} "
            "(would leak into the fit — aborting; no artefact written)",
            error_code=ErrorCode.TRAIN_LEAKAGE,
        )


def _normalise_frame(
    frame: pd.DataFrame, *, params: NormalisationParams, config: AppConfig, log_path: Path
) -> pd.DataFrame:
    """Z-score a frame's 28 feature columns via F04's read-only reader (never a refit). Returns a
    copy with feature columns transformed and metadata columns untouched (mirrors F06a)."""
    if len(frame) == 0:
        return frame.copy()
    rows: list[dict[str, float | None]] = []
    for rec in frame.to_dict("records"):
        vector = {fid: cast("float | None", rec.get(fid)) for fid in FEATURE_IDS}
        res = apply_normalisation(
            vector,
            circuit=cast("str", rec[COL_CIRCUIT]),
            params=params,
            config=config,
            log_path=log_path,
        )
        rows.append(res.normalised_vector)
    out = frame.copy()
    norm_df = pd.DataFrame(rows, index=out.index, columns=list(FEATURE_IDS))
    out.loc[:, list(FEATURE_IDS)] = norm_df
    return out


def _project_all(
    known_aggregate: np.ndarray,
    drivers: list[str],
    season_aggregates: dict[str, dict[str, np.ndarray]],
    *,
    scaler: Any,
    umap_5d: Any | None,
    umap_2d: Any | None,
    pca: Any,
    engine: str,
    n5: int,
    n2: int,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, dict[str, list[float]]],
    dict[str, dict[str, list[float]]],
]:
    """Reconstruct known-driver + per-season coords from their persisted RAW aggregates on
    ``engine``. One source of truth (the aggregates, stored unscaled); coords are always a
    forward-pass ``transform`` through the shared ``scaler``-aware path."""
    known5: dict[str, np.ndarray] = {}
    known2: dict[str, np.ndarray] = {}
    for i, d in enumerate(drivers):
        c5, c2 = _project_aggregate(
            known_aggregate[i],
            scaler=scaler,
            umap_5d=umap_5d,
            umap_2d=umap_2d,
            pca=pca,
            engine=engine,
            n5=n5,
            n2=n2,
        )
        known5[d] = np.asarray(c5, dtype=float)
        known2[d] = np.asarray(c2, dtype=float)
    season5: dict[str, dict[str, list[float]]] = {}
    season2: dict[str, dict[str, list[float]]] = {}
    for d, seasons in season_aggregates.items():
        season5[d], season2[d] = {}, {}
        for season, agg in seasons.items():
            c5, c2 = _project_aggregate(
                agg,
                scaler=scaler,
                umap_5d=umap_5d,
                umap_2d=umap_2d,
                pca=pca,
                engine=engine,
                n5=n5,
                n2=n2,
            )
            season5[d][season] = c5
            season2[d][season] = c2
    return known5, known2, season5, season2


# ----------------------------------------------------------------------------------
# Build (offline fit site) — fail-loud
# ----------------------------------------------------------------------------------


def _setup_mlflow(config: AppConfig, *, artefact_dir: Path) -> None:
    """Point MLflow at F08's OWN tracking store + experiment (never the classifier block)."""
    assert config.stylespace is not None
    mlflow.set_tracking_uri(config.stylespace.mlflow_tracking_uri)
    art_loc = (artefact_dir / "mlartifacts").resolve().as_uri()
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(config.stylespace.mlflow_experiment)
    if exp is None:
        client.create_experiment(config.stylespace.mlflow_experiment, artifact_location=art_loc)
    mlflow.set_experiment(config.stylespace.mlflow_experiment)


def build_style_space(
    laps: pd.DataFrame,
    *,
    config: AppConfig,
    teams: Mapping[str, str] | None = None,
    artefact_dir: str | Path | None = None,
    log_path: str | Path | None = None,
) -> StyleSpaceBuildResult:
    """Fit the style space on the training split ONLY and persist both artefacts (+ MLflow log).

    Raises :class:`StyleSpaceBuildError` on: missing config/columns, empty training pool, the
    leakage tripwire (``TRAIN_LEAKAGE``), fewer than ``umap_n_components_5d + 1`` drivers, or a
    UMAP/PCA fit exception (``MODEL_TRAINING_FAILED``) — after logging, with no artefact written.
    """
    if config.stylespace is None or config.normalise is None:
        raise StyleSpaceBuildError(
            "config.stylespace / config.normalise block absent; cannot build style space",
            error_code=ErrorCode.MODEL_TRAINING_FAILED,
        )
    cfg = config.stylespace
    art_dir = Path(artefact_dir) if artefact_dir is not None else Path(config.paths.artefacts_dir)
    art_dir.mkdir(parents=True, exist_ok=True)
    log = Path(log_path) if log_path is not None else Path(config.paths.data_quality_log)
    ledger = _DQLedger(log_path=log)
    seed = config.seeds.numpy
    min_std = _resolve_min_std(config)
    n5, n2 = cfg.umap_n_components_5d, cfg.umap_n_components_2d

    def _fail(detail: str, code: ErrorCode) -> StyleSpaceBuildError:
        ledger.record(
            code=code,
            entity_type=_ENTITY_BUILD,
            entity_id=cfg.mlflow_experiment,
            detail=detail,
            fallback_used=False,
        )
        return StyleSpaceBuildError(detail, error_code=code)

    # 1. Columns.
    missing = sorted(set(_REQUIRED_COLUMNS) - set(laps.columns))
    if missing:
        raise _fail(
            f"build matrix missing required columns: {missing}", ErrorCode.MODEL_TRAINING_FAILED
        )

    # 2. Quarantine held-out + non-train, then the fail-loud leakage tripwire.
    train_mask = (laps[COL_SPLIT] == "train") & (~laps[COL_CIRCUIT].isin(config.held_out_circuits))
    train = cast("pd.DataFrame", laps.loc[train_mask].copy())
    try:
        _assert_no_leakage(train, config)
    except StyleSpaceBuildError as exc:  # log the tripwire, then re-raise (no artefact written)
        ledger.record(
            code=exc.error_code,
            entity_type=_ENTITY_BUILD,
            entity_id=cfg.mlflow_experiment,
            detail=str(exc),
            fallback_used=False,
        )
        raise
    if len(train) == 0:
        raise _fail(
            "no training rows after excluding held-out / non-train split",
            ErrorCode.MODEL_TRAINING_FAILED,
        )

    # 3. Normalise with F04's loaded params (no refit).
    try:
        params = load_normalisation_params(art_dir / config.normalise.artefact_filename)
    except NormalisationError as exc:
        raise _fail(f"{REASON_PARAMS_UNAVAILABLE}: {exc}", ErrorCode.MODEL_TRAINING_FAILED) from exc
    train_norm = _normalise_frame(train, params=params, config=config, log_path=log)

    # 4. Aggregate per driver (sorted, deterministic label order) -> A (n_drivers × 56).
    drivers = sorted(train_norm[COL_DRIVER].unique().tolist())
    known_aggregate = np.vstack(
        [aggregate_driver(train_norm.loc[train_norm[COL_DRIVER] == d]) for d in drivers]
    )
    n_drivers = len(drivers)

    # 5. n_drivers guards.
    if n_drivers < n5 + 1:
        raise _fail(
            f"only {n_drivers} drivers; need >= {n5 + 1} to fit a {n5}D embedding",
            ErrorCode.MODEL_TRAINING_FAILED,
        )
    # A clamp is an adaptation (reported via effective_n_neighbors + DEGRADED status), not a
    # failure — so it is deliberately NOT logged as a MODEL_TRAINING_FAILED data-quality row.
    effective_n_neighbors = min(cfg.umap_n_neighbors, n_drivers - 1)

    # 6. Fit (all seeded). The 56-dim aggregate mixes z-scored (~unit-scale) and raw-physical-unit
    #    columns spanning ~4 orders of magnitude, so the euclidean UMAP/PCA geometry is first
    #    rebalanced by a train-only aggregate StandardScaler — a documented master gap-fill (§9.7).
    #    Train-only by construction: the leakage tripwire ran before the aggregate was built.
    #    scale_ is floored at min_std so a (near-)constant column cannot explode the transform;
    #    a NaN reaching the scaler fit becomes a coded MODEL_TRAINING_FAILED, not an untyped crash.
    try:
        scaler = StandardScaler().fit(known_aggregate)
        scaler.scale_ = np.maximum(np.asarray(scaler.scale_, dtype=float), min_std)
        known_scaled = scaler.transform(known_aggregate)
        umap_5d = umap.UMAP(
            n_components=n5,
            n_neighbors=effective_n_neighbors,
            min_dist=cfg.umap_min_dist,
            metric=cfg.umap_metric,
            random_state=seed,
        ).fit(known_scaled)
        umap_2d = umap.UMAP(
            n_components=n2,
            n_neighbors=effective_n_neighbors,
            min_dist=cfg.umap_min_dist,
            metric=cfg.umap_metric,
            random_state=seed,
        ).fit(known_scaled)
        pca = PCA(n_components=cfg.pca_n_components, svd_solver="full", random_state=seed).fit(
            known_scaled
        )
    except Exception as exc:  # noqa: BLE001 - any fit fault is a coded build failure (#1)
        raise _fail(f"UMAP/PCA fit failed: {exc}", ErrorCode.MODEL_TRAINING_FAILED) from exc

    # 7. OOD basis (in the scaled space, like the PCA it standardises) + per-feature stats (RAW —
    #    consumed by _risk_flags, which z-scores per feature itself and stays interpretable in
    #    physical units; _raw_nn scales through the handle's scaler at transform time).
    pc_scores = pca.transform(known_scaled)
    pc_std = pc_scores.std(axis=0, ddof=_AGGREGATE_DDOF)
    feature_mean = known_aggregate.mean(axis=0)
    feature_std = known_aggregate.std(axis=0, ddof=_AGGREGATE_DDOF)

    # 8b. Per-(driver, season) aggregates — emitted via the forward-pass transform (NOT a new fit).
    season_aggregates: dict[str, dict[str, np.ndarray]] = {}
    for group_key, grp in train_norm.groupby([COL_DRIVER, COL_SEASON], sort=True):
        drv, season = cast("tuple[Any, Any]", group_key)
        season_aggregates.setdefault(str(drv), {})[str(season)] = aggregate_driver(grp)

    # 8. Known + per-season coords via the shared forward pass (engine="umap" here).
    try:
        known5, known2, season5, season2 = _project_all(
            known_aggregate,
            drivers,
            season_aggregates,
            scaler=scaler,
            umap_5d=umap_5d,
            umap_2d=umap_2d,
            pca=pca,
            engine=_ENGINE_UMAP,
            n5=n5,
            n2=n2,
        )
    except Exception as exc:  # noqa: BLE001 - a transform fault after fit is a coded build failure
        raise _fail(f"coord projection failed: {exc}", ErrorCode.MODEL_TRAINING_FAILED) from exc

    # 9. Persist deterministically. The aggregates are the single source of truth for coords; the
    #    PCA bundle carries them so a UMAP->PCA fallback at load recomputes coords cleanly.
    umap_path = art_dir / cfg.umap_model_filename
    pca_path = art_dir / cfg.pca_model_filename
    joblib_dump(
        {
            "umap_5d": umap_5d,
            "umap_2d": umap_2d,
            "feature_order": list(FEATURE_IDS),
            "seed": seed,
            "n_neighbors": effective_n_neighbors,
            # Cross-bundle fingerprint: ties this UMAP to the scaler it was fit against, so a
            # partial rebuild (old UMAP + new PCA bundle) is detected at load and degraded.
            "scaler_sha256": _scaler_sha(scaler),
        },
        umap_path,
    )
    umap_sha = sha256(umap_path.read_bytes()).hexdigest()
    (art_dir / f"{cfg.umap_model_filename}.sha256").write_text(umap_sha, encoding="utf-8")
    joblib_dump(
        {
            "pca": pca,
            "scaler": scaler,
            "pc_std": pc_std,
            "known_aggregate": known_aggregate,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "drivers": drivers,
            "season_aggregates": season_aggregates,
        },
        pca_path,
    )

    # 10. Stability + proximity (best-effort — a failure degrades, never aborts the build).
    #     Proximity needs a driver->team map; when the caller supplies one it is computed +
    #     logged, else it is left None (evaluated formally in F20, which has the team mapping).
    stability = stability_report(laps, config=config, artefact_dir=art_dir, log_path=log)
    proximity = teammate_proximity(known5, teams, config=config) if teams else None

    # MLflow (F08's own experiment). A logging fault degrades the result, never aborts the write.
    mlflow_run_id: str | None = None
    mlflow_degraded = False
    try:
        _setup_mlflow(config, artefact_dir=art_dir)
        with mlflow.start_run(run_name="stylespace-build") as run:
            mlflow.log_params(
                {
                    "seed": seed,
                    "n_neighbors": effective_n_neighbors,
                    "n_components_5d": n5,
                    "n_components_2d": n2,
                    "metric": cfg.umap_metric,
                    "n_drivers": n_drivers,
                }
            )
            mlflow.log_metric("n_drivers", float(n_drivers))
            if stability.procrustes_variance is not None:
                mlflow.log_metric("procrustes_variance", stability.procrustes_variance)
            mlflow.log_metric("stability_passes", float(stability.passes))
            if proximity is not None:
                mlflow.log_metric("teammate_proximity_pass", float(proximity.passes))
            mlflow.log_artifact(str(umap_path))
            mlflow.log_artifact(str(pca_path))
            mlflow_run_id = run.info.run_id
    except Exception as exc:  # noqa: BLE001 - MLflow fault degrades, does not abort the pkl write
        mlflow_degraded = True
        ledger.record(
            code=ErrorCode.MODEL_TRAINING_FAILED,
            entity_type=_ENTITY_BUILD,
            entity_id=cfg.mlflow_experiment,
            detail=f"{REASON_MLFLOW}: {exc}",
            fallback_used=True,
        )

    # status: OK only when both artefacts written cleanly AND stability passes AND MLflow logged;
    # else DEGRADED with error_code=None (research finding / adaptation, not a registered error).
    degraded = (
        mlflow_degraded or not stability.passes or effective_n_neighbors < cfg.umap_n_neighbors
    )
    flags = [ErrorCode.DATA_QUALITY_LOG_UNWRITABLE] if ledger.failed else []
    return StyleSpaceBuildResult(
        status=Status.DEGRADED if degraded else Status.OK,
        error_code=None,
        flags=flags,
        logged=ledger.rows > 0,
        umap_artefact_path=str(umap_path),
        umap_sha256=umap_sha,
        pca_artefact_path=str(pca_path),
        drivers=drivers,
        coords_5d={d: [float(v) for v in known5[d]] for d in drivers},
        coords_2d={d: [float(v) for v in known2[d]] for d in drivers},
        season_coords_5d=season5,
        season_coords_2d=season2,
        n_drivers=n_drivers,
        n_features=_N_FEATURES,
        effective_n_neighbors=effective_n_neighbors,
        stability=stability,
        proximity=proximity,
        mlflow_run_id=mlflow_run_id,
    )


# ----------------------------------------------------------------------------------
# Load (startup) — fail-loud only when NO engine is usable
# ----------------------------------------------------------------------------------


def load_style_space(
    *,
    config: AppConfig,
    artefact_dir: str | Path | None = None,
    log_path: str | Path | None = None,
) -> StyleSpaceHandle:
    """Load the serving bundle. A corrupt/missing UMAP degrades to PCA (``UMAP_FALLBACK``); PCA
    and/or normalisation missing/corrupt → raise :class:`StyleSpaceLoadError` (``MODEL_NOT_FOUND``),
    after logging (mirrors F06b: refuse to serve rather than project from nothing)."""
    if config.stylespace is None or config.normalise is None:
        raise StyleSpaceLoadError("config.stylespace / config.normalise block absent; cannot load")
    cfg = config.stylespace
    art_dir = Path(artefact_dir) if artefact_dir is not None else Path(config.paths.artefacts_dir)
    log = Path(log_path) if log_path is not None else Path(config.paths.data_quality_log)
    ledger = _DQLedger(log_path=log)
    n5, n2 = cfg.umap_n_components_5d, cfg.umap_n_components_2d

    # UMAP engine — verify sha256 sidecar; any fault degrades to PCA (never a hard failure).
    umap_path = art_dir / cfg.umap_model_filename
    sha_path = art_dir / f"{cfg.umap_model_filename}.sha256"
    umap_5d: Any | None = None
    umap_2d: Any | None = None
    engine = _ENGINE_PCA
    umap_bytes: bytes | None = None
    umap_scaler_sha: str | None = None
    if not umap_path.is_file() or not sha_path.is_file():
        ledger.record(
            code=ErrorCode.UMAP_FALLBACK,
            entity_type=_ENTITY_MODEL,
            entity_id="startup",
            detail=f"{REASON_UMAP_ABSENT}: {umap_path.name} / sidecar",
            fallback_used=True,
        )
    else:
        raw = umap_path.read_bytes()
        expected = sha_path.read_text(encoding="utf-8").strip()
        if sha256(raw).hexdigest() != expected:
            ledger.record(
                code=ErrorCode.UMAP_FALLBACK,
                entity_type=_ENTITY_MODEL,
                entity_id="startup",
                detail=f"{REASON_UMAP_CORRUPT}: sha256 mismatch",
                fallback_used=True,
            )
        else:
            try:
                bundle = joblib_load(umap_path)
                u5, u2 = bundle["umap_5d"], bundle["umap_2d"]
                # Dimension check vs the CURRENT config: a stale bundle after a config bump would
                # otherwise silently serve wrong-length coordinates.
                if (
                    getattr(u5, "n_components", None) != n5
                    or getattr(u2, "n_components", None) != n2
                ):
                    raise ValueError(
                        f"umap bundle dims {getattr(u5, 'n_components', None)}/"
                        f"{getattr(u2, 'n_components', None)} != config {n5}/{n2}"
                    )
                umap_5d, umap_2d = u5, u2
                umap_scaler_sha = bundle.get("scaler_sha256")
                engine = _ENGINE_UMAP
                umap_bytes = raw
            except Exception as exc:  # noqa: BLE001 - corrupt pickle -> PCA fallback, not a crash
                ledger.record(
                    code=ErrorCode.UMAP_FALLBACK,
                    entity_type=_ENTITY_MODEL,
                    entity_id="startup",
                    detail=f"{REASON_UMAP_CORRUPT}: {exc}",
                    fallback_used=True,
                )

    # PCA fallback — REQUIRED. Its absence/corruption means no usable engine.
    pca_path = art_dir / cfg.pca_model_filename
    if not pca_path.is_file():
        raise _load_fail(ledger, f"{REASON_PCA_ABSENT}: {pca_path}")
    try:
        pca_bundle = joblib_load(pca_path)
        pca = pca_bundle["pca"]
        known_aggregate = np.asarray(pca_bundle["known_aggregate"], dtype=float)
        pc_std = np.asarray(pca_bundle["pc_std"], dtype=float)
        feature_mean = np.asarray(pca_bundle["feature_mean"], dtype=float)
        feature_std = np.asarray(pca_bundle["feature_std"], dtype=float)
        drivers = list(pca_bundle["drivers"])
        season_aggregates = pca_bundle["season_aggregates"]
    except Exception as exc:  # noqa: BLE001 - corrupt PCA fallback = no usable engine
        raise _load_fail(ledger, f"{REASON_PCA_CORRUPT}: {exc}") from exc

    # PCA dims must match the CURRENT config: a stale bundle after a config bump would otherwise
    # silently serve truncated PCA-engine coordinates (p[:n5] of a shorter transform).
    if getattr(pca, "n_components_", None) != cfg.pca_n_components:
        raise _load_fail(
            ledger,
            f"{REASON_PCA_CORRUPT}: pca n_components {getattr(pca, 'n_components_', None)} "
            f"!= config {cfg.pca_n_components} — stale artefact vs config; rebuild required",
        )

    # Aggregate scaler — REQUIRED, with its fitted stats. A pre-scaler bundle (or a wrong-width /
    # stats-less scaler) must never serve: skipping the transform would silently project raw
    # aggregates through models fit in the scaled space, and a scaler without mean_/scale_ would
    # escape as an untyped AttributeError in the fingerprint check below.
    scaler = pca_bundle.get("scaler")
    if (
        scaler is None
        or getattr(scaler, "n_features_in_", None) != _N_FEATURES
        or getattr(scaler, "mean_", None) is None
        or getattr(scaler, "scale_", None) is None
    ):
        raise _load_fail(
            ledger,
            f"{REASON_SCALER_ABSENT}: pca bundle lacks a fitted {_N_FEATURES}-dim aggregate "
            "scaler (pre-scaler artefact?) — rebuild required",
        )

    # Cross-bundle skew: a UMAP fitted against a DIFFERENT aggregate scaler (partial rebuild /
    # pre-scaler umap bundle) has meaningless geometry in this scaled space -> degrade to PCA.
    if engine == _ENGINE_UMAP and umap_scaler_sha != _scaler_sha(scaler):
        ledger.record(
            code=ErrorCode.UMAP_FALLBACK,
            entity_type=_ENTITY_MODEL,
            entity_id="startup",
            detail=f"{REASON_BUNDLE_SKEW}: umap bundle was fitted against a different scaler",
            fallback_used=True,
        )
        engine, umap_5d, umap_2d, umap_bytes = _ENGINE_PCA, None, None, None

    # Normalisation — REQUIRED (project_unknown normalises with it).
    try:
        params = load_normalisation_params(art_dir / config.normalise.artefact_filename)
    except NormalisationError as exc:
        raise _load_fail(ledger, f"{REASON_PARAMS_UNAVAILABLE}: {exc}") from exc

    # Reconstruct coords from the aggregates. A loaded-but-unusable UMAP (its transform raises)
    # degrades to the PCA engine rather than escaping untyped; if PCA also fails there is no engine.
    try:
        known5, known2, season5, season2 = _project_all(
            known_aggregate,
            drivers,
            season_aggregates,
            scaler=scaler,
            umap_5d=umap_5d,
            umap_2d=umap_2d,
            pca=pca,
            engine=engine,
            n5=n5,
            n2=n2,
        )
    except Exception as exc:  # noqa: BLE001 - unusable UMAP transform -> PCA, never an untyped escape
        if engine == _ENGINE_UMAP:
            ledger.record(
                code=ErrorCode.UMAP_FALLBACK,
                entity_type=_ENTITY_MODEL,
                entity_id="startup",
                detail=f"{REASON_UMAP_CORRUPT}: transform unusable: {exc}",
                fallback_used=True,
            )
            engine, umap_5d, umap_2d, umap_bytes = _ENGINE_PCA, None, None, None
            try:
                known5, known2, season5, season2 = _project_all(
                    known_aggregate,
                    drivers,
                    season_aggregates,
                    scaler=scaler,
                    umap_5d=None,
                    umap_2d=None,
                    pca=pca,
                    engine=_ENGINE_PCA,
                    n5=n5,
                    n2=n2,
                )
            except Exception as exc2:  # noqa: BLE001 - PCA reconstruction failed too -> no engine
                raise _load_fail(
                    ledger, f"{REASON_PCA_CORRUPT}: coord reconstruction: {exc2}"
                ) from exc2
        else:
            raise _load_fail(ledger, f"{REASON_PCA_CORRUPT}: coord reconstruction: {exc}") from exc
    version_bytes = umap_bytes if umap_bytes is not None else pca_path.read_bytes()
    model_version = f"sha256:{sha256(version_bytes).hexdigest()[:_VERSION_HASH_LEN]}"
    return StyleSpaceHandle(
        umap_5d=umap_5d,
        umap_2d=umap_2d,
        pca=pca,
        scaler=scaler,
        drivers=drivers,
        known_aggregate=known_aggregate,
        known_coords_5d=known5,
        known_coords_2d=known2,
        season_coords_5d=season5,
        season_coords_2d=season2,
        pc_std=pc_std,
        feature_mean=feature_mean,
        feature_std=feature_std,
        normalisation_params=params,
        engine=engine,
        model_version=model_version,
        min_std=_resolve_min_std(config),
    )


def _load_fail(ledger: _DQLedger, detail: str) -> StyleSpaceLoadError:
    """Log ``MODEL_NOT_FOUND`` and build the fail-loud load error (no usable engine)."""
    ledger.record(
        code=ErrorCode.MODEL_NOT_FOUND,
        entity_type=_ENTITY_MODEL,
        entity_id="startup",
        detail=detail,
        fallback_used=False,
    )
    return StyleSpaceLoadError(detail)


# ----------------------------------------------------------------------------------
# Project one unknown driver (per call) — never raises
# ----------------------------------------------------------------------------------


def _normalise_unknown_laps(
    laps: pd.DataFrame,
    *,
    default_circuit: str,
    handle: StyleSpaceHandle,
    config: AppConfig,
    log: Path,
) -> tuple[pd.DataFrame, bool, bool]:
    """Normalise each unknown lap (a per-lap ``circuit`` column overrides ``default_circuit``).
    Returns ``(normalised feature frame, any_used_global, any_logged)``."""
    has_circuit = COL_CIRCUIT in laps.columns
    rows: list[dict[str, float | None]] = []
    used_global = False
    logged = False
    for rec in laps.to_dict("records"):
        circ = cast("str", rec[COL_CIRCUIT]) if has_circuit else default_circuit
        vector = {fid: cast("float | None", rec.get(fid)) for fid in FEATURE_IDS}
        res = apply_normalisation(
            vector, circuit=circ, params=handle.normalisation_params, config=config, log_path=log
        )
        rows.append(res.normalised_vector)
        used_global = used_global or res.used_global
        logged = logged or res.logged
    return pd.DataFrame(rows, columns=list(FEATURE_IDS)), used_global, logged


def _raw_nn(agg: np.ndarray, handle: StyleSpaceHandle, k: int) -> list[Neighbour]:
    """Nearest neighbours by cosine in the standardised 56-dim aggregate space (the
    PROJECTION_FALLBACK path). Both the query and the known rows go through the handle's
    train-only scaler read-only — unscaled cosine would be dominated by the largest-magnitude
    physical-unit columns, the same defect the aggregate scaler fixes for the manifold fit
    (§9.7 applies to this path too; the ranking IS this path's entire output)."""
    q = handle.scaler.transform(np.asarray(agg, dtype=float).reshape(1, -1))[0]
    known_scaled = handle.scaler.transform(np.asarray(handle.known_aggregate, dtype=float))
    known = {d: known_scaled[i] for i, d in enumerate(handle.drivers)}
    return nearest_neighbours(q, known, k)


def project_unknown(
    laps: pd.DataFrame,
    *,
    circuit: str,
    handle: StyleSpaceHandle,
    config: AppConfig,
    log_path: str | Path | None = None,
) -> ProjectionResult:
    """Project one unknown driver's clean lap-feature rows into the style space. **Never raises** —
    every fault (sparse, OOD, transform failure, unexpected internal error) is a coded result."""
    log = Path(log_path) if log_path is not None else Path(config.paths.data_quality_log)
    ledger = _DQLedger(log_path=log)
    # Size the input defensively BEFORE the try so even a non-sized `laps` cannot make the
    # never-raises backstop itself raise when it re-reads the count (§4).
    try:
        n_input = int(len(laps))
    except Exception:  # noqa: BLE001 - a non-sized laps input -> treated as zero laps
        n_input = 0
    # Best-effort defaults for the backstop; each is refined as the pipeline progresses so a late
    # fault reports what was actually established (signal counts, sparsity, normalisation
    # fallback) instead of fabricated values. Before any lap is processed, nothing was "used".
    n_used = 0
    laps_rejected = n_input
    sparse = False
    used_global = False
    try:
        if config.stylespace is None:
            raise ValueError("config.stylespace block absent")
        cfg = config.stylespace
        min_std = handle.min_std
        n5, n2 = cfg.umap_n_components_5d, cfg.umap_n_components_2d
        k = cfg.nearest_neighbours_k

        # Zero clean laps -> raw-NN fallback immediately (no aggregate possible).
        if n_input == 0:
            ledger.record(
                code=ErrorCode.PROJECTION_FALLBACK,
                entity_type=_ENTITY_UNKNOWN,
                entity_id=circuit,
                detail=f"{REASON_ZERO_LAPS}: no clean laps to aggregate",
                fallback_used=True,
            )
            return _projection_fallback(
                nn=[],
                ood=False,
                ood_max=0.0,
                sparse=True,
                used_global=False,
                n_laps=0,
                rejected=0,
                ledger=ledger,
            )

        # 1. Normalise (unknown circuit -> global params + NORMALISATION_FALLBACK).
        norm_df, used_global, norm_logged = _normalise_unknown_laps(
            laps, default_circuit=circuit, handle=handle, config=config, log=log
        )
        if used_global:
            ledger.record(
                code=ErrorCode.NORMALISATION_FALLBACK,
                entity_type=_ENTITY_UNKNOWN,
                entity_id=circuit,
                detail="unknown circuit; global normalisation params used",
                fallback_used=True,
            )

        # Count laps that carried >=1 finite feature: a lap that is entirely NULL contributes no
        # signal, so the sparse gate + n_laps_used use THIS count (not the raw row count) — a pile
        # of empty laps cannot slip past SPARSE_PROJECTION as a confident projection, and an
        # all-NULL input degrades to a coded fallback rather than a confident origin coordinate.
        n_used = int(norm_df.notna().any(axis=1).sum())
        laps_rejected = n_input - n_used
        if n_used == 0:
            ledger.record(
                code=ErrorCode.PROJECTION_FALLBACK,
                entity_type=_ENTITY_UNKNOWN,
                entity_id=circuit,
                detail=f"{REASON_ZERO_LAPS}: no lap carried a finite feature",
                fallback_used=True,
            )
            return _projection_fallback(
                nn=[],
                ood=False,
                ood_max=0.0,
                sparse=True,
                used_global=used_global,
                n_laps=0,
                rejected=laps_rejected,
                ledger=ledger,
            )

        # 2. Aggregate -> 56-dim. 2b. Non-finite guard (defence-in-depth).
        agg = aggregate_driver(norm_df)
        if not bool(np.all(np.isfinite(agg))):
            ledger.record(
                code=ErrorCode.PROJECTION_FALLBACK,
                entity_type=_ENTITY_UNKNOWN,
                entity_id=circuit,
                detail=f"{REASON_NON_FINITE_AGGREGATE}: aggregate held a non-finite value",
                fallback_used=True,
            )
            # A non-finite query cannot be ranked: an inf slips past the zero-norm guard and
            # yields NaN cosine similarities — return no neighbours rather than garbage ones.
            return _projection_fallback(
                nn=[],
                ood=False,
                ood_max=0.0,
                sparse=n_used < config.thresholds.min_clean_laps,
                used_global=used_global,
                n_laps=n_used,
                rejected=laps_rejected,
                ledger=ledger,
            )

        # 3. Sparse gate (on the count of laps that carried real feature signal).
        sparse = n_used < config.thresholds.min_clean_laps
        if sparse:
            ledger.record(
                code=ErrorCode.SPARSE_PROJECTION,
                entity_type=_ENTITY_UNKNOWN,
                entity_id=circuit,
                detail=f"{REASON_SPARSE}: {n_used} laps < {config.thresholds.min_clean_laps}",
                fallback_used=False,
            )

        # 4. OOD gate (the check standardises the raw aggregate through the handle's scaler).
        ood, ood_max = ood_check(
            agg, handle.pca, handle.pc_std, cfg.ood_sd_threshold, min_std, scaler=handle.scaler
        )
        if ood:
            ledger.record(
                code=ErrorCode.OUT_OF_DISTRIBUTION,
                entity_type=_ENTITY_UNKNOWN,
                entity_id=circuit,
                detail=f"{REASON_OOD}: max |PC z| {ood_max:.3f} > {cfg.ood_sd_threshold}",
                fallback_used=False,
            )

        # 5. Project (UMAP or PCA engine); ANY exception -> raw-NN fallback.
        try:
            coords_5d, coords_2d = _project_aggregate(
                agg,
                scaler=handle.scaler,
                umap_5d=handle.umap_5d,
                umap_2d=handle.umap_2d,
                pca=handle.pca,
                engine=handle.engine,
                n5=n5,
                n2=n2,
            )
            if not (np.all(np.isfinite(coords_5d)) and np.all(np.isfinite(coords_2d))):
                # A non-finite coord is never shipped silently -> raw-NN fallback (§4).
                raise ValueError("transform produced a non-finite coordinate")
            nn = nearest_neighbours(np.asarray(coords_5d, dtype=float), handle.known_coords_5d, k)
        except Exception as exc:  # noqa: BLE001 - transform fault -> raw-NN, never a crash (§5.2 FM4)
            ledger.record(
                code=ErrorCode.PROJECTION_FALLBACK,
                entity_type=_ENTITY_UNKNOWN,
                entity_id=circuit,
                detail=f"{REASON_TRANSFORM_RAISED}: {exc}",
                fallback_used=True,
            )
            nn = _raw_nn(agg, handle, k)
            return _projection_fallback(
                nn=nn,
                ood=ood,
                ood_max=ood_max,
                sparse=sparse,
                used_global=used_global,
                n_laps=n_used,  # honest ledger: signal-carrying laps, not the raw row count
                rejected=laps_rejected,
                ledger=ledger,
            )

        if handle.engine == _ENGINE_PCA:
            ledger.record(
                code=ErrorCode.UMAP_FALLBACK,
                entity_type=_ENTITY_UNKNOWN,
                entity_id=circuit,
                detail="projected via PCA engine (UMAP unavailable at load)",
                fallback_used=True,
            )

        # 7. Risk flags. 8. Precedence for state/code/status.
        risk = _risk_flags(
            agg, handle.feature_mean, handle.feature_std, cfg.ood_sd_threshold, min_std
        )
        umap_fallback_engine = handle.engine == _ENGINE_PCA

        if ood:
            state = ConfidenceState.OUT_OF_DISTRIBUTION
        elif sparse:
            state = ConfidenceState.LOW_CONFIDENCE
        else:
            state = ConfidenceState.HIGH
        error_code = ErrorCode.OUT_OF_DISTRIBUTION if ood else None

        flags: list[ErrorCode] = []
        if used_global:
            flags.append(ErrorCode.NORMALISATION_FALLBACK)
        if sparse:
            flags.append(ErrorCode.SPARSE_PROJECTION)
        if umap_fallback_engine:
            flags.append(ErrorCode.UMAP_FALLBACK)
        if ledger.failed:
            flags.append(ErrorCode.DATA_QUALITY_LOG_UNWRITABLE)

        degraded = sparse or ood or umap_fallback_engine or used_global
        return ProjectionResult(
            status=Status.DEGRADED if degraded else Status.OK,
            error_code=error_code,
            flags=_dedup(flags),
            logged=(ledger.rows > 0) or norm_logged,
            coords_5d=coords_5d,
            coords_2d=coords_2d,
            nearest_neighbours=nn,
            confidence_state=state,
            sparse=sparse,
            ood=ood,
            ood_max_pc_sd=ood_max,
            projection_engine=handle.engine,
            risk_flags=risk,
            n_laps_used=n_used,
            laps_rejected=laps_rejected,
        )
    except Exception as exc:  # noqa: BLE001 - the never-raises backstop (§4)
        try:
            ledger.record(
                code=ErrorCode.PROJECTION_FALLBACK,
                entity_type=_ENTITY_UNKNOWN,
                entity_id=circuit,
                detail=f"{REASON_TRANSFORM_RAISED}: {type(exc).__name__}: {exc}",
                fallback_used=True,
            )
        except Exception:  # noqa: BLE001 - even the logger must not break never-raises
            pass
        return _projection_fallback(
            nn=[],
            ood=False,
            ood_max=0.0,
            sparse=sparse,  # best-effort values captured above — never re-derefs a bad `laps`,
            used_global=used_global,  # never contradicts the DQ rows already written
            n_laps=n_used,
            rejected=laps_rejected,
            ledger=ledger,
        )


def _projection_fallback(
    *,
    nn: list[Neighbour],
    ood: bool,
    ood_max: float,
    sparse: bool,
    used_global: bool,
    n_laps: int,
    rejected: int,
    ledger: _DQLedger,
) -> ProjectionResult:
    """Build the raw-NN ``PROJECTION_FALLBACK`` result: no coordinate, LOW_CONFIDENCE,
    ``status=DEGRADED``. ``PROJECTION_FALLBACK`` is the operative code even when OOD co-occurs (it
    is retained on ``flags``); ``confidence_state`` stays populated."""
    flags: list[ErrorCode] = []
    if used_global:
        flags.append(ErrorCode.NORMALISATION_FALLBACK)
    if sparse:
        flags.append(ErrorCode.SPARSE_PROJECTION)
    if ood:
        flags.append(ErrorCode.OUT_OF_DISTRIBUTION)
    if ledger.failed:
        flags.append(ErrorCode.DATA_QUALITY_LOG_UNWRITABLE)
    return ProjectionResult(
        status=Status.DEGRADED,
        error_code=ErrorCode.PROJECTION_FALLBACK,
        flags=_dedup(flags),
        logged=ledger.rows > 0,
        coords_5d=None,
        coords_2d=None,
        nearest_neighbours=nn,
        confidence_state=ConfidenceState.LOW_CONFIDENCE,
        sparse=sparse,
        ood=ood,
        ood_max_pc_sd=ood_max,
        projection_engine=_ENGINE_RAW_NN,
        risk_flags=[],
        n_laps_used=n_laps,
        laps_rejected=rejected,
    )


# ----------------------------------------------------------------------------------
# Serving reads + health
# ----------------------------------------------------------------------------------


def get_style_space(handle: StyleSpaceHandle) -> StyleSpaceResult:
    """Pure read of the known-driver style space (driver-level + per-season)."""
    return StyleSpaceResult(
        status=Status.OK,
        drivers=list(handle.drivers),
        coords_5d={d: [float(v) for v in handle.known_coords_5d[d]] for d in handle.drivers},
        coords_2d={d: [float(v) for v in handle.known_coords_2d[d]] for d in handle.drivers},
        season_coords_5d=handle.season_coords_5d,
        season_coords_2d=handle.season_coords_2d,
        engine=handle.engine,
    )


def health(handle: StyleSpaceHandle | None) -> StyleSpaceHealthResult:
    """Readiness for F13's ``GET /health``. Not-ready → ``HEALTH_FAIL`` (→ 503) with
    ``MODEL_NOT_FOUND`` reason; ready → OK + ``model_version`` + ``engine`` (mirrors F06b)."""
    if handle is None or not handle.ready:
        return StyleSpaceHealthResult(
            status=Status.FAILED,
            error_code=ErrorCode.HEALTH_FAIL,
            ready=False,
            reason=ErrorCode.MODEL_NOT_FOUND,
        )
    return StyleSpaceHealthResult(
        status=Status.OK, ready=True, model_version=handle.model_version, engine=handle.engine
    )


# ----------------------------------------------------------------------------------
# Dissertation-only functions (§5.2/§11.2) — never raise, never block
# ----------------------------------------------------------------------------------


def stability_report(
    laps: pd.DataFrame,
    *,
    config: AppConfig,
    artefact_dir: str | Path | None = None,
    log_path: str | Path | None = None,
) -> StabilityReport:
    """UMAP Procrustes stability over ``stability_n_runs`` fits with the fixed seed sequence
    ``seeds.numpy + i`` (§5.2/§11.2). A **dissertation-only second fit site** — train-only, guarded
    by :func:`_assert_no_leakage`, never refits normalisation. **Never raises**: a leakage tripwire
    or fit/alignment fault degrades (``procrustes_variance=None``) — a research finding, not a fail.
    """
    log = Path(log_path) if log_path is not None else Path(config.paths.data_quality_log)
    ledger = _DQLedger(log_path=log)
    try:
        if config.stylespace is None or config.normalise is None:
            raise ValueError("config.stylespace / config.normalise block absent")
        cfg = config.stylespace
        art_dir = (
            Path(artefact_dir) if artefact_dir is not None else Path(config.paths.artefacts_dir)
        )
        seed = config.seeds.numpy
        n5 = cfg.umap_n_components_5d

        train_mask = (laps[COL_SPLIT] == "train") & (
            ~laps[COL_CIRCUIT].isin(config.held_out_circuits)
        )
        train = cast("pd.DataFrame", laps.loc[train_mask].copy())
        _assert_no_leakage(
            train, config
        )  # caught by the backstop -> DEGRADED, never fits on leaked
        params = load_normalisation_params(art_dir / config.normalise.artefact_filename)
        train_norm = _normalise_frame(train, params=params, config=config, log_path=log)
        drivers = sorted(train_norm[COL_DRIVER].unique().tolist())
        agg = np.vstack(
            [aggregate_driver(train_norm.loc[train_norm[COL_DRIVER] == d]) for d in drivers]
        )
        n_drivers = len(drivers)
        eff = min(cfg.umap_n_neighbors, max(n_drivers - 1, _MIN_UMAP_NEIGHBORS))

        # Same train-only aggregate scaling as build_style_space (in-memory, never persisted) so
        # the Procrustes evidence measures the geometry the production model actually ships.
        scaler = StandardScaler().fit(agg)
        scaler.scale_ = np.maximum(np.asarray(scaler.scale_, dtype=float), _resolve_min_std(config))
        agg_scaled = scaler.transform(agg)
        embeddings = [
            umap.UMAP(
                n_components=n5,
                n_neighbors=eff,
                min_dist=cfg.umap_min_dist,
                metric=cfg.umap_metric,
                random_state=seed + i,
            ).fit_transform(agg_scaled)
            for i in range(cfg.stability_n_runs)
        ]
        disparities = [
            float(procrustes(embeddings[0], embeddings[i])[2]) for i in range(1, len(embeddings))
        ]
        variance = float(np.mean(disparities)) if disparities else 0.0
        passes = variance < cfg.stability_variance_threshold
        return StabilityReport(
            status=Status.OK if passes else Status.DEGRADED,
            flags=[ErrorCode.DATA_QUALITY_LOG_UNWRITABLE] if ledger.failed else [],
            logged=ledger.rows > 0,
            procrustes_variance=variance,
            per_run_disparity=disparities,
            n_runs=cfg.stability_n_runs,
            n_drivers=n_drivers,
            passes=passes,
        )
    except Exception as exc:  # noqa: BLE001 - stability failure is a finding, never a raise (§5.2)
        try:
            ledger.record(
                code=ErrorCode.MODEL_TRAINING_FAILED,
                entity_type=_ENTITY_BUILD,
                entity_id="stability_report",
                detail=f"stability degraded: {exc}",
                fallback_used=True,
            )
        except Exception:  # noqa: BLE001
            pass
        return StabilityReport(
            status=Status.DEGRADED, logged=ledger.rows > 0, procrustes_variance=None, passes=False
        )


def teammate_proximity(
    handle_or_coords: StyleSpaceHandle | Mapping[str, np.ndarray],
    teams: Mapping[str, str],
    *,
    config: AppConfig,
) -> TeammateProximityResult:
    """Mean intra-team vs inter-team Euclidean distance in 5D (§5.2/§11.2). Single-driver teams
    contribute no intra pair; no comparable pair → ``None`` (never a fabricated 0.0). No raise."""
    try:
        if isinstance(handle_or_coords, StyleSpaceHandle):
            coords = handle_or_coords.known_coords_5d
        else:
            coords = dict(handle_or_coords)
        intra: list[float] = []
        inter: list[float] = []
        drivers = [d for d in coords if d in teams]
        for i in range(len(drivers)):
            for j in range(i + 1, len(drivers)):
                a, b = drivers[i], drivers[j]
                dist = float(np.linalg.norm(np.asarray(coords[a]) - np.asarray(coords[b])))
                (intra if teams[a] == teams[b] else inter).append(dist)
        mean_intra = float(np.mean(intra)) if intra else None
        mean_inter = float(np.mean(inter)) if inter else None
        passes = mean_intra is not None and mean_inter is not None and mean_intra < mean_inter
        return TeammateProximityResult(
            status=Status.OK if passes else Status.DEGRADED,
            mean_intra_team=mean_intra,
            mean_inter_team=mean_inter,
            passes=passes,
            teams_evaluated=len({teams[d] for d in drivers}),
        )
    except Exception:  # noqa: BLE001 - dissertation evidence never blocks serving
        return TeammateProximityResult(status=Status.DEGRADED, passes=False)
