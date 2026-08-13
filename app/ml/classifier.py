"""F06a — Model 1 (driver classifier) TRAINING: tune, calibrate, log, register the champion.

This is the **offline training half** of F06. Given the assembled 28-feature lap matrix it:

1. quarantines the held-out circuits (Belgium, Italy) **before any fit**, with a fail-loud
   :func:`_assert_no_leakage` tripwire (``TRAIN_LEAKAGE``);
2. produces the real ``artefacts/normalisation.json`` via F04's :func:`fit_normaliser` (the only
   ``.fit`` of a scaler anywhere here), then transforms every row through F04's read-only
   :func:`apply_normalisation` — never a refit;
3. carves a **disjoint** calibration hold-out (excluded from the CV fit folds);
4. trains + logs the three baselines (majority, logistic regression, random forest);
5. runs the Optuna search (``optuna_trials`` ≥ 200 in repo config) under **StratifiedGroupKFold
   grouped by circuit** (Option A — each circuit sits wholly in one fold, so the CV objective is
   leakage-clean; see ``specs/F06a`` §4), optimising macro-averaged F1;
6. refits the champion and wraps it in ``CalibratedClassifierCV(cv="prefit")`` on the disjoint
   hold-out;
7. evaluates macro-F1 / per-driver F1 / confusion matrix / calibration curve on the held-out
   circuits, plus the normalisation ablation;
8. logs every trial + the champion to **MLflow** and registers the champion.

**No component fails silently** (non-negotiable #1): a leakage violation or a study that yields
no valid champion **raises** :class:`ClassifierTrainingError`; an individual trial that raises /
returns NaN is pruned, counted, and logged; a champion that misses the held-out target / a
per-driver floor / a baseline is returned ``DEGRADED`` with a ``MODEL_TRAINING_FAILED`` flag —
never silently shipped as a pass. Every constant (folds, trial count, fractions, targets,
filenames, stratify/group keys) is read from ``config.classifier`` / ``config.seeds`` — no magic
numbers.

**Dependency note (ratified by F06a).** The constitution pins *MLflow 2.x*; APEX runs on Python
3.14 with pandas 3.x, and every full-``mlflow`` release caps ``pandas<3`` (it would downgrade the
F04-ratified stack). F06a therefore uses **``mlflow-skinny``** (the lightweight tracking +
registry client, same ``mlflow.*`` API) on a **sqlite** tracking store — the model registry is
unsupported on ``file://`` stores. See ``specs/F06a`` §8/§9.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import mlflow
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from joblib import dump as joblib_dump
from pydantic import BaseModel, ConfigDict, Field
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline

from app.contracts.config import AppConfig, ClassifierConfig
from app.contracts.enums import Status
from app.contracts.errors import ErrorCode
from app.contracts.logging import DataQualityLogError, log_data_quality
from app.contracts.results import BaseResult
from app.pipeline.features import FEATURE_IDS
from app.pipeline.normalise import (
    NormalisationParams,
    apply_normalisation,
    fit_normaliser,
    load_normalisation_params,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

# Required non-feature columns in the wide training matrix.
COL_DRIVER = "driver"
COL_CIRCUIT = "circuit"
COL_SEASON = "season"
COL_SPLIT = "split"
_REQUIRED_COLUMNS = (COL_DRIVER, COL_CIRCUIT, COL_SEASON, COL_SPLIT, *FEATURE_IDS)

# Canonical baseline keys (mirrors config._KNOWN_BASELINES; validated there at load).
BASELINE_MAJORITY = "majority"
BASELINE_LOGREG = "logistic_regression"
BASELINE_RANDOM_FOREST = "random_forest"

# config.classifier.optuna_objective -> sklearn f1_score `average` argument. The validator pins
# the objective to a key in this map, so an unknown objective is a load-time CONFIG_001.
_OBJECTIVE_AVERAGE = {"macro_f1": "macro"}

# data_quality_log detail tags (mirrors normalise.py's REASON_* convention).
REASON_TRIAL_RAISED = "TRIAL_RAISED"
REASON_TRIAL_NAN = "TRIAL_NAN_OBJECTIVE"
REASON_NO_CHAMPION = "NO_VALID_CHAMPION"
REASON_TARGET_MISSED = "HELD_OUT_TARGET_MISSED"
REASON_DRIVER_FLOOR = "PER_DRIVER_FLOOR_MISSED"
REASON_BASELINE_NOT_BEATEN = "BASELINE_NOT_BEATEN"
REASON_REGISTRATION_FAILED = "REGISTRATION_FAILED"

# Optuna search space for XGBoost — DESIGN constants (the hyperparameter ranges the Bayesian
# search explores), not tunable thresholds, so they live in code and are excluded from the
# config-literal scan (the same treatment normalise.py gives its centre-only fallbacks).
_XGB_N_ESTIMATORS = (50, 400)
_XGB_MAX_DEPTH = (2, 8)
_XGB_LEARNING_RATE = (1e-3, 3e-1)
_XGB_SUBSAMPLE = (0.5, 1.0)
_XGB_COLSAMPLE = (0.5, 1.0)
_XGB_MIN_CHILD_WEIGHT = (1, 10)
_XGB_REG_LAMBDA = (1e-3, 10.0)
_XGB_GAMMA = (0.0, 5.0)
# Safety cap so a pathological run that keeps failing trials cannot loop forever while topping up
# to the trial floor (architectural backstop, not a tunable).
_TRIAL_ATTEMPT_MULTIPLIER = 3
_TRIAL_ATTEMPT_BASE = 5


class ClassifierTrainingError(Exception):
    """Raised on a build-breaking F06a training fault (no result is returned on these paths).

    Carries an :class:`ErrorCode` — ``TRAIN_LEAKAGE`` (held-out circuit / calibration overlap in a
    fit) or ``MODEL_TRAINING_FAILED`` (no valid champion / registration failed). Fail-loud by
    design (non-negotiable #1): the degraded-but-successful conditions do **not** raise — they
    return a ``DEGRADED`` :class:`ClassifierTrainingResult`.
    """

    def __init__(self, message: str, *, error_code: ErrorCode) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code


class TrialRecord(BaseModel):
    """One Optuna trial's outcome (logged to MLflow). ``macro_f1`` is ``None`` for a
    failed / pruned trial; ``state`` is the Optuna trial state name."""

    model_config = ConfigDict(extra="forbid")

    trial_number: int
    params: dict[str, float | int]
    macro_f1: float | None
    state: str


class ClassifierTrainingResult(BaseResult):
    """Summary of a :func:`train_classifier` run.

    ``status`` is ``OK`` when the champion meets the held-out target, every per-driver floor, and
    beats all baselines; otherwise ``DEGRADED`` with ``error_code=MODEL_TRAINING_FAILED`` naming
    the (reported, non-fatal) shortfall via ``flags``. The hard-failure paths raise
    :class:`ClassifierTrainingError` before this object is built.
    """

    mlflow_run_id: str
    model_uri: str
    model_artefact_path: str
    labels_artefact_path: str = ""  # F06b seam — artefacts/labels.json (index->driver-name map)
    champion_params: dict[str, float | int] = Field(default_factory=dict)
    n_trials: int = 0
    n_trials_failed: int = 0
    cv_strategy: str = ""
    cv_folds: int = 0
    train_circuits: list[str] = Field(default_factory=list)
    excluded_circuits: list[str] = Field(default_factory=list)
    calibration_method: str = ""
    calibration_holdout_frac: float = 0.0
    held_out_macro_f1: float = 0.0
    held_out_per_driver_f1: dict[str, float] = Field(default_factory=dict)
    confusion_matrix_path: str = ""
    calibration_curve_path: str = ""
    baseline_macro_f1: dict[str, float] = Field(default_factory=dict)
    beats_all_baselines: bool = False
    # The champion's representation (bundle_meta.json mirror): True = per-circuit z-scored,
    # False = raw (the 2026-08-13 ablation re-scoping). The ablation metric below always reports
    # the COUNTERFACTUAL representation, whichever this is.
    uses_circuit_normalisation: bool = True
    normalisation_ablation_macro_f1: float = 0.0
    target_macro_f1: float = 0.0
    target_met: bool = False
    per_driver_floor: float = 0.0
    per_driver_floor_met: bool = False
    seeds: dict[str, int] = Field(default_factory=dict)
    normalisation_content_hash: str = ""


@dataclass
class _DQLedger:
    """Writes one ``data_quality_log`` row per training fallback / trial failure / shortfall.

    Mirrors F04's parameterised ledger. ``entity_type`` is ``"training_run"`` for run-scope rows
    and ``"trial"`` for per-trial rows. Never propagates a sink failure — a lost audit row is
    counted in ``failed`` so the caller can surface ``DATA_QUALITY_LOG_UNWRITABLE`` on the result
    rather than crash (non-negotiable #1)."""

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
class _Splits:
    """The leakage-safe partitioning of the matrix (all already normalised)."""

    fit: pd.DataFrame  # CV fit pool (train circuits minus calibration hold-out)
    calibration: pd.DataFrame  # disjoint calibration hold-out (train circuits)
    held_out: pd.DataFrame  # Belgium + Italy — only ever evaluated, never fitted
    train_circuits: list[str]


# ----------------------------------------------------------------------------------
# Leakage guard
# ----------------------------------------------------------------------------------


def _assert_no_leakage(fit_pool: pd.DataFrame, config: AppConfig) -> None:
    """Fail loud if any held-out circuit reached the trainable pool (non-negotiable #2).

    Called after the held-out circuits are set aside; a tripwire that only fires if the exclusion
    logic were broken. Raises :class:`ClassifierTrainingError` (``TRAIN_LEAKAGE``) — no model is
    written on a poisoned run. A missing ``config.classifier`` is *not* leakage, so it surfaces as
    ``MODEL_TRAINING_FAILED`` (a typed failure, not a bare ``AttributeError``).
    """
    if config.classifier is None:
        raise ClassifierTrainingError(
            "config.classifier block absent; cannot train classifier",
            error_code=ErrorCode.MODEL_TRAINING_FAILED,
        )
    held_out = set(config.held_out_circuits)
    present = set(fit_pool[COL_CIRCUIT].unique()) if len(fit_pool) else set()
    leaked = sorted(held_out & present)
    if leaked:
        raise ClassifierTrainingError(
            f"held-out circuits present in the trainable pool: {leaked} "
            "(would leak into the fit — aborting; no champion registered)",
            error_code=ErrorCode.TRAIN_LEAKAGE,
        )


def _assert_calibration_covers_classes(
    y_cal: np.ndarray, *, n_classes: int, labels: Sequence[str]
) -> None:
    """Fail loud if the calibration hold-out is missing a class. ``CalibratedClassifierCV`` derives
    ``classes_`` from the calibration labels only, so a missing driver would be silently dropped
    from the champion (F06b could never output it) — a typed, coded error instead."""
    present = set(y_cal.tolist())
    if present != set(range(n_classes)):
        missing = [labels[i] for i in range(n_classes) if i not in present]
        raise ClassifierTrainingError(
            f"calibration hold-out is missing classes {missing}; cannot calibrate all drivers",
            error_code=ErrorCode.MODEL_TRAINING_FAILED,
        )


def _assert_calibration_disjoint(fit: pd.DataFrame, calibration: pd.DataFrame) -> None:
    """Fail loud if the calibration hold-out shares any row with the CV fit set."""
    overlap = set(fit.index) & set(calibration.index)
    if overlap:
        raise ClassifierTrainingError(
            f"calibration set leaks into the fit set ({len(overlap)} shared rows)",
            error_code=ErrorCode.TRAIN_LEAKAGE,
        )


# ----------------------------------------------------------------------------------
# Normalisation (F04) — fit once, then read-only apply to every row
# ----------------------------------------------------------------------------------


def _fit_load_params(
    laps: pd.DataFrame, *, config: AppConfig, artefact_dir: Path, log_path: Path
) -> tuple[NormalisationParams, str]:
    """Produce the F04 artefact once (the only scaler fit) and load it back.

    ``fit_normaliser`` filters internally to ``split == "train"`` & non-held-out before any
    aggregation, so the artefact is fit on the training circuits only. Called *after* the
    leakage-safe partition so no artefact is written on a poisoned run.
    """
    assert config.normalise is not None  # guaranteed by caller / fit_normaliser guards
    artefact_path = artefact_dir / config.normalise.artefact_filename
    fit_res = fit_normaliser(laps, config=config, artefact_path=artefact_path, log_path=log_path)
    return load_normalisation_params(artefact_path), fit_res.content_hash


def _apply_to_frame(
    frame: pd.DataFrame, *, params: NormalisationParams, config: AppConfig, log_path: Path
) -> pd.DataFrame:
    """Z-score a frame's 28 feature columns via F04's read-only :func:`apply_normalisation`.

    Loops over rows (modest N — laps, not telemetry samples) calling the contractual F04 reader
    rather than re-deriving the maths here; it never fits a scaler. Returns a copy with the feature
    columns transformed and all other columns untouched.
    """
    if len(frame) == 0:
        return frame.copy()
    normalised_rows: list[dict[str, float | None]] = []
    for rec in frame.to_dict("records"):
        vector = {fid: cast("float | None", rec[fid]) for fid in FEATURE_IDS}
        res = apply_normalisation(
            vector,
            circuit=cast("str", rec[COL_CIRCUIT]),
            params=params,
            config=config,
            log_path=log_path,
        )
        normalised_rows.append(res.normalised_vector)
    out = frame.copy()
    norm_df = pd.DataFrame(normalised_rows, index=out.index, columns=list(FEATURE_IDS))
    out.loc[:, list(FEATURE_IDS)] = norm_df
    return out


# ----------------------------------------------------------------------------------
# Splitting / calibration carve
# ----------------------------------------------------------------------------------


def _partition(laps: pd.DataFrame, *, config: AppConfig) -> _Splits:
    """Quarantine held-out circuits, then carve a disjoint, driver-stratified calibration set."""
    assert config.classifier is not None
    # Reset to a unique RangeIndex first: the calibration carve selects rows by index label, so a
    # non-unique index (e.g. per-circuit frames concatenated without ignore_index) would let
    # `index.isin(...)` over-select rows from other drivers/circuits and silently corrupt the carve.
    laps = laps.reset_index(drop=True)
    rng = np.random.default_rng(config.seeds.numpy)
    held_out_names = set(config.held_out_circuits)

    held_out = cast("pd.DataFrame", laps.loc[laps[COL_CIRCUIT].isin(held_out_names)].copy())
    # The trainable pool is non-held-out AND split == "train": a val/test row from a training
    # circuit must NOT enter the fit folds or calibration (spec §4 — "partition by the split
    # column"; F04 fits on split=="train" only, so the model must mirror that budget).
    pool_mask = (~laps[COL_CIRCUIT].isin(held_out_names)) & (laps[COL_SPLIT] == "train")
    pool = cast("pd.DataFrame", laps.loc[pool_mask].copy())
    _assert_no_leakage(pool, config)
    if len(pool) == 0:
        raise ClassifierTrainingError(
            "no trainable rows after quarantining held-out circuits / non-train splits",
            error_code=ErrorCode.MODEL_TRAINING_FAILED,
        )

    frac = config.classifier.calibration_holdout_frac
    # Stratify the calibration carve by driver: take ``frac`` of each driver's rows (at least one
    # where a driver has any rows) so every class is represented in calibration.
    calib_idx: list[Any] = []
    for _driver, grp in pool.groupby(COL_DRIVER, sort=True):
        n = len(grp)
        k = max(1, int(round(n * frac))) if n > 1 else 0
        if k:
            chosen = rng.choice(grp.index.to_numpy(), size=min(k, n - 1), replace=False)
            calib_idx.extend(chosen.tolist())
    calibration = cast("pd.DataFrame", pool.loc[pool.index.isin(calib_idx)].copy())
    fit = cast("pd.DataFrame", pool.loc[~pool.index.isin(calib_idx)].copy())
    _assert_calibration_disjoint(fit, calibration)

    train_circuits = sorted(pool[COL_CIRCUIT].unique().tolist())
    return _Splits(
        fit=fit, calibration=calibration, held_out=held_out, train_circuits=train_circuits
    )


# ----------------------------------------------------------------------------------
# Label encoding + matrix helpers
# ----------------------------------------------------------------------------------


def _label_space(laps: pd.DataFrame, *, driver_col: str = COL_DRIVER) -> list[str]:
    """Deterministic sorted list of driver labels (stable class index across runs)."""
    return sorted(laps[driver_col].unique().tolist())


def _assert_known_drivers(
    held_out: pd.DataFrame, labels: Sequence[str], *, driver_col: str = COL_DRIVER
) -> None:
    """Fail loud if a held-out driver was never seen in training (else ``_xy`` would raise a bare
    ``KeyError`` — a silent failure). The 8 known drivers are defined by the training pool; an
    unknown held-out driver is unrepresentable by this classifier and is a typed, coded error."""
    if len(held_out) == 0:
        return
    unknown = sorted(set(held_out[driver_col]) - set(labels))
    if unknown:
        raise ClassifierTrainingError(
            f"held-out driver(s) absent from the training label space: {unknown}",
            error_code=ErrorCode.MODEL_TRAINING_FAILED,
        )


def _xy(
    laps: pd.DataFrame,
    *,
    labels: Sequence[str],
    driver_col: str = COL_DRIVER,
    group_col: str = COL_CIRCUIT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X features, y integer labels, groups) for a frame.

    ``driver_col`` (the stratify/target key) and ``group_col`` (the leakage-grouping key) are read
    from ``config.classifier.cv_stratify_by`` / ``cv_group_by`` by the caller, so those config
    fields actually drive the split rather than being inert.
    """
    index = {name: i for i, name in enumerate(labels)}
    feats = cast("pd.DataFrame", laps.loc[:, list(FEATURE_IDS)])
    x = np.asarray(feats.to_numpy(), dtype=float)
    y = np.asarray([index[d] for d in laps[driver_col]], dtype=int)
    groups = np.asarray(laps[group_col].to_numpy())
    return x, y, groups


# ----------------------------------------------------------------------------------
# CV + Optuna
# ----------------------------------------------------------------------------------


def build_cv(config: AppConfig) -> StratifiedGroupKFold:
    """Construct the Option-A splitter: ``StratifiedGroupKFold`` (stratify driver, group circuit).

    Each circuit lands wholly in one fold, so no circuit appears in both the train and validation
    index of any fold — the CV objective is a leakage-clean unseen-training-circuit estimate.
    """
    assert config.classifier is not None
    return StratifiedGroupKFold(
        n_splits=config.classifier.cv_folds, shuffle=True, random_state=config.seeds.xgboost
    )


def _study_key(x: np.ndarray, y: np.ndarray, groups: np.ndarray, config: AppConfig) -> str:
    """Short content hash of the fit inputs + tuning config — the Optuna study's identity.

    Changes whenever the fit features/labels/circuit-groups, the fold count or trial floor, or the
    seeds change, so a re-run on new data or a retuned CV starts a FRESH study (no stale-tuning)
    while identical inputs resume the same study (the §9 restart-persistence contract)."""
    assert config.classifier is not None
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(x, dtype=float).tobytes())
    h.update(np.ascontiguousarray(y, dtype=np.int64).tobytes())
    h.update("\x00".join(map(str, groups.tolist())).encode())  # circuit grouping affects the folds
    h.update(f"{config.classifier.cv_folds}-{config.classifier.optuna_trials}".encode())
    h.update(f"{config.seeds.numpy}-{config.seeds.xgboost}-{config.seeds.optuna}".encode())
    return h.hexdigest()[:16]


def _base_xgb_params(*, seed: int, n_classes: int) -> dict[str, Any]:
    """The fixed XGBoost params shared by the search, the champion refit, and the ablation.

    ``n_jobs=1`` pins single-threaded training: multi-threaded `hist` sums gradients in
    thread-order-dependent float order, so unpinned threading would make ``held_out_macro_f1``
    non-reproducible across machines/core counts (non-negotiable #3). One definition avoids the two
    copies drifting apart.
    """
    return {
        "objective": "multi:softprob",
        "num_class": n_classes,
        "tree_method": "hist",
        "eval_metric": "mlogloss",
        "random_state": seed,
        "n_jobs": 1,
    }


def _suggest_params(trial: optuna.Trial, *, seed: int, n_classes: int) -> dict[str, Any]:
    """Sample one XGBoost hyperparameter set from the search space (seed fixed for repro)."""
    return {
        **_base_xgb_params(seed=seed, n_classes=n_classes),
        "n_estimators": trial.suggest_int("n_estimators", *_XGB_N_ESTIMATORS),
        "max_depth": trial.suggest_int("max_depth", *_XGB_MAX_DEPTH),
        "learning_rate": trial.suggest_float("learning_rate", *_XGB_LEARNING_RATE, log=True),
        "subsample": trial.suggest_float("subsample", *_XGB_SUBSAMPLE),
        "colsample_bytree": trial.suggest_float("colsample_bytree", *_XGB_COLSAMPLE),
        "min_child_weight": trial.suggest_int("min_child_weight", *_XGB_MIN_CHILD_WEIGHT),
        "reg_lambda": trial.suggest_float("reg_lambda", *_XGB_REG_LAMBDA, log=True),
        "gamma": trial.suggest_float("gamma", *_XGB_GAMMA),
    }


def _cv_macro_f1(
    params: dict[str, Any],
    *,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cv: StratifiedGroupKFold,
    average: str = "macro",
) -> float:
    """Mean F1 (``average`` from ``config.optuna_objective``) across the StratifiedGroupKFold
    folds for one parameter set."""
    scores: list[float] = []
    for train_idx, val_idx in cv.split(x, y, groups):
        model = xgb.XGBClassifier(**params)
        model.fit(x[train_idx], y[train_idx])
        pred = model.predict(x[val_idx])
        scores.append(float(f1_score(y[val_idx], pred, average=average)))
    return float(np.mean(scores)) if scores else math.nan


# ----------------------------------------------------------------------------------
# Baselines + champion fit + evaluation
# ----------------------------------------------------------------------------------


def _baseline_estimator(name: str, *, seed: int) -> Any:
    """Build one baseline estimator. The sklearn baselines (logreg/forest) cannot consume the NaNs
    that F04's ``apply_normalisation`` legitimately leaves for NULL features (XGBoost can), so they
    are wrapped in a median-:class:`SimpleImputer` pipeline — a train-only fit, no leakage — rather
    than crashing the whole run with an uncoded ``ValueError`` (non-negotiable #1). The majority
    baseline ignores X, so it needs no imputation."""
    if name == BASELINE_MAJORITY:
        return DummyClassifier(strategy="most_frequent")
    if name == BASELINE_LOGREG:
        return make_pipeline(
            SimpleImputer(strategy="median"), LogisticRegression(max_iter=1000, random_state=seed)
        )
    if name == BASELINE_RANDOM_FOREST:
        return make_pipeline(
            SimpleImputer(strategy="median"), RandomForestClassifier(random_state=seed, n_jobs=1)
        )
    raise ClassifierTrainingError(  # pragma: no cover - config validator rejects unknown names
        f"unknown baseline {name!r}", error_code=ErrorCode.MODEL_TRAINING_FAILED
    )


def _fit_baseline(name: str, *, x: np.ndarray, y: np.ndarray, seed: int) -> Any:
    """Instantiate + fit one baseline on the (normalised) fit features."""
    model = _baseline_estimator(name, seed=seed)
    model.fit(x, y)
    return model


def _macro_f1_on(model: Any, *, x: np.ndarray, y: np.ndarray) -> float:
    return float(f1_score(y, model.predict(x), average="macro"))


def _per_driver_f1(
    model: Any, *, x: np.ndarray, y: np.ndarray, labels: Sequence[str]
) -> dict[str, float]:
    per_class = f1_score(y, model.predict(x), average=None, labels=list(range(len(labels))))
    return {labels[i]: float(per_class[i]) for i in range(len(labels))}


# ----------------------------------------------------------------------------------
# MLflow helpers (skinny + sqlite store)
# ----------------------------------------------------------------------------------


def _setup_mlflow(config: AppConfig, *, artefact_dir: Path) -> None:
    """Point MLflow at the DB-backed tracking store + a tmp-contained artifact location."""
    assert config.classifier is not None
    mlflow.set_tracking_uri(config.classifier.mlflow_tracking_uri)
    art_loc = (artefact_dir / "mlartifacts").resolve().as_uri()
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(config.classifier.mlflow_experiment)
    if exp is None:
        client.create_experiment(config.classifier.mlflow_experiment, artifact_location=art_loc)
    mlflow.set_experiment(config.classifier.mlflow_experiment)


def _log_json_artifact(payload: dict[str, Any] | list[Any], *, name: str, tmp_dir: Path) -> str:
    """Write a JSON artifact to the active run and return its run-relative path."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    p = tmp_dir / name
    p.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    mlflow.log_artifact(str(p))
    return name


# ----------------------------------------------------------------------------------
# The entry point
# ----------------------------------------------------------------------------------


def train_classifier(
    laps: pd.DataFrame,
    *,
    config: AppConfig,
    artefact_dir: str | Path | None = None,
    log_path: str | Path | None = None,
) -> ClassifierTrainingResult:
    """Train, tune, calibrate, log and register the F06a champion classifier.

    See the module docstring for the pipeline. Raises :class:`ClassifierTrainingError` on leakage
    or a study that yields no valid champion; otherwise returns a :class:`ClassifierTrainingResult`
    (``OK`` when the champion meets target + floors + baselines, else ``DEGRADED``).
    """
    cfg = config.classifier
    if cfg is None or config.normalise is None:
        raise ClassifierTrainingError(
            "config.classifier / config.normalise block absent; cannot train",
            error_code=ErrorCode.MODEL_TRAINING_FAILED,
        )
    art_dir = Path(artefact_dir) if artefact_dir is not None else Path(config.paths.artefacts_dir)
    art_dir.mkdir(parents=True, exist_ok=True)
    log = Path(log_path) if log_path is not None else Path(config.paths.data_quality_log)
    ledger = _DQLedger(log_path=log)

    missing = sorted(set(_REQUIRED_COLUMNS) - set(laps.columns))
    if missing:
        raise ClassifierTrainingError(
            f"training matrix missing required columns: {missing}",
            error_code=ErrorCode.MODEL_TRAINING_FAILED,
        )
    driver_col, group_col = cfg.cv_stratify_by, cfg.cv_group_by
    average = _OBJECTIVE_AVERAGE[cfg.optuna_objective]

    # 1. Partition the RAW matrix leakage-safely FIRST — quarantine held-out, filter to the train
    #    split, carve the disjoint calibration set, and run the leakage tripwires — BEFORE any
    #    artefact is written, so a poisoned run leaves no model or normalisation.json on disk.
    splits = _partition(laps, config=config)
    labels = sorted(set(splits.fit[driver_col]) | set(splits.calibration[driver_col]))
    _assert_known_drivers(splits.held_out, labels, driver_col=driver_col)

    # 2. Produce the F04 artefact (the one scaler fit — still written for provenance and for the
    #    downstream consumers that keep the normalised pipeline), then choose the CLASSIFIER's
    #    representation: per-circuit z-scored (legacy) or raw (the 2026-08-13 ablation re-scoping —
    #    raw 0.614 vs normalised 0.413 on never-seen circuits). The choice is recorded in
    #    bundle_meta.json so serving follows the bundle, never the live config.
    params, norm_hash = _fit_load_params(laps, config=config, artefact_dir=art_dir, log_path=log)
    use_norm = cfg.use_circuit_normalisation
    if use_norm:
        fit_n = _apply_to_frame(splits.fit, params=params, config=config, log_path=log)
        cal_n = _apply_to_frame(splits.calibration, params=params, config=config, log_path=log)
        test_n = _apply_to_frame(splits.held_out, params=params, config=config, log_path=log)
    else:
        fit_n, cal_n, test_n = splits.fit, splits.calibration, splits.held_out

    xy_cols = {"driver_col": driver_col, "group_col": group_col}
    x_fit, y_fit, g_fit = _xy(fit_n, labels=labels, **xy_cols)
    x_cal, y_cal, _ = _xy(cal_n, labels=labels, **xy_cols)
    x_test, y_test, _ = _xy(test_n, labels=labels, **xy_cols)
    n_classes = len(labels)
    cv = build_cv(config)

    _assert_calibration_covers_classes(y_cal, n_classes=n_classes, labels=labels)

    _setup_mlflow(config, artefact_dir=art_dir)

    # 4. Baselines (logged + evaluated on held-out for the beats_all_baselines comparison).
    baseline_macro_f1: dict[str, float] = {}
    for name in cfg.baselines:
        model = _fit_baseline(name, x=x_fit, y=y_fit, seed=config.seeds.xgboost)
        score = _macro_f1_on(model, x=x_test, y=y_test) if len(x_test) else math.nan
        baseline_macro_f1[name] = score
        with mlflow.start_run(run_name=f"baseline-{name}"):
            mlflow.log_param("model", name)
            mlflow.log_metric("held_out_macro_f1", score)

    # 5. Optuna study (StratifiedGroupKFold CV objective), with per-trial failure handling.
    sampler = optuna.samplers.TPESampler(seed=config.seeds.optuna)
    # The study_name is keyed to THIS run's training data + tuning config (a content hash of the
    # fit features/labels, the trial floor, and the seeds). With load_if_exists=True this satisfies
    # the §9 persistence contract two ways: a crashed run on the SAME data resumes its trials, but a
    # re-run on CHANGED data gets a fresh study — never silently shipping a champion tuned on stale
    # data. (Earlier a fixed name resumed across data changes; that bug is fixed here.)
    study_name = f"{cfg.mlflow_experiment}-{_study_key(x_fit, y_fit, g_fit, config)}"
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        storage=cfg.optuna_storage,
        load_if_exists=True,
    )
    trial_records, n_failed, n_complete = _run_study(
        study, x=x_fit, y=y_fit, groups=g_fit, cv=cv, config=config, ledger=ledger
    )
    if n_complete < cfg.optuna_trials:
        ledger.record(
            code=ErrorCode.MODEL_TRAINING_FAILED,
            entity_type="training_run",
            entity_id=cfg.mlflow_experiment,
            detail=(
                f"{REASON_NO_CHAMPION}: {n_complete} complete trials < floor {cfg.optuna_trials}"
            ),
            fallback_used=False,
        )
        raise ClassifierTrainingError(
            f"Optuna produced {n_complete} valid trials, below the floor {cfg.optuna_trials}",
            error_code=ErrorCode.MODEL_TRAINING_FAILED,
        )

    # 5b. Log the FULL search history to MLflow — one run per trial (params + CV score + state) —
    #     so the tuning decision is auditable/reproducible from MLflow (non-negotiable #3), not just
    #     the champion + baselines. (On a rare same-data study resume these trial-N runs re-emit;
    #     harmless duplicate audit rows, never wrong data.)
    for tr in trial_records:
        with mlflow.start_run(run_name=f"trial-{tr.trial_number}"):
            mlflow.log_params(tr.params)
            mlflow.set_tag("trial_state", tr.state)
            if tr.macro_f1 is not None:
                mlflow.log_metric("cv_macro_f1", tr.macro_f1)

    # 6. Champion fit + calibration on the disjoint hold-out.
    best_params = _suggest_from_best(
        study.best_params, seed=config.seeds.xgboost, n_classes=n_classes
    )
    base_model = xgb.XGBClassifier(**best_params)
    base_model.fit(x_fit, y_fit)
    # sklearn >=1.6 replaced ``cv="prefit"`` with wrapping the already-fit model in
    # ``FrozenEstimator``: the base model is frozen (never refit) and only the calibrator is fit on
    # the disjoint calibration hold-out — identical leakage-safe behaviour to the old prefit mode.
    champion = CalibratedClassifierCV(FrozenEstimator(base_model), method=cfg.calibration_method)
    champion.fit(x_cal, y_cal)

    # 8. Held-out evaluation + normalisation ablation.
    held_out_macro_f1 = _macro_f1_on(champion, x=x_test, y=y_test) if len(x_test) else math.nan
    per_driver = (
        _per_driver_f1(champion, x=x_test, y=y_test, labels=labels) if len(x_test) else {}
    )
    cmatrix = (
        confusion_matrix(y_test, champion.predict(x_test), labels=list(range(n_classes))).tolist()
        if len(x_test)
        else []
    )
    # Ablation trains the COUNTERFACTUAL representation on the same fit rows — equal data budget,
    # so the comparison isolates the effect of normalisation, not of extra data. With the champion
    # normalised the counterfactual is raw (§11.1); with the champion raw it is normalised.
    ablation_frame = (
        splits.fit if use_norm
        else _apply_to_frame(splits.fit, params=params, config=config, log_path=log)
    )
    ablation = _normalisation_ablation(
        ablation_frame, config=config, labels=labels, driver_col=driver_col,
        group_col=group_col, average=average,
    )

    # 7/9. Champion MLflow run: log everything, register, persist pkl for F06b. A registry failure
    # (sqlite lock, schema/backend error) is surfaced as a typed, logged MODEL_TRAINING_FAILED —
    # never a raw uncoded MLflow exception (spec §6 FM4 "or registration fails").
    model_artefact_path = art_dir / cfg.model_artefact_filename
    labels_artefact_path = art_dir / cfg.labels_filename
    meta_artefact_path = art_dir / cfg.bundle_meta_filename
    try:
        with mlflow.start_run(run_name="champion") as run:
            mlflow.log_params({k: v for k, v in best_params.items() if k != "num_class"})
            mlflow.log_metric("held_out_macro_f1", held_out_macro_f1)
            for drv, f1v in per_driver.items():
                mlflow.log_metric(f"per_driver_f1.{drv}", f1v)
            mlflow.log_metric("normalisation_ablation_macro_f1", ablation)
            cm_path = _log_json_artifact(
                {"labels": labels, "matrix": cmatrix},
                name="confusion_matrix.json",
                tmp_dir=art_dir / "_tmp",
            )
            cal_path = _log_json_artifact(
                _calibration_curve(champion, x=x_test, y=y_test, n_classes=n_classes),
                name="calibration_curve.json",
                tmp_dir=art_dir / "_tmp",
            )
            joblib_dump(champion, model_artefact_path)
            mlflow.log_artifact(str(model_artefact_path))
            # F06b seam: persist the index->driver-name map. The champion's classes_ are integers
            # [0..n-1]; `labels` is the sorted driver-name list F06a encoded against, so labels[i]
            # is the name of class i. Written deterministically (sorted list, no timestamp) next to
            # classifier.pkl so F06b can map predict_proba columns to driver names offline.
            labels_artefact_path.write_text(json.dumps(labels), encoding="utf-8")
            mlflow.log_artifact(str(labels_artefact_path))
            # The representation record (F06b seam): serving reads THIS, never the live config —
            # a bundle trained raw is served raw even if config later flips (and vice versa).
            meta_artefact_path.write_text(
                json.dumps({"uses_circuit_normalisation": use_norm}, sort_keys=True),
                encoding="utf-8",
            )
            mlflow.log_artifact(str(meta_artefact_path))
            # serialization_format="cloudpickle": mlflow 3.x defaults the sklearn flavor to skops,
            # which is not bundled with mlflow-skinny; cloudpickle ships with skinny and round-trips
            # the CalibratedClassifierCV(FrozenEstimator(XGBClassifier)) champion.
            mlflow.sklearn.log_model(
                champion,
                name="model",
                serialization_format="cloudpickle",
                registered_model_name=cfg.mlflow_registered_model,
            )
            run_id = run.info.run_id
            model_uri = f"runs:/{run_id}/model"
    except ClassifierTrainingError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface ANY registry/log failure as a typed code
        # Remove the orphan champion pkl + labels sidecar so F06b never loads a model bundle from a
        # run we report as failed (mirrors the leakage / no-champion paths — no model on disk).
        model_artefact_path.unlink(missing_ok=True)
        labels_artefact_path.unlink(missing_ok=True)
        meta_artefact_path.unlink(missing_ok=True)
        ledger.record(
            code=ErrorCode.MODEL_TRAINING_FAILED,
            entity_type="training_run",
            entity_id=cfg.mlflow_registered_model,
            detail=f"{REASON_REGISTRATION_FAILED}: {exc}",
            fallback_used=False,
        )
        raise ClassifierTrainingError(
            f"MLflow champion logging/registration failed: {exc}",
            error_code=ErrorCode.MODEL_TRAINING_FAILED,
        ) from exc

    # Degraded signalling — reported, never silently shipped as a pass.
    outcome = _evaluate_outcome(
        held_out_macro_f1=held_out_macro_f1,
        per_driver=per_driver,
        baseline_macro_f1=baseline_macro_f1,
        cfg=cfg,
    )
    for reason, detail in outcome.reasons:
        ledger.record(
            code=ErrorCode.MODEL_TRAINING_FAILED,
            entity_type="training_run",
            entity_id=cfg.mlflow_registered_model,
            detail=f"{reason}: {detail}",
            fallback_used=False,
        )
    flags = list(outcome.flags)
    if ledger.failed:
        flags.append(ErrorCode.DATA_QUALITY_LOG_UNWRITABLE)

    return ClassifierTrainingResult(
        status=outcome.status,
        error_code=ErrorCode.MODEL_TRAINING_FAILED if outcome.status is Status.DEGRADED else None,
        flags=flags,
        logged=ledger.rows > 0,
        mlflow_run_id=run_id,
        model_uri=model_uri,
        model_artefact_path=str(model_artefact_path),
        labels_artefact_path=str(labels_artefact_path),
        champion_params={k: v for k, v in study.best_params.items()},
        n_trials=len(trial_records),
        n_trials_failed=n_failed,
        cv_strategy=cfg.cv_strategy,
        cv_folds=cfg.cv_folds,
        train_circuits=splits.train_circuits,
        excluded_circuits=sorted(config.held_out_circuits),
        calibration_method=cfg.calibration_method,
        calibration_holdout_frac=cfg.calibration_holdout_frac,
        held_out_macro_f1=held_out_macro_f1,
        held_out_per_driver_f1=per_driver,
        confusion_matrix_path=cm_path,
        calibration_curve_path=cal_path,
        baseline_macro_f1=baseline_macro_f1,
        beats_all_baselines=outcome.beats_all,
        uses_circuit_normalisation=use_norm,
        normalisation_ablation_macro_f1=ablation,
        target_macro_f1=cfg.target_macro_f1,
        target_met=outcome.target_met,
        per_driver_floor=cfg.per_driver_floor,
        per_driver_floor_met=outcome.floor_met,
        seeds={
            "numpy": config.seeds.numpy,
            "xgboost": config.seeds.xgboost,
            "optuna": config.seeds.optuna,
        },
        normalisation_content_hash=norm_hash,
    )


@dataclass
class _Outcome:
    """The reported (non-fatal) quality decision for a trained champion."""

    target_met: bool
    floor_met: bool
    beats_all: bool
    status: Status
    flags: list[ErrorCode]
    reasons: list[tuple[str, str]]  # (reason_tag, detail) — one data_quality_log row each


def _evaluate_outcome(
    *,
    held_out_macro_f1: float,
    per_driver: dict[str, float],
    baseline_macro_f1: dict[str, float],
    cfg: ClassifierConfig,
) -> _Outcome:
    """Decide OK vs DEGRADED from the held-out metrics — a pure function (no I/O), so the
    target/floor/baseline gate logic and its boundaries (0.78 / 0.70) are unit-testable without
    training. DEGRADED is a reported research finding, never a silent pass."""
    target_met = bool(math.isfinite(held_out_macro_f1) and held_out_macro_f1 >= cfg.target_macro_f1)
    floor_met = bool(per_driver) and all(v >= cfg.per_driver_floor for v in per_driver.values())
    beats_all = bool(baseline_macro_f1) and all(
        held_out_macro_f1 > b for b in baseline_macro_f1.values()
    )
    reasons: list[tuple[str, str]] = []
    if not target_met:
        reasons.append(
            (REASON_TARGET_MISSED, f"{held_out_macro_f1:.4f} < target {cfg.target_macro_f1}")
        )
    if not floor_met:
        failing = {d: round(v, 4) for d, v in per_driver.items() if v < cfg.per_driver_floor}
        reasons.append(
            (REASON_DRIVER_FLOOR, f"drivers below floor {cfg.per_driver_floor}: {failing}")
        )
    if not beats_all:
        reasons.append(
            (REASON_BASELINE_NOT_BEATEN, f"did not beat all baselines {baseline_macro_f1}")
        )
    status = Status.DEGRADED if reasons else Status.OK
    flags = [ErrorCode.MODEL_TRAINING_FAILED] if reasons else []
    return _Outcome(target_met, floor_met, beats_all, status, flags, reasons)


def _run_study(
    study: optuna.Study,
    *,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cv: StratifiedGroupKFold,
    config: AppConfig,
    ledger: _DQLedger,
    fail_hook: Callable[[int], None] | None = None,
) -> tuple[list[TrialRecord], int, int]:
    """Run the study until ``optuna_trials`` COMPLETE trials accrue (topping up past failures).

    A trial that raises or yields NaN is caught, logged (``MODEL_TRAINING_FAILED`` at trial
    scope), counted, and the study continues — never silently dropped. Returns the trial records,
    the failed-trial count, and the COMPLETE-trial count. ``fail_hook`` (tests only) may raise to
    inject a failing trial.
    """
    assert config.classifier is not None
    cfg = config.classifier
    target = cfg.optuna_trials
    max_attempts = target * _TRIAL_ATTEMPT_MULTIPLIER + _TRIAL_ATTEMPT_BASE

    def objective(trial: optuna.Trial) -> float:
        if fail_hook is not None:
            fail_hook(trial.number)  # may raise to simulate a failing trial
        params = _suggest_params(trial, seed=config.seeds.xgboost, n_classes=len(np.unique(y)))
        score = _cv_macro_f1(
            params, x=x, y=y, groups=groups, cv=cv, average=_OBJECTIVE_AVERAGE[cfg.optuna_objective]
        )
        if not math.isfinite(score):
            ledger.record(
                code=ErrorCode.MODEL_TRAINING_FAILED,
                entity_type="trial",
                entity_id=str(trial.number),
                detail=f"{REASON_TRIAL_NAN}: non-finite macro-F1",
                fallback_used=False,
            )
            raise optuna.TrialPruned
        return score

    def _on_fail(_study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state is optuna.trial.TrialState.FAIL:
            ledger.record(
                code=ErrorCode.MODEL_TRAINING_FAILED,
                entity_type="trial",
                entity_id=str(trial.number),
                detail=f"{REASON_TRIAL_RAISED}: trial raised",
                fallback_used=False,
            )

    attempts = 0
    while (
        len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))) < target
        and attempts < max_attempts
    ):
        study.optimize(objective, n_trials=1, catch=(Exception,), callbacks=[_on_fail])
        attempts += 1

    records: list[TrialRecord] = []
    n_failed = 0
    n_complete = 0
    for t in study.trials:
        completed = t.state is optuna.trial.TrialState.COMPLETE
        if completed:
            n_complete += 1
        else:
            n_failed += 1
        records.append(
            TrialRecord(
                trial_number=t.number,
                params={k: cast("float | int", v) for k, v in t.params.items()},
                macro_f1=float(t.value) if completed and t.value is not None else None,
                state=t.state.name,
            )
        )
    return records, n_failed, n_complete


def _suggest_from_best(best_params: dict[str, Any], *, seed: int, n_classes: int) -> dict[str, Any]:
    """Rebuild the full XGBoost param dict (fixed keys + best tuned values) for the champion fit."""
    return {**_base_xgb_params(seed=seed, n_classes=n_classes), **best_params}


def _normalisation_ablation(
    fit_laps: pd.DataFrame,
    *,
    config: AppConfig,
    labels: Sequence[str],
    driver_col: str,
    group_col: str,
    average: str,
) -> float:
    """Train-circuit macro-F1 with normalisation OFF (§11.1) — a reported number, not a gate.

    Fits a default XGBoost on the *raw* (un-normalised) fit-set features and scores it on a
    StratifiedGroupKFold of that same set — the identical row budget the champion's CV used — so
    the comparison isolates the effect of normalisation, not of data volume. ``fit_laps`` is the
    raw (un-normalised) fit split, already free of held-out circuits and the calibration hold-out.
    """
    x, y, groups = _xy(fit_laps, labels=labels, driver_col=driver_col, group_col=group_col)
    params = _suggest_from_best({}, seed=config.seeds.xgboost, n_classes=len(labels))
    return _cv_macro_f1(params, x=x, y=y, groups=groups, cv=build_cv(config), average=average)


def _calibration_curve(
    model: Any, *, x: np.ndarray, y: np.ndarray, n_classes: int
) -> dict[str, list[float]]:
    """Reliability data: mean predicted top-class prob vs empirical accuracy in 10 bins."""
    if len(x) == 0:
        return {"bin_confidence": [], "bin_accuracy": [], "bin_count": []}
    proba = model.predict_proba(x)
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, 11)
    conf_out: list[float] = []
    acc_out: list[float] = []
    cnt_out: list[float] = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        cnt = int(mask.sum())
        cnt_out.append(float(cnt))
        conf_out.append(float(conf[mask].mean()) if cnt else 0.0)
        acc_out.append(float(correct[mask].mean()) if cnt else 0.0)
    return {"bin_confidence": conf_out, "bin_accuracy": acc_out, "bin_count": cnt_out}
