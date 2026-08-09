"""F06b — Model 1 (driver classifier) INFERENCE: load the champion, gate, serve a PredictionResult.

This is the **inference half** of F06. It LOADS the champion F06a registered and serves a gated
prediction; it **never trains, tunes, or re-fits** (a static test scan enforces this). The seam is
the on-disk champion bundle F06a writes: ``classifier.pkl`` (joblib) + ``normalisation.json`` (F04)
+ ``labels.json`` (the index→driver-name map for the champion's integer ``classes_``).

Gating (master §5.1), in fixed order so the outcome is total and deterministic:
1. **NULL gate (model not called):** > ``thresholds.null_ceiling`` NULLs → ``INVALID_INPUT``.
2. **Normalise** via F04 ``apply_normalisation`` (unknown circuit → ``NORMALISATION_FALLBACK``).
3. **Guarded** ``predict_proba`` — any internal fault → ``INFERENCE_ERROR`` (never a 500).
4. **Entropy gate:** Shannon entropy > ``thresholds.entropy_ambiguous_bits`` → ``AMBIGUOUS``.
5. **Confidence gate:** max-prob < ``thresholds.confidence_gate`` → ``LOW_CONFIDENCE`` (top-3).
6. **Accepted band:** ``HIGH`` if max-prob ≥ ``classifier.confidence_high`` else ``MEDIUM``.
7. **Latency:** > ``classifier.latency_warn_ms`` → ``PERF_WARN`` flag.

No component fails silently (non-negotiable #1): ``predict`` **never raises** (every fault is a
coded :class:`PredictionResult`); only :func:`load_model` raises (``MODEL_NOT_FOUND``) at startup,
after logging. ``confidence`` is never NULL; ``confidence_state`` is non-NULL on every predicted
path and ``None`` only when ``status=FAILED`` (model not run / failed) — those results are never
persisted to the ``predictions`` table, so §8.4's "no NULL confidence" (a constraint on persisted
rows) holds. Every threshold is read from ``config`` — no magic numbers.
"""

from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from joblib import load as joblib_load
from pydantic import BaseModel, ConfigDict, Field

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
    from collections.abc import Sequence

# data_quality_log detail tags (mirrors normalise.py / classifier.py REASON_* convention).
REASON_MODEL_ABSENT = "MODEL_ABSENT"
REASON_MODEL_CORRUPT = "MODEL_CORRUPT"
REASON_PARAMS_UNAVAILABLE = "PARAMS_UNAVAILABLE"
REASON_LABELS_ABSENT = "LABELS_ABSENT"
REASON_LABELS_MISMATCH = "LABELS_MISMATCH"
REASON_FEATURE_COUNT = "FEATURE_COUNT_MISMATCH"
REASON_NULL_EXCESS = "NULL_EXCESS"
REASON_PREDICT_RAISED = "PREDICT_RAISED"

_ENTITY = "prediction"
# Number of candidates surfaced below the gate. A spec-pinned design constant (master §5.1
# "top-3 candidates"), structurally coupled to the ``top_3`` PredictionResult field + the §8.4
# predictions schema — NOT a tunable threshold, so it lives in code like classifier.py's XGBoost
# search-space constants, not in config.thresholds.
_TOP_K = 3
_VERSION_HASH_LEN = 16  # hex chars of the sha256 champion digest used as the model_version id


class ModelLoadError(Exception):
    """Raised by :func:`load_model` when the champion bundle is absent / corrupt / mismatched.

    Carries ``ErrorCode.MODEL_NOT_FOUND``. Fail-loud at startup (non-negotiable #1): the process
    must refuse to serve rather than predict from a missing or mislabelled model. A
    ``data_quality_log`` row is written before the raise (master §5.1: "Log MODEL_NOT_FOUND")."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = ErrorCode.MODEL_NOT_FOUND


@dataclass(frozen=True)
class ModelHandle:
    """The loaded, immutable champion bundle shared read-only across predictions."""

    model: Any  # CalibratedClassifierCV(FrozenEstimator(XGBClassifier)) — sklearn estimator
    normalisation_params: NormalisationParams
    drivers: list[str]  # index i -> driver name for the champion's integer class i
    model_version: str
    ready: bool = True


class HealthResult(BaseResult):
    """Readiness for F13's ``GET /health``. ``error_code=HEALTH_FAIL`` (→ 503) when not ready,
    with ``MODEL_NOT_FOUND`` as the carried reason (master §5.1)."""

    ready: bool
    model_version: str = ""
    reason: ErrorCode | None = None


class ConfidenceBreakdown(BaseModel):
    """The propagated-confidence composition (system-architecture input #6): a **presentational
    composition of already-served signals** — it never participates in the §5.1 gate ordering and
    never overrides ``confidence_state``. Each factor is a distance from a boundary the system
    already enforces (no new thresholds are introduced):

    - ``feature_quality`` = 1 − n_nulls / ``thresholds.null_ceiling`` — distance from the NULL
      rejection boundary (0 nulls → 1.0; at the ceiling → 0.0).
    - ``distribution`` = 1 − entropy_bits / ``thresholds.entropy_ambiguous_bits``, floored at 0 —
      distance from the AMBIGUOUS boundary, using the *same* entropy the gate compared.
    - ``model_confidence`` = the calibrated max-probability, verbatim.
    - ``overall`` = the product of the three — any weak link drags the composite down; all strong
      → high. A multiplicative composition, so no arbitrary weights exist to justify."""

    model_config = ConfigDict(extra="forbid")

    feature_quality: float
    distribution: float
    model_confidence: float
    overall: float


class PredictionResult(BaseResult):
    """One gated prediction (master §5.1 / §8.5). ``confidence`` is never NULL; ``confidence_state``
    is ``None`` only on a ``FAILED`` result (model not run / failed), which is never persisted."""

    model_config = ConfigDict(arbitrary_types_allowed=False)

    probabilities: dict[str, float] = Field(default_factory=dict)  # all classes, sums to 1.0
    top_prediction: str | None = None
    confidence: float = 0.0
    confidence_state: ConfidenceState | None = None
    top_3: list[tuple[str, float]] = Field(default_factory=list)
    shap_values: dict[str, float] | None = None  # F07 fills it
    shap_available: bool = False  # F07 sets it
    latency_ms: float = 0.0
    model_version: str = ""
    null_features: list[str] = Field(default_factory=list)
    # None on FAILED results (no distribution to score) — mirrors confidence_state=None there.
    confidence_breakdown: ConfidenceBreakdown | None = None


@dataclass
class _DQLedger:
    """One ``data_quality_log`` row per gated/failed/fallback event; never propagates a sink
    failure (mirrors F04/F06a). A lost row is counted so the caller can flag
    ``DATA_QUALITY_LOG_UNWRITABLE`` rather than crash (non-negotiable #1)."""

    log_path: Path
    rows: int = 0
    failed: int = 0

    def record(
        self,
        *,
        code: ErrorCode,
        entity_id: str,
        detail: str,
        fallback_used: bool,
        entity_type: str = _ENTITY,
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


# ----------------------------------------------------------------------------------
# Load (startup, once) — fail-loud
# ----------------------------------------------------------------------------------


def _resolve_model_version(model_bytes: bytes) -> str:
    """A content-addressed version id = ``sha256`` of the champion bytes. Deterministic, tied to the
    EXACT loaded artefact, and computed offline with no I/O to an MLflow tracking server — so
    ``predictions.model_version`` is always populated and never misattributes a prediction to a
    different registered version, and ``load_model`` cannot hang on an unreachable registry."""
    return f"sha256:{sha256(model_bytes).hexdigest()[:_VERSION_HASH_LEN]}"


def load_model(
    *,
    config: AppConfig,
    artefact_dir: str | Path | None = None,
    log_path: str | Path | None = None,
) -> ModelHandle:
    """Load + validate the champion bundle (champion + normalisation params + labels) ONCE.

    Raises :class:`ModelLoadError` (``MODEL_NOT_FOUND``), after logging, on any of: missing config
    block, absent/corrupt ``classifier.pkl``, missing/corrupt ``normalisation.json``, absent
    ``labels.json``, a label count that does not match the champion's class count, non-integer
    ``classes_``, or a feature count ≠ 28. A stale/mismatched labels file therefore **refuses to
    serve** rather than silently mislabel predictions.
    """
    art_dir = Path(artefact_dir) if artefact_dir is not None else Path(config.paths.artefacts_dir)
    log = Path(log_path) if log_path is not None else Path(config.paths.data_quality_log)
    ledger = _DQLedger(log_path=log)

    def _fail(reason: str, detail: str) -> ModelLoadError:
        ledger.record(
            code=ErrorCode.MODEL_NOT_FOUND,
            entity_type="model",
            entity_id="startup",
            detail=f"{reason}: {detail}",
            fallback_used=False,
        )
        return ModelLoadError(detail)

    if config.classifier is None or config.normalise is None:
        raise _fail("CONFIG", "config.classifier / config.normalise block absent; cannot load")

    model_path = art_dir / config.classifier.model_artefact_filename
    if not model_path.is_file():
        raise _fail(REASON_MODEL_ABSENT, f"champion artefact not found: {model_path}")
    try:
        # Read the champion bytes ONCE: deserialise the model AND derive the content-addressed
        # model_version from the same buffer (no second full-file read of a multi-MB artefact).
        model_bytes = model_path.read_bytes()
        model = joblib_load(io.BytesIO(model_bytes))
    except Exception as exc:  # noqa: BLE001 - any unpickling fault is a coded load failure
        raise _fail(REASON_MODEL_CORRUPT, f"champion artefact unreadable: {exc}") from exc

    try:
        params = load_normalisation_params(art_dir / config.normalise.artefact_filename)
    except NormalisationError as exc:
        raise _fail(REASON_PARAMS_UNAVAILABLE, f"normalisation params unavailable: {exc}") from exc

    labels_path = art_dir / config.classifier.labels_filename
    if not labels_path.is_file():
        raise _fail(REASON_LABELS_ABSENT, f"labels sidecar not found: {labels_path}")
    try:
        drivers = json.loads(labels_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _fail(REASON_LABELS_ABSENT, f"labels sidecar unreadable: {exc}") from exc

    classes = list(getattr(model, "classes_", []))
    if not isinstance(drivers, list) or len(drivers) != len(classes):
        raise _fail(
            REASON_LABELS_MISMATCH,
            f"labels.json count ({len(drivers) if isinstance(drivers, list) else 'n/a'}) != "
            f"champion classes_ ({len(classes)}); refusing to serve mislabelled predictions",
        )
    if len(set(drivers)) != len(drivers):
        # Duplicate names would collapse the name-keyed probabilities dict (mass silently lost).
        raise _fail(REASON_LABELS_MISMATCH, f"labels.json has duplicate driver names: {drivers}")
    if list(classes) != list(range(len(classes))):
        raise _fail(
            REASON_LABELS_MISMATCH,
            f"champion classes_ are not the integer range [0..n-1]: {classes}",
        )
    n_features = int(getattr(model, "n_features_in_", -1))
    if n_features != len(FEATURE_IDS):
        raise _fail(
            REASON_FEATURE_COUNT,
            f"champion n_features_in_={n_features} != expected {len(FEATURE_IDS)}",
        )

    version = _resolve_model_version(model_bytes)
    return ModelHandle(
        model=model, normalisation_params=params, drivers=list(drivers), model_version=version
    )


def health(handle: ModelHandle | None) -> HealthResult:
    """Readiness check F13's ``GET /health`` consumes. Not-loaded → ``HEALTH_FAIL`` (→ 503) with
    ``MODEL_NOT_FOUND`` as the reason; loaded → OK + the champion's ``model_version``."""
    if handle is None or not handle.ready:
        return HealthResult(
            status=Status.FAILED,
            error_code=ErrorCode.HEALTH_FAIL,
            ready=False,
            reason=ErrorCode.MODEL_NOT_FOUND,
        )
    return HealthResult(status=Status.OK, ready=True, model_version=handle.model_version)


# ----------------------------------------------------------------------------------
# Predict (per call) — never raises
# ----------------------------------------------------------------------------------


def _unwritable_flags(ledger: _DQLedger) -> list[ErrorCode]:
    """Surface ``DATA_QUALITY_LOG_UNWRITABLE`` whenever the ledger lost a row — so even an
    early-return (rejected) result flags a silent audit-sink failure rather than hiding it."""
    return [ErrorCode.DATA_QUALITY_LOG_UNWRITABLE] if ledger.failed else []


def _dedup(flags: list[ErrorCode]) -> list[ErrorCode]:
    """Order-preserving de-duplication — F04's carried flags + this layer's own can both emit
    ``DATA_QUALITY_LOG_UNWRITABLE`` when the shared sink is down; a code should appear once."""
    return list(dict.fromkeys(flags))


def _entropy_bits(proba: np.ndarray) -> float:
    """Shannon entropy in bits, ignoring zero-probability classes (0·log0 = 0)."""
    nz = proba[proba > 0.0]
    return float(-np.sum(nz * np.log2(nz))) if nz.size else 0.0


def _rejected(
    *,
    code: ErrorCode,
    latency_ms: float,
    model_version: str,
    null_features: Sequence[str] = (),
    logged: bool,
    flags: list[ErrorCode] | None = None,
) -> PredictionResult:
    """Build a FAILED result (model not run / failed): no distribution, confidence_state=None,
    confidence=0.0 (never NULL). Never persisted to the predictions table."""
    return PredictionResult(
        status=Status.FAILED,
        error_code=code,
        flags=_dedup(flags or []),
        logged=logged,
        probabilities={},
        top_prediction=None,
        confidence=0.0,
        confidence_state=None,
        top_3=[],
        latency_ms=latency_ms,
        model_version=model_version,
        null_features=list(null_features),
    )


def predict(
    feature_vector: dict[str, float | None],
    *,
    circuit: str,
    handle: ModelHandle,
    config: AppConfig,
    log_path: str | Path | None = None,
) -> PredictionResult:
    """Return a gated, calibrated :class:`PredictionResult` for one lap vector. **Never raises** —
    every fault (rejected input, unknown circuit, internal model error) is a coded result so the
    F13 layer never 500s on a model output (master §8.5)."""
    log = Path(log_path) if log_path is not None else Path(config.paths.data_quality_log)
    ledger = _DQLedger(log_path=log)
    start = time.perf_counter()

    def _ms() -> float:
        return (time.perf_counter() - start) * 1000.0

    try:
        if config.classifier is None:  # misconfiguration -> coded result, never a raise
            raise ValueError("config.classifier block absent")
        cfg = config.classifier
        th = config.thresholds
        # Normalise the input to exactly the 28 keys FIRST — a caller may OMIT a key rather than
        # set it None; treat omitted == None so the NULL gate, apply_normalisation, and vector build
        # all agree (a missing key must not KeyError into a spurious INFERENCE_ERROR).
        fv: dict[str, float | None] = {fid: feature_vector.get(fid) for fid in FEATURE_IDS}

        # 1. NULL gate — before any model call.
        nulls = [fid for fid in FEATURE_IDS if fv[fid] is None]
        if len(nulls) > th.null_ceiling:
            ledger.record(
                code=ErrorCode.INVALID_INPUT,
                entity_id=circuit,
                detail=f"{REASON_NULL_EXCESS}: {len(nulls)} > ceiling {th.null_ceiling}: {nulls}",
                fallback_used=False,
            )
            return _rejected(
                code=ErrorCode.INVALID_INPUT,
                latency_ms=_ms(),
                model_version=handle.model_version,
                null_features=nulls,
                logged=ledger.rows > 0,
                flags=_unwritable_flags(ledger),
            )

        # 2. Normalise (read-only F04 reader; never raises). Carry F04's own flags (e.g. a lost
        # audit row) forward. Only an UNKNOWN/skipped circuit (used_global) is a genuine
        # NORMALISATION_FALLBACK degradation; a NULL among the six on a KNOWN circuit is just the
        # tolerated-NULL passthrough, surfaced via null_features rather than mislabelled as a
        # circuit fallback.
        nres = apply_normalisation(
            fv, circuit=circuit, params=handle.normalisation_params, config=config, log_path=log
        )
        flags: list[ErrorCode] = list(nres.flags)
        degraded = bool(nres.used_global)
        if nres.used_global:
            flags.append(ErrorCode.NORMALISATION_FALLBACK)
        vec = nres.normalised_vector
        x = np.array(
            [[(vec.get(fid) if vec.get(fid) is not None else np.nan) for fid in FEATURE_IDS]],
            dtype=float,
        )

        # 3. Guarded predict_proba -> INFERENCE_ERROR on any internal fault (never 500).
        try:
            proba = np.asarray(handle.model.predict_proba(x)[0], dtype=float)
        except Exception as exc:  # noqa: BLE001 - any model fault becomes a coded result
            ledger.record(
                code=ErrorCode.INFERENCE_ERROR,
                entity_id=circuit,
                detail=f"{REASON_PREDICT_RAISED}: {exc}",
                fallback_used=False,
            )
            return _rejected(
                code=ErrorCode.INFERENCE_ERROR,
                latency_ms=_ms(),
                model_version=handle.model_version,
                logged=ledger.rows > 0,
                flags=[*flags, *_unwritable_flags(ledger)],
            )

        drivers = handle.drivers
        probabilities = {drivers[i]: float(p) for i, p in enumerate(proba)}
        confidence = float(proba.max())
        order = np.argsort(proba)[::-1][:_TOP_K]
        top_3 = [(drivers[int(i)], float(proba[int(i)])) for i in order]

        # 4-6. Entropy gate (precedence) -> confidence gate -> accepted band.
        h_bits = _entropy_bits(proba)
        if h_bits > th.entropy_ambiguous_bits:
            state: ConfidenceState = ConfidenceState.AMBIGUOUS
            top_prediction: str | None = None
            error_code: ErrorCode | None = ErrorCode.AMBIGUOUS
            status = Status.DEGRADED
            ledger.record(
                code=ErrorCode.AMBIGUOUS, entity_id=circuit,
                detail=f"entropy {h_bits:.3f} > {th.entropy_ambiguous_bits} bits",
                fallback_used=False,
            )
        elif confidence < th.confidence_gate:
            state = ConfidenceState.LOW_CONFIDENCE
            top_prediction = None
            error_code = ErrorCode.LOW_CONFIDENCE
            status = Status.DEGRADED
            ledger.record(
                code=ErrorCode.LOW_CONFIDENCE, entity_id=circuit,
                detail=f"max-prob {confidence:.4f} < gate {th.confidence_gate}",
                fallback_used=False,
            )
        else:
            top_prediction = drivers[int(order[0])]
            high = confidence >= cfg.confidence_high
            state = ConfidenceState.HIGH if high else ConfidenceState.MEDIUM
            error_code = None
            # a used_global normalisation fallback overlays DEGRADED on an otherwise-OK path
            status = Status.DEGRADED if degraded else Status.OK

        # 6b. Propagated-confidence composition (system input #6). Presentational only — computed
        # AFTER the gate from the gate's own inputs; nothing reads it back. Denominators are the
        # boundaries steps 1 and 4 already enforced (config values, referenced never re-declared),
        # so the composition introduces no new threshold.
        feature_quality = 1.0 - (len(nulls) / th.null_ceiling) if th.null_ceiling > 0 else 1.0
        distribution = (
            max(0.0, 1.0 - (h_bits / th.entropy_ambiguous_bits))
            if th.entropy_ambiguous_bits > 0
            else 1.0
        )
        breakdown = ConfidenceBreakdown(
            feature_quality=feature_quality,
            distribution=distribution,
            model_confidence=confidence,
            overall=feature_quality * distribution * confidence,
        )

        # 7. Latency flag (does not by itself degrade status).
        latency_ms = _ms()
        if latency_ms > cfg.latency_warn_ms:
            flags.append(ErrorCode.PERF_WARN)
            ledger.record(
                code=ErrorCode.PERF_WARN, entity_id=circuit,
                detail=f"latency {latency_ms:.1f}ms > {cfg.latency_warn_ms}ms",
                fallback_used=False,
            )

        if ledger.failed:
            flags.append(ErrorCode.DATA_QUALITY_LOG_UNWRITABLE)

        return PredictionResult(
            status=status,
            error_code=error_code,
            flags=_dedup(flags),
            logged=(ledger.rows > 0) or nres.logged,
            probabilities=probabilities,
            top_prediction=top_prediction,
            confidence=confidence,
            confidence_state=state,
            top_3=top_3,
            latency_ms=latency_ms,
            model_version=handle.model_version,
            null_features=nulls,  # provenance: which features were missing/imputed on a served lap
            confidence_breakdown=breakdown,
        )
    except Exception as exc:  # noqa: BLE001 - the never-raises backstop (master §8.5)
        try:
            ledger.record(
                code=ErrorCode.INFERENCE_ERROR, entity_id=circuit,
                detail=f"{REASON_PREDICT_RAISED}: {type(exc).__name__}: {exc}",
                fallback_used=False,
            )
        except Exception:  # noqa: BLE001 - even the logger must not break the never-raises contract
            pass
        return _rejected(
            code=ErrorCode.INFERENCE_ERROR,
            latency_ms=_ms(),
            model_version=handle.model_version,
            logged=ledger.rows > 0,
            flags=_unwritable_flags(ledger),
        )
