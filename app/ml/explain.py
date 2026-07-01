"""F07 — SHAP explainability: beeswarm (global) · force (local) · radar · NL templates · stability.

Turns every F06 classifier output into a SHAP decomposition (master §6). Explainability is a
first-class output, but it is **best-effort and never blocks a prediction**: a SHAP fault degrades
to ``SHAP_UNAVAILABLE`` and the F06b :class:`~app.ml.inference.PredictionResult` is preserved (§6.2
FM1). This module is **read-only** — it loads the F06 champion and explains it; it never trains,
tunes, or re-fits (a static test scan enforces no ``.fit(`` / ``fit_normaliser`` / ``optuna`` here).

What it produces:

* **Local** — :func:`explain_prediction` returns the 28-feature signed SHAP for the predicted
  driver, a force list (which features pushed *toward* / *away*), a 6-axis radar, and a
  natural-language style sentence. Fills F06b's ``shap_values`` / ``shap_available``.
* **Global** — :func:`build_global_explainability` writes the beeswarm artefact once after training
  and returns per-driver global SHAP (→ ``fingerprints.shap_global``) + per-driver radar (→
  ``fingerprints.radar_values``, 6D).
* **Multi-season stability** — :func:`season_stability` scores year-on-year fingerprint drift via
  Jensen-Shannon divergence between a driver's per-season SHAP distributions (→
  ``fingerprints.stability_score``).

The three §6.2 failure modes are all handled + logged + surfaced; no new error code is added
(``SHAP_UNAVAILABLE`` / ``SHAP_CONSISTENCY_WARN`` are pre-seeded in ``errors.py``). Every threshold
is read from ``config.shap`` — no magic numbers. TreeExplainer explains the champion's **base
XGBoost margins** (the calibrator is a monotone post-hoc transform), which is the standard,
documented approach (see ``specs/F07`` §9).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import shap
import xgboost as xgb
from pydantic import BaseModel, ConfigDict, Field
from scipy.spatial.distance import jensenshannon

from app.contracts.config import AppConfig
from app.contracts.enums import Status
from app.contracts.errors import ErrorCode
from app.contracts.logging import DataQualityLogError, log_data_quality
from app.contracts.results import BaseResult
from app.pipeline.features import FEATURE_IDS
from app.pipeline.normalise import apply_normalisation

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from app.ml.inference import ModelHandle

# ----------------------------------------------------------------------------------
# Structural constants (from master §4.2 / §6.1) — NOT tunable, so they live in code like
# FEATURE_IDS, never in config. A domain is not a threshold.
# ----------------------------------------------------------------------------------

# The six radar domains and their feature membership (§6.1 "the 6 feature domains (throttle,
# braking, speed, gear, engine, cornering)"; membership from the §4.2 Domain column). F22–F28
# (Interaction / Phase / Degradation) are deliberately NOT part of any radar axis — they appear in
# the beeswarm/force (all 28) but the master names exactly six radar domains (see specs/F07 §9).
DOMAIN_FEATURES: dict[str, tuple[str, ...]] = {
    "throttle": ("F01", "F02", "F03", "F04", "F05"),
    "braking": ("F06", "F07", "F08", "F09", "F10"),
    "speed": ("F11", "F12", "F13", "F14"),
    "gear": ("F15", "F16"),
    "engine": ("F17", "F18"),
    "cornering": ("F19", "F20", "F21"),
}
RADAR_DOMAINS: tuple[str, ...] = tuple(DOMAIN_FEATURES)

# Human-readable feature names (master §4.2 Name column) for the force-plot display.
FEATURE_NAMES: dict[str, str] = {
    "F01": "throttle_mean", "F02": "throttle_std", "F03": "throttle_open_rate",
    "F04": "throttle_full_fraction", "F05": "throttle_lift_fraction", "F06": "brake_peak_mean",
    "F07": "brake_duration_mean", "F08": "brake_release_rate", "F09": "brake_entry_speed_mean",
    "F10": "brake_overlap_fraction", "F11": "speed_min_mean", "F12": "speed_variance",
    "F13": "speed_p10", "F14": "speed_p90", "F15": "gear_change_rate",
    "F16": "gear_hold_duration_mean", "F17": "rpm_redline_fraction", "F18": "rpm_std",
    "F19": "lateral_g_mean", "F20": "lateral_g_peak_mean", "F21": "steering_correction_rate",
    "F22": "throttle_speed_corr", "F23": "brake_speed_corr", "F24": "throttle_brake_transition",
    "F25": "corner_entry_balance", "F26": "lap_phase_throttle_delta",
    "F27": "tyre_age_throttle_slope", "F28": "tyre_age_brake_slope",
}

# Natural-language phrase per dominant domain (§6.1 template engine). Composed for the top-N
# domains; the generic fallback text is the master's verbatim §6.2 FM3 string.
_DOMAIN_PHRASE: dict[str, str] = {
    "throttle": "committed, decisive throttle application",
    "braking": "aggressive late braking",
    "speed": "high mid-corner and top-end speed",
    "gear": "busy, deliberate gear usage",
    "engine": "sustained high-RPM engine loading",
    "cornering": "high-load, precise cornering",
}
_GENERIC_STYLE = (
    "This driver shows balanced characteristics across all domains with "
    "no single dominant style trait."
)

# data_quality_log detail tags (mirrors inference.py / normalise.py REASON_* convention).
REASON_EXPLAINER_BUILD = "EXPLAINER_BUILD_FAILED"
REASON_SHAP_RAISED = "SHAP_RAISED"
REASON_UNKNOWN_CLASS = "PREDICTED_CLASS_UNKNOWN"
REASON_CONSISTENCY = "ADDITIVITY_MISMATCH"
REASON_GLOBAL_SHAP = "GLOBAL_SHAP_FAILED"
REASON_STABILITY = "STABILITY_INSUFFICIENT_SEASONS"

_ENTITY = "explanation"


def _validate_domain_features() -> None:
    """Fail loud at import if ``DOMAIN_FEATURES`` does not partition F01..F21 exactly (§6.1).

    The six radar axes and their membership are a contract the radar + NL derivation rests on; a
    typo (a duplicated or dropped feature) would silently skew a domain score, so make it a
    build-breaking error at module load, not a wrong number at runtime (non-negotiable #1)."""
    if tuple(DOMAIN_FEATURES) != ("throttle", "braking", "speed", "gear", "engine", "cornering"):
        raise ValueError(
            f"DOMAIN_FEATURES keys must be the six §6.1 domains, got {list(DOMAIN_FEATURES)}"
        )
    seen: list[str] = [fid for ids in DOMAIN_FEATURES.values() for fid in ids]
    expected = tuple(f"F{n:02d}" for n in range(1, 22))  # F01..F21
    if tuple(sorted(seen)) != expected:
        raise ValueError(
            f"DOMAIN_FEATURES must partition F01..F21 exactly (no overlap/gap); got {sorted(seen)}"
        )
    if len(seen) != len(set(seen)):
        raise ValueError(f"DOMAIN_FEATURES has a feature in two domains: {seen}")


_validate_domain_features()


# ----------------------------------------------------------------------------------
# Result objects (typed — no bare dicts across boundaries)
# ----------------------------------------------------------------------------------


class Contribution(BaseModel):
    """One feature's local contribution to the predicted driver (the force-plot payload row)."""

    model_config = ConfigDict(extra="forbid")

    feature_id: str
    name: str
    shap: float
    feature_value: float | None
    direction: str  # "toward" (shap > 0) | "away" (shap <= 0)


class StyleDescription(BaseModel):
    """Natural-language style sentence + which domains drove it (§6.1)."""

    model_config = ConfigDict(extra="forbid")

    sentence: str
    dominant_domains: list[str] = Field(default_factory=list)
    generic: bool = False


class ExplanationResult(BaseResult):
    """One prediction's local SHAP decomposition (master §6). ``shap_available`` is ``False`` only
    on ``SHAP_UNAVAILABLE`` (prediction preserved, no SHAP); ``True`` even with a consistency
    warning (values are shown *with* the warning, §6.2 FM2)."""

    model_config = ConfigDict(extra="forbid")

    shap_values: dict[str, float] | None = None  # 28 F-id -> signed SHAP for predicted_class
    base_value: float = 0.0
    predicted_class: str = ""
    force: list[Contribution] = Field(default_factory=list)
    radar: dict[str, float] = Field(default_factory=dict)  # 6 domain scores
    dominant_domains: list[str] = Field(default_factory=list)
    nl_description: str = ""
    shap_available: bool = False
    consistency_ok: bool = True
    consistency_delta: float = 0.0
    latency_ms: float = 0.0


class GlobalExplainabilityResult(BaseResult):
    """Global explainability computed once after training (§6.1): the beeswarm artefact + per-driver
    global SHAP (→ ``fingerprints.shap_global``) + per-driver radar (→ ``radar_values``)."""

    model_config = ConfigDict(extra="forbid")

    beeswarm_artefact_path: str = ""
    per_driver_shap: dict[str, dict[str, float]] = Field(default_factory=dict)
    per_driver_radar: dict[str, dict[str, float]] = Field(default_factory=dict)
    n_background: int = 0
    shap_available: bool = False
    latency_ms: float = 0.0


class SeasonStabilityResult(BaseResult):
    """Multi-season fingerprint drift (§6.1): pairwise Jensen-Shannon divergence between a driver's
    per-season SHAP distributions. ``stability_score`` is the mean pairwise JS (lower = more stable
    / less drift); ``None`` when fewer than two seasons make any pair impossible."""

    model_config = ConfigDict(extra="forbid")

    stability_score: float | None = None
    pairwise_js: dict[str, float] = Field(default_factory=dict)
    seasons_available: int = 0
    partial: bool = False


@dataclass(frozen=True)
class ShapExplainer:
    """The loaded, immutable TreeExplainer bundle shared read-only across explanations.

    ``ready=False`` marks a failed build — every :func:`explain_prediction` then degrades to
    ``SHAP_UNAVAILABLE`` (the prediction is still served)."""

    explainer: Any  # shap.TreeExplainer over the base XGBClassifier
    base_model: Any  # the base XGBClassifier (for the independent additivity/margin check)
    base_values: dict[str, float] = field(default_factory=dict)  # driver -> tree base margin
    drivers: list[str] = field(default_factory=list)
    ready: bool = True


# ----------------------------------------------------------------------------------
# DQ ledger (mirrors inference.py — never propagates a sink failure)
# ----------------------------------------------------------------------------------


@dataclass
class _DQLedger:
    """One ``data_quality_log`` row per SHAP fault / warning; never propagates a sink failure. A
    lost row is counted so the caller can flag ``DATA_QUALITY_LOG_UNWRITABLE`` rather than crash."""

    log_path: Path
    rows: int = 0
    failed: int = 0

    def record(
        self, *, code: ErrorCode, entity_id: str, detail: str, entity_type: str = _ENTITY
    ) -> None:
        try:
            log_data_quality(
                error_code=code,
                entity_type=entity_type,
                entity_id=entity_id,
                error_detail=detail,
                fallback_used=False,
                log_path=self.log_path,
            )
            self.rows += 1
        except DataQualityLogError:
            self.failed += 1


def _unwritable_flags(ledger: _DQLedger) -> list[ErrorCode]:
    return [ErrorCode.DATA_QUALITY_LOG_UNWRITABLE] if ledger.failed else []


# ----------------------------------------------------------------------------------
# Base-model extraction + explainer build (startup, once) — never raises
# ----------------------------------------------------------------------------------


def _base_tree_model(champion: Any) -> xgb.XGBClassifier:
    """Extract the fitted base ``XGBClassifier`` from the calibrated champion.

    TreeExplainer cannot explain the calibrated wrapper, so we navigate
    ``champion.calibrated_classifiers_[0].estimator`` (the ``FrozenEstimator``) and unwrap its
    ``.estimator``. Prefit calibration yields exactly one ``calibrated_classifiers_`` entry, so
    index 0 is total. Raises ``TypeError`` if the structure is unexpected — the caller catches it
    and marks the explainer not-ready (never an ``AttributeError``, never a wrong model)."""
    calibrated = getattr(champion, "calibrated_classifiers_", None)
    if not calibrated:
        raise TypeError("champion has no calibrated_classifiers_; not a fitted calibrated model")
    inner = calibrated[0].estimator
    inner = getattr(inner, "estimator", inner)  # unwrap FrozenEstimator -> the fitted base
    if not isinstance(inner, xgb.XGBClassifier):
        raise TypeError(f"resolved base model is {type(inner).__name__}, expected XGBClassifier")
    return inner


def build_explainer(
    handle: ModelHandle, *, config: AppConfig, log_path: str | Path | None = None
) -> ShapExplainer:
    """Build the exact TreeExplainer over the champion's base XGBoost, ONCE at startup.

    Never raises (§6.2): on any failure it logs ``SHAP_UNAVAILABLE`` and returns a not-ready
    :class:`ShapExplainer` so every downstream explanation degrades cleanly while predictions keep
    serving."""
    log = Path(log_path) if log_path is not None else Path(config.paths.data_quality_log)
    ledger = _DQLedger(log_path=log)
    try:
        base = _base_tree_model(handle.model)
        # tree_path_dependent -> exact tree SHAP (§6.1 "exact, not approximate"), no background set.
        explainer = shap.TreeExplainer(base, feature_perturbation="tree_path_dependent")
        expected = np.atleast_1d(np.asarray(explainer.expected_value, dtype=float))
        base_values = {handle.drivers[i]: float(expected[i]) for i in range(len(handle.drivers))}
        return ShapExplainer(
            explainer=explainer,
            base_model=base,
            base_values=base_values,
            drivers=list(handle.drivers),
            ready=True,
        )
    except Exception as exc:  # noqa: BLE001 - any build fault -> not-ready, never blocks serving
        ledger.record(
            code=ErrorCode.SHAP_UNAVAILABLE,
            entity_id="startup",
            entity_type="explainer",
            detail=f"{REASON_EXPLAINER_BUILD}: {type(exc).__name__}: {exc}",
        )
        return ShapExplainer(
            explainer=None, base_model=None, base_values={}, drivers=list(handle.drivers),
            ready=False,
        )


# ----------------------------------------------------------------------------------
# Pure helpers — radar aggregation + NL template engine (no I/O; unit-testable in isolation)
# ----------------------------------------------------------------------------------


def aggregate_domains(shap_by_feature: Mapping[str, float]) -> dict[str, float]:
    """Aggregate signed per-feature SHAP into the six §6.1 radar domain scores.

    Each domain score is ``Σ |shap|`` over that domain's §4.2 features (the magnitude of the
    domain's contribution). Returns exactly the six ``RADAR_DOMAINS`` keys; F22–F28 contribute to
    none of them (the master names exactly six domains)."""
    return {
        domain: float(sum(abs(float(shap_by_feature.get(fid, 0.0))) for fid in ids))
        for domain, ids in DOMAIN_FEATURES.items()
    }


def describe_style(radar: Mapping[str, float], *, config: AppConfig) -> StyleDescription:
    """Compose the natural-language style sentence from the top-N radar domains (§6.1).

    If the six domain scores are degenerate — all within ``shap.nl_dominance_margin`` (10%) of the
    largest — return the master's verbatim generic sentence and **no** dominant domain (§6.2 FM3:
    do not fabricate a dominant trait). Otherwise compose the top ``shap.nl_top_domains`` domains'
    template phrases. A pure function (no I/O)."""
    cfg = _require_shap(config)
    scores = {d: float(radar.get(d, 0.0)) for d in RADAR_DOMAINS}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    hi = ranked[0][1]
    lo = ranked[-1][1]
    # Degenerate spread (incl. all-zero) -> generic; do not invent a dominant trait.
    if hi <= 0.0 or (hi - lo) <= cfg.nl_dominance_margin * hi:
        return StyleDescription(sentence=_GENERIC_STYLE, dominant_domains=[], generic=True)
    # Take the top-N, but drop any domain with a zero (or negative) contribution — naming a domain
    # that contributed nothing as a "dominant trait" is exactly the fabrication §6.2 FM3 forbids.
    top = [d for d, s in ranked[: cfg.nl_top_domains] if s > 0.0]
    phrases = [_DOMAIN_PHRASE[d] for d in top]
    if len(phrases) == 1:
        joined = phrases[0]
    elif len(phrases) == 2:
        joined = " and ".join(phrases)
    else:  # Oxford-style: "a, b and c" for nl_top_domains > 2
        joined = ", ".join(phrases[:-1]) + " and " + phrases[-1]
    return StyleDescription(
        sentence=f"This driver is characterised by {joined}.",
        dominant_domains=top,
        generic=False,
    )


def _require_shap(config: AppConfig) -> Any:
    """Return the validated ``shap`` config block or raise a typed error (never a bare ``None``
    attribute access). Callers that must not raise wrap this in their try/except."""
    if config.shap is None:
        raise ValueError("config.shap block absent; F07 explainability not configured")
    return config.shap


# ----------------------------------------------------------------------------------
# Local explanation (per prediction) — never raises
# ----------------------------------------------------------------------------------


def _shap_unavailable(
    *, predicted_class: str, latency_ms: float, logged: bool, flags: list[ErrorCode] | None = None
) -> ExplanationResult:
    """A degraded result carrying no SHAP — the prediction is preserved upstream (§6.2 FM1)."""
    return ExplanationResult(
        status=Status.DEGRADED,
        error_code=ErrorCode.SHAP_UNAVAILABLE,
        flags=list(flags or []),
        logged=logged,
        shap_values=None,
        base_value=0.0,
        predicted_class=predicted_class,
        force=[],
        radar={},
        dominant_domains=[],
        nl_description="",
        shap_available=False,
        consistency_ok=True,
        consistency_delta=0.0,
        latency_ms=latency_ms,
    )


def explain_prediction(
    feature_vector: dict[str, float | None],
    *,
    circuit: str,
    predicted_class: str,
    explainer: ShapExplainer,
    handle: ModelHandle,
    config: AppConfig,
    log_path: str | Path | None = None,
) -> ExplanationResult:
    """Return the local SHAP decomposition for one accepted prediction. **Never raises** (§6.2).

    Takes the **raw** feature vector + circuit and re-normalises via F04's read-only
    :func:`apply_normalisation` (F06b's ``predict`` does not expose the normalised vector), so F07
    explains exactly the 28 features the model consumed. Called only on an accepted (HIGH/MEDIUM)
    prediction — a gated result carries no SHAP."""
    log = Path(log_path) if log_path is not None else Path(config.paths.data_quality_log)
    ledger = _DQLedger(log_path=log)
    start = time.perf_counter()

    def _ms() -> float:
        return (time.perf_counter() - start) * 1000.0

    # Explainer failed to build -> degrade (prediction already served upstream).
    if not explainer.ready:
        ledger.record(
            code=ErrorCode.SHAP_UNAVAILABLE, entity_id=circuit,
            detail=f"{REASON_EXPLAINER_BUILD}: explainer not ready",
        )
        return _shap_unavailable(
            predicted_class=predicted_class, latency_ms=_ms(),
            logged=ledger.rows > 0, flags=_unwritable_flags(ledger),
        )

    try:
        cfg = _require_shap(config)
        # Guard against an explainer/handle mismatch (e.g. a future F13 hot-swap reusing a stale
        # explainer): the class column k is keyed off explainer.drivers, so if that disagrees with
        # the handle the model actually ran, k would attribute SHAP to the wrong driver.
        if list(explainer.drivers) != list(handle.drivers):
            ledger.record(
                code=ErrorCode.SHAP_UNAVAILABLE, entity_id=circuit,
                detail=f"{REASON_UNKNOWN_CLASS}: explainer/handle driver order disagree",
            )
            return _shap_unavailable(
                predicted_class=predicted_class, latency_ms=_ms(),
                logged=ledger.rows > 0, flags=_unwritable_flags(ledger),
            )
        if predicted_class not in explainer.drivers:
            ledger.record(
                code=ErrorCode.SHAP_UNAVAILABLE, entity_id=circuit,
                detail=f"{REASON_UNKNOWN_CLASS}: {predicted_class!r} not in champion drivers",
            )
            return _shap_unavailable(
                predicted_class=predicted_class, latency_ms=_ms(),
                logged=ledger.rows > 0, flags=_unwritable_flags(ledger),
            )
        k = explainer.drivers.index(predicted_class)

        # Re-normalise to exactly the 28 features the model saw (read-only F04 reader).
        fv: dict[str, float | None] = {fid: feature_vector.get(fid) for fid in FEATURE_IDS}
        nres = apply_normalisation(
            fv, circuit=circuit, params=handle.normalisation_params, config=config, log_path=log
        )
        vec = nres.normalised_vector
        x = np.array(
            [[(vec.get(fid) if vec.get(fid) is not None else np.nan) for fid in FEATURE_IDS]],
            dtype=float,
        )

        # Guarded SHAP — any fault -> SHAP_UNAVAILABLE, prediction preserved (§6.2 FM1).
        try:
            explanation = explainer.explainer(x)
            values = np.asarray(explanation.values, dtype=float)  # (1, 28, n_classes)
            shap_row = values[0, :, k]
            # The base value is the tree's constant expected margin for class k, recorded once at
            # build time (== explanation.base_values[0, k] on every call) — use the stored value as
            # the single source of truth rather than re-reading it per call.
            base_value = float(explainer.base_values[predicted_class])
        except Exception as exc:  # noqa: BLE001 - any SHAP fault becomes a coded, logged result
            ledger.record(
                code=ErrorCode.SHAP_UNAVAILABLE, entity_id=circuit,
                detail=f"{REASON_SHAP_RAISED}: {type(exc).__name__}: {exc}",
            )
            return _shap_unavailable(
                predicted_class=predicted_class, latency_ms=_ms(),
                logged=ledger.rows > 0, flags=_unwritable_flags(ledger),
            )

        shap_values = {fid: float(shap_row[i]) for i, fid in enumerate(FEATURE_IDS)}
        order = np.argsort(np.abs(shap_row))[::-1]
        force = [
            Contribution(
                feature_id=FEATURE_IDS[i], name=FEATURE_NAMES[FEATURE_IDS[i]],
                shap=float(shap_row[i]),
                feature_value=(None if not np.isfinite(x[0, i]) else float(x[0, i])),
                direction="toward" if shap_row[i] > 0 else "away",
            )
            for i in order
        ]

        # Consistency (additivity) check (§6.2 FM2): base + Σshap ?= base-tree margin for class k.
        # NOTE: this verifies SHAP's own additivity residual; because all three operands are indexed
        # by the SAME k, it cannot detect a wrong class index. The protection against a mis-keyed
        # class is upstream — F06b's load_model pins champion.classes_ == [0..n-1] and labels.json
        # order, so drivers[k] is class k. Do not remove that invariant.
        flags: list[ErrorCode] = list(nres.flags)
        status = Status.OK
        error_code: ErrorCode | None = None
        margin_k = float(
            np.asarray(explainer.base_model.predict(x, output_margin=True), dtype=float)[0, k]
        )
        consistency_delta = float(abs(base_value + float(shap_row.sum()) - margin_k))
        consistency_ok = consistency_delta <= cfg.consistency_tolerance
        if not consistency_ok:
            flags.append(ErrorCode.SHAP_CONSISTENCY_WARN)
            status = Status.DEGRADED
            error_code = ErrorCode.SHAP_CONSISTENCY_WARN
            ledger.record(
                code=ErrorCode.SHAP_CONSISTENCY_WARN, entity_id=circuit,
                detail=f"{REASON_CONSISTENCY}: |base+Σshap-margin|={consistency_delta:.4g} "
                f"> tol {cfg.consistency_tolerance}",
            )

        radar = aggregate_domains(shap_values)
        style = describe_style(radar, config=config)

        latency_ms = _ms()
        if latency_ms > cfg.single_lap_warn_ms:
            flags.append(ErrorCode.PERF_WARN)
            ledger.record(
                code=ErrorCode.PERF_WARN, entity_id=circuit,
                detail=f"single-lap SHAP {latency_ms:.1f}ms > {cfg.single_lap_warn_ms}ms",
            )

        if ledger.failed:
            flags.append(ErrorCode.DATA_QUALITY_LOG_UNWRITABLE)

        return ExplanationResult(
            status=status,
            error_code=error_code,
            flags=list(dict.fromkeys(flags)),
            logged=(ledger.rows > 0) or nres.logged,
            shap_values=shap_values,
            base_value=base_value,
            predicted_class=predicted_class,
            force=force,
            radar=radar,
            dominant_domains=style.dominant_domains,
            nl_description=style.sentence,
            shap_available=True,
            consistency_ok=consistency_ok,
            consistency_delta=consistency_delta,
            latency_ms=latency_ms,
        )
    except Exception as exc:  # noqa: BLE001 - the never-raises backstop (§6.2 FM1)
        try:
            ledger.record(
                code=ErrorCode.SHAP_UNAVAILABLE, entity_id=circuit,
                detail=f"{REASON_SHAP_RAISED}: {type(exc).__name__}: {exc}",
            )
        except Exception:  # noqa: BLE001 - even the logger must not break the contract
            pass
        return _shap_unavailable(
            predicted_class=predicted_class, latency_ms=_ms(),
            logged=ledger.rows > 0, flags=_unwritable_flags(ledger),
        )


# ----------------------------------------------------------------------------------
# Global explainability (offline, once after training) — never raises
# ----------------------------------------------------------------------------------


def build_global_explainability(
    explainer: ShapExplainer,
    background: Any,  # pd.DataFrame with FEATURE_IDS columns + a driver column
    driver_labels: Sequence[str],
    *,
    config: AppConfig,
    artefact_dir: str | Path | None = None,
    log_path: str | Path | None = None,
    driver_col: str = "driver",
) -> GlobalExplainabilityResult:
    """Compute the global beeswarm artefact + per-driver global SHAP / radar (§6.1), ONCE.

    ``background`` MUST already be the **normalised training-fit** matrix (train circuits only —
    Belgium/Italy excluded; the caller's contract, mirrored by F06a's fit split). Never raises: a
    SHAP fault → ``DEGRADED`` + ``SHAP_UNAVAILABLE`` + empty payloads + logged, so the artefact
    build never crashes the training pipeline."""
    import pandas as pd  # local import: pandas only needed on this offline path

    log = Path(log_path) if log_path is not None else Path(config.paths.data_quality_log)
    art_dir = Path(artefact_dir) if artefact_dir is not None else Path(config.paths.artefacts_dir)
    ledger = _DQLedger(log_path=log)
    start = time.perf_counter()

    def _ms() -> float:
        return (time.perf_counter() - start) * 1000.0

    if not explainer.ready:
        ledger.record(
            code=ErrorCode.SHAP_UNAVAILABLE, entity_id="global",
            detail=f"{REASON_EXPLAINER_BUILD}: explainer not ready", entity_type="explainer",
        )
        return GlobalExplainabilityResult(
            status=Status.DEGRADED, error_code=ErrorCode.SHAP_UNAVAILABLE,
            flags=_unwritable_flags(ledger), logged=ledger.rows > 0, shap_available=False,
            latency_ms=_ms(),
        )
    try:
        cfg = _require_shap(config)
        frame = pd.DataFrame(background).reset_index(drop=True)
        x = np.asarray(frame[list(FEATURE_IDS)].to_numpy(), dtype=float)
        # The SHAP class column MUST be indexed by the model's OWN class order (explainer.drivers),
        # NOT the caller-supplied driver_labels — a re-ordered/rebuilt roster would otherwise map
        # each driver to another driver's SHAP column and silently corrupt every fingerprint.
        idx = {name: i for i, name in enumerate(explainer.drivers)}
        declared = set(driver_labels)  # the roster the caller intends to report

        explanation = explainer.explainer(x)
        values = np.asarray(explanation.values, dtype=float)  # (n, 28, n_classes)

        points: list[dict[str, Any]] = []
        per_driver_shap: dict[str, dict[str, float]] = {}
        per_driver_radar: dict[str, dict[str, float]] = {}
        driver_series = frame[driver_col].astype(str).to_numpy()
        for name in sorted(set(driver_series.tolist())):
            if name not in idx or name not in declared:
                continue  # a background driver absent from the champion label space is skipped
            cls = idx[name]
            rows = np.flatnonzero(driver_series == name)
            # Global importance = mean |SHAP| for THIS driver's own class over this driver's laps.
            mean_abs = np.abs(values[rows][:, :, cls]).mean(axis=0)
            shap_map = {fid: float(mean_abs[i]) for i, fid in enumerate(FEATURE_IDS)}
            per_driver_shap[name] = shap_map
            per_driver_radar[name] = aggregate_domains(shap_map)
            # Beeswarm points: one dot per lap per feature (SHAP + the feature value carried).
            for r in rows.tolist():
                for i, fid in enumerate(FEATURE_IDS):
                    fv = x[r, i]
                    points.append({
                        "driver": name, "feature_id": fid,
                        "shap": round(float(values[r, i, cls]), 8),
                        "feature_value": (None if not np.isfinite(fv) else round(float(fv), 8)),
                    })

        # Deterministic artefact — sorted keys, stable point order, no timestamp (byte-identical
        # on re-run for the same background).
        points.sort(key=lambda p: (p["driver"], p["feature_id"], p["shap"], -1.0
                                   if p["feature_value"] is None else p["feature_value"]))
        payload = {
            "schema_version": "F07-beeswarm-1",
            "features": list(FEATURE_IDS),
            "drivers": sorted(per_driver_shap),
            "points": points,
        }
        art_dir.mkdir(parents=True, exist_ok=True)
        artefact_path = art_dir / cfg.beeswarm_filename
        artefact_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

        latency_ms = _ms()
        flags: list[ErrorCode] = []
        if latency_ms > cfg.batch_warn_ms:
            flags.append(ErrorCode.PERF_WARN)
            ledger.record(
                code=ErrorCode.PERF_WARN, entity_id="global", entity_type="explainer",
                detail=f"global SHAP {latency_ms:.1f}ms > {cfg.batch_warn_ms}ms",
            )
        # Snapshot the unwritable flag AFTER the PERF_WARN write, so a sink failure on that write is
        # still surfaced (mirrors explain_prediction's post-write `if ledger.failed` check).
        flags.extend(_unwritable_flags(ledger))
        return GlobalExplainabilityResult(
            status=Status.OK,
            flags=list(dict.fromkeys(flags)),
            logged=ledger.rows > 0,
            beeswarm_artefact_path=str(artefact_path),
            per_driver_shap=per_driver_shap,
            per_driver_radar=per_driver_radar,
            n_background=int(len(frame)),
            shap_available=True,
            latency_ms=latency_ms,
        )
    except Exception as exc:  # noqa: BLE001 - offline artefact build never crashes training
        try:
            ledger.record(
                code=ErrorCode.SHAP_UNAVAILABLE, entity_id="global", entity_type="explainer",
                detail=f"{REASON_GLOBAL_SHAP}: {type(exc).__name__}: {exc}",
            )
        except Exception:  # noqa: BLE001
            pass
        return GlobalExplainabilityResult(
            status=Status.DEGRADED, error_code=ErrorCode.SHAP_UNAVAILABLE,
            flags=_unwritable_flags(ledger), logged=ledger.rows > 0, shap_available=False,
            latency_ms=_ms(),
        )


# ----------------------------------------------------------------------------------
# Multi-season stability (Jensen-Shannon drift, §6.1) — never raises
# ----------------------------------------------------------------------------------


def _pair_js(
    a: Mapping[str, Sequence[float]], b: Mapping[str, Sequence[float]], *, bins: int
) -> float:
    """Mean per-feature Jensen-Shannon **divergence** (base-2, ∈ [0,1]) between two seasons' SHAP
    distributions. A feature constant across both seasons contributes 0 (no drift). Returns
    ``nan`` when NO feature is comparable (e.g. a season's SHAP is entirely NaN/empty) — so the
    caller reports missing data rather than silently claiming perfect stability."""
    shared = sorted(set(a) & set(b))
    if not shared:
        return float("nan")
    divs: list[float] = []
    for fid in shared:
        av = np.asarray(a[fid], dtype=float)
        bv = np.asarray(b[fid], dtype=float)
        av = av[np.isfinite(av)]
        bv = bv[np.isfinite(bv)]
        if av.size == 0 or bv.size == 0:
            continue
        lo = float(min(av.min(), bv.min()))
        hi = float(max(av.max(), bv.max()))
        if hi <= lo:  # identical constant -> no divergence
            divs.append(0.0)
            continue
        edges = np.linspace(lo, hi, bins + 1)
        pa, _ = np.histogram(av, bins=edges)
        pb, _ = np.histogram(bv, bins=edges)
        if pa.sum() == 0 or pb.sum() == 0:
            continue
        # scipy returns the JS *distance* (sqrt of divergence); square it -> divergence ∈ [0,1].
        dist = float(jensenshannon(pa, pb, base=2))
        divs.append(0.0 if not np.isfinite(dist) else dist * dist)
    return float(np.mean(divs)) if divs else float("nan")


def season_stability(
    per_season_shap: Mapping[str, Mapping[str, Sequence[float]]],
    *,
    config: AppConfig,
    log_path: str | Path | None = None,
    entity_id: str = "driver",
) -> SeasonStabilityResult:
    """Score one driver's year-on-year fingerprint drift (§6.1). **Never raises.**

    ``per_season_shap`` maps ``season -> {feature_id -> that feature's SHAP values across that
    season's laps}``. Computes pairwise Jensen-Shannon divergence between seasons; the score
    = mean over pairs (lower = more stable). Fewer than ``shap.stability_min_seasons`` seasons →
    ``partial=True``; fewer than two → no pair exists → ``stability_score=None`` (never a fabricated
    0.0)."""
    log = Path(log_path) if log_path is not None else Path(config.paths.data_quality_log)
    ledger = _DQLedger(log_path=log)
    try:
        cfg = _require_shap(config)
        seasons = sorted(per_season_shap)
        n = len(seasons)
        partial = n < cfg.stability_min_seasons
        if n < 2:
            ledger.record(
                code=ErrorCode.SHAP_UNAVAILABLE, entity_id=entity_id, entity_type="fingerprint",
                detail=f"{REASON_STABILITY}: {n} season(s) < 2; no pair to compare",
            )
            return SeasonStabilityResult(
                status=Status.DEGRADED, flags=_unwritable_flags(ledger), logged=ledger.rows > 0,
                stability_score=None, pairwise_js={}, seasons_available=n, partial=True,
            )
        # Keep only pairs with comparable data; a NaN pair (a season's SHAP all-NaN/empty) is
        # missing data, NOT zero drift — excluding it stops us silently reporting perfect stability.
        pairwise: dict[str, float] = {}
        n_uncomparable = 0
        for i in range(n):
            for j in range(i + 1, n):
                js = _pair_js(
                    per_season_shap[seasons[i]], per_season_shap[seasons[j]],
                    bins=cfg.stability_js_bins,
                )
                if np.isfinite(js):
                    pairwise[f"{seasons[i]}|{seasons[j]}"] = float(js)
                else:
                    n_uncomparable += 1
        # No comparable pair -> score is genuinely unknown; report None (never a fabricated 0.0).
        score = float(np.mean(list(pairwise.values()))) if pairwise else None
        if partial or n_uncomparable:
            detail = f"{REASON_STABILITY}: {n} season(s)"
            if partial:
                detail += f" < min {cfg.stability_min_seasons}"
            if n_uncomparable:
                detail += f"; {n_uncomparable} pair(s) had no comparable SHAP data"
            ledger.record(
                code=ErrorCode.SHAP_UNAVAILABLE, entity_id=entity_id, entity_type="fingerprint",
                detail=detail,
            )
        return SeasonStabilityResult(
            status=Status.DEGRADED if (partial or score is None) else Status.OK,
            flags=_unwritable_flags(ledger),
            logged=ledger.rows > 0,
            stability_score=score,
            pairwise_js=pairwise,
            seasons_available=n,
            partial=partial,
        )
    except Exception as exc:  # noqa: BLE001 - never raises
        try:
            ledger.record(
                code=ErrorCode.SHAP_UNAVAILABLE, entity_id=entity_id, entity_type="fingerprint",
                detail=f"{REASON_STABILITY}: {type(exc).__name__}: {exc}",
            )
        except Exception:  # noqa: BLE001
            pass
        return SeasonStabilityResult(
            status=Status.DEGRADED, flags=_unwritable_flags(ledger), logged=ledger.rows > 0,
            stability_score=None, pairwise_js={}, seasons_available=0, partial=True,
        )
