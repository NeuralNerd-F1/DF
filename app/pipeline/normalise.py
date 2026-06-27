"""F04 — circuit-aware normalisation: APEX's train-only leakage chokepoint.

Six of the 28 lap features are dominated by the *circuit* rather than the driver — the
speed-dependent F09/F11/F13/F14 and the lateral-G F19/F20 (master spec §4.3). This module
**Z-score normalises those six per circuit**, fitting the mean/std on the **training split
only**, saving them once to ``artefacts/normalisation.json``, and re-loading them — never
recomputing — for val/test/upload. Every downstream model (F06, F08, F10, F11) consumes this
same artefact.

The architecture is the guarantee against leakage (non-negotiable #2): **one writer, two
readers, no refit path.**

* :func:`fit_normaliser` is the ONLY function that writes the artefact or calls ``groupby`` /
  derives any mean/std. It excludes the held-out circuits (Belgium, Italy) and the non-train
  split **before any aggregation**, then a fail-loud :func:`_assert_no_leakage` raises
  :class:`NormalisationError` if any held-out circuit somehow survived — **no artefact is
  written on a poisoned run.** The artefact is serialised deterministically (sorted keys,
  Python floats, no timestamp, embedded ``content_hash``) so two runs on the same inputs
  produce a byte-identical file (reproducibility, non-negotiable #3).
* :func:`load_normalisation_params` only reads; a missing / corrupt / hash-mismatched artefact
  **raises** :class:`NormalisationError` rather than silently skipping normalisation.
* :func:`apply_normalisation` is a pure transform that **never raises and never fits**: an
  unknown/skipped circuit falls back to the global params, a NULL input feature is left as-is,
  and both paths return a typed :class:`NormalisationResult` with
  ``error_code=NORMALISATION_FALLBACK`` plus a logged ``data_quality_log`` row (non-negotiable
  #1). The other 22 feature keys pass through untouched.

Every constant (the six feature IDs, the grouping key, ``ddof``, ``min_std``, the artefact
filename, the per-circuit fit floor) is read from ``config.yaml`` via ``config.normalise`` /
``config.thresholds`` — there are no magic numbers here.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import cast

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.contracts.config import AppConfig
from app.contracts.enums import Status
from app.contracts.errors import ErrorCode
from app.contracts.logging import DataQualityLogError, log_data_quality
from app.contracts.results import BaseResult

# Structural constants (not tunable thresholds — those live in config). Bumped only when the
# on-disk artefact shape changes; load rejects any other value as a schema mismatch.
SCHEMA_VERSION = "1.0"

# Sentinel entity id for the global (all-circuit) params in fit-time data_quality_log rows.
_GLOBAL_ENTITY = "GLOBAL"

# Centre-only fallbacks: when a stat is degenerate it cannot scale/centre the feature, so we
# substitute a no-op. ``std=1.0`` (global std < min_std / NaN) and ``mean=0.0`` (global mean
# non-finite — feature all-NULL across training) keep the artefact finite + valid JSON instead of
# storing NaN. Architectural constants (spec §4.2), not tunable thresholds — named so they are
# self-documenting and excluded from the config-literal scan.
_CENTRE_ONLY_STD = 1.0
_CENTRE_ONLY_MEAN = 0.0

# data_quality_log detail tags (mirrors features.py's REASON_* convention).
REASON_ZERO_VARIANCE = "ZERO_VARIANCE"
REASON_NON_FINITE_MEAN = "NON_FINITE_MEAN"  # fit: feature all-NULL at a circuit -> mean is NaN
REASON_UNKNOWN_CIRCUIT = "UNKNOWN_CIRCUIT"
REASON_NULL_INPUT = "NULL_INPUT"
REASON_NON_FINITE_INPUT = "NON_FINITE_INPUT"  # apply: a feature value is NaN/Inf, not a real float
REASON_SUB_MIN_CIRCUIT = "SUB_MIN_CIRCUIT"
REASON_UNEXPECTED = "UNEXPECTED"


class NormalisationError(Exception):
    """Raised on F04's fail-loud paths — leakage detected at fit, or a missing/corrupt artefact
    at load.

    Carries ``ErrorCode.NORMALISATION_FALLBACK`` so callers surface a typed failure rather than
    a bare exception (non-negotiable #1). Note: ``apply_normalisation`` never raises this — an
    apply-time fault is always a coded :class:`NormalisationResult` + a logged row, never an
    exception that could be swallowed into an un-normalised vector.
    """

    error_code = ErrorCode.NORMALISATION_FALLBACK

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


# ----------------------------------------------------------------------------------
# Param / result models (live here, mirroring FeatureResult living in features.py)
# ----------------------------------------------------------------------------------


class FeatureStats(BaseModel):
    """One feature's train-fit centre + scale: ``z = (x - mean) / std``."""

    model_config = ConfigDict(extra="forbid")

    mean: float
    std: float


class NormalisationParams(BaseModel):
    """The on-disk ``normalisation.json`` artefact: per-circuit + global ``{mean, std}`` for the
    six normalised features, the train/excluded circuit lists, and a ``content_hash`` over the
    canonical body. Written only by :func:`fit_normaliser`; read only by
    :func:`load_normalisation_params`.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    normalised_features: list[str]
    grouping_key: str
    ddof: int
    train_circuits: list[str]
    excluded_circuits: list[str]
    global_: dict[str, FeatureStats]
    per_circuit: dict[str, dict[str, FeatureStats]]
    content_hash: str


class NormalisationFitResult(BaseResult):
    """Summary of a :func:`fit_normaliser` run. ``status`` is ``OK`` when every train circuit was
    fitted cleanly, else ``DEGRADED`` (some circuit skipped < floor, or a zero-variance fallback
    taken), in which case ``error_code`` carries ``NORMALISATION_FALLBACK`` to name the condition
    (mirroring the per-row data_quality_log code and F03b's ``GeometryFeatureResult``). This is
    *not* a failure — the artefact is valid and written; the only hard-failure paths raise
    :class:`NormalisationError` before this object is built.
    """

    n_train_laps: int
    n_circuits: int
    normalised_features: list[str] = Field(default_factory=list)
    excluded_circuits: list[str] = Field(default_factory=list)
    skipped_circuits: list[str] = Field(default_factory=list)
    artefact_path: str
    content_hash: str


class NormalisationResult(BaseResult):
    """Per-vector output of :func:`apply_normalisation`. ``normalised_vector`` is the full input
    vector with only the six normalised features transformed (the other keys bit-identical).
    ``status`` is ``OK`` for a clean per-circuit transform, else ``DEGRADED`` (global fallback or
    a NULL passthrough), with ``error_code=NORMALISATION_FALLBACK``.
    """

    circuit: str
    normalised_vector: dict[str, float | None] = Field(default_factory=dict)
    normalised_features: list[str] = Field(default_factory=list)
    used_global: bool = False
    fallback_features: list[str] = Field(default_factory=list)


# ----------------------------------------------------------------------------------
# Data-quality ledger — parameterised on entity_type (features.py's hardcodes "lap")
# ----------------------------------------------------------------------------------


@dataclass
class _DQLedger:
    """Writes one ``data_quality_log`` row per fallback / skip / zero-variance event.

    Unlike F03a's ledger (which hardcodes ``entity_type="lap"``), F04 normalises *per circuit*,
    so this ledger is parameterised on ``entity_type`` — ``"circuit"`` for both fit-time and
    apply-time rows.
    """

    log_path: Path
    entity_type: str
    rows: int = 0
    failed: int = 0

    def record(self, *, code: ErrorCode, entity_id: str, detail: str, fallback_used: bool) -> None:
        """Append one row; never propagate a sink failure.

        If the ``data_quality_log`` sink is unwritable, ``log_data_quality`` raises
        ``DataQualityLogError`` — but at this boundary that must NOT crash the caller:
        ``apply_normalisation`` is contractually never-raises, and ``fit_normaliser`` must still
        produce its artefact. We count the loss in ``failed`` so the caller can surface
        ``DATA_QUALITY_LOG_UNWRITABLE`` on the result's ``flags`` (visible, not silent) rather
        than lose the audit signal entirely (non-negotiable #1)."""
        try:
            log_data_quality(
                error_code=code,
                entity_type=self.entity_type,
                entity_id=entity_id,
                error_detail=detail,
                fallback_used=fallback_used,
                log_path=self.log_path,
            )
            self.rows += 1
        except DataQualityLogError:
            self.failed += 1


# ----------------------------------------------------------------------------------
# Leakage guard + deterministic serialisation helpers
# ----------------------------------------------------------------------------------


def _assert_no_leakage(train: pd.DataFrame, config: AppConfig) -> None:
    """Fail loud if any held-out circuit survives into the training set (non-negotiable #2).

    Called inside :func:`fit_normaliser` *after* the exclusion filter as a tripwire (it would
    only fire if the filter logic were broken), and directly testable on an unfiltered frame.
    Raises :class:`NormalisationError` — no artefact is written on a poisoned run. The
    ``config.normalise is None`` precondition is checked here too, so a direct caller gets the
    typed failure rather than a bare ``AttributeError`` (non-negotiable #1).
    """
    if config.normalise is None:
        raise NormalisationError("config.normalise block absent; cannot check for leakage")
    held_out = set(config.held_out_circuits)
    present = set(train[config.normalise.grouping_key].unique()) if len(train) else set()
    leaked = sorted(held_out & present)
    if leaked:
        raise NormalisationError(
            f"held-out circuits present in training input after filter: {leaked} "
            "(would leak into the fit — aborting; no artefact written)"
        )


def _canonical_body(params: NormalisationParams) -> str:
    """Deterministic JSON of the params *without* the hash field — the bytes the hash covers.

    Sorted keys + ``json``'s shortest-round-trip float repr make this stable across runs and
    platforms for identical float values (reproducibility, non-negotiable #3).
    """
    body = params.model_dump(exclude={"content_hash"})
    # allow_nan=False: a non-finite stat must never be written as the bare `NaN`/`Infinity` token
    # (invalid JSON for strict readers). With the fit's centre-only fallbacks this never triggers;
    # it is a fail-loud guard against any future regression that lets a NaN reach serialisation.
    return json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)


# ----------------------------------------------------------------------------------
# The ONE writer
# ----------------------------------------------------------------------------------


def fit_normaliser(
    laps: pd.DataFrame,
    *,
    config: AppConfig,
    artefact_path: str | Path | None = None,
    log_path: str | Path | None = None,
) -> NormalisationFitResult:
    """Fit per-circuit + global Z-score params on the TRAINING split only and write the artefact.

    The only function that writes the artefact or derives any mean/std. Steps:
    exclude held-out + non-train rows **first**, fail-loud leakage assert, per-circuit
    ``groupby`` aggregation (circuits below ``thresholds.min_clean_laps`` are skipped → global
    at apply), global params, zero-variance fallback (``std < min_std`` → global std, or centre
    only), then a deterministic write with an embedded ``content_hash``. Raises
    :class:`NormalisationError` on a poisoned / empty / malformed input (never writes a bad
    artefact).
    """
    ncfg = config.normalise
    if ncfg is None:
        raise NormalisationError("config.normalise block absent; cannot fit normaliser")

    features = list(ncfg.features)
    grouping_key = ncfg.grouping_key
    min_std = ncfg.min_std
    ddof = ncfg.ddof
    min_laps = config.thresholds.min_clean_laps
    held_out = sorted(config.held_out_circuits)

    log = Path(log_path) if log_path is not None else Path(config.paths.data_quality_log)
    ledger = _DQLedger(log_path=log, entity_type="circuit")

    required = {"split", grouping_key, *features}
    missing = sorted(required - set(laps.columns))
    if missing:
        raise NormalisationError(f"fit input missing required columns: {missing}")

    # 1. Exclude held-out + non-train rows BEFORE any reduction (the leakage barrier).
    train_mask = (laps["split"] == "train") & (~laps[grouping_key].isin(config.held_out_circuits))
    train = cast("pd.DataFrame", laps.loc[train_mask])
    # 2. Fail-loud tripwire — raises before any artefact is written.
    _assert_no_leakage(train, config)

    n_train_laps = int(len(train))
    if n_train_laps == 0:
        raise NormalisationError("no training rows after excluding held-out / non-train split")

    # 2b. Reject null circuit keys: pandas groupby silently DROPS NaN-key rows from the per-circuit
    # fit, yet they would still skew the global params — two stat sets fit on inconsistent
    # populations, with no audit row. Fail loud (consistent with the other fit-time guards) rather
    # than fit on a row that has no per-circuit home (non-negotiable #1/#2).
    if bool(train[grouping_key].isna().any()):
        n_null = int(train[grouping_key].isna().sum())
        raise NormalisationError(
            f"{n_null} training row(s) have a null '{grouping_key}' key; refusing to fit "
            "(they would skew global params while being dropped from every per-circuit fit)"
        )

    # 3. Per-circuit aggregation (vectorised; no per-row loops).
    grouped = train.groupby(grouping_key)
    counts = grouped.size()
    means = grouped[features].mean()
    stds = grouped[features].std(ddof=ddof)

    # 4. Global params (the fallback for unknown / skipped circuits).
    global_means = cast("pd.Series", train[features].mean())
    global_stds = cast("pd.Series", train[features].std(ddof=ddof))
    global_stats: dict[str, FeatureStats] = {}
    for f in features:
        gmean = float(global_means[f])
        gstd = float(global_stds[f])
        # A feature that is all-NULL across the whole train split yields a NaN mean; storing it
        # verbatim would ship NaN into every downstream model. Centre to 0.0 and log instead.
        if not math.isfinite(gmean):
            ledger.record(
                code=ErrorCode.NORMALISATION_FALLBACK,
                entity_id=_GLOBAL_ENTITY,
                detail=f"{f} {REASON_NON_FINITE_MEAN} (global mean={gmean}); centred (mean=0.0)",
                fallback_used=True,
            )
            gmean = _CENTRE_ONLY_MEAN
        # `not (gstd >= min_std)` is True for both gstd < min_std AND gstd NaN.
        if not (gstd >= min_std):
            ledger.record(
                code=ErrorCode.NORMALISATION_FALLBACK,
                entity_id=_GLOBAL_ENTITY,
                detail=f"{f} {REASON_ZERO_VARIANCE} (global std={gstd}); centred (std=1.0)",
                fallback_used=True,
            )
            gstd = _CENTRE_ONLY_STD
        global_stats[f] = FeatureStats(mean=gmean, std=gstd)

    # 5. Per-circuit params + skip / zero-variance handling.
    per_circuit: dict[str, dict[str, FeatureStats]] = {}
    skipped: list[str] = []
    train_circuits: list[str] = []
    for circuit in sorted(counts.index):
        name = str(circuit)
        if int(counts.loc[circuit]) < min_laps:
            skipped.append(name)
            ledger.record(
                code=ErrorCode.NORMALISATION_FALLBACK,
                entity_id=name,
                detail=(
                    f"{REASON_SUB_MIN_CIRCUIT}: {int(counts.loc[circuit])} laps < "
                    f"min_clean_laps {min_laps}; skipped (global params at apply)"
                ),
                fallback_used=False,
            )
            continue
        stats: dict[str, FeatureStats] = {}
        for f in features:
            mean = float(means.loc[circuit, f])
            std = float(stds.loc[circuit, f])
            if not math.isfinite(mean):
                mean = global_stats[f].mean  # already finite (centred to 0.0 if global all-NULL)
                ledger.record(
                    code=ErrorCode.NORMALISATION_FALLBACK,
                    entity_id=name,
                    detail=f"{f} {REASON_NON_FINITE_MEAN}; using global mean {mean}",
                    fallback_used=True,
                )
            if not (std >= min_std):
                std = global_stats[f].std  # already >= min_std or centred to 1.0
                ledger.record(
                    code=ErrorCode.NORMALISATION_FALLBACK,
                    entity_id=name,
                    detail=f"{f} {REASON_ZERO_VARIANCE}; using global std {std}",
                    fallback_used=True,
                )
            stats[f] = FeatureStats(mean=mean, std=std)
        per_circuit[name] = stats
        train_circuits.append(name)

    # 6. Build params, hash the canonical body, write deterministically.
    params = NormalisationParams(
        schema_version=SCHEMA_VERSION,
        normalised_features=features,
        grouping_key=grouping_key,
        ddof=ddof,
        train_circuits=train_circuits,
        excluded_circuits=held_out,
        global_=global_stats,
        per_circuit=per_circuit,
        content_hash="",
    )
    params.content_hash = sha256(_canonical_body(params).encode("utf-8")).hexdigest()

    path = (
        Path(artefact_path)
        if artefact_path is not None
        else Path(config.paths.artefacts_dir) / ncfg.artefact_filename
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(params.model_dump(), sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8")

    degraded = bool(skipped) or ledger.rows > 0 or ledger.failed > 0
    flags = [ErrorCode.DATA_QUALITY_LOG_UNWRITABLE] if ledger.failed else []
    return NormalisationFitResult(
        n_train_laps=n_train_laps,
        n_circuits=len(per_circuit),
        normalised_features=features,
        excluded_circuits=held_out,
        skipped_circuits=skipped,
        artefact_path=str(path),
        content_hash=params.content_hash,
        flags=flags,
        status=Status.DEGRADED if degraded else Status.OK,
        # A skipped circuit / zero-variance fallback names a specific degradation condition, so
        # the result carries the code (consistent with F03b's GeometryFeatureResult on a DEGRADED
        # result, and the spec §6 table) — not a failure: the artefact is valid and written.
        error_code=ErrorCode.NORMALISATION_FALLBACK if degraded else None,
        logged=ledger.rows > 0,
    )


# ----------------------------------------------------------------------------------
# Reader 1 — fail-loud load
# ----------------------------------------------------------------------------------


def load_normalisation_params(artefact_path: str | Path) -> NormalisationParams:
    """Read and verify the saved artefact. Read-only — performs no fitting and no aggregation.

    A missing file, invalid JSON, a non-mapping body, a schema mismatch, or a ``content_hash``
    that does not match the body → raise :class:`NormalisationError`. Never returns empty params
    / silently skips normalisation (non-negotiable #1) — the caller must get real params or a
    typed failure.
    """
    p = Path(artefact_path)
    if not p.is_file():
        raise NormalisationError(f"normalisation artefact not found: {p}")
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NormalisationError(f"normalisation artefact unreadable/corrupt: {exc}") from exc
    if not isinstance(raw, dict):
        raise NormalisationError("normalisation artefact must be a JSON object")
    try:
        params = NormalisationParams(**raw)
    except ValidationError as exc:
        raise NormalisationError(f"normalisation artefact schema mismatch: {exc}") from exc
    if params.schema_version != SCHEMA_VERSION:
        raise NormalisationError(
            f"normalisation artefact schema_version {params.schema_version!r} != "
            f"expected {SCHEMA_VERSION!r}"
        )
    recomputed = sha256(_canonical_body(params).encode("utf-8")).hexdigest()
    if params.content_hash != recomputed:
        raise NormalisationError("normalisation artefact content_hash mismatch (corrupt/tampered)")
    return params


# ----------------------------------------------------------------------------------
# Reader 2 — pure transform (never raises, never fits)
# ----------------------------------------------------------------------------------


def apply_normalisation(
    feature_vector: dict[str, float | None],
    *,
    circuit: str,
    params: NormalisationParams,
    config: AppConfig,
    log_path: str | Path | None = None,
) -> NormalisationResult:
    """Z-score the six normalised features at ``circuit``; pass the other keys through unchanged.

    Read-only — performs no fitting and no aggregation; the centre/scale come entirely from the
    loaded ``params``. An unknown / skipped circuit → global params (``used_global``,
    ``NORMALISATION_FALLBACK``, logged, ``DEGRADED``). A NULL among the six → left as-is, added
    to ``fallback_features``, logged. The full input vector is always returned. **Never raises.**
    ``config`` supplies the default ``data_quality_log`` path (overridable via ``log_path``),
    consistent with every other pipeline entry point.
    """
    log = Path(log_path) if log_path is not None else Path(config.paths.data_quality_log)
    ledger = _DQLedger(log_path=log, entity_type="circuit")

    out: dict[str, float | None] = dict(feature_vector)
    features = list(params.normalised_features)

    circuit_stats = params.per_circuit.get(circuit)
    used_global = circuit_stats is None
    if used_global:
        circuit_stats = params.global_
        ledger.record(
            code=ErrorCode.NORMALISATION_FALLBACK,
            entity_id=circuit,
            detail=f"{REASON_UNKNOWN_CIRCUIT}: '{circuit}' not in training set; global params",
            fallback_used=True,
        )

    fallback_features: list[str] = []
    for f in features:
        x = out.get(f)
        if x is None:
            fallback_features.append(f)
            ledger.record(
                code=ErrorCode.NORMALISATION_FALLBACK,
                entity_id=circuit,
                detail=f"{f} {REASON_NULL_INPUT}; left un-normalised (passthrough)",
                fallback_used=True,
            )
            continue
        try:
            xf = float(x)
        except (TypeError, ValueError) as exc:
            out[f] = None  # non-numeric value -> NULL (typed result stays valid)
            fallback_features.append(f)
            ledger.record(
                code=ErrorCode.NORMALISATION_FALLBACK,
                entity_id=circuit,
                detail=f"{f} {REASON_UNEXPECTED}: {exc!r}; left un-normalised",
                fallback_used=True,
            )
            continue
        if not math.isfinite(xf):
            # A NaN/Inf input (e.g. a missing channel surfaced as np.nan) is NOT a real value:
            # normalising it would ship NaN downstream with status OK. Treat as missing -> NULL.
            out[f] = None
            fallback_features.append(f)
            ledger.record(
                code=ErrorCode.NORMALISATION_FALLBACK,
                entity_id=circuit,
                detail=f"{f} {REASON_NON_FINITE_INPUT} ({xf}); left un-normalised",
                fallback_used=True,
            )
            continue
        try:
            st = circuit_stats[f]  # type: ignore[index]
            out[f] = (xf - st.mean) / st.std
        except Exception as exc:  # noqa: BLE001 — last resort: NULL it, never propagate (#1)
            out[f] = None  # un-normalisable value -> NULL (typed result stays valid)
            fallback_features.append(f)
            ledger.record(
                code=ErrorCode.NORMALISATION_FALLBACK,
                entity_id=circuit,
                detail=f"{f} {REASON_UNEXPECTED}: {exc!r}; left un-normalised",
                fallback_used=True,
            )

    degraded = used_global or bool(fallback_features)
    # A DQ-sink failure (ledger.failed) is surfaced on flags, not raised — apply never raises.
    flags = [ErrorCode.DATA_QUALITY_LOG_UNWRITABLE] if ledger.failed else []
    return NormalisationResult(
        circuit=circuit,
        normalised_vector=out,
        normalised_features=features,
        used_global=used_global,
        fallback_features=fallback_features,
        flags=flags,
        status=Status.DEGRADED if degraded else Status.OK,
        error_code=ErrorCode.NORMALISATION_FALLBACK if degraded else None,
        logged=ledger.rows > 0,
    )
