"""F01 — data ingestion (FastF1 / OpenF1 / Ergast) with caching + circuit breakers.

Pulls race telemetry and context for the driver-circuit-season grid and makes it
available as a local Parquet cache the rest of the pipeline reads. The whole module
exists to honour the constitution's first non-negotiable — *no component fails
silently*: every external call goes through :func:`call_with_breaker`, and every
failure or fallback returns a typed :class:`IngestResult` carrying an explicit error
code AND writes one row to the ``data_quality_log``. Nothing is ever invented or
silently substituted.

External I/O is injected as callables (``fastf1_fetch`` / ``openf1_fetch`` /
``ergast_fetch``), so the orchestration is unit-tested with no network and no
``fastf1`` import. The real adapters (added alongside this module) own the HTTP
clients and apply each source's ``timeout_s``; they translate their faults into the
two typed errors below.

Cached column contract — the F01 -> F02 handoff
-----------------------------------------------
A persisted session Parquet (``cache/<season>/<circuit>/<driver>.parquet``) ALWAYS
holds one row per telemetry sample with EVERY one of these columns present (a column
is never dropped — an unavailable source fills it with the documented null marker):

* FastF1 telemetry (~50 Hz): ``Speed``, ``Throttle``, ``Brake``, ``RPM``, ``Gear``,
  ``DRS``, ``X``, ``Y`` and the ``lap`` number the sample belongs to.
* FastF1 ``Laps.TrackStatus`` (per lap, left-joined on ``lap``): ``track_status`` — the
  RAW status code stored verbatim (no green-flag interpretation here; that is F02's
  job). Null for affected laps when track status is unavailable (logged, never fatal).
* OpenF1 enrichment (per lap, left-joined on ``lap``): ``compound``, ``pit_in``,
  ``pit_out``.
* Ergast / derived lap context (per lap, left-joined on ``lap``): ``lap_time``,
  ``stint``, ``tyre_age``.

Provenance (which source each column came from, and the fallback flags) is NOT
recoverable from the frame's shape, so it is persisted in a sidecar JSON next to the
Parquet (see :func:`cache_meta_path`). A cache read restores that provenance verbatim
— a session ingested via a fallback is replayed as ``DEGRADED`` with the same flags,
never laundered back to ``OK``.

Batch ingestion: build the breakers ONCE with :func:`build_breakers` and pass the same
dict to every :func:`ingest_session` call, so an open breaker short-circuits across the
whole batch. Calling ``ingest_session`` without a ``breakers`` argument gives each call
a fresh breaker (correct only for one-off ingests).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TypeVar

import pandas as pd
import pyarrow
from pydantic import BaseModel, ConfigDict

from app.contracts.config import AppConfig, BreakerConfig
from app.contracts.enums import Status
from app.contracts.errors import ErrorCode
from app.contracts.logging import log_data_quality
from app.contracts.results import BaseResult

T = TypeVar("T")

SECONDS_PER_DAY = 86_400.0

# Errors a corrupt / unwritable / incomplete Parquet cache can raise. Caught so a bad
# cache is a typed, logged outcome — never an exception escaping ingest_session (#1).
_CACHE_IO_ERRORS = (OSError, ValueError, KeyError, pyarrow.ArrowInvalid)

# The per-lap context columns the F01 -> F02 contract guarantees, with the null marker
# used when a source could not supply them (so the column is present but explicitly empty).
_CONTEXT_DEFAULTS: dict[str, object] = {
    "track_status": pd.NA,
    "compound": "UNKNOWN",
    "pit_in": False,
    "pit_out": False,
    "lap_time": pd.NA,
    "stint": pd.NA,
    "tyre_age": pd.NA,
}

# ----------------------------------------------------------------------------------
# Typed inputs / outputs
# ----------------------------------------------------------------------------------


class CompoundSource(StrEnum):
    """Where a session's tyre compound was sourced from (IngestResult field)."""

    OPENF1 = "OPENF1"
    ERGAST = "ERGAST"
    UNKNOWN = "UNKNOWN"


class TyreAgeSource(StrEnum):
    """Where a session's tyre age was sourced from (IngestResult field)."""

    ERGAST = "ERGAST"
    DERIVED = "DERIVED"


class SessionRequest(BaseModel):
    """A single session to ingest. ``session_type='R'`` (race) by default."""

    model_config = ConfigDict(extra="forbid")

    season: int
    circuit: str
    driver: str
    session_type: str = "R"

    @property
    def entity_id(self) -> str:
        """Stable id used for the ``data_quality_log`` entity_id column."""
        return f"{self.season}/{self.circuit}/{self.driver}/{self.session_type}"


class IngestResult(BaseResult):
    """Typed outcome of ingesting one session.

    Inherits ``status`` / ``error_code`` / ``flags`` / ``logged`` from
    :class:`BaseResult`. ``status`` is ``OK`` on a clean fetch (or fresh clean cache
    hit), ``DEGRADED`` when a documented fallback fired (the ``flags`` say which), and
    ``FAILED`` when no usable telemetry could be produced (``error_code`` says why).
    """

    source_used: str = ""
    rows: int = 0
    compound_source: CompoundSource = CompoundSource.UNKNOWN
    tyre_age_source: TyreAgeSource = TyreAgeSource.ERGAST
    cache_path: Path | None = None


# ----------------------------------------------------------------------------------
# Source-error contract — adapters raise exactly one of these
# ----------------------------------------------------------------------------------


class IngestSourceError(Exception):
    """A transport / availability fault (timeout, connection error, 5xx).

    These are *retried* inside one :func:`call_with_breaker` invocation and, once a
    call is fully exhausted, counted as one failure toward the breaker.
    """


class SessionUnavailableError(Exception):
    """The requested session genuinely has no data (e.g. driver did not race).

    A data-existence condition, NOT a transport fault: never retried, never counted
    by the breaker. Surfaces as ``DATA_001`` (skip, never substitute).
    """


class SourceUnavailableError(Exception):
    """Raised by :func:`call_with_breaker` when a source yields nothing.

    ``short_circuited`` is ``True`` when the breaker was already open (no call was
    made — the <500 ms fallback path), ``False`` when the call's retries were fully
    exhausted.
    """

    def __init__(self, source: str, *, short_circuited: bool, cause: Exception | None = None):
        self.source = source
        self.short_circuited = short_circuited
        self.cause = cause
        state = "breaker open" if short_circuited else "retries exhausted"
        super().__init__(f"{source} unavailable ({state})")


# ----------------------------------------------------------------------------------
# Circuit breaker
# ----------------------------------------------------------------------------------


@dataclass
class CircuitBreaker:
    """Per-source retry + circuit-breaker policy (§8.3).

    The clock and the backoff sleeper are injected so tests never touch wall time
    (the freshness clock in :func:`ingest_session` is injected for the same reason).

    Semantics (locked by F01): retries happen *inside* one :func:`call_with_breaker`
    invocation; only a fully-exhausted invocation counts as one failure. The breaker
    counts *consecutive* exhausted failures and opens once it reaches ``open_after`` —
    counting consecutively rather than over a wall-clock window is deliberate, because
    a source's own retry backoff (e.g. FastF1's 5+15+30 s) can exceed ``window_s``, so
    a sliding-window count could never reach the threshold during a real outage. A
    ``success`` resets the streak; an idle gap longer than ``window_s`` between failures
    also resets it (an isolated blip is forgotten). Once open, calls short-circuit
    immediately for ``window_s`` (the cooldown), after which one trial call is allowed.
    """

    name: str
    config: BreakerConfig
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    _consecutive: int = field(default=0, repr=False)
    _last_failure_at: float | None = field(default=None, repr=False)
    _opened_at: float | None = field(default=None, repr=False)

    def failure_count(self) -> int:
        """Consecutive post-retry failures recorded (the count that trips the breaker)."""
        return self._consecutive

    def is_open(self) -> bool:
        """True while the breaker is open and still inside its ``window_s`` cooldown."""
        if self._opened_at is None:
            return False
        # Cooldown elapsed -> half-open: allow one trial call through.
        return (self.clock() - self._opened_at) < self.config.window_s

    def record_success(self) -> None:
        """A successful call closes the breaker and clears the failure streak."""
        self._consecutive = 0
        self._last_failure_at = None
        self._opened_at = None

    def record_failure(self) -> None:
        """A fully-exhausted call: extend the streak and open if it hits ``open_after``."""
        now = self.clock()
        if self._last_failure_at is not None and now - self._last_failure_at > self.config.window_s:
            # Failures spaced further apart than the window are not a sustained outage.
            self._consecutive = 0
            self._opened_at = None
        self._consecutive += 1
        self._last_failure_at = now
        if self._consecutive >= self.config.open_after:
            self._opened_at = now


def call_with_breaker(breaker: CircuitBreaker, fn: Callable[[], T]) -> T:
    """Invoke ``fn`` under ``breaker``'s retry + circuit policy.

    Returns ``fn``'s result on success. Raises :class:`SourceUnavailableError` if the
    breaker is already open (short-circuit, no call made) or if every attempt fails
    with an :class:`IngestSourceError`. Any other exception (notably
    :class:`SessionUnavailableError`) is *not* caught — it propagates so the caller can
    distinguish "no such session" from "source down".
    """
    if breaker.is_open():
        # Breaker open: return immediately with no source call and no sleep — the
        # spec's "fallback within 500 ms, no hang" guarantee.
        raise SourceUnavailableError(breaker.name, short_circuited=True)

    attempts = breaker.config.retries + 1
    backoff = breaker.config.backoff_s
    last_exc: IngestSourceError | None = None
    for i in range(attempts):
        try:
            result = fn()
        except IngestSourceError as exc:  # transport fault — retry, then count once
            last_exc = exc
            if i < attempts - 1:
                breaker.sleep(backoff[min(i, len(backoff) - 1)])
        else:
            breaker.record_success()
            return result
    # Every attempt failed: this whole invocation is exactly one failure.
    breaker.record_failure()
    raise SourceUnavailableError(breaker.name, short_circuited=False, cause=last_exc)


def build_breakers(
    config: AppConfig,
    *,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, CircuitBreaker]:
    """Construct the three source breakers from the validated ``ingest:`` config.

    Precondition: ``config.ingest`` is present (callers that load a config without an
    ``ingest:`` block must handle that before calling this — :func:`ingest_session`
    does, returning a typed ``CONFIG_001`` result).
    """
    if config.ingest is None:
        raise ValueError("config.ingest is missing; cannot build breakers")
    b = config.ingest.breakers
    return {
        "fastf1": CircuitBreaker("fastf1", b.fastf1, clock, sleep),
        "openf1": CircuitBreaker("openf1", b.openf1, clock, sleep),
        "ergast": CircuitBreaker("ergast", b.ergast, clock, sleep),
    }


# ----------------------------------------------------------------------------------
# Cache (Parquet frame + provenance sidecar)
# ----------------------------------------------------------------------------------


def cache_path_for(request: SessionRequest, cache_dir: Path) -> Path:
    """Resolve the Parquet cache location: ``<cache_dir>/<season>/<circuit>/<driver>.parquet``."""
    return cache_dir / str(request.season) / request.circuit / f"{request.driver}.parquet"


def cache_meta_path(cpath: Path) -> Path:
    """Sidecar path carrying a cached session's provenance: ``<cpath>.meta.json``."""
    return cpath.with_name(cpath.name + ".meta.json")


def _cache_is_fresh(path: Path, max_age_days: int, now_epoch: float) -> bool:
    """True if ``path`` exists and is younger than ``max_age_days`` at ``now_epoch``."""
    if not path.is_file():
        return False
    age_days = (now_epoch - path.stat().st_mtime) / SECONDS_PER_DAY
    return age_days < max_age_days


def _write_cache(
    cpath: Path,
    telemetry: pd.DataFrame,
    *,
    compound_source: CompoundSource,
    tyre_age_source: TyreAgeSource,
    flags: list[ErrorCode],
) -> None:
    """Persist the telemetry frame and its provenance sidecar atomically-ish."""
    cpath.parent.mkdir(parents=True, exist_ok=True)
    telemetry.to_parquet(cpath, index=False)
    meta = {
        "compound_source": compound_source.value,
        "tyre_age_source": tyre_age_source.value,
        "flags": [f.value for f in flags],
    }
    cache_meta_path(cpath).write_text(json.dumps(meta), encoding="utf-8")


def _read_cached_session(
    cpath: Path,
) -> tuple[pd.DataFrame, CompoundSource, TyreAgeSource, list[ErrorCode]]:
    """Read a cached frame and its provenance sidecar.

    Raises one of ``_CACHE_IO_ERRORS`` if the Parquet or the sidecar is missing,
    unreadable, or malformed — so an incomplete cache is treated as corrupt rather than
    served with guessed provenance.
    """
    df = pd.read_parquet(cpath)
    mpath = cache_meta_path(cpath)
    if not mpath.is_file():
        raise ValueError(f"cache provenance sidecar missing for {cpath}")
    meta = json.loads(mpath.read_text(encoding="utf-8"))
    compound_source = CompoundSource(meta["compound_source"])
    tyre_age_source = TyreAgeSource(meta["tyre_age_source"])
    flags = [ErrorCode(f) for f in meta["flags"]]
    return df, compound_source, tyre_age_source, flags


# ----------------------------------------------------------------------------------
# Enrichment helpers (vectorised; merge per-lap context onto telemetry on `lap`)
# ----------------------------------------------------------------------------------


def _merge_on_lap(
    telemetry: pd.DataFrame, context: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    """Left-join the named per-lap ``columns`` of ``context`` onto ``telemetry`` by ``lap``.

    Defensive against malformed adapter frames: if ``context`` has no ``lap`` key or is
    empty, nothing is merged (the caller's fallback then fills the columns). The context
    is de-duplicated on ``lap`` first so a source returning multiple rows per lap can
    never multiply the telemetry row count.
    """
    present = [c for c in columns if c in context.columns]
    if "lap" not in context.columns or context.empty or not present:
        return telemetry
    keep_cols: list[str] = ["lap", *present]
    per_lap = context.reindex(columns=keep_cols).drop_duplicates(subset="lap", keep="first")
    return telemetry.merge(per_lap, on="lap", how="left")


def _ensure_columns(df: pd.DataFrame, defaults: dict[str, object]) -> pd.DataFrame:
    """Guarantee every key in ``defaults`` is a column, filling absent ones with its value."""
    missing = {c: v for c, v in defaults.items() if c not in df.columns}
    if not missing:
        return df
    df = df.copy()
    for col, value in missing.items():
        df[col] = value
    return df


def _derive_stint_and_age_from_pit(telemetry: pd.DataFrame) -> pd.DataFrame:
    """Derive per-lap ``stint`` and ``tyre_age`` from ``pit_out`` flags (Ergast-down fallback).

    Stint increments on each pit-out lap; tyre age = laps completed since that stint's
    out-lap. Vectorised over the unique lap list — no per-row Python loop.
    """
    out = telemetry.copy().drop(columns=["stint", "tyre_age"], errors="ignore")
    laps = pd.Index(sorted(out["lap"].unique()), name="lap")
    pit_out = out.groupby("lap")["pit_out"].max().reindex(laps).fillna(False).astype(bool)
    stint = pit_out.cumsum().clip(lower=1)  # 1-based stint index
    age = pit_out.groupby(stint).cumcount()
    per_lap = pd.DataFrame({"lap": laps, "stint": stint.to_numpy(), "tyre_age": age.to_numpy()})
    return out.merge(per_lap, on="lap", how="left")


# ----------------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------------


@dataclass
class _Ledger:
    """Accumulates data_quality rows for one ingest so logging stays in one place."""

    log_path: Path
    entity_id: str
    flags: list[ErrorCode] = field(default_factory=list)
    wrote: bool = False

    def record(
        self, code: ErrorCode, detail: str, *, fallback_used: bool, flag: bool = True
    ) -> None:
        log_data_quality(
            error_code=code,
            entity_type="session",
            entity_id=self.entity_id,
            error_detail=detail,
            fallback_used=fallback_used,
            log_path=self.log_path,
        )
        self.wrote = True
        if flag:
            self.flags.append(code)


def _config_missing_result(request: SessionRequest, config: AppConfig) -> IngestResult:
    """Typed result when the loaded config has no ``ingest:`` block (no bare raise, #1)."""
    log_data_quality(
        error_code=ErrorCode.CONFIG_001,
        entity_type="session",
        entity_id=request.entity_id,
        error_detail="config.ingest block absent; cannot ingest",
        fallback_used=False,
        log_path=Path(config.paths.data_quality_log),
    )
    return IngestResult(
        status=Status.FAILED, error_code=ErrorCode.CONFIG_001, source_used="", rows=0, logged=True
    )


def _data_001(ledger: _Ledger, detail: str) -> IngestResult:
    """FastF1 session unavailable / unusable: skip, never substitute."""
    ledger.record(ErrorCode.DATA_001, detail, fallback_used=False, flag=False)
    return IngestResult(
        status=Status.FAILED, error_code=ErrorCode.DATA_001, source_used="fastf1", rows=0,
        logged=True,
    )


def ingest_session(
    request: SessionRequest,
    config: AppConfig,
    *,
    fastf1_fetch: Callable[[SessionRequest], pd.DataFrame],
    openf1_fetch: Callable[[SessionRequest], pd.DataFrame],
    ergast_fetch: Callable[[SessionRequest], pd.DataFrame],
    fastf1_laps_fetch: Callable[[SessionRequest], pd.DataFrame] | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    now_epoch: Callable[[], float] = time.time,
    breakers: dict[str, CircuitBreaker] | None = None,
) -> IngestResult:
    """Ingest one session: cache-first, then FastF1 + OpenF1 + Ergast with fallbacks.

    Every external call is wrapped by :func:`call_with_breaker`; every failure or
    fallback writes one ``data_quality_log`` row and is reflected in the returned
    :class:`IngestResult`. The cache is always consulted before any network call.

    ``fastf1_laps_fetch`` supplies the per-lap FastF1 ``Laps.TrackStatus`` frame
    (``[lap, track_status]``); when omitted, track status is recorded as unavailable.

    The ``clock`` (breaker), ``now_epoch`` (cache freshness) and ``sleep`` (backoff) are
    injected so the whole flow is deterministic under test. For batch ingestion, build
    ``breakers`` once and pass the same dict to every call (see the module docstring).
    """
    if config.ingest is None:
        return _config_missing_result(request, config)

    log_path = Path(config.paths.data_quality_log)
    cpath = cache_path_for(request, Path(config.paths.cache_dir))
    ledger = _Ledger(log_path=log_path, entity_id=request.entity_id)
    if breakers is None:
        breakers = build_breakers(config, clock=clock, sleep=sleep)

    # 1) Cache-first: a fresh cache hit is served without any network call. Provenance
    #    is restored from the sidecar, so a previously-degraded session is NOT laundered
    #    to OK. A corrupt/incomplete cache is logged and treated as a miss (refetch).
    if _cache_is_fresh(cpath, config.ingest.cache_max_age_days, now_epoch()):
        try:
            cached, compound_source, tyre_age_source, flags = _read_cached_session(cpath)
        except _CACHE_IO_ERRORS as exc:
            ledger.record(
                ErrorCode.CACHE_CORRUPT, f"fresh cache unreadable, refetching: {exc}",
                fallback_used=True,
            )
        else:
            cached = _ensure_columns(cached, _CONTEXT_DEFAULTS)  # honour the column contract
            return IngestResult(
                status=Status.DEGRADED if flags else Status.OK,
                flags=flags,
                source_used="cache",
                rows=int(len(cached)),
                compound_source=compound_source,
                tyre_age_source=tyre_age_source,
                cache_path=cpath,
                logged=bool(flags),  # degradations were logged at original fetch time
            )

    # 2) FastF1 — primary telemetry. Distinguish "no such session" from "source down".
    try:
        telemetry = call_with_breaker(breakers["fastf1"], lambda: fastf1_fetch(request))
    except SessionUnavailableError:
        return _data_001(ledger, "FastF1 session unavailable")
    except SourceUnavailableError:
        return _fastf1_down(cpath, ledger)

    if telemetry is None or len(telemetry) == 0:
        return _data_001(ledger, "FastF1 returned an empty frame")
    if "lap" not in telemetry.columns:
        return _data_001(ledger, "FastF1 frame missing the 'lap' key column")

    # 2b) Attach the raw per-lap FastF1 track status (no green-flag interpretation).
    telemetry = _attach_track_status(request, telemetry, fastf1_laps_fetch, ledger)

    # 3) Enrich compound (+ pit flags) and lap context (lap_time/stint/tyre_age).
    #    OpenF1 and Ergast are each fetched at most once (Ergast serves both the
    #    compound fallback and the lap context).
    telemetry, compound_source, tyre_age_source = _enrich(
        request, telemetry, openf1_fetch, ergast_fetch, breakers, ledger
    )

    # Guarantee the full F01 -> F02 column contract before persisting.
    telemetry = _ensure_columns(telemetry, _CONTEXT_DEFAULTS)

    # 5) Persist telemetry + provenance. The cache is an optimisation, never a hard
    #    dependency: a write failure is logged and degrades the result, but the
    #    freshly-fetched telemetry is still returned.
    written_path: Path | None = cpath
    try:
        _write_cache(
            cpath, telemetry, compound_source=compound_source,
            tyre_age_source=tyre_age_source, flags=list(ledger.flags),
        )
    except _CACHE_IO_ERRORS as exc:
        ledger.record(ErrorCode.CACHE_CORRUPT, f"cache write failed: {exc}", fallback_used=False)
        written_path = None

    return IngestResult(
        status=Status.DEGRADED if ledger.flags else Status.OK,
        flags=list(ledger.flags),
        source_used="fastf1",
        rows=int(len(telemetry)),
        compound_source=compound_source,
        tyre_age_source=tyre_age_source,
        cache_path=written_path,
        logged=ledger.wrote,
    )


def _fastf1_down(cpath: Path, ledger: _Ledger) -> IngestResult:
    """FastF1 source is down: serve stale cache (STALE_DATA) or fail (DATA_SOURCE_UNAVAILABLE)."""
    if cpath.is_file():
        try:
            cached, compound_source, tyre_age_source, carried = _read_cached_session(cpath)
        except _CACHE_IO_ERRORS as exc:
            # Source down AND the only cache is corrupt/incomplete -> no usable telemetry.
            ledger.record(
                ErrorCode.DATA_SOURCE_UNAVAILABLE, f"FastF1 down and cache unreadable: {exc}",
                fallback_used=False, flag=False,
            )
            return IngestResult(
                status=Status.FAILED, error_code=ErrorCode.DATA_SOURCE_UNAVAILABLE,
                source_used="fastf1", rows=0, logged=True,
            )
        ledger.record(ErrorCode.STALE_DATA, "FastF1 down; served stale cache", fallback_used=True)
        cached = _ensure_columns(cached, _CONTEXT_DEFAULTS)  # honour the column contract
        # Carry the cached session's own fallback flags forward, plus the stale flag.
        flags = [*carried, *(f for f in ledger.flags if f not in carried)]
        return IngestResult(
            status=Status.DEGRADED, flags=flags, source_used="cache", rows=int(len(cached)),
            compound_source=compound_source, tyre_age_source=tyre_age_source,
            cache_path=cpath, logged=True,
        )
    ledger.record(
        ErrorCode.DATA_SOURCE_UNAVAILABLE, "FastF1 down and no cache present",
        fallback_used=False, flag=False,
    )
    return IngestResult(
        status=Status.FAILED, error_code=ErrorCode.DATA_SOURCE_UNAVAILABLE,
        source_used="fastf1", rows=0, logged=True,
    )


def _try_source(
    breaker: CircuitBreaker, fn: Callable[[], pd.DataFrame]
) -> pd.DataFrame | None:
    """Call an enrichment source through its breaker; return ``None`` if unavailable.

    Both "source down" and "session has no data" map to ``None`` for an *enrichment*
    source — the caller then falls back. (Only FastF1, the primary, distinguishes them.)
    """
    try:
        return call_with_breaker(breaker, fn)
    except (SourceUnavailableError, SessionUnavailableError):
        return None


def _has_values(df: pd.DataFrame, column: str) -> bool:
    """True if ``column`` exists in ``df`` and holds at least one non-null value."""
    return column in df.columns and bool(df[column].notna().any())


def _attach_track_status(
    request: SessionRequest,
    telemetry: pd.DataFrame,
    fastf1_laps_fetch: Callable[[SessionRequest], pd.DataFrame] | None,
    ledger: _Ledger,
) -> pd.DataFrame:
    """Left-join the raw per-lap FastF1 ``track_status`` onto telemetry (no interpretation).

    The status code is stored verbatim — green/SC/VSC classification is F02's job. If
    track status is unavailable (no fetcher, a fetch fault, or no ``track_status``
    values), the column is left null for every lap and one ``data_quality_log`` row is
    written; the session is NOT failed (track status is non-critical context).
    """
    laps: pd.DataFrame | None = None
    if fastf1_laps_fetch is not None:
        try:
            laps = fastf1_laps_fetch(request)
        except (IngestSourceError, SourceUnavailableError, SessionUnavailableError):
            # Covers a direct adapter fault AND an adapter that wraps the fetch in
            # call_with_breaker (which raises SourceUnavailableError when the breaker
            # is open / retries exhausted) — none may escape untyped.
            laps = None
    if laps is not None:
        telemetry = _merge_on_lap(telemetry, laps, ["track_status"])
    if _has_values(telemetry, "track_status"):
        return telemetry
    telemetry = _ensure_columns(telemetry, {"track_status": pd.NA})
    ledger.record(
        ErrorCode.TRACK_STATUS_UNAVAILABLE,
        "FastF1 Laps.TrackStatus unavailable; track_status left null",
        fallback_used=False,
    )
    return telemetry


def _enrich(
    request: SessionRequest,
    telemetry: pd.DataFrame,
    openf1_fetch: Callable[[SessionRequest], pd.DataFrame],
    ergast_fetch: Callable[[SessionRequest], pd.DataFrame],
    breakers: dict[str, CircuitBreaker],
    ledger: _Ledger,
) -> tuple[pd.DataFrame, CompoundSource, TyreAgeSource]:
    """Attach compound + pit flags + lap context, fetching each source at most once.

    Always returns telemetry carrying every per-lap contract column; an unavailable
    source is logged (COMPOUND_FALLBACK / COMPOUND_UNKNOWN / DERIVED_AGE) and its
    columns are filled with the documented null marker rather than dropped.
    """
    openf1 = _try_source(breakers["openf1"], lambda: openf1_fetch(request))
    ergast = _try_source(breakers["ergast"], lambda: ergast_fetch(request))

    # --- compound (+ pit flags from OpenF1) ---
    if openf1 is not None:
        telemetry = _merge_on_lap(telemetry, openf1, ["compound", "pit_in", "pit_out"])
    if _has_values(telemetry, "compound"):
        compound_source = CompoundSource.OPENF1
    elif ergast is not None and "compound" in ergast.columns:
        telemetry = _merge_on_lap(telemetry, ergast, ["compound"])
        if _has_values(telemetry, "compound"):
            compound_source = CompoundSource.ERGAST
            ledger.record(
                ErrorCode.COMPOUND_FALLBACK, "OpenF1 unavailable; compound sourced from Ergast",
                fallback_used=True,
            )
        else:
            compound_source = CompoundSource.UNKNOWN
    else:
        compound_source = CompoundSource.UNKNOWN

    if compound_source is CompoundSource.UNKNOWN:
        # Excluded from compound-stratified analysis only — core telemetry untouched.
        telemetry = telemetry.copy()
        telemetry["compound"] = "UNKNOWN"
        ledger.record(
            ErrorCode.COMPOUND_UNKNOWN, "OpenF1 and Ergast both unavailable for compound",
            fallback_used=False,
        )
    telemetry = _ensure_columns(telemetry, {"pit_in": False, "pit_out": False})

    # --- lap context: lap_time, stint, tyre_age ---
    if ergast is not None and "tyre_age" in ergast.columns:
        telemetry = _merge_on_lap(telemetry, ergast, ["lap_time", "stint", "tyre_age"])
    if _has_values(telemetry, "tyre_age"):
        tyre_age_source = TyreAgeSource.ERGAST
    else:
        # Ergast down / no tyre age: derive stint + age from pit flags, tag the laps.
        # lap_time has no fallback source -> left as the documented null marker.
        telemetry = _derive_stint_and_age_from_pit(telemetry)
        tyre_age_source = TyreAgeSource.DERIVED
        ledger.record(
            ErrorCode.DERIVED_AGE, "Ergast unavailable; stint and tyre age derived from pit flags",
            fallback_used=True,
        )

    return telemetry, compound_source, tyre_age_source
