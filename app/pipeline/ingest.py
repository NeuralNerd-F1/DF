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
A persisted session Parquet (``cache/<season>/<circuit>/<driver>.parquet``) holds one
row per telemetry sample with these columns:

* FastF1 telemetry (~50 Hz): ``Speed``, ``Throttle``, ``Brake``, ``RPM``, ``Gear``,
  ``DRS``, ``X``, ``Y`` and the ``lap`` number the sample belongs to.
* OpenF1 enrichment (per lap, left-joined on ``lap``): ``compound``, ``pit_in``,
  ``pit_out``.
* Ergast / derived lap context (per lap, left-joined on ``lap``): ``lap_time``,
  ``stint``, ``tyre_age``.

When a source is unavailable the affected columns are still present but carry the
documented fallback marker (``compound == "UNKNOWN"``, tyre age derived from pit
flags) and the result's ``flags`` / ``data_quality_log`` record exactly which.
"""

from __future__ import annotations

import time
from collections import deque
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

# Errors a corrupt / unwritable Parquet cache can raise. Caught so a bad cache file is
# a typed, logged outcome — never an exception escaping ingest_session (non-negotiable #1).
_CACHE_IO_ERRORS = (OSError, ValueError, pyarrow.ArrowInvalid)

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
    :class:`BaseResult`. ``status`` is ``OK`` on a clean fetch (or fresh cache hit),
    ``DEGRADED`` when a documented fallback fired (the ``flags`` say which), and
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
    call is fully exhausted, counted as one failure toward the breaker window.
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

    Retry/breaker relationship (locked by F01): retries happen *inside* one
    :func:`call_with_breaker` invocation; only a fully-exhausted invocation counts as
    a single failure toward the window. The breaker opens once ``open_after`` such
    failures fall inside ``window_s``; while open, calls short-circuit immediately
    (no source call, no sleep) and never add further failures.
    """

    name: str
    config: BreakerConfig
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    _failures: deque[float] = field(default_factory=deque, repr=False)

    def _prune(self, now: float) -> None:
        """Drop failure timestamps that have aged out of the rolling window."""
        horizon = now - self.config.window_s
        while self._failures and self._failures[0] <= horizon:
            self._failures.popleft()

    def failure_count(self) -> int:
        """Number of post-retry failures currently inside the window."""
        self._prune(self.clock())
        return len(self._failures)

    def is_open(self) -> bool:
        """True once the in-window failure count reaches ``open_after``."""
        return self.failure_count() >= self.config.open_after

    def _record_failure(self) -> None:
        self._failures.append(self.clock())


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
            return fn()
        except IngestSourceError as exc:  # transport fault — retry, then count once
            last_exc = exc
            if i < attempts - 1:
                breaker.sleep(backoff[min(i, len(backoff) - 1)])
    # Every attempt failed: this whole invocation is exactly one failure.
    breaker._record_failure()
    raise SourceUnavailableError(breaker.name, short_circuited=False, cause=last_exc)


def build_breakers(
    config: AppConfig,
    *,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, CircuitBreaker]:
    """Construct the three source breakers from the validated ``ingest:`` config."""
    if config.ingest is None:  # pragma: no cover - guarded by config validation
        raise ValueError("config.ingest is missing; cannot build breakers")
    b = config.ingest.breakers
    return {
        "fastf1": CircuitBreaker("fastf1", b.fastf1, clock, sleep),
        "openf1": CircuitBreaker("openf1", b.openf1, clock, sleep),
        "ergast": CircuitBreaker("ergast", b.ergast, clock, sleep),
    }


# ----------------------------------------------------------------------------------
# Cache
# ----------------------------------------------------------------------------------


def cache_path_for(request: SessionRequest, cache_dir: Path) -> Path:
    """Resolve the Parquet cache location: ``<cache_dir>/<season>/<circuit>/<driver>.parquet``."""
    return cache_dir / str(request.season) / request.circuit / f"{request.driver}.parquet"


def _cache_is_fresh(path: Path, max_age_days: int, now_epoch: float) -> bool:
    """True if ``path`` exists and is younger than ``max_age_days`` at ``now_epoch``."""
    if not path.is_file():
        return False
    age_days = (now_epoch - path.stat().st_mtime) / 86400.0
    return age_days < max_age_days


# ----------------------------------------------------------------------------------
# Enrichment helpers (vectorised; merge per-lap context onto telemetry on `lap`)
# ----------------------------------------------------------------------------------


def _merge_on_lap(
    telemetry: pd.DataFrame, context: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    """Left-join the named per-lap ``columns`` of ``context`` onto ``telemetry`` by ``lap``."""
    take = ["lap", *[c for c in columns if c in context.columns]]
    return telemetry.merge(context[take], on="lap", how="left")


def _derive_tyre_age_from_pit(telemetry: pd.DataFrame) -> pd.DataFrame:
    """Derive per-lap ``tyre_age`` from ``pit_out`` flags (Ergast-down fallback).

    Tyre age = laps completed since the most recent pit-out (a fresh stint starts at
    age 0 on its out-lap). Vectorised over the unique lap list — no per-row Python loop.
    """
    out = telemetry.copy()
    out = out.drop(columns=["tyre_age"], errors="ignore")
    laps = pd.Index(sorted(out["lap"].unique()), name="lap")
    pit_out = out.groupby("lap")["pit_out"].max().reindex(laps).fillna(False).astype(bool)
    # Stint index increments on each pit-out lap; age = position within the stint.
    stint = pit_out.cumsum()
    age_by_lap = pit_out.groupby(stint).cumcount().rename("tyre_age").reset_index()
    return out.merge(age_by_lap, on="lap", how="left")


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


def ingest_session(
    request: SessionRequest,
    config: AppConfig,
    *,
    fastf1_fetch: Callable[[SessionRequest], pd.DataFrame],
    openf1_fetch: Callable[[SessionRequest], pd.DataFrame],
    ergast_fetch: Callable[[SessionRequest], pd.DataFrame],
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    now_epoch: Callable[[], float] = time.time,
    breakers: dict[str, CircuitBreaker] | None = None,
) -> IngestResult:
    """Ingest one session: cache-first, then FastF1 + OpenF1 + Ergast with fallbacks.

    Every external call is wrapped by :func:`call_with_breaker`; every failure or
    fallback writes one ``data_quality_log`` row and is reflected in the returned
    :class:`IngestResult`. The cache is always consulted before any network call.

    The ``clock`` (breaker window) and ``now_epoch`` (cache freshness) clocks plus the
    backoff ``sleep`` are injected so the whole flow is deterministic under test.
    """
    if config.ingest is None:  # pragma: no cover - guarded by config validation
        raise ValueError("config.ingest is missing; cannot ingest")

    cache_dir = Path(config.paths.cache_dir)
    log_path = Path(config.paths.data_quality_log)
    cpath = cache_path_for(request, cache_dir)
    ledger = _Ledger(log_path=log_path, entity_id=request.entity_id)
    if breakers is None:
        breakers = build_breakers(config, clock=clock, sleep=sleep)

    # 1) Cache-first: a fresh cache hit is served without any network call. A corrupt
    #    cache is not fatal — it is logged and treated as a miss, so we fetch fresh.
    if _cache_is_fresh(cpath, config.ingest.cache_max_age_days, now_epoch()):
        try:
            cached = pd.read_parquet(cpath)
        except _CACHE_IO_ERRORS as exc:
            ledger.record(
                ErrorCode.CACHE_CORRUPT,
                f"fresh cache unreadable, refetching: {exc}",
                fallback_used=True,
            )
        else:
            return IngestResult(
                status=Status.OK,
                source_used="cache",
                rows=int(len(cached)),
                compound_source=_compound_source_of(cached),
                tyre_age_source=_tyre_age_source_of(cached),
                cache_path=cpath,
            )

    # 2) FastF1 — primary telemetry. Distinguish "no such session" from "source down".
    try:
        telemetry = call_with_breaker(breakers["fastf1"], lambda: fastf1_fetch(request))
    except SessionUnavailableError:
        # Session genuinely unavailable: skip, never substitute (not even from cache).
        ledger.record(
            ErrorCode.DATA_001, "FastF1 session unavailable", fallback_used=False, flag=False
        )
        return IngestResult(
            status=Status.FAILED, error_code=ErrorCode.DATA_001,
            source_used="fastf1", rows=0, logged=True,
        )
    except SourceUnavailableError:
        # Source down (breaker open or retries exhausted). Serve stale cache if present.
        return _fastf1_down(request, cpath, ledger)

    if telemetry is None or len(telemetry) == 0:
        # Empty frame is the same outcome as an unavailable session.
        ledger.record(
            ErrorCode.DATA_001, "FastF1 returned an empty frame", fallback_used=False, flag=False
        )
        return IngestResult(
            status=Status.FAILED, error_code=ErrorCode.DATA_001,
            source_used="fastf1", rows=0, logged=True,
        )

    # 3) Enrich compound + pit flags from OpenF1; fall back to Ergast for compound.
    telemetry, compound_source = _enrich_compound(
        request, telemetry, openf1_fetch, ergast_fetch, breakers, ledger
    )

    # 4) Lap context (lap_time, stint, tyre_age) from Ergast; derive age if Ergast down.
    telemetry, tyre_age_source = _enrich_lap_context(
        request, telemetry, ergast_fetch, breakers, ledger
    )

    # 5) Persist the (possibly fallback-marked) telemetry to the Parquet cache. The
    #    cache is an optimisation, never a hard dependency: a write failure is logged
    #    and degrades the result, but the freshly-fetched telemetry is still returned.
    written_path: Path | None = cpath
    try:
        cpath.parent.mkdir(parents=True, exist_ok=True)
        telemetry.to_parquet(cpath, index=False)
    except _CACHE_IO_ERRORS as exc:
        ledger.record(
            ErrorCode.CACHE_CORRUPT, f"cache write failed: {exc}", fallback_used=False
        )
        written_path = None

    status = Status.DEGRADED if ledger.flags else Status.OK
    return IngestResult(
        status=status,
        flags=list(ledger.flags),
        source_used="fastf1",
        rows=int(len(telemetry)),
        compound_source=compound_source,
        tyre_age_source=tyre_age_source,
        cache_path=written_path,
        logged=ledger.wrote,
    )


def _fastf1_down(request: SessionRequest, cpath: Path, ledger: _Ledger) -> IngestResult:
    """FastF1 source is down: serve stale cache (STALE_DATA) or fail (DATA_SOURCE_UNAVAILABLE)."""
    if cpath.is_file():
        try:
            cached = pd.read_parquet(cpath)
        except _CACHE_IO_ERRORS as exc:
            # Source down AND the only cache is corrupt -> no usable telemetry.
            ledger.record(
                ErrorCode.DATA_SOURCE_UNAVAILABLE,
                f"FastF1 down and cache unreadable: {exc}",
                fallback_used=False,
                flag=False,
            )
            return IngestResult(
                status=Status.FAILED,
                error_code=ErrorCode.DATA_SOURCE_UNAVAILABLE,
                source_used="fastf1",
                rows=0,
                logged=True,
            )
        ledger.record(
            ErrorCode.STALE_DATA, "FastF1 down; served stale cache", fallback_used=True
        )
        return IngestResult(
            status=Status.DEGRADED,
            flags=list(ledger.flags),
            source_used="cache",
            rows=int(len(cached)),
            compound_source=_compound_source_of(cached),
            tyre_age_source=_tyre_age_source_of(cached),
            cache_path=cpath,
            logged=True,
        )
    ledger.record(
        ErrorCode.DATA_SOURCE_UNAVAILABLE,
        "FastF1 down and no cache present",
        fallback_used=False,
        flag=False,
    )
    return IngestResult(
        status=Status.FAILED,
        error_code=ErrorCode.DATA_SOURCE_UNAVAILABLE,
        source_used="fastf1",
        rows=0,
        logged=True,
    )


def _enrich_compound(
    request: SessionRequest,
    telemetry: pd.DataFrame,
    openf1_fetch: Callable[[SessionRequest], pd.DataFrame],
    ergast_fetch: Callable[[SessionRequest], pd.DataFrame],
    breakers: dict[str, CircuitBreaker],
    ledger: _Ledger,
) -> tuple[pd.DataFrame, CompoundSource]:
    """Attach compound (+ pit flags) from OpenF1; on OpenF1 failure fall back to Ergast."""
    try:
        openf1 = call_with_breaker(breakers["openf1"], lambda: openf1_fetch(request))
        telemetry = _merge_on_lap(telemetry, openf1, ["compound", "pit_in", "pit_out"])
        return telemetry, CompoundSource.OPENF1
    except (SourceUnavailableError, SessionUnavailableError):
        pass  # OpenF1 unavailable -> try Ergast for compound (and keep pit flags absent).

    try:
        ergast = call_with_breaker(breakers["ergast"], lambda: ergast_fetch(request))
    except (SourceUnavailableError, SessionUnavailableError):
        ergast = None

    if ergast is not None and "compound" in ergast.columns:
        telemetry = _merge_on_lap(telemetry, ergast, ["compound"])
        ledger.record(
            ErrorCode.COMPOUND_FALLBACK,
            "OpenF1 unavailable; compound sourced from Ergast",
            fallback_used=True,
        )
        return telemetry, CompoundSource.ERGAST

    # Both sources failed to supply compound: mark UNKNOWN, exclude from compound
    # stratification only — core telemetry is untouched.
    telemetry = telemetry.copy()
    telemetry["compound"] = "UNKNOWN"
    ledger.record(
        ErrorCode.COMPOUND_UNKNOWN,
        "OpenF1 and Ergast both unavailable for compound",
        fallback_used=False,
    )
    return telemetry, CompoundSource.UNKNOWN


def _enrich_lap_context(
    request: SessionRequest,
    telemetry: pd.DataFrame,
    ergast_fetch: Callable[[SessionRequest], pd.DataFrame],
    breakers: dict[str, CircuitBreaker],
    ledger: _Ledger,
) -> tuple[pd.DataFrame, TyreAgeSource]:
    """Attach lap_time/stint/tyre_age from Ergast; derive tyre age from pit flags if down."""
    try:
        ergast = call_with_breaker(breakers["ergast"], lambda: ergast_fetch(request))
        telemetry = _merge_on_lap(telemetry, ergast, ["lap_time", "stint", "tyre_age"])
        if "tyre_age" in ergast.columns:
            return telemetry, TyreAgeSource.ERGAST
    except (SourceUnavailableError, SessionUnavailableError):
        pass

    # Ergast down (or no tyre_age): derive age from pit flags and tag the laps.
    if "pit_out" not in telemetry.columns:
        telemetry = telemetry.copy()
        telemetry["pit_out"] = False
    telemetry = _derive_tyre_age_from_pit(telemetry)
    ledger.record(
        ErrorCode.DERIVED_AGE,
        "Ergast unavailable; tyre age derived from pit flags",
        fallback_used=True,
    )
    return telemetry, TyreAgeSource.DERIVED


def _compound_source_of(df: pd.DataFrame) -> CompoundSource:
    """Infer the compound provenance of a cached frame from its columns."""
    if "compound" not in df.columns:
        return CompoundSource.UNKNOWN
    if df["compound"].astype(str).eq("UNKNOWN").all():
        return CompoundSource.UNKNOWN
    return CompoundSource.OPENF1


def _tyre_age_source_of(df: pd.DataFrame) -> TyreAgeSource:
    """A cached frame carries Ergast-quality tyre age unless it was derived."""
    return TyreAgeSource.ERGAST if "tyre_age" in df.columns else TyreAgeSource.DERIVED
