"""The async session-ingestion batch — the path that populates ``predictions`` + ``fingerprints``.

``run_ingestion`` is the **tested core** (§4.2 of the spec): it takes a list of :class:`LapRecord`
(one per lap, already feature-extracted + normalised by F01–F04) and, inside a single atomic F12
``session_scope``, persists the circuits/sessions/drivers/laps/lap_features graph, runs the F06b
classifier per clean lap → ``predictions``, and aggregates → F08 projection → ``fingerprints`` —
reporting progress + terminal status through a :class:`JobStore`. It **never fails silently**: a
fault rolls back the whole session (no partial write) and marks the job ``FAILED`` with
``INGEST_TASK_FAILED`` logged to ``data_quality_log``.

The live **FastF1 → F01–F04** pipeline that produces the ``LapRecord``s is injected as a
``feature_provider`` (F01's ``ingest_session`` itself takes injected fetchers, so this is faithful
dependency injection, not a stub). The offline suite drives ``run_ingestion`` with synthetic
records, fake handles, and aiosqlite (F13 handles, F12 aiosqlite — the established pattern). Wiring
the live provider as the default + populating real fingerprints is completed in F20's evaluation
harness, which runs F01–F04 end-to-end over the dataset (divergence #38; spec §4.0-c / §9 limit 6).
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from contextlib import suppress
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

from app.contracts.errors import ErrorCode
from app.contracts.logging import log_data_quality

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from app.contracts.config import AppConfig
    from app.db.breaker import DbCircuitBreaker
    from app.ml.inference import PredictionResult
    from app.ml.stylespace import ProjectionResult

    PredictFn = Callable[[dict[str, float | None], str], PredictionResult]
    ProjectFn = Callable[[pd.DataFrame, str, str | None], ProjectionResult]


class JobState(StrEnum):
    """The four ingest job states surfaced at ``GET /ingest/status`` (master §8.5)."""

    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


@dataclass(frozen=True)
class LapRecord:
    """One validated + feature-extracted lap (the F01–F03 pipeline output contract).

    This is the intermediate the ingestion task persists. ``feature_vector`` is the **raw** 28-key
    F03 vector — each downstream model (F06b classifier, F08 style space, F10 monitor) applies the
    F04 ``normalisation.json`` internally, so raw features are stored and never pre-normalised.
    ``is_clean`` + ``error_code`` come from F02; ``null_count`` / ``imputed_features`` from F03a's
    6-NULL tolerance gate."""

    driver_code: str
    driver_name: str
    team: str | None
    season: int
    round: int
    circuit_name: str
    circuit_type: str
    session_type: str
    lap_number: int
    tyre_compound: str | None
    tyre_age: int | None
    is_clean: bool
    error_code: str
    has_interpolation: bool
    feature_vector: dict[str, float | None]
    null_count: int
    imputed_features: list[str] = field(default_factory=list)


@dataclass
class JobStatus:
    """A job's live status. ``result`` carries the terminal ``IngestSessionResponse`` dict when
    ``COMPLETE`` (§4.0-b — the terminal report the 202 could not include inline)."""

    status: JobState
    progress: float
    error_detail: str | None = None
    result: dict[str, Any] | None = None


@dataclass
class IngestOutcome:
    """The terminal result of ``run_ingestion`` — the fields of ``IngestSessionResponse`` plus the
    counts the ``quality_report`` surfaces."""

    session_id: str
    laps_accepted: int
    laps_rejected: int
    rejection_summary: dict[str, int]
    quality_report: dict[str, Any]

    def as_response_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "laps_accepted": self.laps_accepted,
            "laps_rejected": self.laps_rejected,
            "rejection_summary": self.rejection_summary,
            "quality_report": self.quality_report,
        }


class JobStore(Protocol):
    """Where job status lives so ``/ingest/status`` can read what a worker wrote. The in-process
    impl suffices for eager mode (same process); production uses the Redis-backed impl (shared
    across the API + worker processes)."""

    async def create(self, session_id: str) -> None: ...
    async def update(
        self,
        session_id: str,
        *,
        status: JobState,
        progress: float,
        error_detail: str | None = None,
        result: dict[str, Any] | None = None,
    ) -> None: ...
    async def get(self, session_id: str) -> JobStatus | None: ...


class InProcessJobStore:
    """A dict-backed job store — used by the offline suite and Celery eager mode (same process)."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobStatus] = {}

    async def create(self, session_id: str) -> None:
        self._jobs[session_id] = JobStatus(status=JobState.PENDING, progress=0.0)

    async def update(
        self,
        session_id: str,
        *,
        status: JobState,
        progress: float,
        error_detail: str | None = None,
        result: dict[str, Any] | None = None,
    ) -> None:
        self._jobs[session_id] = JobStatus(
            status=status, progress=progress, error_detail=error_detail, result=result
        )

    async def get(self, session_id: str) -> JobStatus | None:
        return self._jobs.get(session_id)


class RedisJobStore:
    """A Redis-hash-backed job store (production). Status is a JSON blob under
    ``{prefix}:job:{session_id}`` with the configured result TTL. Every client call is bounded by
    ``timeout_s`` (so a hung/partitioned Redis can never hang ``/ingest/session`` or
    ``/ingest/status``), and a fault is **logged** — never silently swallowed (§1). A read fault
    still degrades to ``None`` (the route maps that to 404), but the outage is recorded, not
    invisible."""

    def __init__(
        self,
        client: Any,
        *,
        key_prefix: str,
        ttl_s: int,
        timeout_s: float = 2.0,
        log_path: str | None = None,
    ) -> None:
        self._client = client
        self._prefix = key_prefix
        self._ttl_s = ttl_s
        self._timeout_s = timeout_s
        self._log_path = log_path

    def _key(self, session_id: str) -> str:
        return f"{self._prefix}:job:{session_id}"

    def _log(self, session_id: str, detail: str) -> None:
        if self._log_path is None:
            return
        with suppress(Exception):
            log_data_quality(
                error_code=ErrorCode.CACHE_UNAVAILABLE,
                entity_type="job_store",
                entity_id=session_id,
                error_detail=detail,
                fallback_used=True,
                log_path=self._log_path,
            )

    async def create(self, session_id: str) -> None:
        await self.update(session_id, status=JobState.PENDING, progress=0.0)

    async def update(
        self,
        session_id: str,
        *,
        status: JobState,
        progress: float,
        error_detail: str | None = None,
        result: dict[str, Any] | None = None,
    ) -> None:
        blob = json.dumps(
            {
                "status": status.value,
                "progress": progress,
                "error_detail": error_detail,
                "result": result,
            }
        )
        try:
            await asyncio.wait_for(
                self._client.set(self._key(session_id), blob, ex=self._ttl_s), self._timeout_s
            )
        except Exception as exc:  # never hang, never swallow silently — log the write fault (§1)
            self._log(session_id, f"job-store write failed: {type(exc).__name__}: {exc}")

    async def get(self, session_id: str) -> JobStatus | None:
        try:
            raw = await asyncio.wait_for(
                self._client.get(self._key(session_id)), self._timeout_s
            )
        except Exception as exc:  # log the backend outage, then degrade to "unknown" (404)
            self._log(session_id, f"job-store read failed: {type(exc).__name__}: {exc}")
            return None
        if raw is None:
            return None
        try:
            d = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
            return JobStatus(
                status=JobState(d["status"]),
                progress=float(d["progress"]),
                error_detail=d.get("error_detail"),
                result=d.get("result"),
            )
        except (ValueError, KeyError):
            return None


# --- persistence helpers (get-or-create → idempotent, atomic re-ingest) ------------------------
async def _get_or_create_circuit(session: AsyncSession, *, name: str, circuit_type: str) -> Any:
    from sqlalchemy import select

    from app.db import models

    row = (
        await session.execute(select(models.Circuit).where(models.Circuit.name == name))
    ).scalar_one_or_none()
    if row is not None:
        return row
    row = models.Circuit(name=name, type=models.CircuitType(circuit_type))
    session.add(row)
    await session.flush()
    return row


async def _get_or_create_driver(
    session: AsyncSession, *, code: str, full_name: str, team: str | None, season: int
) -> Any:
    from sqlalchemy import select

    from app.db import models

    row = (
        await session.execute(
            select(models.Driver).where(
                models.Driver.code == code, models.Driver.season == season
            )
        )
    ).scalar_one_or_none()
    if row is not None:
        return row
    row = models.Driver(code=code, full_name=full_name, team=team, season=season)
    session.add(row)
    await session.flush()
    return row


async def _get_or_create_session_row(
    session: AsyncSession, *, circuit_id: Any, season: int, rnd: int, session_type: str
) -> Any:
    from sqlalchemy import select

    from app.db import models

    row = (
        await session.execute(
            select(models.RaceSession).where(
                models.RaceSession.circuit_id == circuit_id,
                models.RaceSession.season == season,
                models.RaceSession.round == rnd,
                models.RaceSession.session_type == session_type,
            )
        )
    ).scalar_one_or_none()
    if row is not None:
        return row
    row = models.RaceSession(
        circuit_id=circuit_id, season=season, round=rnd, session_type=session_type
    )
    session.add(row)
    await session.flush()
    return row


def _driver_code_for_prediction(name: str) -> str:
    """A stable 3-char code for a predicted driver name (drivers.code is CHAR(3))."""
    letters = "".join(ch for ch in name.upper() if ch.isalpha())
    return (letters + "XXX")[:3]


async def run_ingestion(
    records: list[LapRecord],
    *,
    session_id: str,
    config: AppConfig,
    session_factory: async_sessionmaker[AsyncSession],
    db_breaker: DbCircuitBreaker,
    predict_fn: PredictFn | None = None,
    project_fn: ProjectFn | None = None,
    job_store: JobStore | None = None,
    log_path: str | None = None,
) -> IngestOutcome:
    """Persist a session's laps + predictions + fingerprints atomically, reporting progress.

    All-or-nothing: the whole session is written inside one ``session_scope`` — a DB fault rolls it
    back (no partial write, §11.4) and the exception propagates so the caller marks the job FAILED.
    ``predict_fn`` / ``project_fn`` are the F06b / F08 runners (injected; a ``None`` handle simply
    skips that stage). Idempotent: circuits/drivers/sessions/fingerprints are get-or-created, so a
    re-ingest of the same session updates rather than duplicates.
    """
    import pandas as pd
    from sqlalchemy import select

    from app.db import models
    from app.db.repository import insert_lap_features, persist
    from app.db.session import session_scope

    async def _progress(status: JobState, p: float) -> None:
        if job_store is not None:
            await job_store.update(session_id, status=status, progress=p)

    accepted = sum(1 for r in records if r.is_clean)
    rejected = sum(1 for r in records if not r.is_clean)
    rejection_summary = dict(Counter(r.error_code for r in records if not r.is_clean))
    predictions_written = 0
    fingerprints_written = 0

    await _progress(JobState.PROCESSING, 0.05)

    async with session_scope(session_factory, db_breaker) as session:
        # 1. Master data + per-lap rows (one atomic graph).
        driver_rows: dict[str, Any] = {}
        circuit_rows: dict[str, Any] = {}
        lap_id_by_key: dict[tuple[str, int], Any] = {}
        for r in records:
            if r.circuit_name not in circuit_rows:
                circuit_rows[r.circuit_name] = await _get_or_create_circuit(
                    session, name=r.circuit_name, circuit_type=r.circuit_type
                )
            circuit = circuit_rows[r.circuit_name]
            if r.driver_code not in driver_rows:
                driver_rows[r.driver_code] = await _get_or_create_driver(
                    session, code=r.driver_code, full_name=r.driver_name, team=r.team,
                    season=r.season,
                )
            driver = driver_rows[r.driver_code]
            race = await _get_or_create_session_row(
                session, circuit_id=circuit.id, season=r.season, rnd=r.round,
                session_type=r.session_type,
            )
            existing = (
                await session.execute(
                    select(models.Lap).where(
                        models.Lap.session_id == race.id,
                        models.Lap.driver_id == driver.id,
                        models.Lap.lap_number == r.lap_number,
                    )
                )
            ).scalar_one_or_none()
            if existing is not None:
                lap_id_by_key[(r.driver_code, r.lap_number)] = existing.id
                continue
            lap = models.Lap(
                session_id=race.id, driver_id=driver.id, lap_number=r.lap_number,
                is_clean=r.is_clean, error_code=r.error_code, tyre_compound=r.tyre_compound,
                tyre_age=r.tyre_age, has_interpolation=r.has_interpolation,
            )
            await persist(session, lap)
            lap_id_by_key[(r.driver_code, r.lap_number)] = lap.id
            await insert_lap_features(
                session, lap_id=lap.id, feature_vector=r.feature_vector,
                null_count=r.null_count, imputed_features=r.imputed_features,
            )
        await _progress(JobState.PROCESSING, 0.4)

        # 2. Classifier → predictions (clean laps only; skip gated laps with no top prediction).
        #    Get-or-update by lap_id so a re-ingest updates rather than duplicating (idempotent —
        #    predictions is 1:1 with a lap; no UNIQUE at the DB layer, so dedup here).
        if predict_fn is not None:
            for r in records:
                if not r.is_clean:
                    continue
                result = predict_fn(r.feature_vector, r.circuit_name)
                cs = result.confidence_state
                if result.top_prediction is None or cs is None:
                    continue  # gated / no state → cannot satisfy the NOT-NULL predictions row
                pred_driver = await _get_or_create_driver(
                    session, code=_driver_code_for_prediction(result.top_prediction),
                    full_name=result.top_prediction, team=None, season=r.season,
                )
                lap_id = lap_id_by_key[(r.driver_code, r.lap_number)]
                existing_pred = (
                    await session.execute(
                        select(models.Prediction).where(models.Prediction.lap_id == lap_id)
                    )
                ).scalar_one_or_none()
                if existing_pred is not None:
                    existing_pred.predicted_driver_id = pred_driver.id
                    existing_pred.confidence = result.confidence
                    existing_pred.confidence_state = cs
                    existing_pred.shap_values = result.shap_values
                    existing_pred.shap_available = result.shap_available
                else:
                    await persist(
                        session,
                        models.Prediction(
                            lap_id=lap_id,
                            predicted_driver_id=pred_driver.id,
                            confidence=result.confidence,
                            confidence_state=cs,
                            shap_values=result.shap_values,
                            shap_available=result.shap_available,
                        ),
                    )
                predictions_written += 1
        await _progress(JobState.PROCESSING, 0.7)

        # 3. Fingerprint per (driver, season, circuit) from the driver's clean laps (get-or-create).
        if project_fn is not None:
            by_group: dict[tuple[str, int, str], list[LapRecord]] = {}
            for r in records:
                if r.is_clean:
                    by_group.setdefault((r.driver_code, r.season, r.circuit_name), []).append(r)
            for (dcode, season, cname), grp in by_group.items():
                frame = pd.DataFrame([g.feature_vector for g in grp])
                proj = project_fn(frame, cname, grp[0].circuit_type)
                if proj.coords_5d is None:
                    continue  # projection unavailable → no fingerprint vector to write
                driver = driver_rows[dcode]
                circuit = circuit_rows[cname]
                vector = {f"d{i}": float(v) for i, v in enumerate(proj.coords_5d)}
                existing_fp = (
                    await session.execute(
                        select(models.Fingerprint).where(
                            models.Fingerprint.driver_id == driver.id,
                            models.Fingerprint.season == season,
                            models.Fingerprint.circuit_id == circuit.id,
                        )
                    )
                ).scalar_one_or_none()
                if existing_fp is not None:
                    existing_fp.vector = vector
                else:
                    await persist(
                        session,
                        models.Fingerprint(
                            driver_id=driver.id, season=season, circuit_id=circuit.id,
                            vector=vector,
                        ),
                    )
                fingerprints_written += 1

    quality_report = {
        "state": JobState.COMPLETE.value,
        "laps_total": len(records),
        "predictions_written": predictions_written,
        "fingerprints_written": fingerprints_written,
        "drivers": sorted({r.driver_code for r in records}),
    }
    outcome = IngestOutcome(
        session_id=session_id, laps_accepted=accepted, laps_rejected=rejected,
        rejection_summary=rejection_summary, quality_report=quality_report,
    )
    if job_store is not None:
        await job_store.update(
            session_id, status=JobState.COMPLETE, progress=1.0,
            result=outcome.as_response_dict(),
        )
    return outcome


def log_ingest_failure(session_id: str, exc: BaseException, *, log_path: str | None) -> None:
    """Record an ingestion-task fault to ``data_quality_log`` (``INGEST_TASK_FAILED``) — the job
    never dies silently (§1). The distinct DB codes keep their own identity where they apply."""
    if log_path is None:
        return
    code = getattr(exc, "error_code", None)
    if not isinstance(code, ErrorCode):
        code = ErrorCode.INGEST_TASK_FAILED
    with suppress(Exception):
        log_data_quality(
            error_code=code,
            entity_type="ingest_session",
            entity_id=session_id,
            error_detail=f"{type(exc).__name__}: {exc}",
            fallback_used=False,
            log_path=log_path,
        )
