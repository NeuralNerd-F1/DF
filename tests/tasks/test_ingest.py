"""F14 ``run_ingestion`` — the atomic persist + predict + fingerprint batch (§4.2, §8.4)."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
from app.contracts.config import AppConfig
from app.db import models
from app.db.breaker import DbCircuitBreaker
from app.db.results import DatabaseUnavailableError, DbIntegrityError
from app.tasks.ingest import InProcessJobStore, JobState, run_ingestion
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from tests.tasks.conftest import fake_predict, fake_project, make_records


async def _count(session_factory: async_sessionmaker, model: type) -> int:
    async with session_factory() as s:
        return (await s.execute(select(func.count()).select_from(model))).scalar_one()


async def test_happy_path_persists_full_graph(
    cfg: AppConfig, session_factory: async_sessionmaker, db_breaker: DbCircuitBreaker
) -> None:
    store = InProcessJobStore()
    await store.create("s1")
    out = await run_ingestion(
        make_records(n_clean=5, n_rejected=1),
        session_id="s1",
        config=cfg,
        session_factory=session_factory,
        db_breaker=db_breaker,
        predict_fn=fake_predict(),
        project_fn=fake_project(),
        job_store=store,
    )
    assert out.laps_accepted == 5 and out.laps_rejected == 1
    assert out.rejection_summary == {"LAP_001": 1}
    assert await _count(session_factory, models.Lap) == 6
    assert await _count(session_factory, models.LapFeatures) == 6
    assert await _count(session_factory, models.Prediction) == 5  # clean laps only
    assert await _count(session_factory, models.Fingerprint) == 1
    status = await store.get("s1")
    assert status is not None and status.status is JobState.COMPLETE
    assert status.progress == 1.0 and status.result is not None


async def test_gated_prediction_writes_no_prediction_row(
    cfg: AppConfig, session_factory: async_sessionmaker, db_breaker: DbCircuitBreaker
) -> None:
    out = await run_ingestion(
        make_records(n_clean=3),
        session_id="s2",
        config=cfg,
        session_factory=session_factory,
        db_breaker=db_breaker,
        predict_fn=fake_predict(top=None),  # gated → no predicted_driver_id
        project_fn=fake_project(),
    )
    assert out.laps_accepted == 3
    assert await _count(session_factory, models.Prediction) == 0


async def test_no_projection_writes_no_fingerprint(
    cfg: AppConfig, session_factory: async_sessionmaker, db_breaker: DbCircuitBreaker
) -> None:
    def project_none(frame: object, circuit: str, ctype: str | None) -> SimpleNamespace:
        return SimpleNamespace(coords_5d=None)  # projection unavailable

    await run_ingestion(
        make_records(n_clean=3),
        session_id="s3",
        config=cfg,
        session_factory=session_factory,
        db_breaker=db_breaker,
        predict_fn=fake_predict(),
        project_fn=project_none,
    )
    assert await _count(session_factory, models.Fingerprint) == 0


async def test_null_count_mismatch_rolls_back_whole_session(
    cfg: AppConfig, session_factory: async_sessionmaker, db_breaker: DbCircuitBreaker
) -> None:
    recs = make_records(n_clean=3)
    # Corrupt one record's null_count so insert_lap_features raises DB_INTEGRITY mid-batch.
    recs[1] = dataclasses.replace(recs[1], null_count=recs[1].null_count + 5)
    with pytest.raises(DbIntegrityError):
        await run_ingestion(
            recs,
            session_id="s4",
            config=cfg,
            session_factory=session_factory,
            db_breaker=db_breaker,
            predict_fn=fake_predict(),
            project_fn=fake_project(),
        )
    # Atomic: the whole session rolled back — NO partial write.
    assert await _count(session_factory, models.Lap) == 0
    assert await _count(session_factory, models.LapFeatures) == 0


async def test_db_outage_rolls_back_no_partial_write(
    cfg: AppConfig, session_factory: async_sessionmaker, db_breaker: DbCircuitBreaker
) -> None:
    # Pre-open the breaker so session_scope short-circuits → DatabaseUnavailableError, no writes.
    for _ in range(db_breaker.config.open_after):
        db_breaker.record_failure()
    assert db_breaker.is_open()
    with pytest.raises(DatabaseUnavailableError):
        await run_ingestion(
            make_records(n_clean=3),
            session_id="s5",
            config=cfg,
            session_factory=session_factory,
            db_breaker=db_breaker,
            predict_fn=fake_predict(),
            project_fn=fake_project(),
        )


async def test_reingest_is_idempotent(
    cfg: AppConfig, session_factory: async_sessionmaker, db_breaker: DbCircuitBreaker
) -> None:
    recs = make_records(n_clean=4)
    for sid in ("a", "b"):
        await run_ingestion(
            recs,
            session_id=sid,
            config=cfg,
            session_factory=session_factory,
            db_breaker=db_breaker,
            predict_fn=fake_predict(),
            project_fn=fake_project(),
        )
    # Second run updates rather than duplicates (get-or-create / get-or-update).
    assert await _count(session_factory, models.Lap) == 4
    assert await _count(session_factory, models.Fingerprint) == 1
    assert await _count(session_factory, models.Prediction) == 4  # NOT 8 — predictions deduped


async def test_empty_session_completes_with_zero_counts(
    cfg: AppConfig, session_factory: async_sessionmaker, db_breaker: DbCircuitBreaker
) -> None:
    store = InProcessJobStore()
    await store.create("s6")
    out = await run_ingestion(
        [],
        session_id="s6",
        config=cfg,
        session_factory=session_factory,
        db_breaker=db_breaker,
        predict_fn=fake_predict(),
        project_fn=fake_project(),
        job_store=store,
    )
    assert out.laps_accepted == 0 and out.laps_rejected == 0
    status = await store.get("s6")
    assert status is not None and status.status is JobState.COMPLETE  # 0 laps is not an error
