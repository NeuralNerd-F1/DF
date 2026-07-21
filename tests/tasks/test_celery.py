"""F14 Celery app + task shell + enqueue (§8.2). Offline: eager task body + fake broker."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from app.contracts.config import AppConfig
from app.contracts.errors import ErrorCode
from app.db.breaker import DbCircuitBreaker
from app.tasks.celery import (
    IngestContext,
    QueueUnavailableError,
    _ingest_session_body,
    enqueue_ingestion,
    make_celery_app,
    set_ingest_context,
)
from app.tasks.ingest import InProcessJobStore, JobState
from sqlalchemy.ext.asyncio import async_sessionmaker

from tests.tasks.conftest import fake_predict, fake_project, make_records


def test_make_celery_app_honours_always_eager(cfg: AppConfig) -> None:
    cfg.celery.always_eager = True
    app = make_celery_app(cfg)
    assert app.conf.task_always_eager is True
    assert "apex.ingest_session" in app.tasks


def test_task_body_runs_ingestion(
    cfg: AppConfig,
    session_factory: async_sessionmaker,
    db_breaker: DbCircuitBreaker,
    tmp_path: Path,
) -> None:
    """The sync task body (called outside an event loop) runs run_ingestion → COMPLETE."""
    store = InProcessJobStore()
    ctx = IngestContext(
        config=cfg,
        session_factory=session_factory,
        db_breaker=db_breaker,
        job_store=store,
        feature_provider=lambda y, e, s: make_records(n_clean=4),
        predict_fn=fake_predict(),
        project_fn=fake_project(),
        log_path=str(tmp_path / "dq.jsonl"),
    )
    set_ingest_context(ctx)
    try:
        import asyncio

        asyncio.run(store.create("j1"))
        result = _ingest_session_body("j1", 2023, "Bahrain", "R")
        assert result["laps_accepted"] == 4
        status = asyncio.run(store.get("j1"))
        assert status is not None and status.status is JobState.COMPLETE
    finally:
        set_ingest_context(None)


def test_task_body_failure_marks_failed_and_logs(
    cfg: AppConfig,
    session_factory: async_sessionmaker,
    db_breaker: DbCircuitBreaker,
    tmp_path: Path,
) -> None:
    log = tmp_path / "dq.jsonl"
    store = InProcessJobStore()

    def boom(y: int, e: str, s: str) -> list:
        raise RuntimeError("provider exploded")

    ctx = IngestContext(
        config=cfg,
        session_factory=session_factory,
        db_breaker=db_breaker,
        job_store=store,
        feature_provider=boom,
        log_path=str(log),
    )
    set_ingest_context(ctx)
    try:
        import asyncio

        asyncio.run(store.create("j2"))
        with pytest.raises(RuntimeError):
            _ingest_session_body("j2", 2023, "Bahrain", "R")
        status = asyncio.run(store.get("j2"))
        assert status is not None and status.status is JobState.FAILED
        rows = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if x]
        assert any(r["error_code"] == ErrorCode.INGEST_TASK_FAILED.value for r in rows)
    finally:
        set_ingest_context(None)


async def test_enqueue_registers_job_and_sends(cfg: AppConfig) -> None:
    store = InProcessJobStore()
    sent: list[dict] = []

    class FakeCelery:
        def send_task(self, name: str, *, args: list, task_id: str) -> None:
            sent.append({"name": name, "args": args, "task_id": task_id})

    sid = await enqueue_ingestion(FakeCelery(), store, year=2023, event="Bahrain", session_type="R")
    assert sent and sent[0]["task_id"] == sid
    status = await store.get(sid)
    assert status is not None and status.status is JobState.PENDING


async def test_enqueue_broker_down_raises_queue_unavailable(cfg: AppConfig) -> None:
    store = InProcessJobStore()

    class DownCelery:
        def send_task(self, name: str, *, args: list, task_id: str) -> None:
            raise ConnectionError("broker unreachable")

    with pytest.raises(QueueUnavailableError) as exc:
        await enqueue_ingestion(DownCelery(), store, year=2023, event="X", session_type="R")
    assert exc.value.error_code is ErrorCode.QUEUE_UNAVAILABLE


def test_enqueue_via_real_app_registers_pending(cfg: AppConfig) -> None:
    """The real Celery app + a memory broker: ``enqueue_ingestion`` registers the job and dispatches
    without crashing. The worker is separate (``send_task`` does not run eagerly, by Celery design),
    so the job sits PENDING — the honest "queued until a worker consumes it" state (§8.2)."""
    import asyncio

    app = make_celery_app(cfg)
    store = InProcessJobStore()
    sid = asyncio.run(enqueue_ingestion(app, store, year=2023, event="Bahrain", session_type="R"))
    status = asyncio.run(store.get(sid))
    assert status is not None and status.status is JobState.PENDING
