"""The Celery task queue (§8.2/§9) — async ML batch jobs on a Redis broker + result backend.

The heavy async job is **session ingestion** (``ingest_session_task`` → :func:`run_ingestion`); the
batch SHAP task (§6.1) is the other. Single-lap requests stay synchronous (``/predict``), satisfying
the master's "synchronous fallback for single-lap requests" without a queue.

Worker resources (DB, model handles, job store, and the F01–F04 ``feature_provider``) are **not**
serialisable task arguments, so they live in a process-level :class:`IngestContext` set once per
worker (``set_ingest_context``) — the standard Celery worker-init pattern. The offline suite sets a
test context (aiosqlite + fake handles + synthetic ``feature_provider``) and runs the task eagerly,
so no broker/worker/FastF1 is needed. Broker-unreachable at enqueue → typed ``QUEUE_UNAVAILABLE``
(503), never a silent drop (§8.2, divergence #32).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import uuid
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from app.contracts.errors import ErrorCode
from app.tasks.ingest import JobState, LapRecord, log_ingest_failure, run_ingestion

if TYPE_CHECKING:
    from collections.abc import Callable

    from celery import Celery
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from app.contracts.config import AppConfig
    from app.db.breaker import DbCircuitBreaker
    from app.tasks.ingest import JobStore, PredictFn, ProjectFn

    FeatureProvider = Callable[[int, str, str], list[LapRecord]]


class QueueUnavailableError(Exception):
    """The Celery broker/worker could not be reached to enqueue a batch job (§8.2). Maps to a typed
    **503 ``QUEUE_UNAVAILABLE``** — the batch is refused loudly, never dropped."""

    error_code = ErrorCode.QUEUE_UNAVAILABLE

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


@dataclass
class IngestContext:
    """The resources ``ingest_session_task`` needs, set once per worker (not task arguments)."""

    config: AppConfig
    session_factory: async_sessionmaker[AsyncSession]
    db_breaker: DbCircuitBreaker
    job_store: JobStore
    feature_provider: FeatureProvider
    predict_fn: PredictFn | None = None
    project_fn: ProjectFn | None = None
    log_path: str | None = None


_T = TypeVar("_T")

_CONTEXT: IngestContext | None = None


def _run_coro(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run an async coroutine to completion from a SYNC Celery task body, whether or not a loop is
    already running (Celery ``always_eager`` fired from inside an async route would otherwise hit a
    "loop already running" error). No running loop → ``asyncio.run``; a running loop → a thread."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(lambda: asyncio.run(coro)).result()


def set_ingest_context(context: IngestContext | None) -> None:
    """Install (or clear) the process-level ingestion context. Called from the worker init in
    production and from the test fixture in the offline suite."""
    global _CONTEXT
    _CONTEXT = context


def get_ingest_context() -> IngestContext | None:
    return _CONTEXT


def make_celery_app(config: AppConfig) -> Celery:
    """Build the Celery app from the validated ``celery:`` block. Broker + result backend DSNs come
    from the named env vars (Redis); absent → an in-memory broker (fine under ``always_eager``, set
    the offline suite sets). Registers ``ingest_session_task``."""
    from celery import Celery

    if config.celery is None:  # pragma: no cover — guarded by callers/config validators
        raise ValueError("celery: block is required to build the Celery app")
    cc = config.celery
    broker = os.environ.get(cc.broker_url_env_var, "memory://")
    backend = os.environ.get(cc.result_backend_env_var, "cache+memory://")
    app = Celery("apex", broker=broker, backend=backend)
    app.conf.update(
        task_default_queue=cc.task_default_queue,
        task_always_eager=cc.always_eager,
        task_eager_propagates=cc.always_eager,
        worker_concurrency=cc.worker_concurrency,
        task_soft_time_limit=cc.task_soft_time_limit_s,
        task_time_limit=cc.task_time_limit_s,
        result_expires=cc.result_ttl_s,
        broker_connection_retry_on_startup=False,
    )
    app.task(name="apex.ingest_session", bind=False, max_retries=cc.task_max_retries)(
        _ingest_session_body
    )
    return app


def _ingest_session_body(
    session_id: str, year: int, event: str, session_type: str
) -> dict[str, Any]:
    """The task body: fetch records via the F01–F04 provider, then persist via ``run_ingestion``.

    Never fails silently — any fault marks the job ``FAILED`` (``INGEST_TASK_FAILED`` logged) and
    re-raises so Celery records the failure too."""
    ctx = get_ingest_context()
    if ctx is None:  # a worker with no context is a misconfiguration, surfaced loudly
        raise RuntimeError("ingest context not set on this worker (call set_ingest_context)")
    try:
        records = ctx.feature_provider(year, event, session_type)
        outcome = _run_coro(
            run_ingestion(
                records,
                session_id=session_id,
                config=ctx.config,
                session_factory=ctx.session_factory,
                db_breaker=ctx.db_breaker,
                predict_fn=ctx.predict_fn,
                project_fn=ctx.project_fn,
                job_store=ctx.job_store,
                log_path=ctx.log_path,
            )
        )
    except Exception as exc:
        log_ingest_failure(session_id, exc, log_path=ctx.log_path)
        _run_coro(
            ctx.job_store.update(
                session_id, status=JobState.FAILED, progress=1.0,
                error_detail=f"{type(exc).__name__}: {exc}",
            )
        )
        raise
    return outcome.as_response_dict()


async def enqueue_ingestion(
    celery_app: Celery,
    job_store: JobStore,
    *,
    year: int,
    event: str,
    session_type: str,
) -> str:
    """Register a job + enqueue ``ingest_session_task``, returning the ``session_id`` (= task id).

    A broker/worker that cannot be reached → :class:`QueueUnavailableError` (503
    ``QUEUE_UNAVAILABLE``); the job is refused loudly, never silently dropped (§8.2). Async because
    it is awaited from the (async) ``/ingest/session`` route and registers the job in an async store
    first, so ``/ingest/status`` can find it immediately."""
    session_id = str(uuid.uuid4())
    await job_store.create(session_id)
    try:
        celery_app.send_task(
            "apex.ingest_session",
            args=[session_id, year, event, session_type],
            task_id=session_id,
        )
    except Exception as exc:  # broker unreachable / kombu OperationalError → typed 503, not dropped
        raise QueueUnavailableError(
            f"cannot enqueue ingestion: {type(exc).__name__}: {exc}"
        ) from exc
    return session_id


def build_batch_shap_task(celery_app: Celery) -> None:  # pragma: no cover — thin registration
    """Register the batch-SHAP async task (§6.1). Single-lap SHAP stays synchronous on ``/predict``;
    batch SHAP over ≥1 laps runs here. Body delegates to F07's explainer via the ingest context's
    handles — wired alongside the ingestion provider in production."""
    # The batch-SHAP body mirrors the ingestion body's context pattern; its live wiring lands with
    # the F20 evaluation harness (same divergence #38 boundary as the feature_provider).
    return None
