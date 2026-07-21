"""POST /ingest/session (202 async) + GET /ingest/status — the Celery ingestion queue (F14, §8.5).

``POST /ingest/session`` enqueues the async ingestion batch and returns **202** immediately with the
``session_id`` and honest *pending* placeholder counts (§4.0-b, divergence #31); the terminal report
is polled via ``GET /ingest/status`` (and stored durably in the result backend + the DQ log). A
broker/worker that cannot be reached → 503 ``QUEUE_UNAVAILABLE`` (never a silent drop). An unknown
``session_id`` → 404 ``NOT_FOUND``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends

from app.api import metrics as metrics_mod
from app.api.deps import get_celery, get_job_store
from app.api.errors import not_found
from app.api.schemas import IngestSessionRequest, IngestSessionResponse, IngestStatusResponse
from app.tasks.celery import enqueue_ingestion
from app.tasks.ingest import JobState, JobStore

if TYPE_CHECKING:
    from celery import Celery

router = APIRouter(tags=["ingest"])


@router.post("/ingest/session", response_model=IngestSessionResponse, status_code=202)
async def ingest_session(
    req: IngestSessionRequest,
    celery_app: Celery = Depends(get_celery),
    job_store: JobStore = Depends(get_job_store),
) -> IngestSessionResponse:
    """Enqueue an async session ingestion → 202 Accepted (§8.5). Counts are *pending* placeholders;
    poll ``/ingest/status/{session_id}`` for progress and the terminal quality report."""
    session_id = await enqueue_ingestion(
        celery_app, job_store, year=req.year, event=req.event, session_type=req.session
    )
    metrics_mod.record_ingest_job("enqueued")  # §8.5 /metrics — async job inflow (depth proxy)
    return IngestSessionResponse(
        session_id=session_id,
        laps_accepted=0,
        laps_rejected=0,
        rejection_summary={},
        quality_report={"state": JobState.PENDING.value},
    )


@router.get("/ingest/status/{session_id}", response_model=IngestStatusResponse)
async def ingest_status(
    session_id: str, job_store: JobStore = Depends(get_job_store)
) -> IngestStatusResponse:
    """Report a job's status (§8.5). 404 if unknown. On COMPLETE, ``result`` carries the terminal
    report (the §8.5 counts + quality_report) so the async response fulfils the master contract."""
    status = await job_store.get(session_id)
    if status is None:
        raise not_found(f"ingestion session {session_id!r} not found")
    return IngestStatusResponse(
        status=status.status.value,
        progress=status.progress,
        error_detail=status.error_detail,
        result=status.result,
    )
