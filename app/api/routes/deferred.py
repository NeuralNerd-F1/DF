"""Endpoints whose backing engine lands in F14 (Redis/Celery/replay) or F05 (upload).

Their Pydantic response schemas ship in F13 (so the OpenAPI surface is complete), but the live
routes honestly return **503 `HEALTH_FAIL`** "not yet available" rather than fabricating data
(divergence #28). F14/F05 replace these bodies with real implementations.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.errors import service_unavailable
from app.api.schemas import (
    IngestSessionResponse,
    IngestStatusResponse,
    ReplayResponse,
    UploadResponse,
)

router = APIRouter(tags=["deferred"])

_DEFERRED = "endpoint not yet available (backing service lands in a later feature)"


@router.post("/ingest/session", response_model=IngestSessionResponse)
async def ingest_session() -> IngestSessionResponse:
    raise service_unavailable(f"POST /ingest/session — {_DEFERRED} (F14 Celery)")


@router.get("/ingest/status/{session_id}", response_model=IngestStatusResponse)
async def ingest_status(session_id: str) -> IngestStatusResponse:
    raise service_unavailable(f"GET /ingest/status — {_DEFERRED} (F14)")


@router.post("/ingest/upload", response_model=UploadResponse)
async def ingest_upload() -> UploadResponse:
    raise service_unavailable(f"POST /ingest/upload — {_DEFERRED} (F05 manual upload)")


@router.get("/replay/{session_id}/{driver_id}", response_model=ReplayResponse)
async def replay(session_id: str, driver_id: str) -> ReplayResponse:
    raise service_unavailable(f"GET /replay — {_DEFERRED} (F14 replay engine)")
