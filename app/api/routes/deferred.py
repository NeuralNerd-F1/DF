"""The one remaining deferred endpoint: POST /ingest/upload (manual CSV upload → F05).

F14 filled the async ingestion + replay routes (see ``ingest.py`` / ``replay.py``); only the manual
upload path still awaits its feature (F05). Its Pydantic response schema ships in F13 (so the
OpenAPI surface is complete), but the live route honestly returns **503 `HEALTH_FAIL`** "not yet
available" rather than fabricating data (divergence #28). F05 replaces this body with the real one.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.errors import service_unavailable
from app.api.schemas import UploadResponse

router = APIRouter(tags=["deferred"])

_DEFERRED = "endpoint not yet available (backing service lands in a later feature)"


@router.post("/ingest/upload", response_model=UploadResponse)
async def ingest_upload() -> UploadResponse:
    raise service_unavailable(f"POST /ingest/upload — {_DEFERRED} (F05 manual upload)")
