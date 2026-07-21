"""GET /metrics — Prometheus exposition, internal-only (403 external, §8.5)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Response

from app.api import metrics as metrics_mod
from app.api.deps import AppState, get_apex_state
from app.api.errors import forbidden

router = APIRouter(tags=["metrics"])


@router.get("/metrics")
async def get_metrics(request: Request, state: AppState = Depends(get_apex_state)) -> Response:
    if not metrics_mod.is_internal(request, state.config):
        raise forbidden("metrics endpoint is internal only")
    metrics_mod.update_cache_breaker_gauge(state.cache)  # F14 — refresh the breaker gauge at scrape
    return Response(content=metrics_mod.render(), media_type=metrics_mod.CONTENT_TYPE)
