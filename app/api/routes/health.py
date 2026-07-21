"""GET /health — real component status (§8.1/§8.3). 503 if any critical component is down."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response

from app.api.deps import AppState, get_apex_state
from app.api.schemas import HealthResponse
from app.ml.inference import health as classifier_health
from app.ml.stylespace import health as stylespace_health
from app.monitoring.detector import health as degradation_health
from app.personalisation.engine import health as personalisation_health

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def get_health(
    response: Response, state: AppState = Depends(get_apex_state)
) -> HealthResponse:
    """Report DB + model-handle status. Critical = DB connected AND every model ready; cache
    (Redis, F14) is reported but never critical. Any critical failure → HTTP 503."""
    models_loaded = {
        "classifier": classifier_health(state.model_handle).ready,
        "stylespace": stylespace_health(state.style_handle).ready,
        "degradation": degradation_health(state.monitor_handle).ready,
        "personalisation": personalisation_health(state.personalisation_handle).ready,
    }

    db_connected = False
    if state.session_factory is not None:
        from app.db.health import check_database

        db_health = await check_database(
            state.session_factory, state.config, breaker=state.db_breaker
        )
        db_connected = db_health.ok

    # Cache (F14 Redis) is reported but NEVER critical (constitution: an optimisation, not a
    # dependency). None when unconfigured; else a real PING through the breaker.
    cache_connected: bool | None = None
    if state.cache is not None:
        cache_connected = await state.cache.ping()

    healthy = db_connected and all(models_loaded.values())
    response.status_code = 200 if healthy else 503
    return HealthResponse(
        status="healthy" if healthy else "unhealthy",
        models_loaded=models_loaded,
        db_connected=db_connected,
        cache_connected=cache_connected,
        version=state.config.api.version if state.config.api is not None else "unknown",
    )
