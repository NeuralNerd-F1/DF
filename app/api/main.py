"""``create_app()`` — the APEX FastAPI application factory (§8.5/§9).

Builds the app, wires the best-effort lifespan (loads every model handle + the DB engine ONCE),
registers the typed exception handlers, CORS, Prometheus instrumentation, and the routers. The app
boots even when model artefacts are absent — ``/health`` then reports the outage as 503.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import metrics as metrics_mod
from app.api.deps import load_state
from app.api.errors import register_exception_handlers
from app.api.routes import (
    drivers,
    health,
    ingest,
    metrics,
    predict,
    replay,
    sessions,
    style,
    upload,
)
from app.contracts.config import AppConfig, ConfigError, load_config


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load handles + DB once at startup (best-effort); dispose the engine at shutdown."""
    config: AppConfig = app.state.apex_config
    app.state.apex = load_state(config)
    try:
        yield
    finally:
        state = app.state.apex
        if state.engine is not None:
            await state.engine.dispose()
        if state.cache is not None:  # F14 — release the Redis connection pool (never raises)
            await state.cache.close()


def create_app(config: AppConfig | None = None) -> FastAPI:
    """Build the APEX API. ``config`` defaults to ``load_config()``; the ``api:`` block required."""
    cfg = config if config is not None else load_config()
    if cfg.api is None:
        raise ConfigError("api: block is required to build the FastAPI app")

    docs_on = cfg.api.docs_enabled
    app = FastAPI(
        title="APEX API",
        version=cfg.api.version,
        lifespan=lifespan,
        docs_url="/docs" if docs_on else None,
        redoc_url="/redoc" if docs_on else None,
        # Gate the schema too — else /openapi.json leaks the full endpoint map even with docs off.
        openapi_url="/openapi.json" if docs_on else None,
    )
    app.state.apex_config = cfg

    register_exception_handlers(app)
    metrics_mod.instrument(app)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.api.cors_origins,
        # F13 has no cookies/auth, so credentials are off by default; a validator forbids the
        # unsafe '*'-origin-with-credentials combination (ApiConfig._scalars_sane).
        allow_credentials=cfg.api.allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(predict.router)
    app.include_router(drivers.router)
    app.include_router(style.router)
    app.include_router(metrics.router)
    app.include_router(ingest.router)  # F14 — POST /ingest/session (202) + GET /ingest/status
    app.include_router(replay.router)  # F14 — GET /replay/{session_id}/{driver_id}
    app.include_router(sessions.router)  # Screen 6 session catalog — GET /sessions (div #77)
    app.include_router(upload.router)  # F05 — POST /ingest/upload (manual CSV upload)
    return app
