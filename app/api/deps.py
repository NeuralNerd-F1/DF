"""Application state + FastAPI dependency providers + the best-effort startup loader.

The handles are loaded ONCE at startup into an ``AppState`` and injected into routes via ``Depends``
— so tests override the providers with fake handles (`app.dependency_overrides`), the only way to
exercise endpoint logic without committed model artefacts. The loader is **best-effort**: a failed
load (a missing artefact — none are committed — or an absent DSN) stores ``None`` and the app still
boots, so ``/health`` can report the outage as 503 rather than crashing at startup (§8.1).

Import-light at module level (only `Request` + config): the heavy ML/DB imports happen inside
``load_state`` so importing this module never pulls pandas/xgboost.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from fastapi import Depends, Request

from app.api.errors import service_unavailable
from app.api.profiles import DriverProfiles, load_driver_profiles
from app.contracts.config import AppConfig
from app.contracts.errors import ErrorCode
from app.contracts.logging import log_data_quality

_T = TypeVar("_T")

# Injected model-runner types. Kept as ``Callable[..., Any]`` at RUNTIME (not forward refs to the
# heavy result classes) so FastAPI's ``get_type_hints`` on the dependencies resolves without pulling
# — or failing to resolve — the ML result types. The precise shapes are documented on each provider.
PredictFn = Callable[..., Any]
ExplainFn = Callable[..., Any]
ProjectFn = Callable[..., Any]
StyleSpaceFn = Callable[..., Any]
RecommendFn = Callable[..., Any]

if TYPE_CHECKING:
    import pandas as pd
    from celery import Celery
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

    from app.cache.redis import RedisCache
    from app.db.breaker import DbCircuitBreaker
    from app.ml.explain import ExplanationResult, ShapExplainer
    from app.ml.inference import ModelHandle, PredictionResult
    from app.ml.stylespace import ProjectionResult, StyleSpaceHandle, StyleSpaceResult
    from app.monitoring.detector import MonitorHandle
    from app.personalisation.engine import PersonalisationHandle, SetupRecommendation
    from app.pipeline.features import FeatureMedians
    from app.tasks.ingest import JobStore


@dataclass
class AppState:
    """Everything a request needs, loaded once at startup. ``None`` = component failed to load."""

    config: AppConfig
    model_handle: ModelHandle | None = None
    explainer: ShapExplainer | None = None
    style_handle: StyleSpaceHandle | None = None
    monitor_handle: MonitorHandle | None = None
    personalisation_handle: PersonalisationHandle | None = None
    engine: AsyncEngine | None = None
    session_factory: async_sessionmaker[AsyncSession] | None = None
    db_breaker: DbCircuitBreaker | None = None
    # F14 platform layer. ``cache`` is non-critical (None/disabled → cache-less). ``celery_app`` +
    # ``job_store`` back the async ingestion queue (None when the ``celery:`` block is absent).
    cache: RedisCache | None = None
    celery_app: Celery | None = None
    job_store: JobStore | None = None
    # F05 manual upload — the train-only imputation medians, injected (there is NO production
    # loader; ``None`` in a fresh checkout → the upload returns the cleaning report with feature
    # vectors unavailable, never fabricated). Tests inject a synthetic table.
    medians: FeatureMedians | None = None
    # Per-driver dataset + evaluation facts read from ``driver_profiles.json`` (build script:
    # scripts/build_driver_profiles.py). ``None`` = the artefact is absent or unreadable, which
    # /drivers surfaces as null profile fields rather than zeroed measurements.
    driver_profiles: DriverProfiles | None = None


def get_apex_state(request: Request) -> AppState:
    """The injected app state. Tests override THIS provider to inject fake handles + a test DB."""
    state: AppState | None = getattr(request.app.state, "apex", None)
    if state is None:  # pragma: no cover — lifespan always sets it
        raise RuntimeError("AppState not initialised — the lifespan did not run")
    return state


async def get_db_session(
    state: AppState = Depends(get_apex_state),
) -> AsyncIterator[AsyncSession]:
    """A per-request DB session inside F12's ``session_scope`` (commit/rollback + breaker).

    Two distinct 503s: DB **never configured** (no engine built at startup — absent DSN) → 503
    ``HEALTH_FAIL``, matching every other not-loaded component (`get_predictor` etc.) and what
    ``/health`` reports. A **runtime** connection fault / open breaker inside the scope raises
    ``DatabaseUnavailableError`` → 503 ``DB_UNAVAILABLE`` (rolled back first — no partial write).
    """
    from app.db.session import session_scope

    if state.session_factory is None or state.db_breaker is None:
        raise service_unavailable("database is not configured")
    async with session_scope(state.session_factory, state.db_breaker) as session:
        yield session


def _try_load(name: str, loader: Callable[[], _T], *, log_path: str) -> _T | None:
    """Run a startup ``loader``, returning ``None`` on failure. NOT a silent swallow: the primary
    fault is written to ``data_quality_log`` (only the secondary log-write is suppressed, mirroring
    F12's ``_safe_rollback``) so a missing component is observable, per non-negotiable #1.

    Note: this writes a **component-level** startup row (``entity_type="startup"``); the inner
    ``load_*`` ALSO logs its own module-level fault row, so a missing artefact yields two rows by
    design — a dashboard on ``data_quality_log`` should de-duplicate on ``entity_type``.
    """
    try:
        return loader()
    except Exception as exc:  # startup best-effort: log + degrade, never crash the server
        code = getattr(exc, "error_code", ErrorCode.HEALTH_FAIL)
        with suppress(Exception):
            log_data_quality(
                error_code=code if isinstance(code, ErrorCode) else ErrorCode.HEALTH_FAIL,
                entity_type="startup",
                entity_id=name,
                error_detail=f"{type(exc).__name__}: {exc}",
                fallback_used=False,
                log_path=log_path,
            )
        return None


# --- Model-computation runners (injectable; tests override with canned results) ----------------
# Each returns a callable that runs the real never-raising model function over the loaded handle. A
# ``None`` handle → 503 (service not ready). Tests override these providers to return a fake runner
# yielding a canned internal ``*Result`` — so the ROUTE's status/response mapping is tested without
# committed artefacts, while the 503-not-loaded path is exercised by NOT overriding.
def get_predictor(state: AppState = Depends(get_apex_state)) -> PredictFn:
    from app.ml.inference import predict

    if state.model_handle is None:
        raise service_unavailable("classifier is not loaded")
    handle, cfg = state.model_handle, state.config

    def _run(feature_vector: dict[str, float | None], circuit: str) -> PredictionResult:
        return predict(feature_vector, circuit=circuit, handle=handle, config=cfg)

    return _run


def get_explainer_fn(state: AppState = Depends(get_apex_state)) -> ExplainFn | None:
    if state.explainer is None or state.model_handle is None:
        return None
    from app.ml.explain import explain_prediction

    explainer, handle, cfg = state.explainer, state.model_handle, state.config

    def _run(
        feature_vector: dict[str, float | None], circuit: str, predicted_class: str
    ) -> ExplanationResult:
        return explain_prediction(
            feature_vector,
            circuit=circuit,
            predicted_class=predicted_class,
            explainer=explainer,
            handle=handle,
            config=cfg,
        )

    return _run


def get_projector(state: AppState = Depends(get_apex_state)) -> ProjectFn:
    from app.ml.stylespace import project_unknown

    if state.style_handle is None:
        raise service_unavailable("style space is not loaded")
    handle, cfg = state.style_handle, state.config

    def _run(laps: pd.DataFrame, circuit: str, circuit_type: str | None) -> ProjectionResult:
        return project_unknown(laps, circuit=circuit, handle=handle, config=cfg)

    return _run


def get_stylespace_fn(state: AppState = Depends(get_apex_state)) -> StyleSpaceFn:
    from app.ml.stylespace import get_style_space

    if state.style_handle is None:
        raise service_unavailable("style space is not loaded")
    handle = state.style_handle

    def _run() -> StyleSpaceResult:
        return get_style_space(handle)

    return _run


def get_recommender(state: AppState = Depends(get_apex_state)) -> RecommendFn:
    from app.personalisation.engine import recommend_setup

    if state.personalisation_handle is None or state.style_handle is None:
        raise service_unavailable("personalisation is not loaded")
    ph, sh, cfg = state.personalisation_handle, state.style_handle, state.config

    def _run(driver: str, circuit: str, circuit_type: str | None) -> SetupRecommendation:
        return recommend_setup(
            driver=driver,
            circuit=circuit,
            circuit_type=circuit_type,
            handle=ph,
            style_handle=sh,
            config=cfg,
        )

    return _run


# --- F14 cache / queue / replay providers ------------------------------------------------------
def get_cache(state: AppState = Depends(get_apex_state)) -> RedisCache | None:
    """The cache, or ``None`` when unconfigured. NEVER raises — the cache is an optimisation, so a
    ``None``/disabled/down cache just means the caller computes from source (§4.0-a)."""
    return state.cache


def get_job_store(state: AppState = Depends(get_apex_state)) -> JobStore:
    """The ingest job store. A ``None`` (no ``celery:`` block) → 503 like any not-configured
    component."""
    if state.job_store is None:
        raise service_unavailable("ingestion queue is not configured")
    return state.job_store


def get_celery(state: AppState = Depends(get_apex_state)) -> Celery:
    """The Celery app for enqueuing ingestion. A ``None`` → 503 (not configured)."""
    if state.celery_app is None:
        raise service_unavailable("ingestion queue is not configured")
    return state.celery_app


def get_monitor_handle(state: AppState = Depends(get_apex_state)) -> MonitorHandle:
    """The F10 degradation handle the replay engine drives. A ``None`` (not loaded) → 503."""
    if state.monitor_handle is None:
        raise service_unavailable("degradation monitor is not loaded")
    return state.monitor_handle


def get_medians(state: AppState = Depends(get_apex_state)) -> FeatureMedians | None:
    """The F05 train-only imputation medians, or ``None`` when unavailable. NEVER raises — the
    upload's cleaning report (accepted/rejected counts) is medians-independent, so a ``None`` table
    just means feature vectors are skipped (honestly), not that the endpoint fails (§4.0-d)."""
    return state.medians


def _build_job_store(config: AppConfig, cache: RedisCache | None) -> JobStore:  # noqa: ARG001
    """The job store shared by the API + worker. Eager / no-backend-DSN → in-process (single
    process); a real Redis result-backend DSN → a Redis-backed store visible across processes."""
    import os

    from app.tasks.ingest import InProcessJobStore, RedisJobStore

    cc = config.celery
    assert cc is not None  # caller guards
    dsn = os.environ.get(cc.result_backend_env_var, "").strip()
    if cc.always_eager or not dsn or dsn.startswith(("memory", "cache+memory")):
        return InProcessJobStore()
    import redis.asyncio as aioredis

    # Bound the client's socket ops so a hung Redis backend can never hang enqueue/status; the store
    # additionally caps each call with asyncio.wait_for and logs faults (never silently swallows).
    timeout = config.cache.breaker.timeout_s if config.cache is not None else 2.0
    client = aioredis.Redis.from_url(dsn, socket_timeout=timeout, socket_connect_timeout=timeout)
    prefix = config.cache.key_prefix if config.cache is not None else "apex"
    return RedisJobStore(
        client, key_prefix=prefix, ttl_s=cc.result_ttl_s, timeout_s=timeout,
        log_path=config.paths.data_quality_log,
    )


def load_state(config: AppConfig) -> AppState:
    """Best-effort startup load of every handle + the DB engine. A failure → ``None`` (logged via
    ``_try_load``; the app boots and ``/health`` reports the outage as 503). Heavy imports are local
    to keep this module import-light."""
    from app.db.breaker import make_db_breaker
    from app.db.session import make_engine, make_session_factory
    from app.ml.explain import build_explainer
    from app.ml.inference import load_model
    from app.ml.stylespace import load_style_space
    from app.monitoring.detector import load_detector
    from app.personalisation.engine import load_personalisation

    lp = config.paths.data_quality_log
    state = AppState(config=config)

    state.model_handle = _try_load("classifier", lambda: load_model(config=config), log_path=lp)
    if state.model_handle is not None:
        handle = state.model_handle
        state.explainer = _try_load(
            "shap", lambda: build_explainer(handle, config=config), log_path=lp
        )
    state.style_handle = _try_load(
        "stylespace", lambda: load_style_space(config=config), log_path=lp
    )
    state.monitor_handle = _try_load(
        "degradation", lambda: load_detector(config=config), log_path=lp
    )
    state.personalisation_handle = _try_load(
        "personalisation", lambda: load_personalisation(config=config), log_path=lp
    )
    # Dataset + evaluation facts for /drivers. Absent artefact → None → null profile fields; the
    # endpoint keeps serving code/team/radar from the DB either way.
    state.driver_profiles = _try_load(
        "driver_profiles", lambda: load_driver_profiles(config=config, log_path=lp), log_path=lp
    )

    # DB engine (env-only DSN; absent → engine None → /health reports db down).
    def _build_db() -> tuple[AsyncEngine, async_sessionmaker[AsyncSession], DbCircuitBreaker]:
        engine = make_engine(config)
        return engine, make_session_factory(engine), make_db_breaker(config)

    db = _try_load("database", _build_db, log_path=lp)
    if db is not None:
        state.engine, state.session_factory, state.db_breaker = db

    # F14 cache (best-effort; a down/absent Redis degrades — cache is NEVER critical).
    from app.cache.redis import build_cache

    state.cache = _try_load("cache", lambda: build_cache(config, log_path=lp), log_path=lp)

    # F14 Celery queue + job store (best-effort). Absent celery: block → queue endpoints 503.
    if config.celery is not None:
        from app.tasks.celery import make_celery_app

        state.celery_app = _try_load("celery", lambda: make_celery_app(config), log_path=lp)
        state.job_store = _try_load(
            "job_store", lambda: _build_job_store(config, state.cache), log_path=lp
        )

    return state
