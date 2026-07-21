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
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

    from app.db.breaker import DbCircuitBreaker
    from app.ml.explain import ExplanationResult, ShapExplainer
    from app.ml.inference import ModelHandle, PredictionResult
    from app.ml.stylespace import ProjectionResult, StyleSpaceHandle, StyleSpaceResult
    from app.monitoring.detector import MonitorHandle
    from app.personalisation.engine import PersonalisationHandle, SetupRecommendation


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

    # DB engine (env-only DSN; absent → engine None → /health reports db down).
    def _build_db() -> tuple[AsyncEngine, async_sessionmaker[AsyncSession], DbCircuitBreaker]:
        engine = make_engine(config)
        return engine, make_session_factory(engine), make_db_breaker(config)

    db = _try_load("database", _build_db, log_path=lp)
    if db is not None:
        state.engine, state.session_factory, state.db_breaker = db

    return state
