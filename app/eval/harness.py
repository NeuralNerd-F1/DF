"""Assembly + fault-injection + measurement primitives for the F15 RQ5 failsafe harness (§11.4).

Self-contained (never imports from ``tests/``): builds the real F13 ``create_app`` over an in-memory
aiosqlite DB with **fake** model handles injected via ``dependency_overrides`` (no champion artefact
is committed — F13's constraint), so the harness drives the assembled system in-process via
``httpx.ASGITransport`` and injects faults per test. F20 reuses these primitives + the suite.
"""

from __future__ import annotations

import statistics
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from app.contracts.enums import ConfidenceState, Status

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from httpx import AsyncClient
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

    from app.contracts.config import AppConfig
    from app.db.breaker import DbCircuitBreaker
    from app.ml.inference import PredictionResult
    from app.ml.stylespace import ProjectionResult


@dataclass
class _FakeHandle:
    """A minimal stand-in for every model handle (only what each ``health()`` reads)."""

    ready: bool = True
    model_version: str = "harness-v1"
    engine: str = "umap"
    drivers: list[str] = field(default_factory=lambda: ["VER", "HAM"])


def fake_prediction(
    *,
    top: str | None = "VER",
    confidence: float = 0.91,
    state: ConfidenceState | None = ConfidenceState.HIGH,
    status: Status = Status.OK,
    error_code: Any = None,
) -> PredictionResult:
    """A canned :class:`PredictionResult` the fake predictor returns (deterministic → Test 7)."""
    from app.ml.inference import PredictionResult

    return PredictionResult(
        status=status,
        error_code=error_code,
        probabilities={"VER": confidence, "HAM": 1 - confidence},
        top_prediction=top,
        confidence=confidence,
        confidence_state=state,
        top_3=[("VER", confidence), ("HAM", 1 - confidence)],
        model_version="harness-v1",
    )


def fake_projection(*, ood: bool = False, coords: list[float] | None = None) -> ProjectionResult:
    """A canned :class:`ProjectionResult`; ``ood=True`` drives Test 6 (computed + flagged)."""
    from app.ml.stylespace import Neighbour, ProjectionResult

    return ProjectionResult(
        coords_5d=coords if coords is not None else [0.1, 0.2, 0.3, 0.4, 0.5],
        coords_2d=[0.1, 0.2],
        nearest_neighbours=[Neighbour(driver="VER", cosine_similarity=0.9, distance=0.1)],
        confidence_state=ConfidenceState.OUT_OF_DISTRIBUTION if ood else ConfidenceState.HIGH,
        sparse=False,
        ood=ood,
        ood_max_pc_sd=6.0 if ood else 1.0,
        projection_engine="umap",
        risk_flags=["F09"] if ood else [],
        n_laps_used=10,
        laps_rejected=0,
    )


@dataclass
class Harness:
    """A ready-to-drive assembled system: an httpx client + the app + its injected state + DB."""

    client: AsyncClient
    app: Any
    state: Any
    session_factory: async_sessionmaker[AsyncSession]
    db_breaker: DbCircuitBreaker
    engine: AsyncEngine


@asynccontextmanager
async def assemble(
    config: AppConfig,
    *,
    cache: Any = None,
    predictor: Callable[..., PredictionResult] | None = None,
    projector: Callable[..., ProjectionResult] | None = None,
) -> AsyncIterator[Harness]:
    """Assemble the real app over in-memory aiosqlite with fake handles + optional fault injection.

    ``cache`` (a ``RedisCache`` or ``None``) is placed on the state; ``predictor``/``projector``
    override the model runners with canned results. Yields a :class:`Harness`; disposes on exit.
    """
    from httpx import ASGITransport, AsyncClient
    from sqlalchemy import event, text
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import StaticPool

    from app.api.deps import AppState, get_apex_state, get_predictor, get_projector
    from app.api.main import create_app
    from app.db.base import Base
    from app.db.breaker import make_db_breaker
    from app.db.health import _head_revision

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine.sync_engine, "connect")
    def _fk_on(dbapi_conn: object, _rec: object) -> None:
        cur = dbapi_conn.cursor()  # type: ignore[attr-defined]
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    head = _head_revision(config.database.migrations_path) if config.database else None
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Stamp the migration head so F12's check_database reports OK (create_all leaves none).
        await conn.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"))
        await conn.execute(
            text("INSERT INTO alembic_version (version_num) VALUES (:v)"), {"v": head}
        )
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    breaker = make_db_breaker(config)

    fake: Any = _FakeHandle()  # a fake stands in for every typed model handle (no artefacts)
    state = AppState(
        config=config,
        model_handle=fake,
        style_handle=fake,
        monitor_handle=fake,
        personalisation_handle=fake,
        engine=engine,
        session_factory=session_factory,
        db_breaker=breaker,
        cache=cache,
    )
    app = create_app(config)
    app.dependency_overrides[get_apex_state] = lambda: state
    if predictor is not None:
        app.dependency_overrides[get_predictor] = lambda: predictor
    if projector is not None:
        app.dependency_overrides[get_projector] = lambda: projector

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    client = AsyncClient(transport=transport, base_url="http://testserver")
    try:
        yield Harness(
            client=client, app=app, state=state, session_factory=session_factory,
            db_breaker=breaker, engine=engine,
        )
    finally:
        await client.aclose()
        await engine.dispose()


# --- measurement -------------------------------------------------------------------------------
async def measure_post(
    client: AsyncClient, path: str, body: dict[str, Any], n: int
) -> tuple[list[float], list[int]]:
    """Fire ``n`` sequential POSTs, returning (per-request seconds, status codes)."""
    times: list[float] = []
    statuses: list[int] = []
    for _ in range(n):
        t0 = time.perf_counter()
        r = await client.post(path, json=body)
        times.append(time.perf_counter() - t0)
        statuses.append(r.status_code)
    return times, statuses


def p95(samples: list[float]) -> float:
    """95th-percentile of ``samples`` (seconds). Inclusive method; single-sample → that sample."""
    if not samples:
        return 0.0
    if len(samples) < 2:
        return samples[0]
    return statistics.quantiles(samples, n=100, method="inclusive")[94]
