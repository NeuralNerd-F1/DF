"""Shared fixtures for the F13 API suite.

No committed model artefacts exist, so endpoint tests inject **fake handles** + **canned internal
results** through ``app.dependency_overrides`` (the runner providers in ``app.api.deps``). The DB is
a real in-memory aiosqlite (migrated via ``create_all``); all tests are ``async`` so the engine and
the ASGI app share ONE event loop (a sync ``TestClient`` + aiosqlite would cross event loops). The
app is driven in-process by ``httpx.AsyncClient`` + ``ASGITransport`` — no live uvicorn, no lifespan
(``get_apex_state`` is overridden, so ``load_state`` never runs).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest_asyncio
from app.api.deps import AppState, get_apex_state
from app.api.main import create_app
from app.contracts.config import load_config
from app.contracts.enums import ConfidenceState, Status
from app.db.base import Base
from app.db.breaker import make_db_breaker
from app.db.health import _head_revision
from app.ml.explain import Contribution, ExplanationResult
from app.ml.inference import PredictionResult
from app.ml.stylespace import Neighbour, ProjectionResult, StyleSpaceResult
from app.personalisation.engine import (
    SetupConfidence,
    SetupOutputKind,
    SetupParameter,
    SetupRecommendation,
)
from httpx import ASGITransport, AsyncClient
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool


async def seed_fingerprint(
    session_factory: object,
    *,
    code: str,
    season: int = 2023,
    radar: dict[str, float] | None = None,
    shap_global: dict[str, float] | None = None,
) -> None:
    """Insert one driver (code, season) + circuit + fingerprint row into the test DB."""
    from app.db import models

    async with session_factory() as s:  # type: ignore[operator]
        driver = models.Driver(code=code, full_name=code, season=season)
        circuit = models.Circuit(
            name=f"Circuit-{code}-{season}", type=models.CircuitType.HIGH_SPEED
        )
        s.add_all([driver, circuit])
        await s.flush()
        s.add(
            models.Fingerprint(
                driver_id=driver.id,
                season=season,
                circuit_id=circuit.id,
                vector={"d0": 0.1, "d1": 0.2, "d2": 0.3, "d3": 0.4, "d4": 0.5},
                radar_values=radar
                or {
                    "throttle": 0.5,
                    "braking": 0.3,
                    "speed": 0.7,
                    "gear": 0.4,
                    "engine": 0.6,
                    "cornering": 0.5,
                },
                shap_global=shap_global or {"throttle": 0.5, "braking": 0.3},
            )
        )
        await s.commit()


@dataclass
class FakeHandle:
    """A minimal stand-in for every model handle: just what each ``health()`` reads (+ drivers)."""

    ready: bool = True
    model_version: str = "test-v1"
    engine: str = "umap"
    drivers: list[str] = field(default_factory=lambda: ["VER", "HAM"])


# --- canned internal results (the route maps these to HTTP; the model compute is faked) ----------
def fake_prediction(
    *,
    top: str | None = "VER",
    confidence: float = 0.91,
    state: ConfidenceState | None = ConfidenceState.HIGH,
    status: Status = Status.OK,
    error_code: object = None,
) -> PredictionResult:
    return PredictionResult(
        status=status,
        error_code=error_code,  # type: ignore[arg-type]
        probabilities={"VER": confidence, "HAM": 1 - confidence},
        top_prediction=top,
        confidence=confidence,
        confidence_state=state,
        top_3=[("VER", confidence), ("HAM", 1 - confidence)],
        model_version="test-v1",
    )


def fake_explanation(*, shap_available: bool = True) -> ExplanationResult:
    return ExplanationResult(
        shap_values={"F01": 0.2} if shap_available else None,
        base_value=0.1,
        predicted_class="VER",
        force=[
            Contribution(
                feature_id="F01", name="throttle", shap=0.2, feature_value=1.0, direction="toward"
            )
        ],
        radar={"throttle": 0.5},
        dominant_domains=["throttle"],
        nl_description="Aggressive on throttle.",
        shap_available=shap_available,
        consistency_ok=True,
        consistency_delta=0.0,
    )


def fake_projection(
    *, sparse: bool = False, ood: bool = False, coords: list[float] | None = None
) -> ProjectionResult:
    return ProjectionResult(
        coords_5d=coords if coords is not None else [0.1, 0.2, 0.3, 0.4, 0.5],
        coords_2d=[0.1, 0.2],
        nearest_neighbours=[Neighbour(driver="VER", cosine_similarity=0.9, distance=0.1)],
        confidence_state=ConfidenceState.HIGH,
        sparse=sparse,
        ood=ood,
        ood_max_pc_sd=4.0 if ood else 1.0,
        projection_engine="umap",
        risk_flags=["F09"] if ood else [],
        n_laps_used=10,
        laps_rejected=1,
    )


def fake_setup(driver: str = "VER", circuit: str = "Monaco") -> SetupRecommendation:
    param = SetupParameter(
        name="front_wing_angle",
        kind=SetupOutputKind.DEGREES,
        value=5.0,
        confidence=0.8,
        rating=SetupConfidence.HIGH,
        unit="deg",
        fia_min=0.0,
        fia_max=12.0,
    )
    return SetupRecommendation(
        driver=driver,
        circuit=circuit,
        circuit_type=None,
        parameters=[param],
        approximation_disclaimer="Setup targets are approximated from telemetry, not measured.",
        model_version="test-v1",
    )


def fake_stylespace(drivers: list[str] | None = None) -> StyleSpaceResult:
    drivers = drivers if drivers is not None else ["VER", "HAM"]
    return StyleSpaceResult(
        drivers=drivers,
        coords_5d={d: [0.1, 0.2, 0.3, 0.4, 0.5] for d in drivers},
        coords_2d={d: [0.1, 0.2] for d in drivers},
        engine="umap",
    )


@pytest_asyncio.fixture
async def api(tmp_path: Path, repo_config_path: Path) -> AsyncIterator[SimpleNamespace]:
    """A ready-to-drive API context: httpx client + app + injected fake state + DB factory."""
    config = load_config(repo_config_path)
    config.paths.data_quality_log = str(tmp_path / "dq.jsonl")  # contain any startup logging

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
        # Stamp the migration head so check_database (F12) reports OK (create_all leaves none).
        await conn.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"))
        await conn.execute(
            text("INSERT INTO alembic_version (version_num) VALUES (:v)"), {"v": head}
        )
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    breaker = make_db_breaker(config)

    state = AppState(
        config=config,
        model_handle=FakeHandle(),
        explainer=None,  # SHAP off by default; /predict tests override get_explainer_fn
        style_handle=FakeHandle(),
        monitor_handle=FakeHandle(),
        personalisation_handle=FakeHandle(),
        engine=engine,
        session_factory=session_factory,
        db_breaker=breaker,
    )

    app = create_app(config)
    app.dependency_overrides[get_apex_state] = lambda: state

    # raise_app_exceptions=False: the global Exception handler returns a typed 500 — the transport
    # must surface THAT response, not re-raise (so the API_ERROR backstop is testable).
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield SimpleNamespace(
            client=client,
            app=app,
            state=state,
            session_factory=session_factory,
            breaker=breaker,
            config=config,
        )
    await engine.dispose()
