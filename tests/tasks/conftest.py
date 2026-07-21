"""Shared fixtures for the F14 Celery/ingestion suite — offline (aiosqlite + fake handles)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio
from app.contracts.config import AppConfig, load_config
from app.contracts.enums import ConfidenceState
from app.db.base import Base
from app.db.breaker import DbCircuitBreaker, make_db_breaker
from app.pipeline.features import FEATURE_IDS
from app.tasks.ingest import LapRecord
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool


@pytest.fixture
def cfg(repo_config_path: Path) -> AppConfig:
    return load_config(repo_config_path)


@pytest_asyncio.fixture
async def db_engine() -> AsyncIterator[AsyncEngine]:
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

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
def session_factory(db_engine: AsyncEngine) -> async_sessionmaker:
    return async_sessionmaker(db_engine, expire_on_commit=False)


@pytest.fixture
def db_breaker(cfg: AppConfig) -> DbCircuitBreaker:
    return make_db_breaker(cfg)


def make_records(
    *, driver: str = "HAM", n_clean: int = 5, n_rejected: int = 0, circuit: str = "Bahrain"
) -> list[LapRecord]:
    """Synthetic per-lap records (F01–F04 output stand-in)."""
    recs: list[LapRecord] = []
    lap = 1
    for _ in range(n_clean):
        fv = {fid: 0.1 * lap for fid in FEATURE_IDS}
        recs.append(_rec(driver, lap, circuit, True, "OK", fv))
        lap += 1
    for _ in range(n_rejected):
        fv = {fid: 0.1 * lap for fid in FEATURE_IDS}
        recs.append(_rec(driver, lap, circuit, False, "LAP_001", fv))
        lap += 1
    return recs


def _rec(
    driver: str, lap: int, circuit: str, clean: bool, code: str, fv: dict[str, float | None]
) -> LapRecord:
    nc = sum(1 for v in fv.values() if v is None)
    return LapRecord(
        driver_code=driver,
        driver_name=f"{driver} Driver",
        team="Team",
        season=2023,
        round=1,
        circuit_name=circuit,
        circuit_type="high_speed",
        session_type="R",
        lap_number=lap,
        tyre_compound="SOFT",
        tyre_age=lap,
        is_clean=clean,
        error_code=code,
        has_interpolation=False,
        feature_vector=fv,
        null_count=nc,
        imputed_features=[],
    )


def fake_predict(top: str | None = "HAM"):
    def _run(fv: dict[str, float | None], circuit: str) -> SimpleNamespace:
        return SimpleNamespace(
            top_prediction=top,
            confidence=0.9,
            confidence_state=ConfidenceState.HIGH,
            shap_values={"F01": 0.2},
            shap_available=True,
        )

    return _run


def fake_project(coords: list[float] | None = None):
    def _run(frame: object, circuit: str, ctype: str | None) -> SimpleNamespace:
        c5 = coords if coords is not None else [0.1, 0.2, 0.3, 0.4, 0.5]
        return SimpleNamespace(coords_5d=c5)

    return _run
