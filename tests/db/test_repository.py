"""Repository writes: the data_quality_log DB sink, the lap_features invariant, generic persist."""

from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy as sa
from app.contracts.config import AppConfig
from app.contracts.enums import Status
from app.contracts.errors import ErrorCode
from app.contracts.logging import ROW_COLUMNS, log_data_quality
from app.db import models, repository
from app.db.breaker import make_db_breaker
from app.db.results import DatabaseUnavailableError, DbIntegrityError
from app.db.session import session_scope
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


async def _seed_lap(session: AsyncSession) -> models.Lap:
    driver = models.Driver(code="VER", full_name="Max", season=2023)
    circuit = models.Circuit(name="Monza", type=models.CircuitType.HIGH_SPEED)
    session.add_all([driver, circuit])
    await session.flush()
    race = models.RaceSession(circuit_id=circuit.id, season=2023, round=1, session_type="R")
    session.add(race)
    await session.flush()
    lap = models.Lap(
        session_id=race.id, driver_id=driver.id, lap_number=1, is_clean=True, error_code="OK"
    )
    session.add(lap)
    await session.flush()
    return lap


# --- data_quality_log DB sink ------------------------------------------------------------------
async def test_write_quality_log_row_persists(db_session: AsyncSession) -> None:
    res = await repository.write_quality_log_row(
        db_session,
        error_code=ErrorCode.LAP_003,
        entity_type="lap",
        entity_id="5",
        error_detail="10% consecutive NaN in Speed",
        fallback_used=False,
    )
    assert res.status is Status.OK
    assert res.written is True
    assert res.entity_type == "data_quality_log"
    row = (await db_session.execute(sa.select(models.DataQualityLog))).scalar_one()
    assert row.error_code == "LAP_003"  # stored as the ErrorCode .value
    assert row.entity_id == "5"
    assert row.fallback_used is False


def test_data_quality_log_columns_match_logging_row_columns() -> None:
    """The DB table is a drop-in for the F00 JSONL writer: same 7 columns, same order."""
    cols = tuple(c.name for c in models.DataQualityLog.__table__.columns)
    assert cols == ROW_COLUMNS


def test_repository_exposes_no_delete_for_quality_log() -> None:
    """§8.4 'Never deleted' — the module offers no delete surface."""
    assert not [name for name in dir(repository) if "delete" in name.lower()]


# --- lap_features invariant --------------------------------------------------------------------
async def test_insert_lap_features_computes_null_count(db_session: AsyncSession) -> None:
    lap = await _seed_lap(db_session)
    res = await repository.insert_lap_features(
        db_session,
        lap_id=lap.id,
        feature_vector={"F01": 1.0, "F02": None, "F03": None},
        imputed_features=["F02", "F03"],
    )
    assert res.written is True
    row = (await db_session.execute(sa.select(models.LapFeatures))).scalar_one()
    assert row.null_count == 2  # computed from the vector


async def test_insert_lap_features_rejects_null_count_mismatch(db_session: AsyncSession) -> None:
    lap = await _seed_lap(db_session)
    with pytest.raises(DbIntegrityError) as exc:
        await repository.insert_lap_features(
            db_session,
            lap_id=lap.id,
            feature_vector={"F01": 1.0, "F02": None},  # 1 null
            null_count=5,  # ...disagrees -> rejected before any DB round-trip
        )
    assert exc.value.error_code is ErrorCode.DB_INTEGRITY


# --- generic persist ---------------------------------------------------------------------------
async def test_persist_returns_entity_id(db_session: AsyncSession) -> None:
    res = await repository.persist(
        db_session, models.Driver(code="HAM", full_name="Lewis", season=2023)
    )
    assert res.written is True
    assert res.entity_id is not None
    assert res.entity_type == "drivers"


async def test_persist_duplicate_raises_db_integrity(db_session: AsyncSession) -> None:
    await repository.persist(db_session, models.Driver(code="LEC", full_name="A", season=2023))
    with pytest.raises(DbIntegrityError) as exc:
        await repository.persist(
            db_session, models.Driver(code="LEC", full_name="B", season=2023)
        )
    assert exc.value.error_code is ErrorCode.DB_INTEGRITY


# --- no silent data loss: DB sink down -> JSONL fallback ----------------------------------------
async def test_quality_log_db_down_falls_back_to_jsonl_no_loss(
    session_factory: async_sessionmaker[AsyncSession], db_config: AppConfig, tmp_path: Path
) -> None:
    """The audit trail is the spine of #1 — if the DB sink is unavailable, the JSONL writer catches
    the row so nothing is silently lost (F13/F14 wire this routing; F12 proves the contract)."""
    breaker = make_db_breaker(db_config)
    for _ in range(3):  # open the breaker (open_after=3) -> DB unavailable
        breaker.record_failure()
    jsonl = tmp_path / "dq.jsonl"

    wrote_to_db = False
    try:
        async with session_scope(session_factory, breaker) as s:
            await repository.write_quality_log_row(
                s,
                error_code=ErrorCode.LAP_003,
                entity_type="lap",
                entity_id="5",
                error_detail="x",
                fallback_used=False,
            )
            wrote_to_db = True
    except DatabaseUnavailableError:
        log_data_quality(
            error_code=ErrorCode.LAP_003,
            entity_type="lap",
            entity_id="5",
            error_detail="db sink unavailable; fell back to jsonl",
            fallback_used=True,
            log_path=jsonl,
        )

    assert wrote_to_db is False
    assert jsonl.exists() and jsonl.read_text(encoding="utf-8").strip()  # the row landed in JSONL
