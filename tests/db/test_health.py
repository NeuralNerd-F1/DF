"""The DB health probe: OK on a migrated DB, HEALTH_FAIL on stale schema / down DB / no config."""

from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from app.contracts.config import AppConfig, load_config
from app.contracts.enums import Status
from app.contracts.errors import ErrorCode
from app.db.breaker import make_db_breaker
from app.db.health import check_database
from app.db.session import make_engine, make_session_factory


@pytest.fixture
def migrated_db_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A temp-file SQLite DB migrated to head. Sync fixture so ``command.upgrade``'s internal
    ``asyncio.run`` runs BEFORE the async test's event loop is entered (they cannot nest)."""
    db_file = tmp_path / "apex.db"
    monkeypatch.setenv("APEX_DATABASE_URL", f"sqlite+aiosqlite:///{db_file}")
    cfg = Config("alembic.ini")
    cfg.set_main_option("script_location", "migrations")
    command.upgrade(cfg, "head")
    return db_file


async def test_health_ok_on_migrated_db(migrated_db_file: Path, db_config: AppConfig) -> None:
    engine = make_engine(db_config, dsn=f"sqlite+aiosqlite:///{migrated_db_file}")
    res = await check_database(make_session_factory(engine), db_config)
    await engine.dispose()
    assert res.status is Status.OK
    assert res.connected is True
    assert res.revision == res.head_revision == "0001"
    assert res.error_code is None
    assert res.latency_ms is not None


async def test_health_stale_revision_is_health_fail(
    migrated_db_file: Path, db_config: AppConfig
) -> None:
    engine = make_engine(db_config, dsn=f"sqlite+aiosqlite:///{migrated_db_file}")
    # Stamp the DB at a bogus revision — schema present but out of date.
    async with engine.begin() as conn:
        await conn.execute(sa.text("UPDATE alembic_version SET version_num = 'deadbeef'"))
    res = await check_database(make_session_factory(engine), db_config)
    await engine.dispose()
    assert res.status is Status.FAILED
    assert res.connected is True
    assert res.revision == "deadbeef"
    assert res.head_revision == "0001"
    assert res.error_code is ErrorCode.HEALTH_FAIL


async def test_health_db_down_is_health_fail_never_raises(
    migrated_db_file: Path, db_config: AppConfig
) -> None:
    engine = make_engine(db_config, dsn=f"sqlite+aiosqlite:///{migrated_db_file}")
    breaker = make_db_breaker(db_config)
    for _ in range(3):  # open the breaker -> probe short-circuits
        breaker.record_failure()
    res = await check_database(make_session_factory(engine), db_config, breaker=breaker)
    await engine.dispose()
    assert res.status is Status.FAILED
    assert res.connected is False
    assert res.error_code is ErrorCode.HEALTH_FAIL


async def test_health_missing_database_block_is_health_fail(valid_config_file: Path) -> None:
    cfg = load_config(valid_config_file)
    assert cfg.database is None
    res = await check_database(None, cfg)  # type: ignore[arg-type]
    assert res.status is Status.FAILED
    assert res.connected is False
    assert res.error_code is ErrorCode.HEALTH_FAIL
