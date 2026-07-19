"""Engine build (env DSN + CONFIG_001) and session_scope (commit / rollback / breaker)."""

from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy as sa
from app.contracts.config import AppConfig, ConfigError, load_config
from app.db import models
from app.db.breaker import make_db_breaker
from app.db.results import DatabaseUnavailableError
from app.db.session import make_engine, make_session_factory, session_scope
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.ext.asyncio import async_sessionmaker

SQLITE_MEMORY = "sqlite+aiosqlite:///:memory:"


# --- make_engine -------------------------------------------------------------------------------
def test_make_engine_with_explicit_dsn(db_config: AppConfig) -> None:
    engine = make_engine(db_config, dsn=SQLITE_MEMORY)
    assert engine.url.get_backend_name() == "sqlite"


def test_make_engine_reads_env_var(
    db_config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(db_config.database.dsn_env_var, SQLITE_MEMORY)  # type: ignore[union-attr]
    engine = make_engine(db_config)
    assert engine.url.get_backend_name() == "sqlite"


def test_make_engine_missing_dsn_raises_config_001(
    db_config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(db_config.database.dsn_env_var, raising=False)  # type: ignore[union-attr]
    with pytest.raises(ConfigError) as exc:
        make_engine(db_config)
    assert exc.value.error_code.value == "CONFIG_001"


def test_make_engine_requires_database_block(valid_config_file: Path) -> None:
    cfg = load_config(valid_config_file)
    assert cfg.database is None
    with pytest.raises(ConfigError):
        make_engine(cfg)


# --- session_scope -----------------------------------------------------------------------------
async def test_session_scope_commits_on_success(
    session_factory: async_sessionmaker, db_config: AppConfig
) -> None:
    breaker = make_db_breaker(db_config)
    async with session_scope(session_factory, breaker) as s:
        s.add(models.Driver(code="VER", full_name="Max", season=2023))
    # New session sees the committed row.
    async with session_factory() as s2:
        n = (await s2.execute(sa.select(sa.func.count()).select_from(models.Driver))).scalar_one()
    assert n == 1
    assert breaker.failure_count() == 0


async def test_session_scope_rolls_back_on_exception_no_partial_write(
    session_factory: async_sessionmaker, db_config: AppConfig
) -> None:
    breaker = make_db_breaker(db_config)
    with pytest.raises(RuntimeError):
        async with session_scope(session_factory, breaker) as s:
            s.add(models.Driver(code="HAM", full_name="Lewis", season=2023))
            await s.flush()  # row is in the transaction...
            raise RuntimeError("boom")  # ...then we blow up
    # Nothing was persisted — the whole unit of work rolled back.
    async with session_factory() as s2:
        n = (await s2.execute(sa.select(sa.func.count()).select_from(models.Driver))).scalar_one()
    assert n == 0


async def test_session_scope_reraises_integrity_error(
    session_factory: async_sessionmaker, db_config: AppConfig
) -> None:
    """Constraint violation on commit → rollback + re-raise (repository maps to DB_INTEGRITY)."""
    breaker = make_db_breaker(db_config)
    async with session_factory() as seed:
        seed.add(models.Driver(code="LEC", full_name="Charles", season=2023))
        await seed.commit()
    with pytest.raises(IntegrityError):
        async with session_scope(session_factory, breaker) as s:
            s.add(models.Driver(code="LEC", full_name="Dup", season=2023))  # dup (code, season)
    assert breaker.failure_count() == 0  # a data fault is not a transport failure


async def test_session_scope_short_circuits_when_breaker_open(
    session_factory: async_sessionmaker, db_config: AppConfig
) -> None:
    breaker = make_db_breaker(db_config)
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_failure()  # open_after=3 -> open
    assert breaker.is_open() is True
    with pytest.raises(DatabaseUnavailableError) as exc:
        async with session_scope(session_factory, breaker):
            pass
    assert exc.value.short_circuited is True


class _BoomSession:
    """A stand-in session whose commit fails with a connection error (DB down at commit)."""

    def __init__(self) -> None:
        self.rolled_back = False
        self.closed = False

    async def commit(self) -> None:
        raise OperationalError("COMMIT", {}, Exception("connection lost"))

    async def rollback(self) -> None:
        self.rolled_back = True

    async def close(self) -> None:
        self.closed = True


async def test_session_scope_connection_error_becomes_db_unavailable(db_config: AppConfig) -> None:
    boom = _BoomSession()
    breaker = make_db_breaker(db_config)

    def factory() -> _BoomSession:
        return boom

    with pytest.raises(DatabaseUnavailableError) as exc:
        async with session_scope(factory, breaker):  # type: ignore[arg-type]
            pass
    assert exc.value.short_circuited is False
    assert isinstance(exc.value.cause, OperationalError)
    assert boom.rolled_back is True  # rolled back before surfacing (no partial write)
    assert boom.closed is True
    assert breaker.failure_count() == 1  # the outage tripped the breaker


class _CloseBoomSession:
    """A stand-in session whose close() fails (closing an already-dead connection)."""

    def __init__(self) -> None:
        self.rolled_back = False

    async def commit(self) -> None:
        pass

    async def rollback(self) -> None:
        self.rolled_back = True

    async def close(self) -> None:
        raise OSError("close on a dead socket")


async def test_session_scope_close_error_does_not_mask_primary(db_config: AppConfig) -> None:
    """A failing close() in the finally must NOT replace the already-determined typed error."""
    breaker = make_db_breaker(db_config)

    def factory() -> _CloseBoomSession:
        return _CloseBoomSession()

    # The body raises RuntimeError; close() then raises OSError. RuntimeError must win.
    with pytest.raises(RuntimeError, match="primary"):
        async with session_scope(factory, breaker):  # type: ignore[arg-type]
            raise RuntimeError("primary")


def test_make_session_factory_returns_sessionmaker(db_config: AppConfig) -> None:
    engine = make_engine(db_config, dsn=SQLITE_MEMORY)
    factory = make_session_factory(engine)
    assert isinstance(factory, async_sessionmaker)
