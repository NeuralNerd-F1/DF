"""The database health probe (consumed by F13's ``/health`` ``db_connected``).

``check_database`` is **never-raising** (mirrors F06b/F10 health): every fault — unreachable DB,
an open breaker, a schema at the wrong Alembic revision, a missing ``database:`` block — becomes a
typed ``DbHealthResult`` with ``HEALTH_FAIL``, never an exception. ``status=OK`` requires BOTH
connectivity AND the applied migration revision equalling the migration head (a DB reachable but
un-migrated is a failure, not a pass — §8.1 "every component has a defined health check").
"""

from __future__ import annotations

import time
from typing import Any

import sqlalchemy as sa
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.contracts.config import AppConfig
from app.contracts.enums import Status
from app.contracts.errors import ErrorCode
from app.db.breaker import DbCircuitBreaker, make_db_breaker
from app.db.results import DatabaseUnavailableError, DbHealthResult


def _head_revision(script_location: str) -> str | None:
    """The migration head revision id from the ``migrations/`` scripts (sync, no DB)."""
    cfg = Config()
    cfg.set_main_option("script_location", script_location)
    return ScriptDirectory.from_config(cfg).get_current_head()


def _current_revision(connection: Any) -> str | None:
    """The revision the DB is stamped at, or ``None`` if never migrated (does not raise)."""
    return MigrationContext.configure(connection).get_current_revision()


async def _probe(session_factory: async_sessionmaker[AsyncSession]) -> str | None:
    """Ping the DB (``SELECT 1``) and read its current Alembic revision in one connection."""
    async with session_factory() as session:
        await session.execute(sa.text("SELECT 1"))
        connection = await session.connection()
        return await connection.run_sync(_current_revision)


async def check_database(
    session_factory: async_sessionmaker[AsyncSession],
    config: AppConfig,
    *,
    breaker: DbCircuitBreaker | None = None,
    script_location: str | None = None,
) -> DbHealthResult:
    """Probe DB connectivity + schema freshness. Never raises — returns a typed result.

    ``OK`` iff connected and ``revision == head``; otherwise ``FAILED`` + ``HEALTH_FAIL`` with a
    ``detail`` saying which (unreachable / stale schema / missing config).
    """
    if config.database is None:
        return DbHealthResult(
            status=Status.FAILED,
            connected=False,
            error_code=ErrorCode.HEALTH_FAIL,
            detail="database: block is not configured",
        )

    location = script_location or config.database.migrations_path
    try:
        head = _head_revision(location)
    except Exception as exc:  # a broken migrations dir must not crash the health check
        return DbHealthResult(
            status=Status.FAILED,
            connected=False,
            error_code=ErrorCode.HEALTH_FAIL,
            detail=f"cannot read migration head: {exc}",
        )

    br = breaker if breaker is not None else make_db_breaker(config)
    start = time.monotonic()
    try:
        current = await br.run(lambda: _probe(session_factory))
    except DatabaseUnavailableError as exc:
        return DbHealthResult(
            status=Status.FAILED,
            connected=False,
            head_revision=head,
            error_code=ErrorCode.HEALTH_FAIL,
            detail=f"database unavailable ({'breaker open' if exc.short_circuited else 'down'})",
        )
    except Exception as exc:  # any other probe fault -> HEALTH_FAIL, never propagate (never-raises)
        return DbHealthResult(
            status=Status.FAILED,
            connected=False,
            head_revision=head,
            error_code=ErrorCode.HEALTH_FAIL,
            detail=f"health probe failed: {exc}",
        )
    latency_ms = (time.monotonic() - start) * 1000.0

    if current != head:
        return DbHealthResult(
            status=Status.FAILED,
            connected=True,
            revision=current,
            head_revision=head,
            latency_ms=latency_ms,
            error_code=ErrorCode.HEALTH_FAIL,
            detail=f"schema revision {current!r} != migration head {head!r}",
        )
    return DbHealthResult(
        status=Status.OK,
        connected=True,
        revision=current,
        head_revision=head,
        latency_ms=latency_ms,
    )
