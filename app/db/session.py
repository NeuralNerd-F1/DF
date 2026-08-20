"""Async engine, session factory, and the transactional ``session_scope``.

The connection layer. Three fail-loud guarantees (non-negotiables #1, §8.3):
- **DSN is env-only** — ``make_engine`` reads it from the env var named by ``database.dsn_env_var``;
  an absent/empty DSN is a build-breaking ``ConfigError`` (``CONFIG_001``), never a silent ``None``
  engine or a committed secret.
- **No partial writes** — ``session_scope`` commits on clean exit and rolls back on ANY exception.
- **DB outage is typed** — a connection-level fault becomes ``DatabaseUnavailableError``
  (``DB_UNAVAILABLE``) and trips the breaker; an open breaker fast-fails the next scope with no
  connection attempt.

The breaker's *retry* loop (``DbCircuitBreaker.run``) is for idempotent operations (health
``SELECT 1``, reads); a write transaction is NOT safely retryable, so ``session_scope`` uses the
breaker's state machine (``is_open`` / ``record_failure`` / ``record_success``) directly, never
re-running a commit.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Any

from sqlalchemy.exc import InterfaceError, OperationalError
from sqlalchemy.exc import TimeoutError as SATimeoutError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from app.contracts.config import AppConfig, ConfigError
from app.db.breaker import DbCircuitBreaker
from app.db.results import DatabaseUnavailableError

# Connection-level faults mapped to DB_UNAVAILABLE (SQLAlchemy's pool TimeoutError included). A data
# fault (IntegrityError etc.) is deliberately NOT here — it rolls back and propagates unchanged to
# the caller.
_CONNECTION_ERRORS: tuple[type[BaseException], ...] = (
    OperationalError,
    InterfaceError,
    SATimeoutError,
)


def _enable_sqlite_fk(engine: AsyncEngine) -> None:
    """Turn on FK enforcement for SQLite (off by default) so offline behaviour matches Postgres."""
    from sqlalchemy import event

    @event.listens_for(engine.sync_engine, "connect")
    def _fk_on(dbapi_conn: Any, _rec: Any) -> None:
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()


def make_engine(config: AppConfig, *, dsn: str | None = None) -> AsyncEngine:
    """Build the async engine. DSN comes from ``dsn`` (tests) else the ``dsn_env_var`` environment
    variable; an absent DSN or missing ``database:`` block raises ``ConfigError`` (``CONFIG_001``).

    SQLite (the offline suite) uses a shared ``StaticPool`` + FK pragma; PostgreSQL uses the
    configured pool sizing and a server-side ``statement_timeout`` (§8.3) via psycopg ``options``.
    """
    if config.database is None:
        raise ConfigError("database: block is required to build the engine")
    db = config.database
    url = dsn if dsn is not None else os.environ.get(db.dsn_env_var, "")
    if not url:
        raise ConfigError(
            f"{db.dsn_env_var} is not set — no database DSN available (secret must come from env)"
        )

    if url.startswith("sqlite"):
        engine = create_async_engine(
            url,
            echo=db.echo,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        _enable_sqlite_fk(engine)
        return engine

    # PostgreSQL (psycopg v3 async). Two independent bounds so an unreachable host cannot hang a
    # write past the §8.3 budget: a server-side statement_timeout (per connection) AND a client-side
    # connect_timeout on the initial TCP handshake — pool_timeout only bounds the wait for a POOLED
    # connection, not the socket connect to a silently-unresponsive host. connect_timeout is bounded
    # by the breaker's per-attempt timeout (§8.3, 5s).
    statement_timeout_ms = int(db.statement_timeout_s * 1000)
    return create_async_engine(
        url,
        echo=db.echo,
        pool_size=db.pool_size,
        max_overflow=db.max_overflow,
        pool_timeout=db.pool_timeout_s,
        connect_args={
            "options": f"-c statement_timeout={statement_timeout_ms}",
            "connect_timeout": max(1, int(db.breaker.timeout_s)),
        },
    )


def make_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """An ``async_sessionmaker`` bound to ``engine`` (``expire_on_commit`` off — safe attribute
    access after commit in async code, no lazy refresh)."""
    return async_sessionmaker(engine, expire_on_commit=False)


async def _safe_rollback(session: AsyncSession) -> None:
    """Best-effort rollback in an error handler. A rollback that itself fails (e.g. the DB is the
    thing that's down) must not mask the original error — suppress the secondary fault only. This is
    cleanup, not a silently-dropped row, so it does not violate non-negotiable #1."""
    with suppress(Exception):
        await session.rollback()


@asynccontextmanager
async def session_scope(
    session_factory: async_sessionmaker[AsyncSession],
    breaker: DbCircuitBreaker,
) -> AsyncIterator[AsyncSession]:
    """A transactional unit of work: commit on clean exit, roll back on any error, fail-fast if the
    breaker is open.

    Raises :class:`DatabaseUnavailableError` (``DB_UNAVAILABLE``) when the breaker is open or a
    connection-level fault occurs — the transaction is rolled back first (**no partial write**). Any
    other exception (notably ``IntegrityError``) is rolled back and re-raised unchanged.
    """
    if breaker.is_open():
        raise DatabaseUnavailableError("postgres", short_circuited=True)

    session = session_factory()
    try:
        yield session
        await session.commit()
    except _CONNECTION_ERRORS as exc:
        await _safe_rollback(session)
        breaker.record_failure()
        raise DatabaseUnavailableError("postgres", short_circuited=False, cause=exc) from exc
    except BaseException:
        await _safe_rollback(session)
        raise
    else:
        breaker.record_success()
    finally:
        # Guard close() the same way as _safe_rollback: closing an already-dead connection can
        # itself raise (a termination packet over a dead socket), and that must NOT replace the
        # already-determined typed error (DB_UNAVAILABLE, or a re-raised body error).
        with suppress(Exception):
            await session.close()
