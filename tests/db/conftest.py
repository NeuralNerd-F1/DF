"""Shared fixtures for the F12 persistence-layer suite.

The whole offline suite runs on **in-memory aiosqlite** — no Postgres required (the house
"runs anywhere" rule). Two SQLite-specific details make it faithful to Postgres:

- ``StaticPool`` + ``check_same_thread=False`` so every async session shares ONE in-memory DB
  (a fresh ``:memory:`` per connection would see an empty schema).
- ``PRAGMA foreign_keys=ON`` per connection so FK constraints are actually enforced (SQLite leaves
  them off by default) — otherwise the FK acceptance criteria could never fire offline.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from app.contracts.config import AppConfig, load_config
from app.db.base import Base
from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

SQLITE_MEMORY = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def db_engine() -> AsyncIterator[AsyncEngine]:
    """A fresh in-memory aiosqlite engine with the schema created and FK enforcement on."""
    engine = create_async_engine(
        SQLITE_MEMORY,
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
def session_factory(db_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """An ``async_sessionmaker`` bound to the in-memory engine (expire_on_commit off for async)."""
    return async_sessionmaker(db_engine, expire_on_commit=False)


@pytest_asyncio.fixture
async def db_session(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """One open ``AsyncSession`` for a test to insert/read through."""
    async with session_factory() as session:
        yield session


@pytest.fixture
def db_config(repo_config_path: Path) -> AppConfig:
    """The real repo config (now carrying the F12 ``database:`` block)."""
    return load_config(repo_config_path)
