"""Alembic environment — async engine, Base.metadata target, env-sourced DSN.

Kept import-light on purpose: this imports ``app.db.base`` / ``app.db.models`` (SQLAlchemy only, no
pandas/xgboost) so running migrations never drags the ML stack. The DSN is resolved from the
environment (the ``database.dsn_env_var`` name, or an explicit ``-x dsn=...``) — never from
``alembic.ini`` (secret hygiene, §9).
"""

from __future__ import annotations

import asyncio
import os
from logging.config import fileConfig
from typing import Any

from alembic import context
from app.contracts.config import ConfigError, load_config
from app.db import models  # noqa: F401 — import registers all tables on Base.metadata
from app.db.base import Base
from sqlalchemy.ext.asyncio import create_async_engine

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _resolve_url() -> str:
    """DSN from ``-x dsn=...`` (explicit) else the env var named by ``database.dsn_env_var``."""
    x = context.get_x_argument(as_dictionary=True)
    if x.get("dsn"):
        return str(x["dsn"])
    cfg = load_config()
    if cfg.database is None:
        raise ConfigError("database: block is required to run migrations")
    url = os.environ.get(cfg.database.dsn_env_var, "")
    if not url:
        raise ConfigError(
            f"{cfg.database.dsn_env_var} is not set — no database DSN available for migrations"
        )
    return url


def _schema_name() -> str | None:
    cfg = load_config()
    return cfg.database.schema_name if cfg.database is not None else None


def do_run_migrations(connection: Any) -> None:
    schema = _schema_name()
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        # Server defaults (now(), 'OK', false) are notoriously noisy under autogenerate; they are
        # verified by the create_all/round-trip tests instead of the diff comparison.
        compare_server_default=False,
        version_table_schema=schema,
        include_schemas=schema is not None,
        # SQLite (the offline suite) can't ALTER in place — batch mode rebuilds the table. Moot for
        # the pure-create_all baseline, but a future F13+ column change under SQLite needs it.
        render_as_batch=connection.dialect.name == "sqlite",
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    engine = create_async_engine(_resolve_url())
    async with engine.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await engine.dispose()


if context.is_offline_mode():
    # The baseline materialises Base.metadata via create_all (needs a live connection), so offline
    # `--sql` generation is not supported — run online migrations against a real DSN.
    raise ConfigError("offline (--sql) migration mode is not supported by the F12 baseline")
else:
    asyncio.run(run_migrations_online())
