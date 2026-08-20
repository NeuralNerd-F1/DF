"""Typed results and exceptions for the persistence layer.

The leaf of the ``app/db`` package: depends only on the contract layer (``BaseResult`` /
``ErrorCode``) so ``breaker``/``session``/``health`` can all import it without a cycle. Follows
the house pattern where infra faults are *typed exceptions* (``ConfigError``,
``DataQualityLogError``) and stage outcomes are Pydantic result objects.
"""

from __future__ import annotations

from app.contracts.errors import ErrorCode
from app.contracts.results import BaseResult


class DatabaseUnavailableError(Exception):
    """Raised when the DB is unreachable — breaker open (short-circuit) or retries exhausted.

    Carries ``ErrorCode.DB_UNAVAILABLE``. ``short_circuited`` is ``True`` when the breaker was
    already open (no connection attempted — the fast-fail path), ``False`` when a connection was
    attempted and every retry failed. F13 maps this to HTTP 503; the write path rolls back first so
    there is **no partial write** (§8.3).
    """

    error_code = ErrorCode.DB_UNAVAILABLE

    def __init__(self, source: str, *, short_circuited: bool, cause: BaseException | None = None):
        self.source = source
        self.short_circuited = short_circuited
        self.cause = cause
        state = "breaker open" if short_circuited else "retries exhausted"
        super().__init__(f"database {source} unavailable ({state})")


class DbHealthResult(BaseResult):
    """DB health probe outcome (consumed by F13's ``/health`` ``db_connected`` field).

    ``status=OK`` only when ``connected`` and the applied Alembic revision equals the migration
    head; otherwise ``FAILED`` + ``HEALTH_FAIL`` with ``detail`` saying which (unreachable / stale
    schema). Never carries a forced answer — the probe never raises.
    """

    connected: bool = False
    revision: str | None = None
    head_revision: str | None = None
    latency_ms: float | None = None
    detail: str | None = None
