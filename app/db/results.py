"""Typed results and exceptions for the persistence layer.

The leaf of the ``app/db`` package: depends only on the contract layer (``BaseResult`` /
``ErrorCode``) so ``breaker``/``session``/``repository``/``health`` can all import it without a
cycle. Follows the house pattern where infra faults are *typed exceptions* (``ConfigError``,
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


class DbIntegrityError(Exception):
    """Raised when a write violates a constraint or an application-layer invariant.

    Carries ``ErrorCode.DB_INTEGRITY`` (F12's one new code). Covers UNIQUE / FK / NOT-NULL / the
    clean-lap CHECK (surfaced from a driver ``IntegrityError``) and the §8.4 ``lap_features``
    ``null_count == count(nulls)`` invariant (raised directly). The transaction is rolled back
    before this propagates.
    """

    error_code = ErrorCode.DB_INTEGRITY

    def __init__(self, message: str, *, cause: BaseException | None = None):
        self.message = message
        self.cause = cause
        super().__init__(message)


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


class PersistResult(BaseResult):
    """Outcome of a repository write.

    Success → ``OK`` + ``written=True`` + ``entity_id``; integrity violation → ``FAILED`` +
    ``DB_INTEGRITY`` + ``written=False``; DB unreachable → ``FAILED`` + ``DB_UNAVAILABLE``.
    """

    entity_type: str
    entity_id: str | None = None
    written: bool = False
