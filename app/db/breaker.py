"""The database circuit breaker (§8.3 Supabase-PostgreSQL policy), async-native.

This deliberately **reimplements** the §8.3 state machine rather than importing F01's
``CircuitBreaker`` from ``app/pipeline/ingest.py``: that module imports pandas + pyarrow at module
top, so importing it would drag the data-pipeline stack — and a backwards ``app.db → app.pipeline``
dependency — into the persistence layer. The semantics here are identical to F01's (retries happen
inside one call; only a fully-exhausted call counts as one failure; consecutive failures open the
breaker; a success or an idle gap > ``window_s`` resets the streak; an open breaker short-circuits
for ``window_s``), verified by its own tests. The one addition is that each attempt is wrapped in
``asyncio.wait_for(fn(), timeout_s)`` — F01's breaker delegates the per-attempt timeout to the
callable, but a DB call has no other timeout home, so ``breaker.timeout_s`` is enforced here.

A DB breaker is its OWN instance (never shares F01's fastf1/openf1/ergast breakers).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TypeVar

from sqlalchemy.exc import InterfaceError, OperationalError
from sqlalchemy.exc import TimeoutError as SATimeoutError

from app.contracts.config import AppConfig, BreakerConfig, ConfigError
from app.db.results import DatabaseUnavailableError

T = TypeVar("T")

# Transport-level faults that a retry might clear. IntegrityError / ProgrammingError etc. are NOT
# here — a data/constraint fault must propagate immediately to the caller, never retried or
# counted as a transport failure. Two distinct timeouts must both be caught: the
# builtin ``TimeoutError`` that ``asyncio.wait_for`` raises (per-attempt cap), AND SQLAlchemy's own
# ``sqlalchemy.exc.TimeoutError`` from pool exhaustion — which does NOT subclass the builtin, so
# omitting it would let a pool-timeout bypass the retry loop and the breaker's failure accounting.
RETRYABLE_DB_ERRORS: tuple[type[BaseException], ...] = (
    OperationalError,
    InterfaceError,
    SATimeoutError,
    TimeoutError,
)


@dataclass
class DbCircuitBreaker:
    """Per-database retry + circuit policy (§8.3). Clock injected so tests never touch wall time."""

    name: str
    config: BreakerConfig
    clock: Callable[[], float] = time.monotonic
    _consecutive: int = field(default=0, repr=False)
    _last_failure_at: float | None = field(default=None, repr=False)
    _opened_at: float | None = field(default=None, repr=False)

    def failure_count(self) -> int:
        """Consecutive post-retry failures recorded (the count that trips the breaker)."""
        return self._consecutive

    def is_open(self) -> bool:
        """True while the breaker is open and still inside its ``window_s`` cooldown."""
        if self._opened_at is None:
            return False
        return (self.clock() - self._opened_at) < self.config.window_s

    def record_success(self) -> None:
        """A successful call closes the breaker and clears the failure streak."""
        self._consecutive = 0
        self._last_failure_at = None
        self._opened_at = None

    def record_failure(self) -> None:
        """A fully-exhausted call: extend the streak and open if it hits ``open_after``."""
        now = self.clock()
        if self._last_failure_at is not None and now - self._last_failure_at > self.config.window_s:
            self._consecutive = 0
            self._opened_at = None
        self._consecutive += 1
        self._last_failure_at = now
        if self._consecutive >= self.config.open_after:
            self._opened_at = now

    async def run(
        self,
        fn: Callable[[], Awaitable[T]],
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> T:
        """Invoke the async ``fn`` under the retry + circuit policy.

        Returns ``fn``'s result on success. Raises :class:`DatabaseUnavailableError` if the breaker
        is already open (short-circuit, no call made) or if every attempt fails with a retryable
        transport fault / timeout. Any non-retryable exception (e.g. ``IntegrityError``) propagates
        uncaught and does NOT count as a transport failure.
        """
        if self.is_open():
            raise DatabaseUnavailableError(self.name, short_circuited=True)

        attempts = self.config.retries + 1
        backoff = self.config.backoff_s
        last_exc: BaseException | None = None
        for i in range(attempts):
            try:
                result = await asyncio.wait_for(fn(), self.config.timeout_s)
            except RETRYABLE_DB_ERRORS as exc:
                last_exc = exc
                if i < attempts - 1:
                    await sleep(backoff[min(i, len(backoff) - 1)])
            else:
                self.record_success()
                return result
        self.record_failure()
        raise DatabaseUnavailableError(self.name, short_circuited=False, cause=last_exc)


def make_db_breaker(
    config: AppConfig, *, clock: Callable[[], float] = time.monotonic
) -> DbCircuitBreaker:
    """Build the DB breaker from the validated ``database:`` config.

    Raises :class:`ConfigError` (``CONFIG_001``) if the ``database:`` block is absent — the DB layer
    cannot fail-loud without it, so this is a build-breaking startup error, never a silent default.
    """
    if config.database is None:
        raise ConfigError("database: block is required to build the DB circuit breaker")
    return DbCircuitBreaker("postgres", config.database.breaker, clock=clock)
