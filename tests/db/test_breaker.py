"""The async DB circuit breaker — §8.3 semantics, DB_UNAVAILABLE, and the retry policy."""

from __future__ import annotations

import asyncio

import pytest
from app.contracts.config import BreakerConfig, ConfigError, load_config
from app.db.breaker import DbCircuitBreaker, make_db_breaker
from app.db.results import DatabaseUnavailableError
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.exc import TimeoutError as SATimeoutError


class FakeClock:
    """Deterministic monotonic clock — tests never touch wall time (mirrors pipeline/conftest)."""

    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


async def _no_sleep(_seconds: float) -> None:
    """An async no-op sleep so retry backoff costs no wall time."""


def _cfg(**over: object) -> BreakerConfig:
    base: dict[str, object] = dict(
        timeout_s=5.0, retries=2, backoff_s=[0.0, 0.0], open_after=3, window_s=30.0
    )
    base.update(over)
    return BreakerConfig(**base)  # type: ignore[arg-type]


def _op_error() -> OperationalError:
    return OperationalError("SELECT 1", {}, Exception("connection lost"))


async def test_success_returns_value_and_closes() -> None:
    br = DbCircuitBreaker("db", _cfg(), clock=FakeClock())

    async def ok() -> int:
        return 42

    assert await br.run(ok, sleep=_no_sleep) == 42
    assert br.failure_count() == 0
    assert br.is_open() is False


async def test_open_breaker_short_circuits_without_calling_fn() -> None:
    clock = FakeClock()
    br = DbCircuitBreaker("db", _cfg(open_after=1), clock=clock)
    br.record_failure()  # open_after=1 -> now open
    assert br.is_open() is True

    called = False

    async def fn() -> int:
        nonlocal called
        called = True
        return 1

    with pytest.raises(DatabaseUnavailableError) as exc:
        await br.run(fn, sleep=_no_sleep)
    assert exc.value.short_circuited is True
    assert called is False  # no connection attempted (fast-fail path)


async def test_retries_exhausted_raises_db_unavailable() -> None:
    br = DbCircuitBreaker("db", _cfg(retries=2, open_after=5), clock=FakeClock())
    calls = 0

    async def always_down() -> int:
        nonlocal calls
        calls += 1
        raise _op_error()

    with pytest.raises(DatabaseUnavailableError) as exc:
        await br.run(always_down, sleep=_no_sleep)
    assert calls == 3  # retries=2 -> 3 attempts
    assert exc.value.short_circuited is False
    assert isinstance(exc.value.cause, OperationalError)
    assert br.failure_count() == 1  # the whole exhausted call = one failure


async def test_breaker_opens_after_configured_consecutive_failures() -> None:
    br = DbCircuitBreaker("db", _cfg(retries=0, open_after=3, backoff_s=[0.0]), clock=FakeClock())

    async def down() -> int:
        raise _op_error()

    for _ in range(3):
        with pytest.raises(DatabaseUnavailableError):
            await br.run(down, sleep=_no_sleep)
    assert br.failure_count() == 3
    assert br.is_open() is True


async def test_timeout_is_retryable_and_surfaces_as_db_unavailable() -> None:
    br = DbCircuitBreaker("db", _cfg(timeout_s=0.001, retries=0, open_after=5), clock=FakeClock())

    async def too_slow() -> int:
        await asyncio.sleep(0.05)  # exceeds the 1 ms timeout -> asyncio.wait_for TimeoutError
        return 1

    with pytest.raises(DatabaseUnavailableError) as exc:
        await br.run(too_slow, sleep=_no_sleep)
    assert isinstance(exc.value.cause, TimeoutError)


async def test_sqlalchemy_pool_timeout_is_retryable_and_counts() -> None:
    """sqlalchemy.exc.TimeoutError (pool exhaustion) does NOT subclass the builtin TimeoutError —
    it must still be caught, retried, and counted, else it bypasses the breaker's accounting."""
    br = DbCircuitBreaker("db", _cfg(retries=1, open_after=5), clock=FakeClock())
    calls = 0

    async def pool_exhausted() -> int:
        nonlocal calls
        calls += 1
        raise SATimeoutError("QueuePool limit of size 5 overflow 10 reached")

    with pytest.raises(DatabaseUnavailableError) as exc:
        await br.run(pool_exhausted, sleep=_no_sleep)
    assert calls == 2  # retries=1 -> 2 attempts (it WAS retried)
    assert isinstance(exc.value.cause, SATimeoutError)
    assert br.failure_count() == 1  # and counted as one transport failure


async def test_integrity_error_is_not_retried_and_propagates() -> None:
    br = DbCircuitBreaker("db", _cfg(retries=2, open_after=5), clock=FakeClock())
    calls = 0

    async def bad_write() -> int:
        nonlocal calls
        calls += 1
        raise IntegrityError("INSERT", {}, Exception("duplicate key"))

    with pytest.raises(IntegrityError):
        await br.run(bad_write, sleep=_no_sleep)
    assert calls == 1  # a data fault is not retried
    assert br.failure_count() == 0  # and not counted as a transport failure


async def test_breaker_recovers_after_cooldown_window() -> None:
    clock = FakeClock()
    br = DbCircuitBreaker("db", _cfg(retries=0, open_after=1, window_s=30.0), clock=clock)

    async def down() -> int:
        raise _op_error()

    with pytest.raises(DatabaseUnavailableError):
        await br.run(down, sleep=_no_sleep)
    assert br.is_open() is True
    clock.advance(31.0)  # past the cooldown window
    assert br.is_open() is False  # half-open: a trial call is allowed through


def test_make_db_breaker_requires_database_block(valid_config_file: object) -> None:
    cfg = load_config(valid_config_file)  # type: ignore[arg-type]
    assert cfg.database is None
    with pytest.raises(ConfigError):
        make_db_breaker(cfg)


def test_make_db_breaker_uses_config_policy(repo_config_path: object) -> None:
    cfg = load_config(repo_config_path)  # type: ignore[arg-type]
    br = make_db_breaker(cfg)
    assert br.config.timeout_s == 5
    assert br.config.open_after == 3
    assert br.config.backoff_s == [1, 2, 4]
