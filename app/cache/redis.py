"""Redis cache — an optimisation, never a hard dependency (§8.2/§8.3/§9, constitution #stores).

The cache accelerates the read endpoints and the model-output path. It is **never critical**: a
miss, a timeout, an open circuit breaker, an unreachable Redis, or a disabled block all degrade to a
graceful miss, and the caller computes from source. :class:`RedisCache` **never raises into a
handler** — every fault is caught, recorded on its own circuit breaker (§8.3: timeout 2s / 1 retry /
1s backoff / open after 2 failures in 30s), translated to a typed ``CACHE_*`` state, and logged to
``data_quality_log`` at outage onset. This is the RQ5 "cache unavailability resilience" guarantee
(§11.4) made structural.

The client is injected as a minimal :class:`RedisClientProtocol`, so the real ``redis.asyncio.Redis``
and the offline ``fakeredis.aioredis.FakeRedis`` (and a ``None`` for the disabled case) are all
first-class — the whole cache suite runs with no Redis server.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast

from app.contracts.errors import ErrorCode
from app.contracts.logging import log_data_quality

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from app.contracts.config import AppConfig, BreakerConfig, CacheConfig

_T = TypeVar("_T")

# Redis-side faults that must degrade gracefully (never propagate). ``redis.exceptions.RedisError``
# is the base for connection/timeout/response errors; ``OSError`` covers raw socket faults; both
# timeout types cover the ``asyncio.wait_for`` per-attempt cap and any client-level timeout.
_CACHE_FAULTS: tuple[type[BaseException], ...] = (OSError, asyncio.TimeoutError, TimeoutError)
with suppress(Exception):  # pragma: no cover — redis is a hard dep of F14; import is defensive only
    import redis.exceptions as _redis_exc

    _CACHE_FAULTS = (_redis_exc.RedisError, *_CACHE_FAULTS)


class RedisClientProtocol(Protocol):
    """The minimal async Redis surface the cache uses. Satisfied by ``redis.asyncio.Redis`` and
    ``fakeredis.aioredis.FakeRedis`` alike — so tests need no live server."""

    async def get(self, key: str) -> Any: ...
    async def set(self, key: str, value: str, *, ex: int | None = None) -> Any: ...
    async def delete(self, *keys: str) -> Any: ...
    async def ping(self) -> Any: ...
    async def aclose(self) -> Any: ...


@dataclass
class CacheResult:
    """The outcome of a cache interaction. ``state`` is ``None`` on a hit or a clean miss; otherwise
    one of the three pre-seeded ``CACHE_*`` codes. Never carries an exception — the cache never
    raises."""

    value: Any | None = None
    hit: bool = False
    state: ErrorCode | None = None


@dataclass
class _CacheBreaker:
    """The §8.3 circuit-breaker state machine, non-raising (mirrors ``DbCircuitBreaker`` but the
    cache turns an open breaker into a graceful miss instead of an error). Clock injected so tests
    never touch wall time. Reimplemented locally to keep ``app/cache`` free of the DB/pandas stack."""

    config: BreakerConfig
    clock: Callable[[], float] = time.monotonic
    _consecutive: int = field(default=0, repr=False)
    _last_failure_at: float | None = field(default=None, repr=False)
    _opened_at: float | None = field(default=None, repr=False)

    def failure_count(self) -> int:
        return self._consecutive

    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        return (self.clock() - self._opened_at) < self.config.window_s

    def record_success(self) -> None:
        self._consecutive = 0
        self._last_failure_at = None
        self._opened_at = None

    def record_failure(self) -> bool:
        """Extend the failure streak; return ``True`` iff this failure just OPENED the breaker
        (the transition the cache logs once, to avoid flooding ``data_quality_log`` during an
        outage)."""
        now = self.clock()
        if self._last_failure_at is not None and now - self._last_failure_at > self.config.window_s:
            self._consecutive = 0
            self._opened_at = None
        was_open = self._opened_at is not None
        self._consecutive += 1
        self._last_failure_at = now
        if self._consecutive >= self.config.open_after:
            self._opened_at = now
        return self._opened_at is not None and not was_open


class RedisCache:
    """A cache-aside wrapper that never fails a request. ``enabled=False`` (or ``client=None``) makes
    every read a graceful ``CACHE_MISS`` — the system runs identically cache-less."""

    def __init__(
        self,
        client: RedisClientProtocol | None,
        *,
        config: CacheConfig,
        api_version: str,
        clock: Callable[[], float] = time.monotonic,
        log_path: str | None = None,
    ) -> None:
        self._client = client
        self._config = config
        self._prefix = f"{config.key_prefix}:v{api_version}"
        self._timeout_s = config.breaker.timeout_s
        self._breaker = _CacheBreaker(config.breaker, clock=clock)
        self._log_path = log_path
        self.enabled = bool(config.enabled and client is not None)
        # Cheap in-process counters the /metrics endpoint reads at scrape (no coupling to prometheus).
        self.hits = 0
        self.misses = 0

    # --- keys ----------------------------------------------------------------------------------
    def key(self, namespace: str, ident: str) -> str:
        """Namespaced + API-versioned key: ``{prefix}:v{version}:{namespace}:{ident}`` (§4.0-h). The
        version segment invalidates the whole cache on an API version bump."""
        return f"{self._prefix}:{namespace}:{ident}"

    @staticmethod
    def hash_ident(payload: Any) -> str:
        """A stable content hash for value-keyed caches (e.g. the model-output cache keyed by the
        feature vector + circuit). Deterministic JSON → sha256 (same input → same key)."""
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    # --- internals -----------------------------------------------------------------------------
    def _log_outage(self, detail: str) -> None:
        if self._log_path is None:
            return
        with suppress(Exception):
            log_data_quality(
                error_code=ErrorCode.CACHE_UNAVAILABLE,
                entity_type="cache",
                entity_id="cache",
                error_detail=detail,
                fallback_used=True,
                log_path=self._log_path,
            )

    async def _guarded(self, coro_factory: Callable[[], Awaitable[_T]]) -> tuple[_T | None, bool]:
        """Run one client call under the timeout. Returns ``(result, ok)``; ``ok=False`` on any
        cache fault (breaker updated, outage logged once at the opening transition)."""
        try:
            result = await asyncio.wait_for(coro_factory(), self._timeout_s)
        except _CACHE_FAULTS as exc:  # graceful degradation — never propagate
            just_opened = self._breaker.record_failure()
            if just_opened or self._breaker.failure_count() < self._config.breaker.open_after:
                self._log_outage(f"{type(exc).__name__}: {exc}")
            return None, False
        self._breaker.record_success()
        return result, True

    # --- reads / writes ------------------------------------------------------------------------
    async def get(self, namespace: str, ident: str) -> CacheResult:
        """Read one key. A hit returns the decoded value; a clean miss / disabled / open-breaker /
        Redis fault all return ``hit=False`` with the right ``CACHE_*`` state. Never raises."""
        if not self.enabled or self._client is None:
            self.misses += 1
            return CacheResult(hit=False, state=ErrorCode.CACHE_MISS)
        if self._breaker.is_open():
            self.misses += 1
            return CacheResult(hit=False, state=ErrorCode.CACHE_MISS)  # short-circuit → graceful miss
        client = self._client
        raw, ok = await self._guarded(lambda: client.get(self.key(namespace, ident)))
        if not ok:
            self.misses += 1
            return CacheResult(hit=False, state=ErrorCode.CACHE_UNAVAILABLE)
        if raw is None:
            self.misses += 1
            return CacheResult(hit=False, state=None)  # clean miss
        try:
            value = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        except (ValueError, AttributeError):
            # Poisoned / unreadable blob → treat as a miss + evict. Logged (a corrupt cache entry is
            # a distinct fault, not a silent miss); never raises.
            self.misses += 1
            self._log_outage(f"poisoned cache value evicted: {self.key(namespace, ident)}")
            with suppress(Exception):
                await self._guarded(lambda: client.delete(self.key(namespace, ident)))
            return CacheResult(hit=False, state=ErrorCode.CACHE_MISS)
        self.hits += 1
        return CacheResult(value=value, hit=True, state=None)

    async def set(self, namespace: str, ident: str, value: Any, ttl_s: int) -> ErrorCode | None:
        """Write one key with a TTL. An open breaker silently drops the write (``CACHE_DEGRADED``);
        a fault returns ``CACHE_UNAVAILABLE``. Never raises. Returns ``None`` on a clean write."""
        if not self.enabled or self._client is None:
            return ErrorCode.CACHE_MISS
        if self._breaker.is_open():
            return ErrorCode.CACHE_DEGRADED  # writes silently dropped while open (§8.3)
        try:
            blob = json.dumps(value, separators=(",", ":"), default=str)
        except (TypeError, ValueError):
            return ErrorCode.CACHE_DEGRADED  # non-serialisable → skip caching, never fail the request
        client = self._client
        _, ok = await self._guarded(lambda: client.set(self.key(namespace, ident), blob, ex=ttl_s))
        return None if ok else ErrorCode.CACHE_UNAVAILABLE

    async def invalidate(self, namespace: str, ident: str) -> None:
        """Best-effort delete of one key (invalidate-on-write, §4.0-h). Never raises."""
        if not self.enabled or self._client is None or self._breaker.is_open():
            return
        client = self._client
        await self._guarded(lambda: client.delete(self.key(namespace, ident)))

    async def get_or_compute(
        self,
        *,
        namespace: str,
        ident: str,
        ttl_s: int,
        compute: Callable[[], Awaitable[_T]],
    ) -> tuple[_T, CacheResult]:
        """Cache-aside: return the cached value on a hit, else ``await compute()`` (the source of
        truth), store it, and return it. The cache is transparent — ``compute`` always runs on a
        miss, so a down Redis never fails the request (only slows it). ``compute`` may raise; that
        is the caller's business, not the cache's."""
        read = await self.get(namespace, ident)
        if read.hit:
            return cast("_T", read.value), read
        value = await compute()
        write_state = await self.set(namespace, ident, value, ttl_s)
        # Surface the more informative state: a read fault (CACHE_UNAVAILABLE) dominates a write drop.
        state = read.state or write_state
        return value, CacheResult(value=value, hit=False, state=state)

    # --- health --------------------------------------------------------------------------------
    async def ping(self) -> bool:
        """True iff the cache is enabled, the breaker is closed, and a ``PING`` succeeds. Used by
        ``/health`` (``cache_connected``) — but the cache is NEVER a critical component."""
        if not self.enabled or self._client is None or self._breaker.is_open():
            return False
        client = self._client
        _, ok = await self._guarded(lambda: client.ping())
        return ok

    def breaker_open(self) -> bool:
        """Whether the cache breaker is currently open (surfaced as a /metrics gauge)."""
        return self._breaker.is_open()

    async def close(self) -> None:
        """Dispose the underlying client (called from the app lifespan shutdown). Never raises."""
        if self._client is None:
            return
        with suppress(Exception):
            await self._client.aclose()


def build_cache(config: AppConfig, *, log_path: str | None = None) -> RedisCache | None:
    """Build the cache from the validated ``cache:`` + ``api:`` blocks.

    Returns ``None`` when the ``cache:`` block is absent, and a **disabled** :class:`RedisCache`
    (all reads → graceful miss) when the block is present but ``enabled: false`` or the DSN env var
    (``cache.url_env_var``) is unset/blank — the system runs cache-less either way, never crashing.
    A live client is created only when enabled AND a DSN is present. This is called best-effort from
    the app lifespan; a bad DSN that fails only at first use degrades via the breaker, not here.
    """
    import os

    if config.cache is None:
        return None
    api_version = config.api.version if config.api is not None else "0"
    if not config.cache.enabled:
        return RedisCache(None, config=config.cache, api_version=api_version, log_path=log_path)
    dsn = os.environ.get(config.cache.url_env_var, "").strip()
    if not dsn:  # no DSN → run cache-less (a normal, non-error condition; §9 secret is env-only)
        return RedisCache(None, config=config.cache, api_version=api_version, log_path=log_path)
    import redis.asyncio as aioredis

    timeout = config.cache.breaker.timeout_s
    client = cast(
        "RedisClientProtocol",
        aioredis.Redis.from_url(dsn, socket_timeout=timeout, socket_connect_timeout=timeout),
    )
    return RedisCache(client, config=config.cache, api_version=api_version, log_path=log_path)
