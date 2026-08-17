"""F14 Redis cache — the never-raising, non-critical cache-aside contract (§8.2/§8.3/§9, RQ5 §11.4).

Runs fully offline: ``fakeredis.aioredis`` for the happy path, hand-rolled failing clients for the
degradation paths. No Redis server required.
"""

from __future__ import annotations

import json
from pathlib import Path

import fakeredis.aioredis
import pytest
from app.cache.redis import RedisCache, build_cache
from app.contracts.config import AppConfig, load_config
from app.contracts.errors import ErrorCode


@pytest.fixture
def cfg(repo_config_path: Path) -> AppConfig:
    config = load_config(repo_config_path)
    # These tests exercise the enabled-cache contract; the repo default (off on this
    # branch) must not decide what the class does when a deployment turns it on.
    config.cache.enabled = True
    return config


def _fake_cache(cfg: AppConfig, *, log_path: str | None = None, clock=None) -> RedisCache:
    client = fakeredis.aioredis.FakeRedis()
    kwargs = {"clock": clock} if clock is not None else {}
    return RedisCache(
        client, config=cfg.cache, api_version=cfg.api.version, log_path=log_path, **kwargs
    )


class _DownClient:
    """A client that fails every op (Redis unreachable)."""

    async def get(self, key: str) -> object:
        raise ConnectionError("redis down")

    async def set(self, key: str, value: str, *, ex: int | None = None) -> object:
        raise ConnectionError("redis down")

    async def delete(self, *keys: str) -> object:
        raise ConnectionError("redis down")

    async def ping(self) -> object:
        raise ConnectionError("redis down")

    async def aclose(self) -> None:
        return None


# --- keys / happy path -------------------------------------------------------------------------
def test_key_is_namespaced_and_versioned(cfg: AppConfig) -> None:
    cache = _fake_cache(cfg)
    key = cache.key("driver_profile", "abc")
    assert key == f"{cfg.cache.key_prefix}:v{cfg.api.version}:driver_profile:abc"


async def test_miss_then_hit(cfg: AppConfig) -> None:
    cache = _fake_cache(cfg)
    r = await cache.get("driver_profile", "x")
    assert r.hit is False and r.state is None  # clean miss
    assert await cache.set("driver_profile", "x", {"v": 1}, 60) is None
    r2 = await cache.get("driver_profile", "x")
    assert r2.hit is True and r2.value == {"v": 1}
    assert cache.hits == 1 and cache.misses == 1


async def test_get_or_compute_computes_once(cfg: AppConfig) -> None:
    cache = _fake_cache(cfg)
    calls = {"n": 0}

    async def compute() -> dict[str, int]:
        calls["n"] += 1
        return {"y": 7}

    v1, r1 = await cache.get_or_compute(namespace="predict", ident="k", ttl_s=60, compute=compute)
    v2, r2 = await cache.get_or_compute(namespace="predict", ident="k", ttl_s=60, compute=compute)
    assert v1 == v2 == {"y": 7}
    assert r1.hit is False and r2.hit is True
    assert calls["n"] == 1  # second call served from cache, compute not re-run


async def test_disabled_cache_is_always_a_miss(cfg: AppConfig) -> None:
    cfg.cache.enabled = False
    cache = RedisCache(None, config=cfg.cache, api_version=cfg.api.version)
    assert cache.enabled is False
    calls = {"n": 0}

    async def compute() -> int:
        calls["n"] += 1
        return 5

    v, r = await cache.get_or_compute(namespace="predict", ident="k", ttl_s=60, compute=compute)
    assert v == 5 and r.state is ErrorCode.CACHE_MISS
    v2, _ = await cache.get_or_compute(namespace="predict", ident="k", ttl_s=60, compute=compute)
    assert calls["n"] == 2  # nothing cached — always recomputes


async def test_poisoned_value_is_treated_as_miss_and_logged(
    cfg: AppConfig, tmp_path: Path
) -> None:
    log = tmp_path / "dq.jsonl"
    client = fakeredis.aioredis.FakeRedis()
    cache_with = RedisCache(
        client, config=cfg.cache, api_version=cfg.api.version, log_path=str(log)
    )
    # Write a non-JSON blob straight past the cache's serialisation.
    await client.set(cache_with.key("predict", "bad"), b"\x00not-json")
    r = await cache_with.get("predict", "bad")
    assert r.hit is False and r.state is ErrorCode.CACHE_MISS
    rows = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if x]
    assert any("poisoned" in (row.get("error_detail") or "") for row in rows)  # not silent


# --- degradation (never raises) ----------------------------------------------------------------
async def test_down_redis_get_is_graceful(cfg: AppConfig) -> None:
    cache = RedisCache(_DownClient(), config=cfg.cache, api_version=cfg.api.version)
    r = await cache.get("predict", "k")  # must NOT raise
    assert r.hit is False and r.state is ErrorCode.CACHE_UNAVAILABLE


async def test_down_redis_get_or_compute_still_returns(cfg: AppConfig) -> None:
    cache = RedisCache(_DownClient(), config=cfg.cache, api_version=cfg.api.version)

    async def compute() -> str:
        return "computed"

    v, r = await cache.get_or_compute(namespace="predict", ident="k", ttl_s=60, compute=compute)
    assert v == "computed"  # a down Redis never fails the request (RQ5)
    assert r.state in {ErrorCode.CACHE_UNAVAILABLE, ErrorCode.CACHE_MISS}


async def test_breaker_opens_after_two_failures(cfg: AppConfig) -> None:
    # breaker: open_after=2, window_s=30 (§8.3). A fixed clock keeps failures inside the window.
    cache = RedisCache(
        _DownClient(), config=cfg.cache, api_version=cfg.api.version, clock=lambda: 100.0
    )
    r1 = await cache.get("predict", "a")  # failure 1
    assert r1.state is ErrorCode.CACHE_UNAVAILABLE
    await cache.get("predict", "b")  # failure 2 → opens
    assert cache.breaker_open() is True
    r3 = await cache.get("predict", "c")  # short-circuits → graceful miss (no call)
    assert r3.state is ErrorCode.CACHE_MISS


async def test_write_dropped_when_breaker_open(cfg: AppConfig) -> None:
    cache = RedisCache(
        _DownClient(), config=cfg.cache, api_version=cfg.api.version, clock=lambda: 5.0
    )
    await cache.get("predict", "a")
    await cache.get("predict", "b")  # breaker now open
    state = await cache.set("predict", "c", {"v": 1}, 60)
    assert state is ErrorCode.CACHE_DEGRADED  # write silently dropped


async def test_cache_unavailable_is_logged(cfg: AppConfig, tmp_path: Path) -> None:
    log = tmp_path / "dq.jsonl"
    cache = RedisCache(
        _DownClient(), config=cfg.cache, api_version=cfg.api.version, log_path=str(log)
    )
    await cache.get("predict", "a")  # first failure logs CACHE_UNAVAILABLE
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line]
    assert any(r["error_code"] == ErrorCode.CACHE_UNAVAILABLE.value for r in rows)


async def test_ping_reflects_liveness(cfg: AppConfig) -> None:
    assert await _fake_cache(cfg).ping() is True
    down = RedisCache(_DownClient(), config=cfg.cache, api_version=cfg.api.version)
    assert await down.ping() is False


# --- build_cache factory -----------------------------------------------------------------------
def test_build_cache_returns_none_without_block(cfg: AppConfig) -> None:
    cfg.cache = None  # type: ignore[assignment]
    assert build_cache(cfg) is None


def test_build_cache_disabled_when_no_dsn(cfg: AppConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(cfg.cache.url_env_var, raising=False)
    cache = build_cache(cfg)
    assert cache is not None and cache.enabled is False  # no DSN → cache-less, not an error
