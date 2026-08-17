"""F14 cache resilience over the API (RQ5 §11.4): Redis down → predictions still 200; hits work."""

from __future__ import annotations

import json
from types import SimpleNamespace

import fakeredis.aioredis
from app.api.deps import get_predictor
from app.cache.redis import RedisCache
from app.contracts.errors import ErrorCode

from tests.api.conftest import fake_prediction


class _DownClient:
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


def _set_predictor(api: SimpleNamespace) -> None:
    pred = fake_prediction()
    api.app.dependency_overrides[get_predictor] = lambda: (lambda fv, circuit: pred)


async def test_100_predictions_succeed_with_redis_down(api: SimpleNamespace) -> None:
    """RQ5: disable Redis, 100 consecutive predictions → zero failures, CACHE_UNAVAILABLE logged."""
    api.config.cache.enabled = True  # the resilience contract is for an ENABLED cache
    api.state.cache = RedisCache(
        _DownClient(),
        config=api.config.cache,
        api_version=api.config.api.version,
        log_path=api.config.paths.data_quality_log,
    )
    _set_predictor(api)
    body = {"feature_vector": {"F01": 0.5, "F02": 0.3}, "circuit": "Bahrain"}
    for _ in range(100):
        r = await api.client.post("/predict", json=body)
        assert r.status_code == 200  # a down cache NEVER fails a prediction
    rows = [
        json.loads(x)
        for x in open(api.config.paths.data_quality_log, encoding="utf-8").read().splitlines()
        if x
    ]
    assert any(r["error_code"] == ErrorCode.CACHE_UNAVAILABLE.value for r in rows)


async def test_predict_consistent_under_cache(api: SimpleNamespace) -> None:
    """Same input → same output, and the second identical request is a cache HIT (reused)."""
    api.config.cache.enabled = True  # the resilience contract is for an ENABLED cache
    api.state.cache = RedisCache(
        fakeredis.aioredis.FakeRedis(),
        config=api.config.cache,
        api_version=api.config.api.version,
    )
    _set_predictor(api)
    body = {"feature_vector": {"F01": 0.5}, "circuit": "Monaco"}
    r1 = await api.client.post("/predict", json=body)
    r2 = await api.client.post("/predict", json=body)
    assert r1.status_code == r2.status_code == 200
    assert r1.json() == r2.json()  # deterministic
    assert api.state.cache.hits >= 1  # the second request was served from cache


async def test_predict_still_works_without_cache(api: SimpleNamespace) -> None:
    """No cache configured (None) → predictions compute directly, still 200."""
    api.state.cache = None
    _set_predictor(api)
    r = await api.client.post("/predict", json={"feature_vector": {"F01": 0.5}, "circuit": "Spa"})
    assert r.status_code == 200
