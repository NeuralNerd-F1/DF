"""GET /health — real component status; 503 iff a critical component is down (§8.1/§8.3)."""

from __future__ import annotations

from types import SimpleNamespace


async def test_health_ok_when_all_ready(api: SimpleNamespace) -> None:
    r = await api.client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert body["db_connected"] is True
    assert all(body["models_loaded"].values())
    assert set(body["models_loaded"]) == {
        "classifier",
        "stylespace",
        "degradation",
        "personalisation",
    }
    assert body["version"] == "0.1.0"


async def test_health_db_down_is_503(api: SimpleNamespace) -> None:
    for _ in range(3):  # open the DB breaker -> check_database reports down
        api.state.db_breaker.record_failure()
    r = await api.client.get("/health")
    assert r.status_code == 503
    assert r.json()["status"] == "unhealthy"
    assert r.json()["db_connected"] is False


async def test_health_model_not_loaded_is_503(api: SimpleNamespace) -> None:
    api.state.model_handle = None  # classifier failed to load
    r = await api.client.get("/health")
    assert r.status_code == 503
    assert r.json()["models_loaded"]["classifier"] is False


async def test_health_cache_down_stays_200(api: SimpleNamespace) -> None:
    r = await api.client.get("/health")
    assert r.status_code == 200
    assert r.json()["cache_connected"] is None  # cache is non-critical (F14)
