"""GET /metrics — Prometheus exposition, internal-only (403 external, §8.5)."""

from __future__ import annotations

from types import SimpleNamespace

from httpx import ASGITransport, AsyncClient


async def test_metrics_internal_ok(api: SimpleNamespace) -> None:
    r = await api.client.get("/metrics")  # no X-Forwarded-For -> internal (testclient host)
    assert r.status_code == 200
    assert "text/plain" in r.headers["content-type"]
    assert "apex_requests_total" in r.text


async def test_metrics_external_is_403(api: SimpleNamespace) -> None:
    r = await api.client.get("/metrics", headers={"X-Forwarded-For": "203.0.113.5"})
    assert r.status_code == 403
    assert r.json()["error_code"] == "FORBIDDEN"


async def test_metrics_external_peer_cannot_spoof_xff(api: SimpleNamespace) -> None:
    """A DIRECT external peer that spoofs ``X-Forwarded-For: 127.0.0.1`` is still 403: the header is
    only trusted from a loopback peer (a co-located proxy), never from a direct external caller."""
    transport = ASGITransport(
        app=api.app, client=("203.0.113.5", 44444), raise_app_exceptions=False
    )
    async with AsyncClient(transport=transport, base_url="http://testserver") as ext:
        r = await ext.get("/metrics", headers={"X-Forwarded-For": "127.0.0.1"})
    assert r.status_code == 403
    assert r.json()["error_code"] == "FORBIDDEN"
