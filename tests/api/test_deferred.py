"""The 4 deferred endpoints — schemas present in OpenAPI, live routes return 503 (F14/F05)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("post", "/ingest/session"),
        ("get", "/ingest/status/abc"),
        ("post", "/ingest/upload"),
        ("get", "/replay/sess/drv"),
    ],
)
async def test_deferred_endpoint_is_503(api: SimpleNamespace, method: str, path: str) -> None:
    r = await getattr(api.client, method)(path)
    assert r.status_code == 503
    assert r.json()["error_code"] == "HEALTH_FAIL"


def test_deferred_endpoints_present_in_openapi(api: SimpleNamespace) -> None:
    paths = api.app.openapi()["paths"]
    for p in ("/ingest/session", "/ingest/status/{session_id}", "/ingest/upload",
              "/replay/{session_id}/{driver_id}"):
        assert p in paths
