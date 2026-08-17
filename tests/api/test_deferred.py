"""Queue endpoints. After F05 every documented route is implemented; the async ingestion + replay
routes 503 ONLY when their backing service is unconfigured (this F13 fixture injects no
``celery_app``/``job_store``; the DB + monitor ARE set, so ``/replay`` reaches the engine). Real F14
route behaviour lives in ``test_ingest_routes.py`` + ``test_replay_route.py``; the upload route
(``/ingest/upload``, no longer deferred) is covered by ``test_upload_route.py``."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    "method_path", [("post", "/ingest/session"), ("get", "/ingest/status/abc")]
)
async def test_ingest_routes_503_when_queue_unconfigured(
    api: SimpleNamespace, method_path: tuple[str, str]
) -> None:
    """No ``celery:`` wiring in this fixture → the queue endpoints 503 (not configured), like any
    other not-loaded component — never a crash."""
    method, path = method_path
    r = await getattr(api.client, method)(path)
    assert r.status_code == 503
    assert r.json()["error_code"] == "HEALTH_FAIL"


async def test_replay_reaches_engine_and_404s_on_unknown(api: SimpleNamespace) -> None:
    """The monitor handle IS injected, so ``/replay`` runs the engine and 404s on an unknown id
    (rather than the old deferred 503)."""
    r = await api.client.get("/replay/sess/drv")
    assert r.status_code == 404
    assert r.json()["error_code"] == "NOT_FOUND"


def test_all_endpoints_present_in_openapi(api: SimpleNamespace) -> None:
    paths = api.app.openapi()["paths"]
    for p in ("/replay/{session_id}/{driver_id}",):
        assert p in paths
    # Queue + upload + unknown-projection endpoints are include_in_schema=False: still routable,
    # outside the presented surface.
    def _all_paths(routes: list) -> set[str]:
        found: set[str] = set()
        for r in routes:
            if (p := getattr(r, "path", None)) is not None:
                found.add(p)
            # FastAPI's _IncludedRouter wraps the real APIRouter in .original_router
            nested = getattr(r, "original_router", r)
            found |= _all_paths(getattr(nested, "routes", []))
        return found

    routed = _all_paths(api.app.routes)
    for p in (
        "/ingest/session",
        "/ingest/status/{session_id}",
        "/ingest/upload",
        "/fingerprint/unknown",
    ):
        assert p not in paths
        assert p in routed
