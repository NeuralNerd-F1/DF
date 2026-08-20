"""Route reachability. Every documented route is implemented; ``/replay`` 503s only when its
backing service is unconfigured (this fixture injects the DB + monitor, so it reaches the engine).
Real route behaviour lives in ``test_replay_route.py``. The unknown-driver projection stays routable
but ``include_in_schema=False`` — it is outside the presented surface, so it must not appear in the
OpenAPI document the frontend types are generated from."""

from __future__ import annotations

from types import SimpleNamespace


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

    # The unknown-projection endpoint is include_in_schema=False: still routable, outside the
    # presented surface.
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
    for p in ("/fingerprint/unknown",):
        assert p not in paths
        assert p in routed


async def test_removed_queue_and_upload_routes_are_gone(api: SimpleNamespace) -> None:
    """The async ingestion queue and the manual CSV upload are out of the build entirely — not
    deferred, not 503, simply not routed. A 404 (never a 503) is the contract."""
    paths = api.app.openapi()["paths"]
    for method, path in (
        ("post", "/ingest/session"),
        ("get", "/ingest/status/abc"),
        ("post", "/ingest/upload"),
    ):
        assert path not in paths
        r = await getattr(api.client, method)(path)
        assert r.status_code == 404, f"{path} still routes ({r.status_code})"
