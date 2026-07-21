"""POST /fingerprint/unknown, GET /compare, GET /stylespace."""

from __future__ import annotations

from types import SimpleNamespace

from app.api.deps import get_projector, get_stylespace_fn

from tests.api.conftest import fake_projection, fake_stylespace, seed_fingerprint

_LAPS_30 = [{"F01": 1.0, "F02": 0.5} for _ in range(30)]


def _projector(api: SimpleNamespace, proj: object) -> None:
    api.app.dependency_overrides[get_projector] = lambda: (lambda laps, circuit, ct: proj)


async def test_fingerprint_unknown_ok(api: SimpleNamespace) -> None:
    _projector(api, fake_projection())
    r = await api.client.post("/fingerprint/unknown", json={"laps": _LAPS_30, "circuit": "Monaco"})
    assert r.status_code == 200
    body = r.json()
    assert body["fingerprint_vector"] == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert body["nearest_neighbours"][0]["driver"] == "VER"
    assert body["warnings"] == []


async def test_fingerprint_unknown_sparse_warns(api: SimpleNamespace) -> None:
    _projector(api, fake_projection(sparse=True))
    r = await api.client.post("/fingerprint/unknown", json={"laps": _LAPS_30, "circuit": "Monaco"})
    assert r.status_code == 200
    assert "SPARSE_PROJECTION" in r.json()["warnings"]


async def test_fingerprint_unknown_ood_warns(api: SimpleNamespace) -> None:
    _projector(api, fake_projection(ood=True))
    r = await api.client.post("/fingerprint/unknown", json={"laps": _LAPS_30, "circuit": "Monaco"})
    assert r.status_code == 200
    assert "OUT_OF_DISTRIBUTION" in r.json()["warnings"]


async def test_fingerprint_unknown_too_few_laps_is_400(api: SimpleNamespace) -> None:
    laps = [{"F01": 1.0} for _ in range(3)]  # < 5
    r = await api.client.post("/fingerprint/unknown", json={"laps": laps, "circuit": "Monaco"})
    assert r.status_code == 400
    assert r.json()["error_code"] == "INVALID_INPUT"


async def test_fingerprint_unknown_too_many_laps_is_400(api: SimpleNamespace) -> None:
    """§11 DoS guard — an oversized laps payload (> api.max_request_laps) is rejected at 400
    before pd.DataFrame materialises it."""
    laps = [{"F01": 1.0} for _ in range(6000)]  # > max_request_laps (5000)
    r = await api.client.post("/fingerprint/unknown", json={"laps": laps, "circuit": "Monaco"})
    assert r.status_code == 400
    assert r.json()["error_code"] == "INVALID_INPUT"


async def test_compare_ok(api: SimpleNamespace) -> None:
    await seed_fingerprint(api.session_factory, code="VER", radar={"throttle": 0.8, "braking": 0.2})
    await seed_fingerprint(api.session_factory, code="HAM", radar={"throttle": 0.4, "braking": 0.6})
    r = await api.client.get("/compare/VER/HAM")
    assert r.status_code == 200
    body = r.json()
    assert body["driver_1_radar"]["throttle"] == 0.8
    assert body["delta_shap"]["throttle"] == 0.8 - 0.4


async def test_compare_missing_driver_is_404(api: SimpleNamespace) -> None:
    await seed_fingerprint(api.session_factory, code="VER")
    r = await api.client.get("/compare/VER/NOPE")
    assert r.status_code == 404
    assert r.json()["error_code"] == "NOT_FOUND"


async def test_stylespace_ok(api: SimpleNamespace) -> None:
    api.app.dependency_overrides[get_stylespace_fn] = lambda: (
        lambda: fake_stylespace(["VER", "HAM"])
    )
    r = await api.client.get("/stylespace")
    assert r.status_code == 200
    body = r.json()
    assert {d["code"] for d in body["drivers"]} == {"VER", "HAM"}
    assert body["unknown_drivers"] == []


async def test_stylespace_empty(api: SimpleNamespace) -> None:
    api.app.dependency_overrides[get_stylespace_fn] = lambda: (lambda: fake_stylespace([]))
    r = await api.client.get("/stylespace")
    assert r.status_code == 200
    assert r.json()["drivers"] == []
