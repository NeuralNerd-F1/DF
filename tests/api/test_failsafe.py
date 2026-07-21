"""RQ5: no unhandled exceptions (500 API_ERROR), DB-down → 503 no partial write, concurrency."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from app.api.deps import get_projector, get_stylespace_fn

from tests.api.conftest import fake_prediction, fake_projection, fake_stylespace
from tests.api.test_predict import _predictor


def _dq_rows(api: SimpleNamespace) -> list[dict[str, object]]:
    """Every row written to the test's data_quality_log sink so far."""
    log = Path(api.config.paths.data_quality_log)
    if not log.exists():
        return []
    return [json.loads(ln) for ln in log.read_text(encoding="utf-8").splitlines() if ln.strip()]


async def test_unhandled_exception_becomes_typed_500(api: SimpleNamespace) -> None:
    """Any uncaught fault on a route → typed 500 API_ERROR, never a raw stack (§11)."""

    def _boom() -> object:
        raise RuntimeError("unexpected internal fault")

    api.app.dependency_overrides[get_stylespace_fn] = _boom
    r = await api.client.get("/stylespace")
    assert r.status_code == 500
    assert r.json()["error_code"] == "API_ERROR"
    assert "Internal error" in r.json()["detail"]


async def test_db_backed_read_db_down_is_503(api: SimpleNamespace) -> None:
    """DB-backed read + breaker open → 503 DB_UNAVAILABLE (no partial write; it is a read)."""
    for _ in range(3):  # open the DB breaker
        api.state.db_breaker.record_failure()
    r = await api.client.get("/drivers")
    assert r.status_code == 503
    assert r.json()["error_code"] == "DB_UNAVAILABLE"


async def test_db_integrity_error_becomes_typed_409(api: SimpleNamespace) -> None:
    """A ``DbIntegrityError`` on any route → typed 409 DB_INTEGRITY (§6). F13 has no write route
    yet (see §4.0/divergence), so the mapping is exercised via a raising dependency — it is wired
    and ready for the F14 write paths."""
    from app.db.results import DbIntegrityError

    def _raise_integrity() -> object:
        raise DbIntegrityError("duplicate key violates unique constraint")

    api.app.dependency_overrides[get_stylespace_fn] = _raise_integrity
    r = await api.client.get("/stylespace")
    assert r.status_code == 409
    assert r.json()["error_code"] == "DB_INTEGRITY"
    assert any(row["error_code"] == "DB_INTEGRITY" for row in _dq_rows(api))  # §6 "Logged? yes"


async def test_stateless_routes_stay_200_when_db_down(api: SimpleNamespace) -> None:
    """§8/§4.0(d) — stateless compute needs no DB, so these routes stay 200 even with the DB down.

    Locks the structural guarantee (no stateless route injects ``get_db_session``) against a future
    regression that accidentally threads a DB dependency into one of them."""
    for _ in range(3):  # open the DB breaker — the DB is "down"
        api.state.db_breaker.record_failure()

    _predictor(api, fake_prediction())
    api.app.dependency_overrides[get_projector] = lambda: (
        lambda laps, circuit, ct: fake_projection()
    )
    api.app.dependency_overrides[get_stylespace_fn] = lambda: (lambda: fake_stylespace(["VER"]))

    laps = [{"F01": 1.0} for _ in range(30)]
    predict = await api.client.post(
        "/predict", json={"feature_vector": {"F01": 1.0}, "circuit": "Monaco"}
    )
    fingerprint = await api.client.post(
        "/fingerprint/unknown", json={"laps": laps, "circuit": "Monaco"}
    )
    stylespace = await api.client.get("/stylespace")

    assert predict.status_code == 200  # stateless compute — unaffected by the DB outage
    assert fingerprint.status_code == 200
    assert stylespace.status_code == 200


async def test_db_route_not_configured_is_503_health_fail(api: SimpleNamespace) -> None:
    """DB never configured at startup (no engine built) → 503 HEALTH_FAIL on a DB-backed route —
    the not-loaded-component code, distinct from a runtime DB_UNAVAILABLE outage."""
    api.state.session_factory = None
    api.state.db_breaker = None
    r = await api.client.get("/drivers")
    assert r.status_code == 503
    assert r.json()["error_code"] == "HEALTH_FAIL"


async def test_db_unavailable_writes_quality_log_row(api: SimpleNamespace) -> None:
    """DB outage on a DB-backed route is observable: a DB_UNAVAILABLE dq row is written (§6)."""
    for _ in range(3):  # open the DB breaker
        api.state.db_breaker.record_failure()
    r = await api.client.get("/drivers")
    assert r.status_code == 503
    assert any(row["error_code"] == "DB_UNAVAILABLE" for row in _dq_rows(api))


async def test_too_few_laps_gate_writes_quality_log_row(api: SimpleNamespace) -> None:
    """The <min_projection_laps gate is observable: an INVALID_INPUT dq row is written (§6, #1)."""
    r = await api.client.post(
        "/fingerprint/unknown", json={"laps": [{"F01": 1.0}], "circuit": "Monaco"}
    )
    assert r.status_code == 400
    assert any(
        row["error_code"] == "INVALID_INPUT" and row["entity_type"] == "fingerprint_unknown"
        for row in _dq_rows(api)
    )


async def test_50_concurrent_predicts_are_consistent(api: SimpleNamespace) -> None:
    """§11 — 50 concurrent predictions all complete with identical bodies for identical input."""
    _predictor(api, fake_prediction())
    body = {"feature_vector": {"F01": 1.0, "F02": 0.5}, "circuit": "Monaco"}
    responses = await asyncio.gather(*(api.client.post("/predict", json=body) for _ in range(50)))
    assert all(r.status_code == 200 for r in responses)
    payloads = [r.json() for r in responses]
    assert all(p == payloads[0] for p in payloads)  # same input -> same output
