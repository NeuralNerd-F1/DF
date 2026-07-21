"""F14 GET /replay/{session_id}/{driver_id} (§8.5). 200 / 206 / 404 / 503, monitor_stint faked."""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Any

import app.monitoring.detector as det
import pytest
from app.db import models
from app.monitoring.detector import StintMonitorState
from app.pipeline.features import FEATURE_IDS


def _fake_monitor(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(frame: Any, **_kw: Any) -> SimpleNamespace:
        last = SimpleNamespace(state=StintMonitorState.NOMINAL, deviation_score=0.02 * len(frame))
        return SimpleNamespace(
            laps=[last], monitor_state=StintMonitorState.NOMINAL,
            fingerprint_snapshot=None, fingerprint_available=False,
        )

    monkeypatch.setattr(det, "monitor_stint", fake)


async def _seed(api: SimpleNamespace, lap_numbers: list[int]) -> tuple[str, str]:
    async with api.session_factory() as s:
        c = models.Circuit(name="Bahrain", type=models.CircuitType.HIGH_SPEED)
        d = models.Driver(code="HAM", full_name="Lewis", team="M", season=2023)
        s.add_all([c, d])
        await s.flush()
        r = models.RaceSession(circuit_id=c.id, season=2023, round=1, session_type="R")
        s.add(r)
        await s.flush()
        for ln in lap_numbers:
            lap = models.Lap(session_id=r.id, driver_id=d.id, lap_number=ln, is_clean=True,
                             error_code="OK", tyre_compound="S", tyre_age=ln,
                             has_interpolation=False)
            s.add(lap)
            await s.flush()
            s.add(models.LapFeatures(lap_id=lap.id, feature_vector={fid: 0.1 * ln for fid in
                                     FEATURE_IDS}, null_count=0, imputed_features=[]))
        await s.commit()
        return str(r.id), str(d.id)


async def test_replay_ok(api: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_monitor(monkeypatch)
    sid, did = await _seed(api, [1, 2, 3, 4, 5])
    r = await api.client.get(f"/replay/{sid}/{did}")
    assert r.status_code == 200
    laps = r.json()["laps"]
    assert [lp["lap_number"] for lp in laps] == [1, 2, 3, 4, 5]
    assert laps[0]["alert_state"] == "NOMINAL"


async def test_replay_unknown_is_404(api: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_monitor(monkeypatch)
    sid, _ = await _seed(api, [1, 2, 3])
    r = await api.client.get(f"/replay/{sid}/{uuid.uuid4()}")
    assert r.status_code == 404
    assert r.json()["error_code"] == "NOT_FOUND"


async def test_replay_partial_is_206(api: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_monitor(monkeypatch)
    sid, did = await _seed(api, [1, 2, 3, 5])  # gap → partial
    r = await api.client.get(f"/replay/{sid}/{did}")
    assert r.status_code == 206
    assert len(r.json()["laps"]) == 4


async def test_replay_db_down_is_503(api: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_monitor(monkeypatch)
    sid, did = await _seed(api, [1, 2, 3])
    for _ in range(api.breaker.config.open_after):  # pre-open the DB breaker
        api.breaker.record_failure()
    r = await api.client.get(f"/replay/{sid}/{did}")
    assert r.status_code == 503
    assert r.json()["error_code"] == "DB_UNAVAILABLE"


async def test_replay_monitor_not_loaded_is_503(
    api: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_monitor(monkeypatch)
    sid, did = await _seed(api, [1, 2, 3])
    api.state.monitor_handle = None  # not loaded → 503
    r = await api.client.get(f"/replay/{sid}/{did}")
    assert r.status_code == 503
