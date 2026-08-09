"""GET /sessions — the Screen 6 session catalog (closes F19 div #77). 200 / 200-empty / 503."""

from __future__ import annotations

from types import SimpleNamespace

from app.db import models
from app.pipeline.features import FEATURE_IDS


async def _seed_two_sessions(api: SimpleNamespace) -> tuple[str, str]:
    """Two sessions on two circuits; VER laps in both, HAM (split across two per-season driver
    rows sharing one code) only in the first. Returns (bahrain_id, monaco_id)."""
    async with api.session_factory() as s:
        bahrain = models.Circuit(name="Bahrain", type=models.CircuitType.HIGH_SPEED)
        monaco = models.Circuit(name="Monaco", type=models.CircuitType.STREET)
        ver = models.Driver(code="VER", full_name="Max", team="RB", season=2023)
        ham23 = models.Driver(code="HAM", full_name="Lewis", team="M", season=2023)
        ham24 = models.Driver(code="HAM", full_name="Lewis", team="F", season=2024)
        s.add_all([bahrain, monaco, ver, ham23, ham24])
        await s.flush()
        r1 = models.RaceSession(circuit_id=bahrain.id, season=2023, round=1, session_type="R")
        r2 = models.RaceSession(circuit_id=monaco.id, season=2024, round=6, session_type="R")
        s.add_all([r1, r2])
        await s.flush()
        fv = {fid: 0.1 for fid in FEATURE_IDS}
        for race, driver, laps in ((r1, ver, 3), (r1, ham23, 2), (r1, ham24, 1), (r2, ver, 2)):
            for ln in range(1, laps + 1):
                lap = models.Lap(session_id=race.id, driver_id=driver.id, lap_number=ln,
                                 is_clean=True, error_code="OK", tyre_compound="S", tyre_age=ln,
                                 has_interpolation=False)
                s.add(lap)
                await s.flush()
                s.add(models.LapFeatures(lap_id=lap.id, feature_vector=fv, null_count=0,
                                         imputed_features=[]))
        await s.commit()
        return str(r1.id), str(r2.id)


async def test_sessions_catalog_lists_sessions_with_drivers(api: SimpleNamespace) -> None:
    bahrain_id, monaco_id = await _seed_two_sessions(api)
    r = await api.client.get("/sessions")
    assert r.status_code == 200
    rows = r.json()
    # Newest season/round first — Monaco 2024 before Bahrain 2023.
    assert [row["id"] for row in rows] == [monaco_id, bahrain_id]
    monaco, bahrain = rows
    assert monaco["circuit"] == "Monaco"
    assert monaco["circuit_type"] == "street"
    assert (monaco["season"], monaco["round"], monaco["session_type"]) == (2024, 6, "R")
    assert monaco["drivers"] == [{"code": "VER", "n_laps": 2}]
    # HAM's two per-season driver rows aggregate under ONE code (matching /replay's resolution).
    assert bahrain["drivers"] == [
        {"code": "HAM", "n_laps": 3},
        {"code": "VER", "n_laps": 3},
    ]


async def test_sessions_empty_db_is_200_empty_list(api: SimpleNamespace) -> None:
    # A fresh checkout has no sessions — honest absence, not an outage.
    r = await api.client.get("/sessions")
    assert r.status_code == 200
    assert r.json() == []


async def test_sessions_db_down_is_503(api: SimpleNamespace) -> None:
    await _seed_two_sessions(api)
    for _ in range(api.breaker.config.open_after):  # pre-open the DB breaker
        api.breaker.record_failure()
    r = await api.client.get("/sessions")
    assert r.status_code == 503
    assert r.json()["error_code"] == "DB_UNAVAILABLE"


async def test_sessions_db_not_configured_is_503_health_fail(api: SimpleNamespace) -> None:
    # DB never configured at startup (no engine built) → HEALTH_FAIL — distinct from a runtime
    # outage.
    api.state.session_factory = None
    api.state.db_breaker = None
    r = await api.client.get("/sessions")
    assert r.status_code == 503
    assert r.json()["error_code"] == "HEALTH_FAIL"
