"""F14 replay engine — growing-prefix replay of a persisted stint through F10 (§4.2/§10.7).

Plumbing tests fake ``monitor_stint`` (validating the reconstructed frame's input contract); one
integration test drives the REAL calibrated detector end-to-end (validating the whole seam).
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import app.monitoring.detector as det
import pytest
import pytest_asyncio
from app.contracts.config import AppConfig, load_config
from app.db import models
from app.db.base import Base
from app.monitoring.detector import _SERVING_REQUIRED, StintMonitorState
from app.replay.engine import replay_stint
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool


@pytest.fixture
def cfg(repo_config_path: Path) -> AppConfig:
    return load_config(repo_config_path)


@pytest_asyncio.fixture
async def session_factory() -> AsyncIterator[async_sessionmaker]:
    engine: AsyncEngine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine.sync_engine, "connect")
    def _fk(dbapi_conn: object, _rec: object) -> None:
        cur = dbapi_conn.cursor()  # type: ignore[attr-defined]
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


async def _seed(
    session_factory: async_sessionmaker,
    *,
    lap_numbers: list[int],
    feature_vectors: list[dict[str, float | None]] | None = None,
    is_clean: bool = True,
    circuit_name: str = "Bahrain",
) -> tuple[str, str]:
    from app.pipeline.features import FEATURE_IDS

    async with session_factory() as s:
        c = models.Circuit(name=circuit_name, type=models.CircuitType.HIGH_SPEED)
        d = models.Driver(code="HAM", full_name="Lewis", team="M", season=2023)
        s.add_all([c, d])
        await s.flush()
        r = models.RaceSession(circuit_id=c.id, season=2023, round=1, session_type="R")
        s.add(r)
        await s.flush()
        for i, ln in enumerate(lap_numbers):
            lap = models.Lap(
                session_id=r.id, driver_id=d.id, lap_number=ln, is_clean=is_clean,
                error_code="OK" if is_clean else "LAP_001", tyre_compound="S", tyre_age=ln,
                has_interpolation=False,
            )
            s.add(lap)
            await s.flush()
            fv = feature_vectors[i] if feature_vectors else {fid: 0.1 * ln for fid in FEATURE_IDS}
            s.add(models.LapFeatures(lap_id=lap.id, feature_vector=fv, null_count=0,
                                     imputed_features=[]))
        await s.commit()
        return str(r.id), str(d.id)


def _fake_monitor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch monitor_stint with a fake that (a) validates the input frame carries the required
    columns, and (b) returns a rising deviation for the prefix's last lap."""

    def fake(frame: Any, *, driver: str, circuit: str, circuit_type: str | None, handle: Any,
             config: Any, style_handle: Any = None, log_path: Any = None) -> SimpleNamespace:
        for col in _SERVING_REQUIRED:
            assert col in frame.columns, f"reconstructed frame missing {col}"
        k = len(frame)
        last = SimpleNamespace(
            state=StintMonitorState.NOMINAL, deviation_score=0.01 * k, threshold=0.5
        )
        return SimpleNamespace(
            laps=[last], monitor_state=StintMonitorState.NOMINAL,
            fingerprint_snapshot=None, fingerprint_available=False, alert=None,
        )

    monkeypatch.setattr(det, "monitor_stint", fake)


async def test_contiguous_stint_returns_full_trace(
    cfg: AppConfig, session_factory: async_sessionmaker, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_monitor(monkeypatch)
    sid, did = await _seed(session_factory, lap_numbers=[1, 2, 3, 4, 5])
    async with session_factory() as s:
        res = await replay_stint(sid, did, session=s, monitor_handle=None, config=cfg)
    assert res.found is True and res.partial is False
    assert len(res.laps) == 5
    assert [lp.lap_number for lp in res.laps] == [1, 2, 3, 4, 5]
    # Growing prefix: lap k's deviation = 0.01*k (last-lap of the k-prefix).
    assert res.laps[4].deviation_score == pytest.approx(0.05)
    assert all(lp.alert_state == "NOMINAL" for lp in res.laps)
    # τ is forwarded per lap (div #72/#73 seam closed); no alert fired → alert stays None.
    assert all(lp.threshold == pytest.approx(0.5) for lp in res.laps)
    assert all(lp.alert is None for lp in res.laps)


async def test_alert_payload_is_forwarded_from_its_firing_lap(
    cfg: AppConfig, session_factory: async_sessionmaker, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once the k-prefix's monitor carries a DegradationAlert, the replayed lap forwards it as a
    serialised dict (div #74 seam closed) — earlier laps stay alert=None, never backfilled."""
    from app.monitoring.detector import AlertConfidence, DegradationAlert

    fire_at = 4

    def fake(frame: Any, **_kw: Any) -> SimpleNamespace:
        k = len(frame)
        fired = k >= fire_at
        state = StintMonitorState.ALERT if fired else StintMonitorState.NOMINAL
        last = SimpleNamespace(state=state, deviation_score=0.01 * k, threshold=0.5)
        alert = (
            DegradationAlert(
                alert_lap=fire_at, first_crossing_lap=fire_at - 1,
                laps_since_threshold_crossed=k - fire_at + 1, deviation_score=0.01 * k,
                threshold=0.5, alert_confidence=AlertConfidence.HIGH,
                feature_attribution={"F01": 0.01 * k}, top_features=["F01"],
                optimal_window_remaining_laps=3,
            )
            if fired
            else None
        )
        return SimpleNamespace(
            laps=[last], monitor_state=state,
            fingerprint_snapshot=None, fingerprint_available=False, alert=alert,
        )

    monkeypatch.setattr(det, "monitor_stint", fake)
    sid, did = await _seed(session_factory, lap_numbers=[1, 2, 3, 4, 5])
    async with session_factory() as s:
        res = await replay_stint(sid, did, session=s, monitor_handle=None, config=cfg)
    assert [lp.alert is None for lp in res.laps] == [True, True, True, False, False]
    fired_lap = res.laps[fire_at - 1]
    assert fired_lap.alert is not None
    assert fired_lap.alert["alert_lap"] == fire_at
    assert fired_lap.alert["alert_confidence"] == "HIGH"  # enum serialised to its value
    assert fired_lap.alert["top_features"] == ["F01"]
    assert fired_lap.alert["optimal_window_remaining_laps"] == 3
    # The as-of-lap payload advances with the prefix (laps_since grows), not a frozen copy.
    assert res.laps[4].alert is not None
    assert res.laps[4].alert["laps_since_threshold_crossed"] == 2


async def test_unknown_session_or_driver_is_not_found(
    cfg: AppConfig, session_factory: async_sessionmaker, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_monitor(monkeypatch)
    sid, did = await _seed(session_factory, lap_numbers=[1, 2, 3])
    async with session_factory() as s:
        r1 = await replay_stint(str(uuid.uuid4()), did, session=s, monitor_handle=None, config=cfg)
        r2 = await replay_stint(sid, str(uuid.uuid4()), session=s, monitor_handle=None, config=cfg)
        r3 = await replay_stint("not-a-uuid", did, session=s, monitor_handle=None, config=cfg)
    assert r1.found is False and r2.found is False and r3.found is False


async def test_driver_resolved_by_code(
    cfg: AppConfig, session_factory: async_sessionmaker, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The serving API identifies a driver by CODE (Screen 6's picker passes the ``/drivers`` id,
    which is the code): replay must resolve "HAM" and the raw driver UUID alike; unknown is 404."""
    _fake_monitor(monkeypatch)
    sid, did = await _seed(session_factory, lap_numbers=[1, 2, 3, 4, 5])
    async with session_factory() as s:
        by_code = await replay_stint(sid, "HAM", session=s, monitor_handle=None, config=cfg)
        by_uuid = await replay_stint(sid, did, session=s, monitor_handle=None, config=cfg)
        unknown = await replay_stint(sid, "ZZZ", session=s, monitor_handle=None, config=cfg)
    assert by_code.found is True
    assert [lp.lap_number for lp in by_code.laps] == [1, 2, 3, 4, 5]
    # Code and raw UUID resolve to the identical trace.
    assert [lp.lap_number for lp in by_code.laps] == [lp.lap_number for lp in by_uuid.laps]
    assert unknown.found is False  # unknown code → 404, never a silent empty replay


async def test_gap_in_laps_is_partial(
    cfg: AppConfig, session_factory: async_sessionmaker, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_monitor(monkeypatch)
    sid, did = await _seed(session_factory, lap_numbers=[1, 2, 3, 5])  # missing lap 4
    async with session_factory() as s:
        res = await replay_stint(sid, did, session=s, monitor_handle=None, config=cfg)
    assert res.found is True and res.partial is True
    assert len(res.laps) == 4


async def test_truncation_is_partial_and_logged(
    cfg: AppConfig, session_factory: async_sessionmaker, monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _fake_monitor(monkeypatch)
    cfg.replay.max_replay_laps = 3
    log = tmp_path / "dq.jsonl"
    sid, did = await _seed(session_factory, lap_numbers=[1, 2, 3, 4, 5])
    async with session_factory() as s:
        res = await replay_stint(
            sid, did, session=s, monitor_handle=None, config=cfg, log_path=str(log)
        )
    assert res.partial is True and len(res.laps) == 3  # capped, no silent truncation
    import json

    rows = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if x]
    assert any(r["error_code"] == "REPLAY_PARTIAL" for r in rows)


# --- integration: the REAL calibrated detector -------------------------------------------------
async def test_real_detector_end_to_end(
    session_factory: async_sessionmaker, tmp_path: Path
) -> None:
    """Persist a real drifting stint (raw F03 vectors) and replay through the REAL monitor_stint —
    validates the whole seam (frame columns, internal F04 normalisation, alert progression)."""
    from app.monitoring.detector import calibrate_stint_variation, load_detector
    from app.pipeline.features import FEATURE_IDS
    from app.pipeline.normalise import fit_normaliser

    from tests.monitoring.conftest import build_training_matrix, make_stint

    cfg = load_config(Path(__file__).resolve().parents[2] / "config.yaml")
    cfg.paths.data_quality_log = str(tmp_path / "dq.jsonl")
    cfg.paths.artefacts_dir = str(tmp_path / "artefacts")
    cfg.monitoring.mlflow_tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    art = Path(cfg.paths.artefacts_dir)

    fit_normaliser(build_training_matrix(), config=cfg)
    calibrate_stint_variation(build_training_matrix(), config=cfg, artefact_dir=art)
    handle = load_detector(config=cfg, artefact_dir=art)

    # A drifting VER stint at Bahrain (raw vectors) → deviation should rise and cross τ.
    stint = make_stint(driver="VER", circuit="Bahrain", circuit_type="high_speed",
                       n_laps=12, drift_from=7, drift_mag=6.0, seed=1)
    vectors = [{fid: float(row[fid]) for fid in FEATURE_IDS} for _, row in stint.iterrows()]

    async with session_factory() as s:
        c = models.Circuit(name="Bahrain", type=models.CircuitType.HIGH_SPEED)
        d = models.Driver(code="VER", full_name="Max", team="RB", season=2023)
        s.add_all([c, d])
        await s.flush()
        r = models.RaceSession(circuit_id=c.id, season=2023, round=1, session_type="R")
        s.add(r)
        await s.flush()
        for i, ln in enumerate(stint["lap_number"].tolist()):
            lap = models.Lap(session_id=r.id, driver_id=d.id, lap_number=int(ln), is_clean=True,
                             error_code="OK", tyre_compound="S", tyre_age=int(ln),
                             has_interpolation=False)
            s.add(lap)
            await s.flush()
            s.add(models.LapFeatures(lap_id=lap.id, feature_vector=vectors[i], null_count=0,
                                     imputed_features=[]))
        await s.commit()
        sid, did = str(r.id), str(d.id)

    async with session_factory() as s:
        res = await replay_stint(sid, did, session=s, monitor_handle=handle, config=cfg)

    assert res.found is True and len(res.laps) == 12
    states = {lp.alert_state for lp in res.laps}
    # Growing-prefix (causal) view: laps before the 8-lap activation floor are INACTIVE (the monitor
    # lacks enough data yet — the honest "as-if-live" state), and the injected drift crosses τ.
    assert "INACTIVE" in states
    assert "ALERT" in states or "WATCH" in states  # drift detected on later laps
    # Real deviations were computed once the monitor activated (not all None).
    assert any(lp.deviation_score is not None for lp in res.laps)
    # The real detector's τ rides along with every scored lap (div #72/#73 seam closed) …
    scored = [lp for lp in res.laps if lp.deviation_score is not None]
    assert scored and all(lp.threshold is not None for lp in scored)
    # … and if the drift confirmed into ALERT, the real DegradationAlert payload rides along too.
    alerted = [lp for lp in res.laps if lp.alert_state == "ALERT"]
    for lp in alerted:
        assert lp.alert is not None
        assert lp.alert["alert_confidence"] in {"LOW", "MEDIUM", "HIGH"}
        assert lp.alert["top_features"]  # attribution names present for the banner
