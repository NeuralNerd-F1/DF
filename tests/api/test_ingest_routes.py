"""F14 POST /ingest/session (202) + GET /ingest/status (§8.5). Fake broker + in-proc job store."""

from __future__ import annotations

from types import SimpleNamespace

from app.tasks.ingest import InProcessJobStore, JobState


class _FakeCelery:
    def __init__(self) -> None:
        self.sent: list[tuple[str, list, str]] = []

    def send_task(self, name: str, *, args: list, task_id: str) -> None:
        self.sent.append((name, args, task_id))


class _DownCelery:
    def send_task(self, name: str, *, args: list, task_id: str) -> None:
        raise ConnectionError("broker unreachable")


async def test_ingest_session_returns_202_and_enqueues(api: SimpleNamespace) -> None:
    celery = _FakeCelery()
    api.state.celery_app = celery
    api.state.job_store = InProcessJobStore()
    r = await api.client.post(
        "/ingest/session", json={"year": 2023, "event": "Bahrain", "session": "R"}
    )
    assert r.status_code == 202
    body = r.json()
    assert body["session_id"] and body["quality_report"]["state"] == "PENDING"
    assert body["laps_accepted"] == 0  # pending placeholder (divergence #31)
    assert celery.sent and celery.sent[0][2] == body["session_id"]


async def test_ingest_session_broker_down_is_503_queue_unavailable(api: SimpleNamespace) -> None:
    api.state.celery_app = _DownCelery()
    api.state.job_store = InProcessJobStore()
    r = await api.client.post(
        "/ingest/session", json={"year": 2023, "event": "Bahrain", "session": "R"}
    )
    assert r.status_code == 503
    assert r.json()["error_code"] == "QUEUE_UNAVAILABLE"


async def test_ingest_session_unconfigured_is_503(api: SimpleNamespace) -> None:
    # No celery_app on state → not configured → 503 HEALTH_FAIL.
    r = await api.client.post(
        "/ingest/session", json={"year": 2023, "event": "Bahrain", "session": "R"}
    )
    assert r.status_code == 503
    assert r.json()["error_code"] == "HEALTH_FAIL"


async def test_ingest_session_malformed_body_is_422(api: SimpleNamespace) -> None:
    api.state.celery_app = _FakeCelery()
    api.state.job_store = InProcessJobStore()
    r = await api.client.post("/ingest/session", json={"event": "Bahrain"})  # missing year
    assert r.status_code == 422


async def test_ingest_status_reports_state(api: SimpleNamespace) -> None:
    store = InProcessJobStore()
    api.state.job_store = store
    await store.update("j1", status=JobState.PROCESSING, progress=0.5)
    r = await api.client.get("/ingest/status/j1")
    assert r.status_code == 200
    assert r.json() == {
        "status": "PROCESSING", "progress": 0.5, "error_detail": None, "result": None,
    }


async def test_ingest_status_complete_carries_terminal_report(api: SimpleNamespace) -> None:
    """On COMPLETE the poll surfaces the real §8.5 counts (div #31 — the master's response contract
    fulfilled, deferred to the poll)."""
    store = InProcessJobStore()
    api.state.job_store = store
    report = {
        "session_id": "j2", "laps_accepted": 42, "laps_rejected": 3,
        "rejection_summary": {"LAP_001": 3}, "quality_report": {"state": "COMPLETE"},
    }
    await store.update("j2", status=JobState.COMPLETE, progress=1.0, result=report)
    r = await api.client.get("/ingest/status/j2")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "COMPLETE"
    assert body["result"]["laps_accepted"] == 42
    assert body["result"]["rejection_summary"] == {"LAP_001": 3}


async def test_ingest_status_unknown_is_404(api: SimpleNamespace) -> None:
    api.state.job_store = InProcessJobStore()
    r = await api.client.get("/ingest/status/nope")
    assert r.status_code == 404
    assert r.json()["error_code"] == "NOT_FOUND"
