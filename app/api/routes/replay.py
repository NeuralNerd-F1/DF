"""GET /replay/{session_id}/{driver_id} — the replay engine (F14, §8.5/§10.7/§12.1).

Reads a persisted ``(session, driver)`` and streams it lap-by-lap through F10 ``monitor_stint``,
returning one ``ReplayLap`` per lap (deviation + alert state + evolving fingerprint snapshot) for
Screen 6's client-side playback. Read-only — writes nothing (divergence #36). Unknown session/driver
→ 404 ``NOT_FOUND``; partial data (non-contiguous laps / ``max_replay_laps`` truncation) → **206**
``REPLAY_PARTIAL``; DB down → 503 ``DB_UNAVAILABLE`` (via ``session_scope``, no partial write).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, Response

from app.api.deps import AppState, get_apex_state, get_db_session, get_monitor_handle
from app.api.errors import not_found
from app.api.schemas import ReplayLap, ReplayResponse
from app.replay.engine import replay_stint

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.monitoring.detector import MonitorHandle

router = APIRouter(tags=["replay"])


@router.get("/replay/{session_id}/{driver_id}", response_model=ReplayResponse)
async def replay(
    session_id: str,
    driver_id: str,
    response: Response,
    session: AsyncSession = Depends(get_db_session),
    monitor_handle: MonitorHandle = Depends(get_monitor_handle),
    state: AppState = Depends(get_apex_state),
) -> ReplayResponse:
    """Replay a persisted stint through the degradation detector (§10.7). 200 full · 206 partial ·
    404 unknown."""
    result = await replay_stint(
        session_id,
        driver_id,
        session=session,
        monitor_handle=monitor_handle,
        config=state.config,
        style_handle=state.style_handle,
        log_path=state.config.paths.data_quality_log,
    )
    if not result.found:
        raise not_found(f"session {session_id!r} / driver {driver_id!r} not found")
    if result.partial:
        response.status_code = 206  # §8.5 — partial data
    return ReplayResponse(
        laps=[
            ReplayLap(
                lap_number=lap.lap_number,
                feature_vector=lap.feature_vector,
                deviation_score=lap.deviation_score,
                alert_state=lap.alert_state,
                fingerprint_snapshot=lap.fingerprint_snapshot,
            )
            for lap in result.laps
        ]
    )
