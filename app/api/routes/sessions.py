"""GET /sessions — the persisted-session catalog (Screen 6 session picker).

Closes F19 divergence #77: sessions were listed nowhere, so the Replay Monitor forced a raw UUID
input. This route lists every persisted session (circuit, season, round, type) with the driver
codes that have replayable laps in it. Read-only. An **empty** DB is honest absence → ``200 []``
(a fresh checkout is not an outage); DB never configured → 503 ``HEALTH_FAIL``; a runtime DB fault
→ 503 ``DB_UNAVAILABLE`` (both via ``get_db_session``, mirroring ``/replay``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends
from sqlalchemy import func, select

from app.api.deps import get_db_session
from app.api.schemas import SessionDriverSchema, SessionSummary
from app.db import models

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(tags=["replay"])


@router.get("/sessions", response_model=list[SessionSummary])
async def sessions(
    session: AsyncSession = Depends(get_db_session),
) -> list[SessionSummary]:
    """List persisted sessions with their replayable drivers (newest season/round first)."""
    rows = (
        await session.execute(
            select(models.RaceSession, models.Circuit)
            .join(models.Circuit, models.RaceSession.circuit_id == models.Circuit.id)
            .order_by(
                models.RaceSession.season.desc(),
                models.RaceSession.round.desc(),
                models.RaceSession.session_type,
            )
        )
    ).all()

    # One grouped pass over laps → per-(session, driver-code) lap counts. Grouping by CODE (not
    # driver id) aggregates across per-season driver rows (UNIQUE(code, season)), matching how
    # /replay resolves a code, so each (session, code) pair arrives exactly once.
    lap_rows = (
        await session.execute(
            select(models.Lap.session_id, models.Driver.code, func.count())
            .join(models.Driver, models.Lap.driver_id == models.Driver.id)
            .group_by(models.Lap.session_id, models.Driver.code)
        )
    ).all()
    by_session: dict[object, dict[str, int]] = {}
    for sess_id, code, n in lap_rows:
        by_session.setdefault(sess_id, {})[str(code)] = int(n)

    out: list[SessionSummary] = []
    for race, circuit in rows:
        counts = by_session.get(race.id, {})
        out.append(
            SessionSummary(
                id=str(race.id),
                circuit=circuit.name,
                circuit_type=circuit.type.value,
                season=int(race.season),
                round=int(race.round),
                session_type=race.session_type,
                drivers=[
                    SessionDriverSchema(code=code, n_laps=n)
                    for code, n in sorted(counts.items())
                ],
            )
        )
    return out
