"""The replay engine — Screen 6 Historical Race Replay Monitor (§10.7 / §5.3 / §12.1).

**Read-only.** Given a persisted ``(session, driver)``, it loads that driver's laps (+ their feature
vectors) from the DB and streams a **growing prefix** through the F10 ``monitor_stint`` detector,
emitting one :class:`ReplayLapData` per lap — its cosine ``deviation_score``, its ``alert_state`` (a
``StintMonitorState``), and the ``fingerprint_snapshot`` *as of that lap*. This is the master's
"replays the race lap by lap, showing the fingerprint evolving" (§10.7): ``monitor_stint`` is pure
over its prefix (§4.0-d), so re-calling it on ``laps[:k]`` gives lap ``k``'s causal deviation; the
whole ``laps`` array is returned in one pull for **client-side** playback (Zustand ``replay
position``; there is no websocket — §13). The engine **writes nothing** (divergence #36): it
recomputes deterministically each call, so the same session always yields the same trace.

The persisted ``laps`` carry no ``track_status`` column (F12 §8.4), but ``monitor_stint`` treats a
**null** ``track_status`` as "defer to ``is_clean``" (detector ``_is_eligible``) and ``is_clean``
already encodes F02's green determination — so the reconstructed frame sets ``track_status=None``,
losslessly (spec §10 R1). A missing session/driver → ``found=False`` (route → 404 ``NOT_FOUND``);
non-contiguous laps or a ``max_replay_laps`` truncation → ``partial=True`` (route → 206
``REPLAY_PARTIAL``); a DB outage propagates as ``DatabaseUnavailableError`` (route → 503).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from app.contracts.errors import ErrorCode

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.contracts.config import AppConfig
    from app.ml.stylespace import StyleSpaceHandle
    from app.monitoring.detector import MonitorHandle


@dataclass
class ReplayLapData:
    """One replayed lap → maps to the API ``ReplayLap`` (§8.5)."""

    lap_number: int
    feature_vector: dict[str, float | None]
    deviation_score: float | None
    alert_state: str
    fingerprint_snapshot: dict[str, Any] | None = None


@dataclass
class ReplayResult:
    """The replay trace + the flags the route maps to a status code. ``found=False`` → 404;
    ``partial=True`` → 206; otherwise 200."""

    laps: list[ReplayLapData] = field(default_factory=list)
    found: bool = True
    partial: bool = False
    flags: list[ErrorCode] = field(default_factory=list)


def _snapshot_to_dict(snapshot: Any) -> dict[str, Any] | None:
    """Serialise a F08 ``ProjectionResult`` snapshot to the compact dict the API returns. Defensive:
    any unexpected shape degrades to ``None`` (the radar simply does not update that lap)."""
    if snapshot is None:
        return None
    try:
        state = getattr(snapshot, "confidence_state", None)
        return {
            "coords_5d": getattr(snapshot, "coords_5d", None),
            "coords_2d": getattr(snapshot, "coords_2d", None),
            "confidence_state": state.value if state is not None else None,
        }
    except Exception:  # pragma: no cover — the snapshot never blocks the deviation trace
        return None


async def replay_stint(
    session_id: str,
    driver_id: str,
    *,
    session: AsyncSession,
    monitor_handle: MonitorHandle,
    config: AppConfig,
    style_handle: StyleSpaceHandle | None = None,
    log_path: str | None = None,
) -> ReplayResult:
    """Replay one persisted ``(session, driver)`` lap-by-lap through F10 ``monitor_stint``.

    ``session_id`` is the DB ``sessions.id`` UUID; ``driver_id`` is the driver **code** (e.g.
    "VER", the serving-API driver identifier) or, tolerated, a raw ``drivers.id`` UUID. Returns a
    :class:`ReplayResult`; never raises on model output (``monitor_stint`` never raises). A DB fault
    propagates (the caller's ``session_scope`` maps it to 503, no partial write)."""
    import pandas as pd
    from sqlalchemy import select

    from app.db import models
    from app.monitoring.detector import monitor_stint
    from app.pipeline.features import FEATURE_IDS

    if config.replay is None:  # pragma: no cover — guarded by config validator + route
        raise ValueError("replay: block is required to run the replay engine")
    max_laps = config.replay.max_replay_laps

    try:
        sess_uuid = uuid.UUID(str(session_id))
    except (ValueError, AttributeError):
        return ReplayResult(found=False)

    race = await session.get(models.RaceSession, sess_uuid)
    if race is None:
        return ReplayResult(found=False)
    circuit = await session.get(models.Circuit, race.circuit_id)
    if circuit is None:  # pragma: no cover — FK guarantees this, defensive only
        return ReplayResult(found=False)

    # Driver identity across the serving API is the driver **code** (e.g. "VER") — see
    # ``app/api/routes/drivers.py`` (``{id}`` is the code) and ``DriverSummary.id``. Screen 6's
    # driver picker is populated from ``/drivers``, so it passes a code here. Resolve by code,
    # tolerating a raw driver UUID too (a code is <= 3 chars, a UUID is 36 — no collision). One code
    # can map to several per-season driver rows (``UNIQUE(code, season)``), so match laps against
    # every driver row for that code. Unknown driver (neither a live UUID nor a known code) → 404.
    driver_ids: list[uuid.UUID] = []
    driver: Any | None = None
    try:
        drv_uuid: uuid.UUID | None = uuid.UUID(str(driver_id))
    except (ValueError, AttributeError):
        drv_uuid = None
    if drv_uuid is not None:
        driver = await session.get(models.Driver, drv_uuid)
        if driver is not None:
            driver_ids = [driver.id]
    if not driver_ids:
        matched = (
            await session.execute(
                select(models.Driver).where(models.Driver.code == str(driver_id))
            )
        ).scalars().all()
        if matched:
            driver = matched[0]
            driver_ids = [d.id for d in matched]
    if driver is None or not driver_ids:
        return ReplayResult(found=False)

    rows = (
        await session.execute(
            select(models.Lap, models.LapFeatures)
            .join(models.LapFeatures, models.LapFeatures.lap_id == models.Lap.id)
            .where(models.Lap.session_id == sess_uuid, models.Lap.driver_id.in_(driver_ids))
            .order_by(models.Lap.lap_number)
        )
    ).all()
    if not rows:
        return ReplayResult(found=False)

    truncated = len(rows) > max_laps
    kept = rows[:max_laps]
    if truncated:
        _log_truncation(session_id, driver_id, len(rows), max_laps, log_path)

    lap_numbers = [int(lap.lap_number) for lap, _ in kept]
    contiguous = lap_numbers == list(range(lap_numbers[0], lap_numbers[0] + len(lap_numbers)))

    # Reconstruct the monitor_stint frame (track_status=None → defer to is_clean; spec §10 R1).
    frame_rows: list[dict[str, Any]] = []
    for lap, feats in kept:
        fv = feats.feature_vector
        row: dict[str, Any] = {
            "lap_number": int(lap.lap_number),
            "tyre_age": lap.tyre_age,
            "track_status": None,
            "is_clean": bool(lap.is_clean),
            "has_interpolation": bool(lap.has_interpolation),
        }
        for fid in FEATURE_IDS:
            row[fid] = fv.get(fid)
        frame_rows.append(row)
    frame = pd.DataFrame(frame_rows)

    driver_name = driver.code
    circuit_name = circuit.name
    circuit_type = circuit.type.value

    out = ReplayResult(partial=truncated or not contiguous)
    if truncated:
        out.flags.append(ErrorCode.REPLAY_PARTIAL)

    # Growing-prefix replay: lap k's causal deviation + the fingerprint as of lap k (§4.0-d).
    for k in range(1, len(frame) + 1):
        res = monitor_stint(
            frame.head(k),
            driver=driver_name,
            circuit=circuit_name,
            circuit_type=circuit_type,
            handle=monitor_handle,
            config=config,
            style_handle=style_handle,
            log_path=log_path,
        )
        last = res.laps[-1] if res.laps else None
        state = last.state if last is not None else res.monitor_state
        snapshot = (
            _snapshot_to_dict(res.fingerprint_snapshot) if res.fingerprint_available else None
        )
        lap_obj, feats_obj = kept[k - 1]
        out.laps.append(
            ReplayLapData(
                lap_number=int(lap_obj.lap_number),
                feature_vector={fid: feats_obj.feature_vector.get(fid) for fid in FEATURE_IDS},
                deviation_score=last.deviation_score if last is not None else None,
                alert_state=state.value,
                fingerprint_snapshot=snapshot,
            )
        )
    return out


def _log_truncation(
    session_id: str, driver_id: str, total: int, cap: int, log_path: str | None
) -> None:
    """Never silently truncate: a ``max_replay_laps`` cap is logged (spec §8 "no silent caps")."""
    from contextlib import suppress

    from app.contracts.logging import log_data_quality

    if log_path is None:
        return
    with suppress(Exception):
        log_data_quality(
            error_code=ErrorCode.REPLAY_PARTIAL,
            entity_type="replay",
            entity_id=f"{session_id}/{driver_id}",
            error_detail=f"replay truncated to {cap} of {total} laps (max_replay_laps)",
            fallback_used=True,
            log_path=log_path,
        )
