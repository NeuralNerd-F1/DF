"""POST /fingerprint/unknown, GET /compare/{id1}/{id2}, GET /stylespace (F08 + F07 + DB)."""

from __future__ import annotations

import uuid
from contextlib import suppress
from typing import NoReturn

import pandas as pd
from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    AppState,
    ProjectFn,
    StyleSpaceFn,
    get_apex_state,
    get_db_session,
    get_projector,
    get_stylespace_fn,
)
from app.api.errors import invalid_input, not_found
from app.api.schemas import (
    CompareResponse,
    DataQualitySchema,
    NeighbourSchema,
    StyleSpaceDriver,
    StyleSpaceResponse,
    UnknownFingerprintRequest,
    UnknownFingerprintResponse,
)
from app.contracts.errors import ErrorCode
from app.contracts.logging import log_data_quality
from app.db import models

router = APIRouter(tags=["style"])

# §8.5/§11 request bounds. Primary source is api.min_projection_laps / api.max_request_laps
# (config.yaml); these are only fallbacks for the api-block-absent case, which create_app forbids at
# build time (no magic numbers in the live path).
_MIN_LAPS_FALLBACK = 5
_MAX_LAPS_FALLBACK = 5000


def _reject_laps(state: AppState, circuit: str, detail: str, message: str) -> NoReturn:
    """Log an input-gate rejection (§6 'Logged? yes') then raise a typed 400. The log-write is
    suppressed so a broken sink can never turn the 400 into a 500 (non-negotiable #1)."""
    with suppress(Exception):
        log_data_quality(
            error_code=ErrorCode.INVALID_INPUT,
            entity_type="fingerprint_unknown",
            entity_id=circuit,
            error_detail=detail,
            fallback_used=False,
            log_path=state.config.paths.data_quality_log,
        )
    raise invalid_input(message)


@router.post("/fingerprint/unknown", response_model=UnknownFingerprintResponse)
async def fingerprint_unknown(
    req: UnknownFingerprintRequest,
    projector: ProjectFn = Depends(get_projector),
    state: AppState = Depends(get_apex_state),
) -> UnknownFingerprintResponse:
    api = state.config.api
    min_laps = api.min_projection_laps if api is not None else _MIN_LAPS_FALLBACK
    max_laps = api.max_request_laps if api is not None else _MAX_LAPS_FALLBACK
    n = len(req.laps)
    if n < min_laps:
        _reject_laps(
            state,
            req.circuit,
            f"{n} laps < {min_laps} projection floor",
            f"at least {min_laps} laps are required to project an unknown driver",
        )
    if n > max_laps:  # §11 DoS guard — reject an oversized payload before pd.DataFrame builds it
        _reject_laps(
            state,
            req.circuit,
            f"{n} laps > {max_laps} cap (DoS guard)",
            f"at most {max_laps} laps may be submitted",
        )
    laps_df = pd.DataFrame(req.laps)
    proj = projector(laps_df, req.circuit, req.circuit_type)

    warnings: list[ErrorCode] = []
    if proj.sparse:
        warnings.append(ErrorCode.SPARSE_PROJECTION)  # §8.5 — <30 laps: 200 with warning
    if proj.ood:
        warnings.append(ErrorCode.OUT_OF_DISTRIBUTION)

    return UnknownFingerprintResponse(
        fingerprint_id=str(uuid.uuid4()),
        fingerprint_vector=proj.coords_5d,
        nearest_neighbours=[
            NeighbourSchema(
                driver=n.driver, cosine_similarity=n.cosine_similarity, distance=n.distance
            )
            for n in proj.nearest_neighbours
        ],
        risk_flags=proj.risk_flags,
        data_quality=DataQualitySchema(
            n_laps_used=proj.n_laps_used, n_laps_rejected=proj.laps_rejected
        ),
        confidence_state=proj.confidence_state,
        warnings=warnings,
    )


async def _radar_for(session: AsyncSession, code: str) -> dict[str, float]:
    """A driver's LATEST fingerprint radar from the DB, or raise 404 if the driver has none. Selects
    the newest (season, circuit) deterministically so /compare agrees with /drivers + /fingerprint
    (§11 same-input-same-output — no oldest-vs-latest divergence, no PG tiebreak ambiguity)."""
    driver_ids = list(
        (
            await session.execute(select(models.Driver.id).where(models.Driver.code == code))
        ).scalars().all()
    )
    if not driver_ids:
        raise not_found(f"driver {code!r} not found")
    fp = (
        await session.execute(
            select(models.Fingerprint)
            .where(models.Fingerprint.driver_id.in_(driver_ids))
            .order_by(models.Fingerprint.season.desc(), models.Fingerprint.circuit_id.desc())
        )
    ).scalars().first()
    if fp is None:
        raise not_found(f"no fingerprint for driver {code!r}")
    return dict(fp.radar_values or {})


@router.get("/compare/{id1}/{id2}", response_model=CompareResponse)
async def compare(
    id1: str, id2: str, session: AsyncSession = Depends(get_db_session)
) -> CompareResponse:
    radar1 = await _radar_for(session, id1)
    radar2 = await _radar_for(session, id2)
    domains = sorted(set(radar1) | set(radar2))
    delta = {d: radar1.get(d, 0.0) - radar2.get(d, 0.0) for d in domains}
    most_distinctive = sorted(domains, key=lambda d: abs(delta[d]), reverse=True)
    compatibility_delta = sum(abs(v) for v in delta.values())
    return CompareResponse(
        driver_1_radar=radar1,
        driver_2_radar=radar2,
        delta_shap=delta,
        most_distinctive_features=most_distinctive,
        compatibility_delta=compatibility_delta,
    )


@router.get("/stylespace", response_model=StyleSpaceResponse)
async def stylespace(style_fn: StyleSpaceFn = Depends(get_stylespace_fn)) -> StyleSpaceResponse:
    space = style_fn()
    drivers = [
        StyleSpaceDriver(
            id=name,
            code=name,
            coords_2d=space.coords_2d.get(name, []),
            coords_5d=space.coords_5d.get(name, []),
        )
        for name in space.drivers
    ]
    return StyleSpaceResponse(drivers=drivers, unknown_drivers=[])
