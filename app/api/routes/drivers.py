"""GET /drivers, /drivers/{id}/fingerprint, /drivers/{id}/setup/{circuit}, /drivers/{id}/stability.

The path ``{id}`` is the driver **code** (e.g. "VER"). The read endpoints serve the F12
``fingerprints`` table (empty → [] / 404 until an F14/F20 batch populates it); ``/setup`` serves the
F11 handle. NL description + multi-season stability are pure F07 functions over the DB rows.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    AppState,
    RecommendFn,
    get_apex_state,
    get_db_session,
    get_recommender,
)
from app.api.errors import not_found, service_unavailable
from app.api.schemas import (
    DriverSummary,
    FingerprintResponse,
    SetupParameterSchema,
    SetupResponse,
    StabilityResponse,
)
from app.db import models

router = APIRouter(tags=["drivers"])

# §8.5 fallback; primary source is config.shap.stability_min_seasons (shap block may be absent).
_DEFAULT_STABILITY_MIN_SEASONS = 3


def _enum_val(x: Any) -> Any:
    return getattr(x, "value", x)


@router.get("/drivers", response_model=list[DriverSummary])
async def list_drivers(session: AsyncSession = Depends(get_db_session)) -> list[DriverSummary]:
    """200 always — an empty array if the fingerprints table is empty (§8.5). Deterministic: each
    driver's radar is its LATEST (season, circuit) row, matching /fingerprint and /compare (§11)."""
    rows = (
        await session.execute(
            select(models.Driver, models.Fingerprint)
            .join(models.Fingerprint, models.Fingerprint.driver_id == models.Driver.id)
            .order_by(
                models.Driver.code, models.Fingerprint.season, models.Fingerprint.circuit_id
            )
        )
    ).all()
    out: dict[str, DriverSummary] = {}
    for driver, fp in rows:
        # ascending order → the last write per code wins = latest (season, circuit), deterministic
        out[driver.code] = DriverSummary(
            id=driver.code,
            code=driver.code,
            team=driver.team,
            radar_values=dict(fp.radar_values or {}),
        )
    return list(out.values())


async def _fps_for_code(session: AsyncSession, code: str) -> list[models.Fingerprint]:
    """Every fingerprint for a driver code, across all per-season driver rows (§8.4
    ``UNIQUE(code, season)`` means one driver row per season). Ordered by (season, circuit_id)
    ascending, so ``fps[-1]`` is the LATEST fingerprint deterministically. 404 if code unknown."""
    driver_ids = list(
        (
            await session.execute(select(models.Driver.id).where(models.Driver.code == code))
        ).scalars().all()
    )
    if not driver_ids:
        raise not_found(f"driver {code!r} not found")
    return list(
        (
            await session.execute(
                select(models.Fingerprint)
                .where(models.Fingerprint.driver_id.in_(driver_ids))
                .order_by(models.Fingerprint.season, models.Fingerprint.circuit_id)
            )
        ).scalars().all()
    )


@router.get("/drivers/{driver_id}/fingerprint", response_model=FingerprintResponse)
async def driver_fingerprint(
    driver_id: str,
    response: Response,
    session: AsyncSession = Depends(get_db_session),
    state: AppState = Depends(get_apex_state),
) -> FingerprintResponse:
    from app.ml.explain import describe_style

    fps = await _fps_for_code(session, driver_id)
    if not fps:
        raise not_found(f"no fingerprint for driver {driver_id!r}")

    latest = fps[-1]
    radar = dict(latest.radar_values or {})
    # describe_style needs the shap block; degrade to no NL if it is absent (mirrors _stability_of).
    nl = (
        describe_style(radar, config=state.config).sentence
        if radar and state.config.shap is not None
        else ""
    )
    stab = _stability_of(fps, state)
    if stab.partial:  # §8.5 — 206 if some seasons missing
        response.status_code = 206

    vec = latest.vector
    vector_5d = list(vec.values()) if isinstance(vec, dict) else list(vec)
    return FingerprintResponse(
        driver_id=driver_id,
        vector_5d=vector_5d,
        shap_radar=radar,
        nl_description=nl,
        stability=StabilityResponse(
            driver_id=driver_id,
            stability_score=stab.stability_score,
            pairwise_js=stab.pairwise_js,
            seasons_available=stab.seasons_available,
            partial=stab.partial,
        ),
    )


def _stability_of(fps: list[models.Fingerprint], state: AppState) -> StabilityResponse:
    """Multi-season stability from the STORED per-fingerprint ``stability_score`` (the batch job
    computes it; F13 serves it). The full per-lap pairwise-JS needs the richer SHAP store (F14/F20),
    so ``pairwise_js`` is empty here. <``shap.stability_min_seasons`` seasons → partial (206)."""
    seasons = {fp.season for fp in fps}
    scores = [fp.stability_score for fp in fps if fp.stability_score is not None]
    min_seasons = (
        state.config.shap.stability_min_seasons
        if state.config.shap is not None
        else _DEFAULT_STABILITY_MIN_SEASONS
    )
    return StabilityResponse(
        driver_id="",
        stability_score=(sum(scores) / len(scores)) if scores else None,
        pairwise_js={},
        seasons_available=len(seasons),
        partial=len(seasons) < min_seasons,
    )


@router.get("/drivers/{driver_id}/stability", response_model=StabilityResponse)
async def driver_stability(
    driver_id: str,
    response: Response,
    session: AsyncSession = Depends(get_db_session),
    state: AppState = Depends(get_apex_state),
) -> StabilityResponse:
    fps = await _fps_for_code(session, driver_id)  # 404 if the driver code is unknown
    stab = _stability_of(fps, state)
    if stab.partial:  # §8.5 — 206 if fewer than 3 seasons available
        response.status_code = 206
    return StabilityResponse(
        driver_id=driver_id,
        stability_score=stab.stability_score,
        pairwise_js=stab.pairwise_js,
        seasons_available=stab.seasons_available,
        partial=stab.partial,
    )


@router.get("/drivers/{driver_id}/setup/{circuit_id}", response_model=SetupResponse)
async def driver_setup(
    driver_id: str,
    circuit_id: str,
    recommender: RecommendFn = Depends(get_recommender),
    state: AppState = Depends(get_apex_state),
) -> SetupResponse:
    # get_recommender guarantees both handles are loaded (else 503 during DI).
    if state.style_handle is None:  # pragma: no cover — guarded by get_recommender
        raise service_unavailable("personalisation is not loaded")
    if driver_id not in state.style_handle.drivers:  # §8.5 — 404 for a driver with no data
        raise not_found(f"no setup data for driver {driver_id!r}")

    rec = recommender(driver_id, circuit_id, None)
    params = [
        SetupParameterSchema(
            name=p.name,
            kind=_enum_val(p.kind),
            value=p.value,
            value_range=p.value_range,
            category=p.category,
            confidence=p.confidence,
            rating=_enum_val(p.rating),
            unit=p.unit,
            fia_min=p.fia_min,
            fia_max=p.fia_max,
            constraint_flags=list(p.constraint_flags),
            shap_top_features=list(p.shap_top_features),
            shap_available=p.shap_available,
        )
        for p in rec.parameters
    ]
    return SetupResponse(
        driver=rec.driver,
        circuit=rec.circuit,
        parameters=params,
        consistency_warning=rec.consistency_warning,
        monaco_override=rec.monaco_override,
        approximation_disclaimer=rec.approximation_disclaimer,
        model_version=rec.model_version,
    )
