"""The 10 §8.4 tables: schema shape, constraints, ENUM domains, and the code-enum drift guards."""

from __future__ import annotations

import uuid

import pytest
import sqlalchemy as sa
from app.contracts.config import load_config
from app.contracts.enums import ConfidenceState
from app.db import models
from app.db.base import Base
from sqlalchemy.exc import IntegrityError, StatementError
from sqlalchemy.ext.asyncio import AsyncSession

EXPECTED_TABLES = {
    "drivers",
    "circuits",
    "sessions",
    "laps",
    "lap_features",
    "fingerprints",
    "predictions",
    "setup_recommendations",
    "degradation_alerts",
    "data_quality_log",
}


async def _seed_base(
    session: AsyncSession,
) -> tuple[models.Driver, models.Circuit, models.RaceSession, models.Lap]:
    """Insert a valid driver→circuit→session→lap chain and flush so ids are populated."""
    driver = models.Driver(code="VER", full_name="Max Verstappen", team="Red Bull", season=2023)
    circuit = models.Circuit(name="Silverstone", type=models.CircuitType.HIGH_SPEED)
    session.add_all([driver, circuit])
    await session.flush()
    race = models.RaceSession(
        circuit_id=circuit.id, season=2023, round=10, session_type="R"
    )
    session.add(race)
    await session.flush()
    lap = models.Lap(
        session_id=race.id,
        driver_id=driver.id,
        lap_number=5,
        is_clean=True,
        error_code="OK",
    )
    session.add(lap)
    await session.flush()
    return driver, circuit, race, lap


# --- Schema shape --------------------------------------------------------------------------------
def test_exactly_ten_tables_defined() -> None:
    assert set(Base.metadata.tables) == EXPECTED_TABLES
    assert len(Base.metadata.tables) == 10


async def test_full_graph_inserts_and_roundtrips(db_session: AsyncSession) -> None:
    driver, circuit, race, lap = await _seed_base(db_session)
    db_session.add_all(
        [
            models.LapFeatures(
                lap_id=lap.id,
                feature_vector={"F01": 1.0, "F02": None},
                null_count=1,
                imputed_features=["F02"],
                corner_type=models.CornerType.SLOW,
            ),
            models.Fingerprint(
                driver_id=driver.id,
                season=2023,
                circuit_id=circuit.id,
                vector={"d0": 0.1},
                shap_global=None,
                radar_values={"throttle": 0.5},
                stability_score=0.9,
            ),
            models.Prediction(
                lap_id=lap.id,
                predicted_driver_id=driver.id,
                confidence=0.91,
                confidence_state=ConfidenceState.HIGH,
                shap_values={"F01": 0.2},
                shap_available=True,
            ),
            models.SetupRecommendation(
                driver_id=driver.id,
                circuit_id=circuit.id,
                setup_vector={"front_wing_angle": 5.0},
                confidence_vector={"front_wing_angle": 0.8},
                constraint_flags={"CONSTRAINT_CLIPPED": ["rear_wing_angle"]},
                model_version="v1",
            ),
            models.DegradationAlert(
                session_id=race.id,
                driver_id=driver.id,
                lap_number=12,
                deviation_score=0.44,
                alert_confidence=models.AlertConfidence.MEDIUM,
                feature_attribution={"F27": 0.3},
                lead_time_actual=None,
            ),
        ]
    )
    await db_session.commit()

    lf = (await db_session.execute(sa.select(models.LapFeatures))).scalar_one()
    assert lf.feature_vector == {"F01": 1.0, "F02": None}
    assert lf.imputed_features == ["F02"]
    assert lf.corner_type is models.CornerType.SLOW
    pred = (await db_session.execute(sa.select(models.Prediction))).scalar_one()
    assert pred.confidence_state is ConfidenceState.HIGH


# --- Constraints ---------------------------------------------------------------------------------
async def test_drivers_code_season_unique(db_session: AsyncSession) -> None:
    db_session.add(models.Driver(code="HAM", full_name="A", season=2023))
    await db_session.flush()
    db_session.add(models.Driver(code="HAM", full_name="B", season=2023))
    with pytest.raises(IntegrityError):
        await db_session.flush()


async def test_lap_features_lap_id_is_unique_one_to_one(db_session: AsyncSession) -> None:
    _, _, _, lap = await _seed_base(db_session)
    db_session.add(models.LapFeatures(lap_id=lap.id, feature_vector={}, null_count=0))
    await db_session.flush()
    db_session.add(models.LapFeatures(lap_id=lap.id, feature_vector={}, null_count=0))
    with pytest.raises(IntegrityError):
        await db_session.flush()


async def test_clean_lap_check_rejects_nonok_error_code(db_session: AsyncSession) -> None:
    """The §8.4 invariant error_code='OK' for clean laps, enforced as a DB CHECK (fail-loud)."""
    driver, _, race, _ = await _seed_base(db_session)
    db_session.add(
        models.Lap(
            session_id=race.id,
            driver_id=driver.id,
            lap_number=99,
            is_clean=True,
            error_code="LAP_001",  # a clean lap may not carry a non-OK code
        )
    )
    with pytest.raises(IntegrityError):
        await db_session.flush()


async def test_dirty_lap_may_carry_error_code(db_session: AsyncSession) -> None:
    driver, _, race, _ = await _seed_base(db_session)
    db_session.add(
        models.Lap(
            session_id=race.id,
            driver_id=driver.id,
            lap_number=98,
            is_clean=False,
            error_code="LAP_001",
        )
    )
    await db_session.flush()  # allowed


async def test_foreign_key_rejects_orphan_lap(db_session: AsyncSession) -> None:
    """FK enforcement is on (PRAGMA foreign_keys=ON) — a lap with dangling ids is rejected."""
    db_session.add(
        models.Lap(
            session_id=uuid.uuid4(),
            driver_id=uuid.uuid4(),
            lap_number=1,
            is_clean=False,
            error_code="LAP_001",
        )
    )
    with pytest.raises(IntegrityError):
        await db_session.flush()


# --- ENUM domains --------------------------------------------------------------------------------
async def test_confidence_state_enum_accepts_every_member(db_session: AsyncSession) -> None:
    driver, _, _, lap = await _seed_base(db_session)
    for i, state in enumerate(ConfidenceState):
        db_session.add(
            models.Prediction(
                lap_id=lap.id,
                predicted_driver_id=driver.id,
                confidence=0.5 + i * 0.01,
                confidence_state=state,
                shap_available=False,
            )
        )
    await db_session.commit()
    count_q = sa.select(sa.func.count()).select_from(models.Prediction)
    n = (await db_session.execute(count_q)).scalar_one()
    assert n == len(list(ConfidenceState))


async def test_confidence_state_enum_rejects_bogus_value(db_session: AsyncSession) -> None:
    driver, _, _, lap = await _seed_base(db_session)
    db_session.add(
        models.Prediction(
            lap_id=lap.id,
            predicted_driver_id=driver.id,
            confidence=0.5,
            confidence_state="NOT_A_STATE",  # type: ignore[arg-type]
            shap_available=False,
        )
    )
    with pytest.raises((LookupError, StatementError, IntegrityError)):
        await db_session.flush()


async def test_corner_type_is_nullable(db_session: AsyncSession) -> None:
    _, _, _, lap = await _seed_base(db_session)
    db_session.add(
        models.LapFeatures(lap_id=lap.id, feature_vector={}, null_count=0, corner_type=None)
    )
    await db_session.flush()  # NULL corner_type is allowed (base 28-vector rows)


# --- Edge cases ----------------------------------------------------------------------------------
async def test_uuid_pks_are_distinct(db_session: AsyncSession) -> None:
    d1 = models.Driver(code="AAA", full_name="A", season=2022)
    d2 = models.Driver(code="BBB", full_name="B", season=2022)
    db_session.add_all([d1, d2])
    await db_session.flush()
    assert isinstance(d1.id, uuid.UUID) and isinstance(d2.id, uuid.UUID)
    assert d1.id != d2.id


async def test_empty_and_all_null_feature_vectors_roundtrip(db_session: AsyncSession) -> None:
    _, _, _, lap = await _seed_base(db_session)
    empty = models.LapFeatures(lap_id=lap.id, feature_vector={}, null_count=0)
    db_session.add(empty)
    await db_session.commit()
    got = (await db_session.execute(sa.select(models.LapFeatures))).scalar_one()
    assert got.feature_vector == {}
    assert got.null_count == 0
    assert got.imputed_features == []  # default empty list


async def test_data_quality_log_created_at_defaults(db_session: AsyncSession) -> None:
    row = models.DataQualityLog(
        entity_type="lap",
        entity_id="5",
        error_code="LAP_003",
        error_detail="x",
        fallback_used=False,
    )
    db_session.add(row)
    await db_session.commit()
    got = (await db_session.execute(sa.select(models.DataQualityLog))).scalar_one()
    assert got.created_at is not None


# --- Code-enum drift guards (divergence #25: DB ENUM domains == the code enums) ------------------
def test_circuit_type_enum_matches_config_circuit_types(repo_config_path: object) -> None:
    cfg = load_config(repo_config_path)  # type: ignore[arg-type]
    assert [c.value for c in models.CircuitType] == cfg.circuit_types


def test_corner_type_mirror_matches_f09() -> None:
    from app.ml.corners import CornerType as F09CornerType

    assert {c.value for c in models.CornerType} == {c.value for c in F09CornerType}


def test_alert_confidence_mirror_matches_f10() -> None:
    from app.monitoring.detector import AlertConfidence as F10AlertConfidence

    assert {c.value for c in models.AlertConfidence} == {c.value for c in F10AlertConfidence}
