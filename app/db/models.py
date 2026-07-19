"""The 10 APEX persistence tables (master §8.4), as SQLAlchemy-2.0 async ORM models.

Every column name, PK, FK, UNIQUE, NOT-NULL, and the clean-lap CHECK is copied from master §8.4.
Two deliberate house choices (see ``specs/F12-postgres-schema.md`` §4.0):

- **ENUM domains follow the CODE enums, not §8.4's prose paraphrase** (divergence #25):
  ``predictions.confidence_state`` = the 5-value ``ConfidenceState`` (incl. ``LOW_CONFIDENCE`` +
  ``OUT_OF_DISTRIBUTION``, non-negotiable #4), not §8.4's ``HIGH/MEDIUM/LOW/AMBIGUOUS``;
  ``corner_type`` = F09's ``CornerType``; ``alert_confidence`` = F10's ``AlertConfidence``. The two
  feature enums are **mirrored locally** here (``CornerType``/``AlertConfidence``) so importing this
  module — and the Alembic env — stays import-light and never pulls the pandas/xgboost ML stack; a
  drift-guard test asserts each mirror equals its owning-feature enum.
- **``laps.error_code`` is a plain VARCHAR** storing an arbitrary ``ErrorCode.value`` (the set is
  large and grows per feature), exactly as ``app/contracts/logging.py`` does — not a DB ENUM.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum

import sqlalchemy as sa
from sqlalchemy import CheckConstraint, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.contracts.enums import ConfidenceState
from app.db.base import JSONB_VARIANT, TEXT_ARRAY_VARIANT, Base, pg_enum, uuid_pk


# --- DB-domain enums --------------------------------------------------------------------------
class CircuitType(StrEnum):
    """The ``circuits.type`` ENUM (master §8.4). Values match ``config.circuit_types`` verbatim —
    a drift-guard test pins the two together (the F11 one-hot vocab and this ENUM must agree)."""

    STREET = "street"
    HIGH_SPEED = "high_speed"
    TECHNICAL = "technical"
    MIXED = "mixed"


class CornerType(StrEnum):
    """Local mirror of ``app.ml.corners.CornerType`` (F09, §4.4) for ``lap_features.corner_type``.

    Mirrored (not imported) to keep the persistence layer free of the ML stack; a drift-guard test
    asserts equality."""

    SLOW = "SLOW"
    MEDIUM = "MEDIUM"
    HIGH_SPEED = "HIGH_SPEED"


class AlertConfidence(StrEnum):
    """Local mirror of ``app.monitoring.detector.AlertConfidence`` (F10, §5.3) for the
    ``degradation_alerts.alert_confidence`` ENUM. Mirrored (not imported); drift-guarded by test."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


# --- Tables (master §8.4, in FK-dependency order) ---------------------------------------------
class Driver(Base):
    """Driver master data. UNIQUE(code, season). 8 known + unlimited unknown drivers."""

    __tablename__ = "drivers"

    id: Mapped[uuid.UUID] = uuid_pk()
    code: Mapped[str] = mapped_column(String(3), nullable=False)
    full_name: Mapped[str] = mapped_column(String, nullable=False)
    team: Mapped[str | None] = mapped_column(String)  # unknown drivers have no team yet
    season: Mapped[int] = mapped_column(sa.Integer, nullable=False)

    __table_args__ = (UniqueConstraint("code", "season"),)


class Circuit(Base):
    """Circuit master data. UNIQUE(name). 10 circuits."""

    __tablename__ = "circuits"

    id: Mapped[uuid.UUID] = uuid_pk()
    name: Mapped[str] = mapped_column(String, nullable=False)
    type: Mapped[CircuitType] = mapped_column(pg_enum(CircuitType, "circuit_type"), nullable=False)

    __table_args__ = (UniqueConstraint("name"),)


class RaceSession(Base):
    """Race session records. Named ``RaceSession`` (table ``sessions``) to avoid confusion with
    SQLAlchemy's ``AsyncSession``. UNIQUE(circuit_id, season, round, session_type)."""

    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = uuid_pk()
    circuit_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("circuits.id"), nullable=False)
    season: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    round: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    session_type: Mapped[str] = mapped_column(String, nullable=False)

    __table_args__ = (UniqueConstraint("circuit_id", "season", "round", "session_type"),)


class Lap(Base):
    """Per-lap metadata with quality flags. NOT NULL on is_clean and error_code; error_code='OK'
    for clean laps (enforced by the CHECK below, fail-loud at the DB, not just the app layer)."""

    __tablename__ = "laps"

    id: Mapped[uuid.UUID] = uuid_pk()
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("sessions.id"), nullable=False)
    driver_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("drivers.id"), nullable=False)
    lap_number: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    is_clean: Mapped[bool] = mapped_column(sa.Boolean, nullable=False)
    # error_code stores an ErrorCode.value string ('OK' for clean laps), NOT a DB ENUM.
    error_code: Mapped[str] = mapped_column(String, nullable=False, server_default=sa.text("'OK'"))
    tyre_compound: Mapped[str | None] = mapped_column(String)
    tyre_age: Mapped[int | None] = mapped_column(sa.Integer)
    has_interpolation: Mapped[bool] = mapped_column(
        sa.Boolean, nullable=False, server_default=sa.false()
    )

    __table_args__ = (
        CheckConstraint("(NOT is_clean) OR (error_code = 'OK')", name="clean_lap_error_code_ok"),
    )


class LapFeatures(Base):
    """The 28-feature lap vector. lap_id is UNIQUE (1:1 with laps). The §8.4 invariant
    ``null_count == count(nulls in feature_vector)`` is "Enforced by application layer" — see
    ``app/db/repository.py::insert_lap_features`` (raises DB_INTEGRITY on mismatch)."""

    __tablename__ = "lap_features"

    id: Mapped[uuid.UUID] = uuid_pk()
    lap_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("laps.id"), nullable=False, unique=True
    )
    feature_vector: Mapped[dict[str, float | None]] = mapped_column(JSONB_VARIANT, nullable=False)
    null_count: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    imputed_features: Mapped[list[str]] = mapped_column(
        TEXT_ARRAY_VARIANT, nullable=False, default=list
    )
    corner_type: Mapped[CornerType | None] = mapped_column(pg_enum(CornerType, "corner_type"))


class Fingerprint(Base):
    """Aggregate fingerprint per driver/season/circuit. UNIQUE(driver_id, season, circuit_id)."""

    __tablename__ = "fingerprints"

    id: Mapped[uuid.UUID] = uuid_pk()
    driver_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("drivers.id"), nullable=False)
    season: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    circuit_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("circuits.id"), nullable=False)
    vector: Mapped[dict[str, float]] = mapped_column(JSONB_VARIANT, nullable=False)  # 5D
    shap_global: Mapped[dict[str, float] | None] = mapped_column(JSONB_VARIANT)
    radar_values: Mapped[dict[str, float] | None] = mapped_column(JSONB_VARIANT)  # 6D
    stability_score: Mapped[float | None] = mapped_column(sa.Float)

    __table_args__ = (UniqueConstraint("driver_id", "season", "circuit_id"),)


class Prediction(Base):
    """Classifier output. confidence is NOT NULL (no NULL confidence, §8.4); confidence_state is the
    5-value ConfidenceState ENUM (divergence #25)."""

    __tablename__ = "predictions"

    id: Mapped[uuid.UUID] = uuid_pk()
    lap_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("laps.id"), nullable=False)
    predicted_driver_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("drivers.id"), nullable=False
    )
    confidence: Mapped[float] = mapped_column(sa.Float, nullable=False)
    confidence_state: Mapped[ConfidenceState] = mapped_column(
        pg_enum(ConfidenceState, "confidence_state"), nullable=False
    )
    shap_values: Mapped[dict[str, float] | None] = mapped_column(JSONB_VARIANT)
    shap_available: Mapped[bool] = mapped_column(sa.Boolean, nullable=False)


class SetupRecommendation(Base):
    """Personalisation output. constraint_flags records any constraint-layer overrides."""

    __tablename__ = "setup_recommendations"

    id: Mapped[uuid.UUID] = uuid_pk()
    driver_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("drivers.id"), nullable=False)
    circuit_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("circuits.id"), nullable=False)
    setup_vector: Mapped[dict[str, float]] = mapped_column(JSONB_VARIANT, nullable=False)
    confidence_vector: Mapped[dict[str, float]] = mapped_column(JSONB_VARIANT, nullable=False)
    constraint_flags: Mapped[dict[str, object] | None] = mapped_column(JSONB_VARIANT)
    model_version: Mapped[str | None] = mapped_column(String)


class DegradationAlert(Base):
    """Degradation monitor output. lead_time_actual is populated post-race for validation."""

    __tablename__ = "degradation_alerts"

    id: Mapped[uuid.UUID] = uuid_pk()
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("sessions.id"), nullable=False)
    driver_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("drivers.id"), nullable=False)
    lap_number: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    deviation_score: Mapped[float] = mapped_column(sa.Float, nullable=False)
    alert_confidence: Mapped[AlertConfidence] = mapped_column(
        pg_enum(AlertConfidence, "alert_confidence"), nullable=False
    )
    feature_attribution: Mapped[dict[str, float] | None] = mapped_column(JSONB_VARIANT)
    lead_time_actual: Mapped[int | None] = mapped_column(sa.Integer)


class DataQualityLog(Base):
    """All pipeline errors and fallbacks. **Never deleted** (§8.4) — the repository exposes no
    delete. Column order + names mirror ``app/contracts/logging.py`` ``ROW_COLUMNS`` exactly, so the
    F00 JSONL writer's rows drop straight in (the "PostgreSQL swap" that docstring promises)."""

    __tablename__ = "data_quality_log"

    id: Mapped[uuid.UUID] = uuid_pk()
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
    )
    entity_type: Mapped[str] = mapped_column(String, nullable=False)
    entity_id: Mapped[str | None] = mapped_column(String)
    error_code: Mapped[str] = mapped_column(String, nullable=False)
    error_detail: Mapped[str | None] = mapped_column(String)
    fallback_used: Mapped[bool] = mapped_column(sa.Boolean, nullable=False)
