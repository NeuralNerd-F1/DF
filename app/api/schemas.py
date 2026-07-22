"""Pydantic v2 request/response contracts for the master-§8.5 REST surface.

Every endpoint's HTTP body is one of these models (§9: "Pydantic v2 for all request/response
validation — invalid inputs rejected at the schema layer before business logic"). They are DTOs, not
pipeline result objects: a route maps the internal never-raising ``*Result`` (from F06b/F07/F08/F10/
F11) onto the response shape. Import-light on purpose — only the contract layer (`enums`, `errors`)
+ pydantic, never the ML stack — so importing `app.api.schemas` never pulls pandas/xgboost. Enum
value fields that live in the heavy feature modules (`SetupConfidence`, `SetupOutputKind`,
`StintMonitorState`) are represented as their ``str`` value here for the same reason.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.contracts.enums import ConfidenceState
from app.contracts.errors import ErrorCode


class ApiModel(BaseModel):
    """Base for every API DTO — ``extra="forbid"`` so a stray request field is a 422, not silently
    ignored (the schema-layer rejection the master §9 requires; house 'no silent failure')."""

    model_config = ConfigDict(extra="forbid")


# --- Error envelope (every non-2xx typed body) --------------------------------------------------
class ErrorResponse(ApiModel):
    """The one error shape. ``error_code`` is the ``ErrorCode`` enum — never a hard-coded string."""

    error_code: ErrorCode
    detail: str
    flags: list[ErrorCode] = Field(default_factory=list)


# --- Shared sub-models --------------------------------------------------------------------------
class TopK(ApiModel):
    driver: str
    probability: float


class NeighbourSchema(ApiModel):
    driver: str
    cosine_similarity: float
    distance: float


class DataQualitySchema(ApiModel):
    n_laps_used: int
    n_laps_rejected: int
    imputed_features: list[str] = Field(default_factory=list)


class SetupParameterSchema(ApiModel):
    name: str
    kind: str  # SetupOutputKind.value (DEGREES/PERCENT/INDEX/INTEGER_LAPS/CATEGORY)
    value: float | str | None = None  # None only when the rating is WIDE
    value_range: tuple[float, float] | None = None  # present iff WIDE
    category: str | None = None
    confidence: float
    rating: str  # SetupConfidence.value (HIGH/MEDIUM/LOW/WIDE)
    unit: str
    fia_min: float | None = None
    fia_max: float | None = None
    constraint_flags: list[ErrorCode] = Field(default_factory=list)
    shap_top_features: list[str] = Field(default_factory=list)
    shap_available: bool = False


# --- GET /health --------------------------------------------------------------------------------
class HealthResponse(ApiModel):
    status: Literal["healthy", "unhealthy"]
    models_loaded: dict[str, bool]
    db_connected: bool
    cache_connected: bool | None = None  # Redis is F14; reported but never critical
    version: str


# --- POST /predict ------------------------------------------------------------------------------
class PredictRequest(ApiModel):
    feature_vector: dict[str, float | None]
    circuit: str


class PredictResponse(ApiModel):
    predicted_driver: str | None
    confidence: float
    confidence_state: ConfidenceState | None
    top_3: list[TopK] = Field(default_factory=list)
    shap_values: dict[str, float] | None = None
    shap_available: bool = False
    model_version: str


# --- GET /drivers, /drivers/{id}/fingerprint, /setup, /stability --------------------------------
class DriverSummary(ApiModel):
    id: str
    code: str
    # F16 seam: the Command Centre card renders a team-colour bar keyed on the driver's team
    # (master §10.2). The team lives on the F12 ``drivers`` table; unknown drivers have none.
    team: str | None = None
    radar_values: dict[str, float] = Field(default_factory=dict)


class StabilityResponse(ApiModel):
    driver_id: str
    stability_score: float | None = None
    pairwise_js: dict[str, float] = Field(default_factory=dict)
    seasons_available: int
    partial: bool


class FingerprintResponse(ApiModel):
    driver_id: str
    vector_5d: list[float]
    shap_radar: dict[str, float] = Field(default_factory=dict)
    nl_description: str = ""
    stability: StabilityResponse | None = None


class SetupResponse(ApiModel):
    driver: str
    circuit: str
    parameters: list[SetupParameterSchema]
    consistency_warning: bool = False
    monaco_override: bool = False
    approximation_disclaimer: str
    model_version: str


# --- POST /fingerprint/unknown, GET /compare, GET /stylespace ----------------------------------
class UnknownFingerprintRequest(ApiModel):
    laps: list[dict[str, float | None]]
    circuit: str
    circuit_type: str | None = None


class UnknownFingerprintResponse(ApiModel):
    fingerprint_id: str
    fingerprint_vector: list[float] | None
    radar_values: dict[str, float] = Field(default_factory=dict)
    style_description: str = ""
    nearest_neighbours: list[NeighbourSchema] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    data_quality: DataQualitySchema
    confidence_state: ConfidenceState | None = None
    warnings: list[ErrorCode] = Field(default_factory=list)  # SPARSE_PROJECTION / OOD


class CompareResponse(ApiModel):
    driver_1_radar: dict[str, float]
    driver_2_radar: dict[str, float]
    delta_shap: dict[str, float]
    most_distinctive_features: list[str]
    compatibility_delta: float


class StyleSpaceDriver(ApiModel):
    id: str
    code: str
    coords_2d: list[float]
    coords_5d: list[float]
    season: str | None = None


class StyleSpaceResponse(ApiModel):
    drivers: list[StyleSpaceDriver] = Field(default_factory=list)
    unknown_drivers: list[StyleSpaceDriver] = Field(default_factory=list)


# --- F14 (Celery/replay) request + response schemas ---------------------------------------------
class IngestSessionRequest(ApiModel):
    """POST /ingest/session body — the FastF1 session coordinates to ingest async (§8.5)."""

    year: int = Field(ge=1950, le=2100)
    event: str = Field(min_length=1, max_length=128)
    session: str = Field(default="R", min_length=1, max_length=8)


class IngestSessionResponse(ApiModel):
    session_id: str
    laps_accepted: int
    laps_rejected: int
    rejection_summary: dict[str, int] = Field(default_factory=dict)
    quality_report: dict[str, Any] = Field(default_factory=dict)


class IngestStatusResponse(ApiModel):
    status: Literal["PENDING", "PROCESSING", "COMPLETE", "FAILED"]
    progress: float
    error_detail: str | None = None
    # The terminal report — the §8.5 IngestSessionResponse fields (laps_accepted, rejection_summary,
    # quality_report, …). Populated once the job is COMPLETE; None while it is still running (F14).
    result: dict[str, Any] | None = None


class UploadResponse(ApiModel):
    upload_id: str
    laps_accepted: int
    laps_rejected: int
    cleaning_report: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class ReplayLap(ApiModel):
    lap_number: int
    feature_vector: dict[str, float | None]
    deviation_score: float | None = None
    alert_state: str
    fingerprint_snapshot: dict[str, Any] | None = None


class ReplayResponse(ApiModel):
    laps: list[ReplayLap] = Field(default_factory=list)
