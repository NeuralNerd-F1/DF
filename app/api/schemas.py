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
    # System input #2 (setup tendencies): the model's mean recommendation across the known grid in
    # the same circuit context (None for CATEGORY kinds / when unavailable), and the design-time
    # §7.3 basis features of the target heuristic. Defaults keep older payloads valid.
    grid_mean: float | None = None
    basis_features: list[str] = Field(default_factory=list)


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


class ConfidenceBreakdownSchema(ApiModel):
    """The propagated-confidence composition (system input #6). Mirrors
    ``app.ml.inference.ConfidenceBreakdown`` field-for-field: three served signals (distance from
    the NULL ceiling, distance from the AMBIGUOUS entropy boundary, calibrated max-probability) and
    their product. Presentational — the gate's ``confidence_state`` stays authoritative."""

    feature_quality: float
    distribution: float
    model_confidence: float
    overall: float


class PredictResponse(ApiModel):
    predicted_driver: str | None
    confidence: float
    confidence_state: ConfidenceState | None
    top_3: list[TopK] = Field(default_factory=list)
    # The FULL calibrated distribution (sums to ~1.0) — the inference result always computes it;
    # serialising it lets the screen show every class instead of an implied "rest is nothing".
    class_probabilities: dict[str, float] = Field(default_factory=dict)
    shap_values: dict[str, float] | None = None
    shap_available: bool = False
    model_version: str
    # None on FAILED results and on cached pre-upgrade payloads (default keeps old caches valid).
    confidence_breakdown: ConfidenceBreakdownSchema | None = None


class TyreCurvePointSchema(ApiModel):
    """One tyre age on the aggregated degradation curve: the median lap-time delta versus each
    stint's own early-lap baseline, with the IQR and the lap count behind the point."""

    age: int
    delta_s: float
    q25: float
    q75: float
    n: int


class TyreCompoundStatsSchema(ApiModel):
    """Stint-degradation statistics for one (driver, compound), from clean train laps.

    All pace numbers are **net** within-stint drift: the matrix carries no fuel-load channel, so
    fuel burn (which makes cars faster) and tyre wear (slower) are combined, and the true tyre
    component is larger than each number — the screen states this rather than correcting it with
    an invented constant. ``cliff_age`` is ``None`` when no acceleration past the linear trend was
    detected; most curves are near-linear and a decorative cliff would be detection theatre.
    ``f27_median``/``f28_median`` are the pipeline's own behavioural tyre-age slopes (throttle /
    brake adaptation per lap of age).
    """

    n_laps: int
    n_stints: int
    median_stint_len: int
    max_stint_len: int
    deg_slope_s_per_lap: float
    total_delta_s: float
    # The stripped per-race-lap fuel term (median across this group's races) — reported so the
    # screen can state the correction it is standing on.
    fuel_slope_s_per_lap: float | None = None
    cliff_age: int | None = None
    curve: list[TyreCurvePointSchema] = Field(default_factory=list)
    f27_median: float | None = None
    f28_median: float | None = None


class TyreWearResponse(ApiModel):
    """GET /tyres — per-driver, per-compound stint degradation from the profiles artefact.
    ``drivers`` is empty when the artefact is absent (the screen explains, nothing is invented)."""

    drivers: dict[str, dict[str, TyreCompoundStatsSchema]] = Field(default_factory=dict)


class SampleSegmentSchema(ApiModel):
    """One real held-out lap for the Prediction screen's picker (profiles artefact). The raw
    per-lap features go straight back into POST /predict; ``true_driver`` is the matrix's ground
    truth so the screen can grade the prediction — a held-out lap, so grading it is a genuine
    generalisation test, not a replay of training data."""

    id: str
    label: str
    circuit: str
    season: int
    session: str = ""
    lap_number: int = 0
    true_driver: str
    features: dict[str, float | None] = Field(default_factory=dict)


# --- GET /drivers, /drivers/{id}/fingerprint, /setup, /stability --------------------------------
class DriverSignalSchema(ApiModel):
    """One feature the champion attributes most to a driver, plus where they rank on it.

    ``attribution`` is mean |SHAP| — a magnitude, so it never implies a direction. ``percentile`` is
    the driver's rank on that feature against the rest of the grid (0 = lowest mean, 1 = highest)
    and is ``None`` when the profiles artefact carries no rank, so the screen names the feature
    without claiming which way it points.
    """

    feature: str
    attribution: float
    percentile: float | None = None


class DriverProfileSchema(ApiModel):
    """Dataset + evaluation facts for one driver, sourced from ``driver_profiles.json``.

    Every optional field is ``None`` for an honest "not measured", never 0.0: a driver with too few
    laps for a within-session SD and a driver who laps identically every time must not render the
    same. ``identification_f1`` is the champion's **held-out** per-driver F1 — a driver the summary
    never scored keeps ``None`` rather than borrowing a training-split number.
    """

    laps_analysed: int
    seasons: list[int] = Field(default_factory=list)
    n_circuits: int = 0
    lap_time_consistency_s: float | None = None
    identification_f1: float | None = None


class DriverSummary(ApiModel):
    id: str
    code: str
    # F16 seam: the Command Centre card renders a team-colour bar keyed on the driver's team
    # (master §10.2). The team lives on the F12 ``drivers`` table; unknown drivers have none.
    team: str | None = None
    full_name: str | None = None
    radar_values: dict[str, float] = Field(default_factory=dict)
    # Both default to absent so the endpoint keeps its contract on a checkout with no profiles
    # artefact and no SHAP fill: the card degrades to code + team + radar, never to fabricated
    # facts. `signature` is [] (nothing to say) and `profile` is null (nothing measured).
    signature: list[DriverSignalSchema] = Field(default_factory=list)
    profile: DriverProfileSchema | None = None


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
    # Mean SIGNED SHAP per feature over the driver's sampled laps (profiles artefact, sourced from
    # the F07 beeswarm): positive = pushes the champion toward this driver, negative = away. Empty
    # when the artefact is absent or predates the signed fill — the profile screen then omits the
    # confirms/argues panel, never derives a sign from the unsigned shap_radar magnitudes.
    shap_signed: dict[str, float] = Field(default_factory=dict)


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


class ReplayAlert(ApiModel):
    """The F10 ``DegradationAlert`` as of one replayed lap (§5.3 / §10.7 banner payload). Mirrors
    ``app.monitoring.detector.DegradationAlert`` field-for-field; closes the F19 alert seam
    (div #74) — the route used to drop it."""

    alert_lap: int
    first_crossing_lap: int
    laps_since_threshold_crossed: int
    deviation_score: float
    threshold: float
    alert_confidence: str
    feature_attribution: dict[str, float] = Field(default_factory=dict)
    top_features: list[str] = Field(default_factory=list)
    optimal_window_remaining_laps: int | None = None


class ReplayLap(ApiModel):
    lap_number: int
    feature_vector: dict[str, float | None]
    deviation_score: float | None = None
    alert_state: str
    fingerprint_snapshot: dict[str, Any] | None = None
    # τ for this stint (constant per stint; null on BASELINE/SKIPPED laps) — closes div #72/#73.
    threshold: float | None = None
    # The DegradationAlert as of this lap's prefix; null until the alert fires — closes div #74.
    alert: ReplayAlert | None = None


class ReplayResponse(ApiModel):
    laps: list[ReplayLap] = Field(default_factory=list)


class SessionDriverSchema(ApiModel):
    """One driver's presence in a persisted session (code + how many laps are replayable)."""

    code: str
    n_laps: int


class SessionSummary(ApiModel):
    """One catalog row for ``GET /sessions`` — the Screen 6 session picker (closes F19 div #77:
    sessions were listed nowhere, forcing a raw UUID input)."""

    id: str
    circuit: str
    circuit_type: str
    season: int
    round: int
    session_type: str
    drivers: list[SessionDriverSchema] = Field(default_factory=list)
