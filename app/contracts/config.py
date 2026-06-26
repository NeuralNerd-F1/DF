"""Typed loader for ``config.yaml`` — the single source of truth for constants.

The constitution bans magic numbers: every threshold, seed, and path is read from
``config.yaml`` through this loader, which validates the foundational blocks with
Pydantic and surfaces a typed ``ConfigError`` (``CONFIG_001``) on any problem rather
than letting a missing key blow up deep in a pipeline stage.

Feature-owned sections (``ingest:``, ``classifier:``, ...) are added to the YAML by
their features. The standing rule (adopted in F01): each feature defines its typed
sub-model *here* in ``config.py`` and wires it onto ``AppConfig`` as an optional typed
field — never as an untyped ``extra="allow"`` passthrough. ``AppConfig`` therefore uses
``extra="forbid"`` so a typo'd top-level section is a build-breaking error, not a
silently-ignored block (non-negotiable #1), and feature code reads its config through a
validated model rather than a raw dict.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.contracts.errors import ErrorCode

DEFAULT_CONFIG_PATH = Path("config.yaml")


class ConfigError(Exception):
    """Raised when ``config.yaml`` is missing or a required key is absent/invalid.

    Carries ``ErrorCode.CONFIG_001`` so callers can surface a typed failure instead
    of a bare exception (non-negotiable #1).
    """

    error_code = ErrorCode.CONFIG_001

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class Seeds(BaseModel):
    """Fixed RNG seeds — reproducibility contract (non-negotiable #3)."""

    model_config = ConfigDict(extra="forbid")

    numpy: int
    xgboost: int
    optuna: int


class Paths(BaseModel):
    """Filesystem locations the system reads/writes."""

    model_config = ConfigDict(extra="forbid")

    artefacts_dir: str
    cache_dir: str
    data_quality_log: str


class Thresholds(BaseModel):
    """The named numeric constants the constitution requires live in config."""

    model_config = ConfigDict(extra="forbid")

    confidence_gate: float  # classifier max-prob floor (§5.1)
    entropy_ambiguous_bits: float  # near-uniform -> AMBIGUOUS (§5.1)
    null_ceiling: int  # > N NULLs -> reject vector; shared F03/F06 (§4.5)
    min_clean_laps: int  # per driver-circuit combination (§3.2)
    sample_rate_floor_hz: int  # effective sampling-rate floor (§3.3)
    laptime_bound_pct: int  # lap-time plausibility bound (§3.3)


class BreakerConfig(BaseModel):
    """Per-source circuit-breaker / retry policy for one external data source (F01)."""

    model_config = ConfigDict(extra="forbid")

    timeout_s: float  # per-attempt wall-clock timeout
    retries: int  # extra attempts after the first within one call_with_breaker call
    backoff_s: list[float]  # sleep between attempts; len must cover the retries
    open_after: int  # failed (post-retry) invocations within window_s before opening
    window_s: float  # rolling window the failure count is measured over


class IngestBreakers(BaseModel):
    """The three source breakers F01 wraps (§8.3)."""

    model_config = ConfigDict(extra="forbid")

    fastf1: BreakerConfig
    openf1: BreakerConfig
    ergast: BreakerConfig


class IngestConfig(BaseModel):
    """Typed view over the ``ingest:`` block (F01).

    Lives here, not in ``app/pipeline/ingest.py``, so the dependency points
    feature -> config and the block is strictly validated (standing rule, F01).
    ``cache_dir`` is intentionally absent — ingestion reads ``paths.cache_dir`` as the
    single source of truth for the cache location.
    """

    model_config = ConfigDict(extra="forbid")

    cache_max_age_days: int  # §8.2 — serve cache fresher than this without fetching
    breakers: IngestBreakers


class ValidateConfig(BaseModel):
    """Typed view over the ``validate:`` block (F02 lap filter chain, §3.2/§3.3).

    Lives here, not in ``app/pipeline/validate.py`` (standing rule, F01): the feature
    reads its config through a strictly-validated model. Holds only the constants
    unique to F02; the shared bounds it also needs — ``sample_rate_floor_hz`` (20 Hz),
    ``laptime_bound_pct`` (115%), ``min_clean_laps`` (30) — are read from
    ``Thresholds`` so this file stays the single source for each value.
    """

    model_config = ConfigDict(extra="forbid")

    required_channels: list[str]  # §3.3 LAP_003 completeness check set
    green_track_status_code: str  # §3.3 LAP_001 — green token; lap green iff every token == this
    max_consecutive_nan_pct: float  # §3.3 LAP_003 — > N% consecutive NaN in a channel -> reject
    interp_gap_max_samples: int  # §3.3 LAP_INTERP — 1..N consecutive-NaN gaps interpolated
    min_moving_speed_kmh: float  # §3.3 LAP_005 — speed must exceed this to count as moving
    max_stationary_fraction: float  # §3.3 LAP_005 — stationary samples must be < this (>99% moving)
    degradation_features: list[str]  # §3.3 LAP_PARTIAL — NULL'd when tyre age UNKNOWN

    @model_validator(mode="after")
    def _speed_channel_present(self) -> ValidateConfig:
        """LAP_005 (speed anomaly) reads the ``Speed`` channel, so it MUST be one of the
        required channels. Enforcing it here turns a bad config into a build-breaking
        ``CONFIG_001`` at load time, rather than a per-lap ``KeyError`` that the validator
        would otherwise launder into a whole-combination exclusion."""
        if "Speed" not in self.required_channels:
            raise ValueError("validate.required_channels must include 'Speed' (LAP_005 reads it)")
        return self


class AppConfig(BaseModel):
    """Validated view over ``config.yaml``.

    Every block — foundational and feature-owned — is a typed field validated
    strictly. ``extra="forbid"`` means a stray or typo'd top-level section is a
    build-breaking error rather than a silently-ignored block. Feature sections are
    optional (``None`` until their feature ships) so the minimal foundation config
    still loads.
    """

    # populate_by_name lets the `lap_validation` field be set by its YAML alias
    # `validate` (which would otherwise shadow a BaseModel attribute) or by name.
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    seeds: Seeds
    paths: Paths
    held_out_circuits: list[str]  # quarantined from all training (§3.2)
    thresholds: Thresholds
    ingest: IngestConfig | None = None  # F01; None until the ingest: block is present
    # YAML key is `validate:`; the Python attribute is `lap_validation` to avoid
    # shadowing pydantic's deprecated BaseModel.validate (F02; None until block present).
    lap_validation: ValidateConfig | None = Field(default=None, alias="validate")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Load and validate ``config.yaml``.

    Raises ``ConfigError`` (``CONFIG_001``) if the file is missing, unreadable, is
    not valid YAML, is not a mapping, or is missing/invalid in any foundational key.
    """
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    try:
        text = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"config file is unreadable: {exc}") from exc
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"config file is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("config file must contain a top-level mapping")
    try:
        return AppConfig(**raw)
    except ValidationError as exc:
        raise ConfigError(f"config is missing or has invalid keys: {exc}") from exc
