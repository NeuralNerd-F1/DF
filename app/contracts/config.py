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


# The 22 scalar feature IDs F03a computes (master spec §4.2: F01–F18, F22–F24, F26).
# F19–F21, F25, F27–F28 are F03b's; the full 28-key vector is assembled by F03a but only
# these 22 carry a valid-range contract here.
SCALAR_FEATURE_IDS: tuple[str, ...] = (
    *(f"F{n:02d}" for n in range(1, 19)),  # F01..F18
    "F22",
    "F23",
    "F24",
    "F26",
)

# The X/Y-derived geometry feature IDs F03b range-checks (master spec §4.2: F19–F21 lateral
# G / steering, F25 corner balance). F27/F28 are degradation slopes — *unconstrained*, so
# they carry no valid-range entry.
GEOMETRY_FEATURE_IDS: tuple[str, ...] = ("F19", "F20", "F21", "F25")

# Every feature that carries a valid-range contract: the 22 scalar (F03a) + the 4 geometry
# (F03b). ``valid_ranges`` must cover exactly these 26 IDs.
RANGE_CHECKED_IDS: tuple[str, ...] = SCALAR_FEATURE_IDS + GEOMETRY_FEATURE_IDS

# The six circuit-dependent feature IDs F04 Z-score normalises **per circuit** (master spec
# §4.3): the speed-dependent F09/F11/F13/F14 (scalar) and the lateral-G F19/F20 (geometry). All
# six are members of ``RANGE_CHECKED_IDS``; F04's config validator pins ``normalise.features`` to
# exactly this tuple so the leakage chokepoint can never be silently widened/narrowed in YAML.
NORMALISED_FEATURE_IDS: tuple[str, ...] = ("F09", "F11", "F13", "F14", "F19", "F20")

# Load-time invariant: every normalised ID must be range-bearing. This is what lets the F04
# validator pin `normalise.features` by exact-order equality alone (no separate per-ID range
# check needed) — and it fails the import loudly if a future edit to the ID tuples breaks it.
assert set(NORMALISED_FEATURE_IDS) <= set(RANGE_CHECKED_IDS), (
    "NORMALISED_FEATURE_IDS must be a subset of RANGE_CHECKED_IDS"
)


class FeaturesConfig(BaseModel):
    """Typed view over the ``features:`` block (F03a scalar extraction, §4.1/§4.2/§4.5).

    Lives here, not in ``app/pipeline/features.py`` (standing rule, F01): the feature reads
    its config through a strictly-validated model. Holds only the constants unique to F03a;
    the NULL ceiling it also needs — ``null_ceiling`` (6) — is read from ``Thresholds`` so
    this file stays the single source for that shared value. ``valid_ranges`` carries the
    §4.2 out-of-range bound per feature (value outside -> ``FeatureComputationError``).
    """

    model_config = ConfigDict(extra="forbid")

    sample_period_ms: int  # §4.2 canonical step for "% per 50ms" rate features (F03, F08)
    throttle_full_pct: float  # §4.2 F04 — sample is full throttle at >= this %
    throttle_lift_pct: float  # §4.2 F05 — sample is a lift below this %
    brake_peak_top_n: int  # §4.2 F06 — mean of top-N brake-pressure peaks
    min_braking_events: int  # §4.2 F06/F09 — fewer braking events -> NULL
    braking_event_threshold_bar: float  # design decision — Brake > this delimits an event
    corner_min_prominence_kmh: float  # design decision — speed-min prominence => corner (F09/F11)
    overlap_throttle_pct: float  # §4.2 F10 — throttle > this AND ...
    overlap_brake_pct: float  # §4.2 F10 — ... brake > this => overlap sample
    overlap_cap: float  # §4.2 F10 — value above -> cap to this and log (not NULL)
    speed_p_low: float  # §4.2 F13 — low percentile
    speed_p_high: float  # §4.2 F14 — high percentile
    redline_pct: float  # §4.2 F17 — fraction of lap above this % of session-max RPM
    min_gear_changes: int  # §4.2 F16 — fewer gear changes -> NULL
    min_transitions: int  # §4.2 F24 — fewer brake->throttle transitions -> NULL
    # --- F03b geometry & degradation (§4.2, §4.4) ---
    lateral_g_peak_top_n: int  # §4.2 F20 — mean of top-N lateral-G peaks
    min_lateral_g_peaks: int  # §4.2 F20 — fewer distinct peaks -> NULL
    lateral_g_peak_min_prominence: float  # design decision — min G prominence for a "distinct" peak
    steering_g_deadband: float  # design decision — |lateral G| below this ignored for F21 reversals
    gravity_m_s2: float  # physical constant — lateral accel (m/s^2) -> G (F19/F20/F21)
    min_stint_laps: int  # §4.2 F27/F28 — stint with fewer usable laps -> NULL
    min_xy_finite_fraction: float  # design decision — X/Y "available" iff >= this fraction finite
    valid_ranges: dict[str, tuple[float, float]]  # §4.2 per-feature [min, max] bound

    @model_validator(mode="after")
    def _ranges_cover_checked_ids(self) -> FeaturesConfig:
        """``valid_ranges`` MUST cover exactly the 26 range-bearing IDs — the 22 scalar (F03a)
        plus the 4 geometry (F03b: F19, F20, F21, F25) — no more, no fewer, each a well-ordered
        ``[min, max]``. F27/F28 are unconstrained slopes and must NOT appear. A missing or stray
        key is a config error surfaced as ``CONFIG_001`` at load time, not a later ``KeyError``."""
        keys = set(self.valid_ranges)
        expected = set(RANGE_CHECKED_IDS)
        if keys != expected:
            missing = sorted(expected - keys)
            extra = sorted(keys - expected)
            raise ValueError(
                f"features.valid_ranges must cover exactly {len(expected)} range-bearing IDs; "
                f"missing={missing} unexpected={extra}"
            )
        for fid, (lo, hi) in self.valid_ranges.items():
            if lo > hi:
                raise ValueError(f"features.valid_ranges[{fid}] is not [min, max]: [{lo}, {hi}]")
        return self

    @model_validator(mode="after")
    def _overlap_cap_matches_f10_range(self) -> FeaturesConfig:
        """``overlap_cap`` and the upper bound of ``valid_ranges['F10']`` encode the same
        physical limit (§4.2 F10). Tie them at load so the cap can never drift above F10's
        range and silently emit an out-of-range value (non-negotiable #1)."""
        f10_hi = self.valid_ranges["F10"][1]
        if self.overlap_cap != f10_hi:
            raise ValueError(
                f"features.overlap_cap ({self.overlap_cap}) must equal the upper bound of "
                f"valid_ranges.F10 ({f10_hi})"
            )
        return self

    @model_validator(mode="after")
    def _geometry_counts_sane(self) -> FeaturesConfig:
        """Guard the F03b count/fraction constants so a mis-edit fails fast at load rather than
        producing a silent wrong feature (e.g. ``lateral_g_peak_top_n=0`` makes the top-N slice
        return *all* peaks). ``min_stint_laps`` needs >= 2 points for a regression slope."""
        if self.lateral_g_peak_top_n < 1:
            raise ValueError("features.lateral_g_peak_top_n must be >= 1")
        if self.min_lateral_g_peaks < 1:
            raise ValueError("features.min_lateral_g_peaks must be >= 1")
        if self.min_stint_laps < 2:
            raise ValueError("features.min_stint_laps must be >= 2 (regression needs >= 2 points)")
        if not 0.0 < self.min_xy_finite_fraction <= 1.0:
            raise ValueError("features.min_xy_finite_fraction must be in (0, 1]")
        return self


class NormaliseConfig(BaseModel):
    """Typed view over the ``normalise:`` block (F04 circuit-aware normalisation, §4.3/§8.2).

    Lives here, not in ``app/pipeline/normalise.py`` (standing rule, F01): the feature reads its
    config through a strictly-validated model. Holds only the constants unique to F04; the
    per-circuit fit floor it also needs — ``min_clean_laps`` (30) — is read from ``Thresholds`` so
    this file stays the single source for that value, and the quarantined ``held_out_circuits`` are
    read from ``AppConfig``. ``features`` is pinned to ``NORMALISED_FEATURE_IDS`` so the leakage
    chokepoint's scope cannot be silently changed in YAML.
    """

    model_config = ConfigDict(extra="forbid")

    features: list[str]  # §4.3 — the six circuit-dependent IDs Z-scored per circuit
    grouping_key: str  # §4.3 — fit mean/std per this key (circuit)
    artefact_filename: str  # §8.2 — written under paths.artefacts_dir
    ddof: int  # delta-degrees-of-freedom for std (0 = population std)
    min_std: float  # std below this -> zero-variance fallback (no division by zero)

    @model_validator(mode="after")
    def _features_are_the_normalised_ids(self) -> NormaliseConfig:
        """``normalise.features`` MUST be exactly the six ``NORMALISED_FEATURE_IDS`` **in canonical
        order**. Pinning the exact *sequence* (not just the set) makes widening, narrowing, *or
        reordering* the leakage chokepoint a build-breaking ``CONFIG_001`` rather than a silent YAML
        edit — and since the artefact stores ``normalised_features`` and the determinism hash covers
        it, a reorder would otherwise silently change ``content_hash`` and break cross-run
        byte-identity (non-negotiable #2/#3). ``NORMALISED_FEATURE_IDS`` is itself a subset of
        ``RANGE_CHECKED_IDS`` (asserted at module load below), so equality here also guarantees
        every ID is range-bearing — no separate check needed."""
        expected = list(NORMALISED_FEATURE_IDS)
        if list(self.features) != expected:
            got = set(self.features)
            missing = sorted(set(expected) - got)
            extra = sorted(got - set(expected))
            raise ValueError(
                f"normalise.features must be exactly {expected} in this order; "
                f"missing={missing} unexpected={extra} got={list(self.features)}"
            )
        return self

    @model_validator(mode="after")
    def _scalars_sane(self) -> NormaliseConfig:
        """Guard the grouping key and the numeric constants so a mis-edit fails fast at load
        rather than producing a silent wrong artefact (e.g. ``min_std <= 0`` re-enabling
        division by zero, or a non-``circuit`` grouping key)."""
        if self.grouping_key != "circuit":
            raise ValueError(f"normalise.grouping_key must be 'circuit', got {self.grouping_key!r}")
        if not self.artefact_filename:
            raise ValueError("normalise.artefact_filename must be non-empty")
        if self.ddof < 0:
            raise ValueError(f"normalise.ddof must be >= 0, got {self.ddof}")
        if self.min_std <= 0:
            raise ValueError(f"normalise.min_std must be > 0, got {self.min_std}")
        return self


_CV_STRATEGY_OPTION_A = "stratified_group_kfold_by_circuit"
_KNOWN_BASELINES = ("majority", "logistic_regression", "random_forest")
_CALIBRATION_METHODS = ("isotonic", "sigmoid")


class ClassifierConfig(BaseModel):
    """Typed view over the ``classifier:`` block (F06a driver-classifier training, §5.1/§11.1).

    Lives here, not in ``app/ml/classifier.py`` (standing rule, F01): training reads its config
    through a strictly-validated model. Holds only the constants unique to F06a; the F06b runtime
    gates (``confidence_gate``, ``entropy_ambiguous_bits``, ``null_ceiling``) live under
    ``Thresholds`` and the seeds / ``held_out_circuits`` under ``AppConfig`` — none are re-declared
    here. The CV-splitter keys are pinned (validators below) so the leakage-clean Option-A wiring
    cannot be silently swapped in YAML; flipping to the literal-composite fallback is a deliberate
    spec+config change, not a typo.
    """

    model_config = ConfigDict(extra="forbid")

    cv_folds: int  # §5.1 K-fold count
    cv_strategy: str  # pinned to Option A (StratifiedGroupKFold grouped by circuit)
    cv_stratify_by: str  # stratify key (driver)
    cv_group_by: str  # group key held wholly within one fold (circuit) — true leakage guard
    optuna_trials: int  # §5.1 minimum trial count (200 in repo config; tests override down)
    optuna_objective: str  # §5.1 macro-averaged F1 on the validation fold
    optuna_storage: str  # §9 Optuna SQLite storage URI (trial persistence across restarts)
    calibration_method: str  # §5.1 CalibratedClassifierCV method
    calibration_holdout_frac: float  # disjoint calibration split, excluded from fit folds
    baselines: list[str]  # §5.1/§11.1 comparators logged to MLflow
    target_macro_f1: float  # §5.1/§11.1 held-out (Belgium, Italy) target
    per_driver_floor: float  # §11.1 per-driver F1 floor
    mlflow_tracking_uri: str  # DB-backed store (registry needs non-file://; sqlite ok)
    mlflow_experiment: str  # MLflow experiment name
    mlflow_registered_model: str  # champion registry name
    model_artefact_filename: str  # calibrated champion persisted under paths.artefacts_dir
    labels_filename: str  # F06b seam — sorted driver-name list (index->name map for classes_)
    confidence_high: float  # F06b — max-prob >= this -> HIGH else MEDIUM (cross-checked vs gate)
    latency_warn_ms: float  # F06b — prediction latency over this -> PERF_WARN flag (§5.1)

    @model_validator(mode="after")
    def _cv_keys_pinned(self) -> ClassifierConfig:
        """Pin the CV splitter to Option A. ``cv_strategy`` selecting the splitter and the
        stratify/group keys being ``driver``/``circuit`` is the contract the spec's fold-membership
        guarantee rests on; a YAML edit here would silently change what "prevent circuit leakage"
        means, so make it a build-breaking ``CONFIG_001`` (non-negotiable #2)."""
        if self.cv_strategy != _CV_STRATEGY_OPTION_A:
            raise ValueError(
                f"classifier.cv_strategy must be {_CV_STRATEGY_OPTION_A!r} (Option A); "
                f"got {self.cv_strategy!r}. Flipping to the literal composite is a spec change."
            )
        if self.cv_stratify_by != "driver":
            raise ValueError(
                f"classifier.cv_stratify_by must be 'driver', got {self.cv_stratify_by!r}"
            )
        if self.cv_group_by != "circuit":
            raise ValueError(
                f"classifier.cv_group_by must be 'circuit', got {self.cv_group_by!r}"
            )
        if self.cv_folds < 2:
            raise ValueError(f"classifier.cv_folds must be >= 2, got {self.cv_folds}")
        return self

    @model_validator(mode="after")
    def _scalars_and_choices_sane(self) -> ClassifierConfig:
        """Fail fast on out-of-range fractions / unknown enum choices so a mis-edit is a load-time
        ``CONFIG_001`` rather than a surprise mid-training."""
        if self.optuna_trials < 1:
            raise ValueError(f"classifier.optuna_trials must be >= 1, got {self.optuna_trials}")
        if self.optuna_objective != "macro_f1":
            raise ValueError(
                f"classifier.optuna_objective must be 'macro_f1', got {self.optuna_objective!r}"
            )
        if self.calibration_method not in _CALIBRATION_METHODS:
            raise ValueError(
                f"classifier.calibration_method must be one of {list(_CALIBRATION_METHODS)}, "
                f"got {self.calibration_method!r}"
            )
        if not 0.0 < self.calibration_holdout_frac < 1.0:
            raise ValueError(
                "classifier.calibration_holdout_frac must be in (0, 1), "
                f"got {self.calibration_holdout_frac}"
            )
        if not 0.0 < self.target_macro_f1 <= 1.0:
            raise ValueError(
                f"classifier.target_macro_f1 must be in (0, 1], got {self.target_macro_f1}"
            )
        if not 0.0 < self.per_driver_floor <= 1.0:
            raise ValueError(
                f"classifier.per_driver_floor must be in (0, 1], got {self.per_driver_floor}"
            )
        if not self.baselines:
            raise ValueError("classifier.baselines must be non-empty")
        unknown = [b for b in self.baselines if b not in _KNOWN_BASELINES]
        if unknown:
            raise ValueError(
                f"classifier.baselines has unknown entries {unknown}; "
                f"known: {list(_KNOWN_BASELINES)}"
            )
        if len(set(self.baselines)) != len(self.baselines):
            raise ValueError(f"classifier.baselines has duplicates: {self.baselines}")
        # F06b serving scalars. The HIGH/MEDIUM split must sit at/above 0 and not exceed 1; the
        # cross-field check that it sits ABOVE thresholds.confidence_gate lives on AppConfig (this
        # sub-model cannot see its sibling `thresholds`).
        if not 0.0 < self.confidence_high <= 1.0:
            raise ValueError(
                f"classifier.confidence_high must be in (0, 1], got {self.confidence_high}"
            )
        if self.latency_warn_ms <= 0:
            raise ValueError(
                f"classifier.latency_warn_ms must be > 0, got {self.latency_warn_ms}"
            )
        for name, val in (
            ("optuna_storage", self.optuna_storage),
            ("mlflow_tracking_uri", self.mlflow_tracking_uri),
            ("mlflow_experiment", self.mlflow_experiment),
            ("mlflow_registered_model", self.mlflow_registered_model),
            ("model_artefact_filename", self.model_artefact_filename),
            ("labels_filename", self.labels_filename),
        ):
            if not val:
                raise ValueError(f"classifier.{name} must be non-empty")
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
    features: FeaturesConfig | None = None  # F03a; None until the features: block is present
    normalise: NormaliseConfig | None = None  # F04; None until the normalise: block is present
    classifier: ClassifierConfig | None = None  # F06a; None until the classifier: block is present

    @model_validator(mode="after")
    def _held_out_circuits_non_empty(self) -> AppConfig:
        """The held-out quarantine list is the spine of the F04 leakage chokepoint: an empty list
        would silently disable *both* the train-mask exclusion and the ``_assert_no_leakage``
        tripwire (``held_out & present`` is always empty), letting held-out data flow into every
        ``.fit``. Make an empty list a build-breaking ``CONFIG_001`` rather than a silent hole
        (non-negotiable #2)."""
        if not self.held_out_circuits:
            raise ValueError(
                "held_out_circuits must be non-empty (it gates the F04 leakage chokepoint)"
            )
        return self

    @model_validator(mode="after")
    def _confidence_high_above_gate(self) -> AppConfig:
        """F06b's HIGH/MEDIUM split must sit strictly ABOVE the LOW confidence gate, else the
        bands are incoherent (a 'HIGH' below the gate is unreachable / contradictory). This is a
        cross-field check between ``classifier.confidence_high`` and ``thresholds.confidence_gate``,
        so it lives here on ``AppConfig`` (the sub-model cannot see its sibling ``thresholds``)."""
        if self.classifier is not None and (
            self.classifier.confidence_high <= self.thresholds.confidence_gate
        ):
            raise ValueError(
                f"classifier.confidence_high ({self.classifier.confidence_high}) must be > "
                f"thresholds.confidence_gate ({self.thresholds.confidence_gate})"
            )
        return self


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
