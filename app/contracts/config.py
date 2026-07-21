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
# F07 — the master (§6.1) names exactly six radar domains (throttle, braking, speed, gear, engine,
# cornering); ``shap.nl_top_domains`` cannot exceed that. The domain->feature membership itself is a
# structural constant in ``app/ml/explain.py`` (DOMAIN_FEATURES), not config.
_RADAR_DOMAIN_COUNT = 6


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


class ShapConfig(BaseModel):
    """Typed view over the ``shap:`` block (F07 SHAP explainability, §6).

    Lives here, not in ``app/ml/explain.py`` (standing rule): the explainer reads its
    thresholds through a strictly-validated model. Holds only the F07 constants; the SHAP error
    codes (``SHAP_UNAVAILABLE`` / ``SHAP_CONSISTENCY_WARN``) are pre-seeded in ``errors.py`` and the
    six radar domains + their feature membership are structural (from §4.2) and live in code as
    ``DOMAIN_FEATURES`` (like ``FEATURE_IDS``), not here — a domain is not a tunable.
    """

    model_config = ConfigDict(extra="forbid")

    consistency_tolerance: float  # §6.2 — |base+Σshap-margin| above this -> SHAP_CONSISTENCY_WARN
    nl_dominance_margin: float  # §6.1/§6.2 — 6 domain scores within this rel. band -> generic desc
    nl_top_domains: int  # §6.1 — top-N domains by |magnitude| generate the style sentence ("Top 2")
    single_lap_warn_ms: float  # §6.1 — single-lap SHAP over this -> PERF_WARN ("< 50ms" budget)
    batch_warn_ms: float  # §6.1 — batch-of-100 SHAP over this -> PERF_WARN ("< 2 seconds" budget)
    beeswarm_filename: str  # §6.1 — global beeswarm artefact under paths.artefacts_dir
    stability_js_bins: int  # §6.1 — histogram bins for the per-feature Jensen-Shannon divergence
    stability_min_seasons: int  # §8.5 — fewer available seasons -> partial stability (F13 -> 206)

    @model_validator(mode="after")
    def _scalars_sane(self) -> ShapConfig:
        """Fail fast on out-of-range constants so a mis-edit is a load-time ``CONFIG_001`` rather
        than a silent mis-explanation (non-negotiable #1)."""
        if self.consistency_tolerance <= 0.0:
            raise ValueError(
                f"shap.consistency_tolerance must be > 0, got {self.consistency_tolerance}"
            )
        if not 0.0 < self.nl_dominance_margin < 1.0:
            raise ValueError(
                f"shap.nl_dominance_margin must be in (0, 1), got {self.nl_dominance_margin}"
            )
        if not 1 <= self.nl_top_domains <= _RADAR_DOMAIN_COUNT:
            raise ValueError(
                f"shap.nl_top_domains must be in 1..{_RADAR_DOMAIN_COUNT}, "
                f"got {self.nl_top_domains}"
            )
        if self.single_lap_warn_ms <= 0:
            raise ValueError(
                f"shap.single_lap_warn_ms must be > 0, got {self.single_lap_warn_ms}"
            )
        if self.batch_warn_ms <= 0:
            raise ValueError(f"shap.batch_warn_ms must be > 0, got {self.batch_warn_ms}")
        if self.stability_js_bins < 2:
            raise ValueError(f"shap.stability_js_bins must be >= 2, got {self.stability_js_bins}")
        if self.stability_min_seasons < 2:
            raise ValueError(
                f"shap.stability_min_seasons must be >= 2, got {self.stability_min_seasons}"
            )
        if not self.beeswarm_filename:
            raise ValueError("shap.beeswarm_filename must be non-empty")
        return self


class StyleSpaceConfig(BaseModel):
    """Typed view over the ``stylespace:`` block (F08 Model 2 — driver style space, §5.2).

    Lives here, not in ``app/ml/stylespace.py`` (standing rule): the style-space module reads
    its knobs through a strictly-validated model. Holds only the F08 constants; the four §5.2
    error codes (``SPARSE_PROJECTION`` / ``OUT_OF_DISTRIBUTION`` / ``UMAP_FALLBACK`` /
    ``PROJECTION_FALLBACK``) are pre-seeded in ``errors.py`` and the sparse floor is
    ``thresholds.min_clean_laps`` (30) — not re-declared here. F08 reads ITS OWN
    ``mlflow_tracking_uri`` (never ``classifier.mlflow_tracking_uri``), preserving its
    independence from the optional ``classifier`` block.
    """

    model_config = ConfigDict(extra="forbid")

    umap_n_components_5d: int  # §5.2 — 5D fingerprint coordinate dimensionality
    umap_n_components_2d: int  # §5.2 — 2D visualisation projection dimensionality
    umap_n_neighbors: int  # §5.2 — UMAP neighbourhood; capped to n_drivers-1 at build
    umap_min_dist: float  # UMAP layout tightness (design decision)
    umap_metric: str  # distance on the 56-dim aggregate (design decision)
    pca_n_components: int  # PCA fallback dims (first 2 == 2D fallback); also the OOD PC basis
    ood_sd_threshold: float  # §5.2 — |z| on ANY PC above this -> OUT_OF_DISTRIBUTION
    nearest_neighbours_k: int  # §7.4 — top-N nearest known drivers (cosine, 5D)
    umap_model_filename: str  # §5.2 — under paths.artefacts_dir (+ .sha256 sidecar)
    pca_model_filename: str  # §5.2 — under paths.artefacts_dir
    stability_n_runs: int  # §5.2/§11.2 — UMAP runs (fixed seed sequence) for Procrustes stability
    stability_variance_threshold: float  # §11.2 — Procrustes-aligned variance target (dissertation)
    mlflow_tracking_uri: str  # #3 — F08's OWN URI (same store, own experiment)
    mlflow_experiment: str  # #3 — build run logged here

    @model_validator(mode="after")
    def _scalars_sane(self) -> StyleSpaceConfig:
        """Fail fast on out-of-range constants so a mis-edit is a load-time ``CONFIG_001`` rather
        than a silent mis-projection (non-negotiable #1). The ``n_drivers``-dependent bounds
        (``n_neighbors < n_drivers``, ``n_components_5d < n_drivers``) are data-dependent and
        enforced fail-loud at BUILD time, not here."""
        if self.umap_n_components_2d < 1:
            raise ValueError(
                f"stylespace.umap_n_components_2d must be >= 1, got {self.umap_n_components_2d}"
            )
        if self.umap_n_components_5d <= self.umap_n_components_2d:
            raise ValueError(
                f"stylespace.umap_n_components_5d ({self.umap_n_components_5d}) must be > "
                f"umap_n_components_2d ({self.umap_n_components_2d})"
            )
        if self.pca_n_components < self.umap_n_components_5d:
            raise ValueError(
                f"stylespace.pca_n_components ({self.pca_n_components}) must be >= "
                f"umap_n_components_5d ({self.umap_n_components_5d}) so PCA serves both fallbacks"
            )
        if self.umap_n_neighbors < 2:
            raise ValueError(
                f"stylespace.umap_n_neighbors must be >= 2, got {self.umap_n_neighbors}"
            )
        if not 0.0 < self.umap_min_dist < 1.0:
            raise ValueError(
                f"stylespace.umap_min_dist must be in (0, 1), got {self.umap_min_dist}"
            )
        if not self.umap_metric:
            raise ValueError("stylespace.umap_metric must be non-empty")
        if self.ood_sd_threshold <= 0.0:
            raise ValueError(
                f"stylespace.ood_sd_threshold must be > 0, got {self.ood_sd_threshold}"
            )
        if self.nearest_neighbours_k < 1:
            raise ValueError(
                f"stylespace.nearest_neighbours_k must be >= 1, got {self.nearest_neighbours_k}"
            )
        if not self.umap_model_filename or not self.pca_model_filename:
            raise ValueError("stylespace.umap_model_filename/pca_model_filename must be non-empty")
        if self.umap_model_filename == self.pca_model_filename:
            raise ValueError(
                "stylespace.umap_model_filename and pca_model_filename must be distinct, "
                f"both are {self.umap_model_filename!r}"
            )
        if self.stability_n_runs < 2:
            raise ValueError(
                f"stylespace.stability_n_runs must be >= 2, got {self.stability_n_runs}"
            )
        if self.stability_variance_threshold <= 0.0:
            raise ValueError(
                "stylespace.stability_variance_threshold must be > 0, "
                f"got {self.stability_variance_threshold}"
            )
        if not self.mlflow_tracking_uri or not self.mlflow_experiment:
            raise ValueError("stylespace.mlflow_tracking_uri / mlflow_experiment must be non-empty")
        return self


class MonitoringConfig(BaseModel):
    """Typed view over the ``monitoring:`` block (F10 Model 3 — tyre degradation detector, §5.3).

    Lives here, not in ``app/monitoring/detector.py`` (standing rule): the detector reads its
    knobs through a strictly-validated model. Holds only the F10 constants; the six §5.3 error
    codes (``MONITOR_INACTIVE`` … ``MONITOR_ERROR``) are pre-seeded in ``errors.py``, the
    activation/slope floor is ``features.min_stint_laps`` (8), the cosine-core exclusion set is
    ``validate.degradation_features``, and the std floor is ``normalise.min_std`` — all
    referenced, never re-declared. F10 reads ITS OWN ``mlflow_tracking_uri`` (never the
    classifier's), preserving independence from the optional ``classifier`` block.
    """

    model_config = ConfigDict(extra="forbid")

    baseline_laps: int  # §5.3 — baseline fingerprint = first N clean laps of stint
    baseline_window_extended: int  # §5.3 failure table — candidate window when < N clean in first N
    rolling_window: int  # §5.3 — N-lap rolling mean of cosine distance
    alert_threshold_sd: float  # §5.3 — τ = μ + N·σ of historical intra-stint variation
    alert_confirm_laps: int  # §5.3 — consecutive laps above τ before the alert fires
    max_consecutive_nan: int  # §5.3 failure table — N consecutive NaN distances -> MONITOR_ERROR
    slope_weight: float  # §5.3 weighted F27/F28 component (master-silent value, divergence #8)
    confidence_medium_sd: float  # alert band: s >= μ + N·σ -> MEDIUM (master-silent, divergence #9)
    confidence_high_sd: float  # alert band: s >= μ + N·σ -> HIGH (master-silent, divergence #9)
    min_calibration_stints: int  # cell usable only with >= N training stints (divergence #12)
    attribution_top_k: int  # alert banner "feature drivers" count (§10.7)
    sensitivity_grid_sd: list[float]  # §11.3 threshold sensitivity grid (verbatim)
    variation_artefact_filename: str  # §5.3 — /artefacts/stint_variation.json
    mlflow_tracking_uri: str  # F10's OWN URI (same store, own experiment)
    mlflow_experiment: str  # calibration runs logged here

    @model_validator(mode="after")
    def _scalars_sane(self) -> MonitoringConfig:
        """Fail fast on out-of-range constants so a mis-edit is a load-time ``CONFIG_001`` rather
        than a silent mis-detection (non-negotiable #1). The band ordering guard keeps the
        LOW/MEDIUM/HIGH alert bands coherent with the τ multiplier (spec §5)."""
        if self.baseline_laps < 1:
            raise ValueError(f"monitoring.baseline_laps must be >= 1, got {self.baseline_laps}")
        if self.baseline_window_extended < self.baseline_laps:
            raise ValueError(
                f"monitoring.baseline_window_extended ({self.baseline_window_extended}) must be "
                f">= baseline_laps ({self.baseline_laps})"
            )
        if self.rolling_window < 1:
            raise ValueError(f"monitoring.rolling_window must be >= 1, got {self.rolling_window}")
        if self.alert_confirm_laps < 1:
            raise ValueError(
                f"monitoring.alert_confirm_laps must be >= 1, got {self.alert_confirm_laps}"
            )
        if self.max_consecutive_nan < 1:
            raise ValueError(
                f"monitoring.max_consecutive_nan must be >= 1, got {self.max_consecutive_nan}"
            )
        if self.alert_threshold_sd <= 0.0:
            raise ValueError(
                f"monitoring.alert_threshold_sd must be > 0, got {self.alert_threshold_sd}"
            )
        if self.slope_weight < 0.0:
            raise ValueError(f"monitoring.slope_weight must be >= 0, got {self.slope_weight}")
        if not self.alert_threshold_sd < self.confidence_medium_sd < self.confidence_high_sd:
            raise ValueError(
                "monitoring alert bands must satisfy alert_threshold_sd < confidence_medium_sd "
                f"< confidence_high_sd, got {self.alert_threshold_sd} / "
                f"{self.confidence_medium_sd} / {self.confidence_high_sd}"
            )
        if self.min_calibration_stints < 1:
            raise ValueError(
                f"monitoring.min_calibration_stints must be >= 1, got "
                f"{self.min_calibration_stints}"
            )
        if self.attribution_top_k < 1:
            raise ValueError(
                f"monitoring.attribution_top_k must be >= 1, got {self.attribution_top_k}"
            )
        if not self.sensitivity_grid_sd:
            raise ValueError("monitoring.sensitivity_grid_sd must be non-empty (§11.3)")
        for grid_sd in self.sensitivity_grid_sd:
            if grid_sd <= 0.0:
                raise ValueError(
                    f"monitoring.sensitivity_grid_sd entries must be > 0, got {grid_sd}"
                )
        if not self.variation_artefact_filename:
            raise ValueError("monitoring.variation_artefact_filename must be non-empty")
        if not self.mlflow_tracking_uri or not self.mlflow_experiment:
            raise ValueError("monitoring.mlflow_tracking_uri / mlflow_experiment must be non-empty")
        return self


# The 10 setup parameters F11 recommends (master §7.3), canonical order. Declared here so the
# ``PersonalisationConfig`` validator can pin ``fia_bounds`` to exactly these keys; the owning
# feature (``app/personalisation/engine.py``) carries the full table (kind + fingerprint basis).
_PERSONALISATION_PARAM_KEYS: tuple[str, ...] = (
    "front_wing_angle",
    "rear_wing_angle",
    "brake_bias",
    "differential_ramp_entry",
    "differential_ramp_exit",
    "throttle_map_curve",
    "suspension_stiffness_front",
    "suspension_stiffness_rear",
    "tyre_warmup_laps",
    "brake_duct_aperture",
)


class PersonalisationConfig(BaseModel):
    """Typed view over the ``personalisation:`` block (F11 car-setup engine, master §7).

    Lives here, not in ``app/personalisation/engine.py`` (standing rule): the engine reads its knobs
    through a strictly-validated model. Holds only the F11 constants; the four §7 error codes
    (``CONSTRAINT_CLIPPED`` … ``WIDE_CONFIDENCE``) are pre-seeded in ``errors.py``, seeds live on
    ``AppConfig``, ``held_out_circuits`` / ``thresholds.min_clean_laps`` / ``normalise.min_std`` are
    referenced not re-declared, and F11 reads its OWN ``mlflow_tracking_uri``.
    """

    model_config = ConfigDict(extra="forbid")

    top_quartile_pct: float  # §7.1 "best-performing stints" = fastest-quartile clean laps (div #14)
    confidence_floor: float  # §7.2 — per-parameter reliability < this → WIDE range
    confidence_medium: float  # rating band: r ≥ this → MEDIUM (master-silent, div #17)
    confidence_high: float  # rating band: r ≥ this → HIGH (master-silent, div #17)
    wide_range_pct: float  # §7.2 — WIDE range = ±N of the predicted value
    ood_sd_threshold: float  # §4.0-c — onboarding OOD attenuation onset (PC-SD)
    ood_floor: float  # §4.0-c — confidence-attenuation floor for a fully-OOD unknown driver
    consistency_front_wing_deg: float  # §7.2 — extreme front-wing threshold for consistency
    consistency_brake_bias_pct: float  # §7.2 — very-low brake-bias threshold
    monaco_circuit: str  # §7.2 — the hard max-downforce override circuit
    shap_top_k: int  # §6.1 — top fingerprint drivers per parameter (Screen 2 panel)
    cv_folds: int  # GroupKFold-by-driver reliability CV (div #17)
    model_artefact_filename: str  # under paths.artefacts_dir
    mlflow_tracking_uri: str  # F11's OWN URI (same store, own experiment)
    mlflow_experiment: str  # training runs logged here
    fia_bounds: dict[str, tuple[float, float]]  # §7.3 per-parameter legal (min, max)

    @model_validator(mode="after")
    def _scalars_sane(self) -> PersonalisationConfig:
        """Fail fast on out-of-range constants so a mis-edit is a load-time ``CONFIG_001`` rather
        than a silent mis-recommendation (non-negotiable #1). The band-ordering guard keeps the
        HIGH/MEDIUM/LOW/WIDE ratings coherent with the confidence floor (§7.2)."""
        if not 0.0 < self.top_quartile_pct <= 1.0:
            raise ValueError(
                f"personalisation.top_quartile_pct must be in (0, 1], got {self.top_quartile_pct}"
            )
        if not 0.0 < self.confidence_floor < self.confidence_medium < self.confidence_high <= 1.0:
            raise ValueError(
                "personalisation confidence bands must satisfy 0 < confidence_floor < "
                f"confidence_medium < confidence_high <= 1, got {self.confidence_floor} / "
                f"{self.confidence_medium} / {self.confidence_high}"
            )
        if not 0.0 < self.wide_range_pct < 1.0:
            raise ValueError(
                f"personalisation.wide_range_pct must be in (0, 1), got {self.wide_range_pct}"
            )
        if self.ood_sd_threshold <= 0.0:
            raise ValueError(
                f"personalisation.ood_sd_threshold must be > 0, got {self.ood_sd_threshold}"
            )
        if not 0.0 <= self.ood_floor <= 1.0:
            raise ValueError(
                f"personalisation.ood_floor must be in [0, 1], got {self.ood_floor}"
            )
        if self.consistency_front_wing_deg <= 0.0 or self.consistency_brake_bias_pct <= 0.0:
            raise ValueError("personalisation consistency thresholds must be > 0")
        if self.shap_top_k < 1:
            raise ValueError(f"personalisation.shap_top_k must be >= 1, got {self.shap_top_k}")
        if self.cv_folds < 2:
            raise ValueError(f"personalisation.cv_folds must be >= 2, got {self.cv_folds}")
        for name, val in (
            ("monaco_circuit", self.monaco_circuit),
            ("model_artefact_filename", self.model_artefact_filename),
            ("mlflow_tracking_uri", self.mlflow_tracking_uri),
            ("mlflow_experiment", self.mlflow_experiment),
        ):
            if not val:
                raise ValueError(f"personalisation.{name} must be non-empty")
        missing = sorted(set(_PERSONALISATION_PARAM_KEYS) - set(self.fia_bounds))
        extra = sorted(set(self.fia_bounds) - set(_PERSONALISATION_PARAM_KEYS))
        if missing or extra:
            raise ValueError(
                f"personalisation.fia_bounds must cover exactly the 10 setup parameters; "
                f"missing={missing}, unexpected={extra}"
            )
        for key, bound in self.fia_bounds.items():
            if bound[0] >= bound[1]:
                raise ValueError(
                    f"personalisation.fia_bounds[{key}] must have min < max, got {bound}"
                )
        return self


class CornersConfig(BaseModel):
    """Typed view over the ``corners:`` block (F09 corner-type decomposition + ablation, §4.4).

    Lives here, not in ``app/ml/corners.py`` (standing rule): the feature reads its knobs through a
    strictly-validated model. Holds only the F09 constants; ``CORNER_001`` is pre-seeded in
    ``errors.py``, the curvature/corner-segmentation machinery + ``min_xy_finite_fraction`` are
    ``features``' (referenced, never re-declared), the ablation CV seed is ``seeds.xgboost``, and
    F09 reads its OWN ``mlflow_tracking_uri``.
    """

    model_config = ConfigDict(extra="forbid")

    radius_slow_max: float  # §4.4 — apex radius (m) below this → SLOW corner
    radius_medium_max: float  # §4.4 — below this → MEDIUM; at/above → HIGH_SPEED
    min_corners_per_type: int  # sub-fingerprint cell needs ≥ N corners of the type (div #20)
    ablation_margin: float  # macro-F1 delta band for IMPROVES/NEUTRAL/DEGRADES (div #21)
    cv_folds: int  # ablation CV (mirrors classifier.cv_folds)
    subfingerprint_artefact_filename: str  # §4.2 — under paths.artefacts_dir
    mlflow_tracking_uri: str  # F09's OWN URI (same store, own experiment)
    mlflow_experiment: str  # sub-fingerprint build + ablation runs logged here

    @model_validator(mode="after")
    def _scalars_sane(self) -> CornersConfig:
        """Fail fast on out-of-range constants so a mis-edit is a load-time ``CONFIG_001`` rather
        than a silent mis-classification (non-negotiable #1). The radius ordering guard keeps the
        slow < medium < high-speed bands coherent (spec §4.4)."""
        if not 0.0 < self.radius_slow_max < self.radius_medium_max:
            raise ValueError(
                "corners radius bands must satisfy 0 < radius_slow_max < radius_medium_max, got "
                f"{self.radius_slow_max} / {self.radius_medium_max}"
            )
        if self.min_corners_per_type < 1:
            raise ValueError(
                f"corners.min_corners_per_type must be >= 1, got {self.min_corners_per_type}"
            )
        if not 0.0 < self.ablation_margin < 1.0:
            raise ValueError(
                f"corners.ablation_margin must be in (0, 1), got {self.ablation_margin}"
            )
        if self.cv_folds < 2:
            raise ValueError(f"corners.cv_folds must be >= 2, got {self.cv_folds}")
        for name, val in (
            ("subfingerprint_artefact_filename", self.subfingerprint_artefact_filename),
            ("mlflow_tracking_uri", self.mlflow_tracking_uri),
            ("mlflow_experiment", self.mlflow_experiment),
        ):
            if not val:
                raise ValueError(f"corners.{name} must be non-empty")
        return self


class DatabaseConfig(BaseModel):
    """Typed view over the ``database:`` block (F12 persistence layer, §8.3/§8.4/§9).

    Lives here, not in ``app/db/`` (standing rule, F01): the DB layer reads its knobs through a
    strictly-validated model. Holds only F12 constants; ``DB_UNAVAILABLE`` / ``HEALTH_FAIL`` /
    ``DATA_QUALITY_LOG_UNWRITABLE`` are pre-seeded in ``errors.py``; ``DB_INTEGRITY`` is F12's.
    The **DSN is a secret** from the env var named by ``dsn_env_var`` — never stored here
    (§9 RLS / secret hygiene); this block carries only the env-var *name* plus non-secret knobs.
    ``breaker`` reuses the shared ``BreakerConfig`` and mirrors the §8.3 Supabase policy verbatim.
    """

    model_config = ConfigDict(extra="forbid")

    dsn_env_var: str  # env var holding the async DSN (postgresql+psycopg://); secret, never in YAML
    test_dsn_env_var: str  # env var whose presence enables the marked Postgres migration test
    schema_name: str | None = None  # None = default schema; §12.2 sets apex_schema (shared inst.)
    migrations_path: str  # Alembic script_location
    pool_size: int  # async engine connection-pool size
    max_overflow: int  # overflow connections beyond pool_size
    pool_timeout_s: float  # wait for a pooled connection before erroring
    statement_timeout_s: float  # §8.3 — server-side per-statement cap (Supabase 5s)
    echo: bool  # SQLAlchemy SQL echo (debug only)
    breaker: BreakerConfig  # §8.3 Supabase-PostgreSQL retry/circuit policy (verbatim)

    @model_validator(mode="after")
    def _scalars_sane(self) -> DatabaseConfig:
        """Fail fast on out-of-range constants so a mis-edit is a load-time ``CONFIG_001`` rather
        than a surprise at first DB use (#1). The breaker knobs the async driver relies on (a
        positive ``timeout_s`` for ``asyncio.wait_for``, a ``>= 1`` open threshold, a positive
        cooldown window, a non-empty backoff list) are guarded here since ``BreakerConfig`` is
        deliberately unvalidated (shared with F01, whose breaker clamps a short backoff list)."""
        for name, val in (
            ("dsn_env_var", self.dsn_env_var),
            ("test_dsn_env_var", self.test_dsn_env_var),
            ("migrations_path", self.migrations_path),
        ):
            if not val:
                raise ValueError(f"database.{name} must be non-empty")
        if self.schema_name is not None and not self.schema_name:
            raise ValueError("database.schema_name must be non-empty when set (null = default)")
        if self.pool_size < 1:
            raise ValueError(f"database.pool_size must be >= 1, got {self.pool_size}")
        if self.max_overflow < 0:
            raise ValueError(f"database.max_overflow must be >= 0, got {self.max_overflow}")
        if self.pool_timeout_s <= 0:
            raise ValueError(f"database.pool_timeout_s must be > 0, got {self.pool_timeout_s}")
        if self.statement_timeout_s <= 0:
            raise ValueError(
                f"database.statement_timeout_s must be > 0, got {self.statement_timeout_s}"
            )
        if self.breaker.timeout_s <= 0:
            raise ValueError(
                f"database.breaker.timeout_s must be > 0, got {self.breaker.timeout_s}"
            )
        if self.breaker.open_after < 1:
            raise ValueError(
                f"database.breaker.open_after must be >= 1, got {self.breaker.open_after}"
            )
        if self.breaker.window_s <= 0:
            raise ValueError(f"database.breaker.window_s must be > 0, got {self.breaker.window_s}")
        if not self.breaker.backoff_s:
            raise ValueError("database.breaker.backoff_s must be non-empty")
        return self


class ApiConfig(BaseModel):
    """Typed view over the ``api:`` block (F13 FastAPI serving layer, §8.5/§8.3/§9).

    Lives here, not in ``app/api/`` (standing rule, F01): the API layer reads its knobs through a
    strictly-validated model. Holds only F13 constants; the three F13 codes (``NOT_FOUND`` /
    ``FORBIDDEN`` / ``API_ERROR``) are in ``errors.py``, the model artefact paths come from the
    per-model blocks, and the DSN stays env-only (F12). Redis is F14 — ``cache_connected`` is
    reported at ``/health`` but is never a critical component.
    """

    model_config = ConfigDict(extra="forbid")

    version: str  # /health version string (single source for the API version)
    cors_origins: list[str]  # §8.5 frontend origins (F16 Next.js dev server)
    docs_enabled: bool  # serve OpenAPI /docs + /redoc
    metrics_internal_only: bool  # §8.5 — /metrics returns 403 to external callers
    latency_slo_ms: float  # §11 p95 target (surfaced as a PERF_WARN metric; formally timed in F20)
    request_timeout_s: float  # per-request wall-clock ceiling (declared; enforced in F14/F20)
    min_projection_laps: int  # §8.5 — /fingerprint/unknown floor: fewer laps -> 400 (no magic num)
    max_request_laps: int  # §11 DoS guard — /fingerprint/unknown laps above this -> 400
    max_feature_keys: int  # §11 DoS guard — /predict feature_vector keys above this -> 400
    allow_credentials: bool = False  # §8.5 CORS — F13 has no auth; credentials off by default

    @model_validator(mode="after")
    def _scalars_sane(self) -> ApiConfig:
        """Fail fast on bad constants so a mis-edit is a load-time ``CONFIG_001`` rather than
        a surprise at first request (non-negotiable #1)."""
        if not self.version:
            raise ValueError("api.version must be non-empty")
        if self.latency_slo_ms <= 0:
            raise ValueError(f"api.latency_slo_ms must be > 0, got {self.latency_slo_ms}")
        if self.request_timeout_s <= 0:
            raise ValueError(f"api.request_timeout_s must be > 0, got {self.request_timeout_s}")
        if self.min_projection_laps < 1:
            raise ValueError(
                f"api.min_projection_laps must be >= 1, got {self.min_projection_laps}"
            )
        if self.max_request_laps < self.min_projection_laps:
            raise ValueError("api.max_request_laps must be >= min_projection_laps")
        if self.max_feature_keys < 1:
            raise ValueError(f"api.max_feature_keys must be >= 1, got {self.max_feature_keys}")
        # A wildcard origin with credentials is the classic any-origin-with-credentials hole (§8.5).
        if self.allow_credentials and "*" in self.cors_origins:
            raise ValueError("api.cors_origins must not contain '*' when allow_credentials is true")
        return self


class CacheConfig(BaseModel):
    """Typed view over the ``cache:`` block (F14 Redis cache, §8.2/§8.3/§9).

    Lives here, not in ``app/cache/`` (standing rule, F01): the cache reads its knobs through a
    strictly-validated model. Holds only F14 cache constants; the three cache codes
    (``CACHE_MISS`` / ``CACHE_UNAVAILABLE`` / ``CACHE_DEGRADED``) are pre-seeded in ``errors.py``.
    The **Redis DSN is a secret** from the env var named by ``url_env_var`` — never stored here
    (§9). The cache is an optimisation, never a hard dependency: a down Redis / open breaker
    degrades to a graceful miss. ``breaker`` reuses the shared ``BreakerConfig`` and mirrors the
    §8.3 Redis policy verbatim (timeout 2s / 1 retry / 1s backoff / open after 2 in 30s).
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool  # false -> RedisCache is a no-op (always CACHE_MISS); system unaffected
    url_env_var: str  # env var holding the Redis DSN (rediss://…@upstash); secret, never in YAML
    key_prefix: str  # §4.0-h namespace root; full key = {prefix}:v{api.version}:{ns}:{id}
    driver_profile_ttl_s: int  # §9 — 1-hour driver-profile TTL (the one master-named TTL)
    model_output_ttl_s: int  # /predict model-output cache TTL (master-silent, div #34)
    normalisation_ttl_s: int  # normalisation-param cache TTL (master-silent, div #34)
    stylespace_ttl_s: int  # style-space read TTL (master-silent, div #34)
    breaker: BreakerConfig  # §8.3 Redis retry/circuit policy (verbatim)

    @model_validator(mode="after")
    def _scalars_sane(self) -> CacheConfig:
        """Fail fast on bad constants so a mis-edit is a load-time ``CONFIG_001`` rather than a
        surprise at first cache use (#1). The breaker knobs the async client relies on are guarded
        here since ``BreakerConfig`` is deliberately unvalidated (shared with F01)."""
        if not self.url_env_var:
            raise ValueError("cache.url_env_var must be non-empty")
        if not self.key_prefix:
            raise ValueError("cache.key_prefix must be non-empty")
        for name, val in (
            ("driver_profile_ttl_s", self.driver_profile_ttl_s),
            ("model_output_ttl_s", self.model_output_ttl_s),
            ("normalisation_ttl_s", self.normalisation_ttl_s),
            ("stylespace_ttl_s", self.stylespace_ttl_s),
        ):
            if val <= 0:
                raise ValueError(f"cache.{name} must be > 0, got {val}")
        if self.breaker.timeout_s <= 0:
            raise ValueError(f"cache.breaker.timeout_s must be > 0, got {self.breaker.timeout_s}")
        if self.breaker.open_after < 1:
            raise ValueError(
                f"cache.breaker.open_after must be >= 1, got {self.breaker.open_after}"
            )
        if self.breaker.window_s <= 0:
            raise ValueError(f"cache.breaker.window_s must be > 0, got {self.breaker.window_s}")
        if not self.breaker.backoff_s:
            raise ValueError("cache.breaker.backoff_s must be non-empty")
        return self


class CeleryConfig(BaseModel):
    """Typed view over the ``celery:`` block (F14 Celery task queue, §8.2/§9).

    Lives here, not in ``app/tasks/`` (standing rule, F01): the queue reads its knobs through a
    strictly-validated model. Redis is broker AND result backend (§9); both DSNs are **secrets**
    from the env vars named here — never stored in YAML. Celery retry/timeout constants are
    master-SILENT (§8.3's numbers are for external HTTP deps) → chosen here (divergence #35). The
    offline test suite sets ``always_eager`` so the task body runs in-process with no broker/worker.
    """

    model_config = ConfigDict(extra="forbid")

    broker_url_env_var: str  # Redis broker DSN env var (secret, never in YAML)
    result_backend_env_var: str  # Redis result backend DSN env var (secret, never in YAML)
    task_default_queue: str  # named queue the ingestion task routes to
    worker_concurrency: int  # worker pool size (design decision)
    task_soft_time_limit_s: int  # soft per-task cap → graceful stop (design decision)
    task_time_limit_s: int  # hard per-task cap → kill (design decision)
    task_max_retries: int  # ingest-task retries on a transient fault (div #35)
    task_retry_backoff_s: list[float]  # backoff between task retries (div #35)
    result_ttl_s: int  # result-backend retention for the terminal IngestSessionResponse
    always_eager: bool = False  # tests set true (synchronous, in-process, no broker)

    @model_validator(mode="after")
    def _scalars_sane(self) -> CeleryConfig:
        """Fail fast on bad constants so a mis-edit is a load-time ``CONFIG_001`` (#1)."""
        for name, val in (
            ("broker_url_env_var", self.broker_url_env_var),
            ("result_backend_env_var", self.result_backend_env_var),
            ("task_default_queue", self.task_default_queue),
        ):
            if not val:
                raise ValueError(f"celery.{name} must be non-empty")
        if self.worker_concurrency < 1:
            raise ValueError(
                f"celery.worker_concurrency must be >= 1, got {self.worker_concurrency}"
            )
        if self.task_soft_time_limit_s < 1:
            raise ValueError("celery.task_soft_time_limit_s must be >= 1")
        if self.task_time_limit_s <= self.task_soft_time_limit_s:
            raise ValueError(
                f"celery.task_time_limit_s ({self.task_time_limit_s}) must be > "
                f"task_soft_time_limit_s ({self.task_soft_time_limit_s})"
            )
        if self.task_max_retries < 0:
            raise ValueError(f"celery.task_max_retries must be >= 0, got {self.task_max_retries}")
        if not self.task_retry_backoff_s:
            raise ValueError("celery.task_retry_backoff_s must be non-empty")
        for b in self.task_retry_backoff_s:
            if b < 0:
                raise ValueError(f"celery.task_retry_backoff_s entries must be >= 0, got {b}")
        if self.result_ttl_s < 1:
            raise ValueError(f"celery.result_ttl_s must be >= 1, got {self.result_ttl_s}")
        return self


class ReplayConfig(BaseModel):
    """Typed view over the ``replay:`` block (F14 replay engine, §5.3/§10.7/§12.1).

    Lives here, not in ``app/replay/`` (standing rule, F01). Read-only: the engine reads persisted
    (session, driver) laps and streams a growing prefix through F10 ``monitor_stint``. ALL Model-3
    constants (``baseline_laps`` … ``alert_confirm_laps``) live under ``monitoring:`` and are
    referenced, never re-declared; playback speed is client-side (Zustand), not server config. Only
    a DoS cap on the returned laps array is unique to F14.
    """

    model_config = ConfigDict(extra="forbid")

    max_replay_laps: int  # DoS guard on the returned laps array (mirrors api.max_request_laps)

    @model_validator(mode="after")
    def _scalars_sane(self) -> ReplayConfig:
        if self.max_replay_laps < 1:
            raise ValueError(f"replay.max_replay_laps must be >= 1, got {self.max_replay_laps}")
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
    circuit_types: list[str] = Field(default_factory=list)  # §8.4 ENUM; F11 one-hot vocabulary
    ingest: IngestConfig | None = None  # F01; None until the ingest: block is present
    # YAML key is `validate:`; the Python attribute is `lap_validation` to avoid
    # shadowing pydantic's deprecated BaseModel.validate (F02; None until block present).
    lap_validation: ValidateConfig | None = Field(default=None, alias="validate")
    features: FeaturesConfig | None = None  # F03a; None until the features: block is present
    normalise: NormaliseConfig | None = None  # F04; None until the normalise: block is present
    classifier: ClassifierConfig | None = None  # F06a; None until the classifier: block is present
    shap: ShapConfig | None = None  # F07; None until the shap: block is present
    stylespace: StyleSpaceConfig | None = None  # F08; None until the stylespace: block is present
    monitoring: MonitoringConfig | None = None  # F10; None until the monitoring: block is present
    personalisation: PersonalisationConfig | None = None  # F11; None until the block is present
    corners: CornersConfig | None = None  # F09; None until the corners: block is present
    database: DatabaseConfig | None = None  # F12; None until the database: block is present
    api: ApiConfig | None = None  # F13; None until the api: block is present
    cache: CacheConfig | None = None  # F14; None until the cache: block is present
    celery: CeleryConfig | None = None  # F14; None until the celery: block is present
    replay: ReplayConfig | None = None  # F14; None until the replay: block is present

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

    @model_validator(mode="after")
    def _monitoring_requires_sibling_blocks(self) -> AppConfig:
        """F10 reads ``features.min_stint_laps`` (activation/slope floor),
        ``validate.degradation_features`` (cosine-core exclusion set), and
        ``normalise.min_std`` (σ/scaler floor) — a ``monitoring:`` block without those
        siblings would crash the detector at first use instead of at config load. Cross-block,
        so it lives here on ``AppConfig`` (spec §5 validator placement)."""
        if self.monitoring is not None and (
            self.features is None or self.lap_validation is None or self.normalise is None
        ):
            missing = [
                name
                for name, block in (
                    ("features", self.features),
                    ("validate", self.lap_validation),
                    ("normalise", self.normalise),
                )
                if block is None
            ]
            raise ValueError(
                f"monitoring: requires the {missing} block(s) — F10 references their keys"
            )
        return self

    @model_validator(mode="after")
    def _personalisation_requires_sibling_blocks(self) -> AppConfig:
        """F11 reads ``normalise`` (to build the setup-target aggregates), ``features`` (the F27/…
        feature-id contract), and a non-empty ``circuit_types`` (the drop-first one-hot vocab) —
        a ``personalisation:`` block without those would crash the engine at first use rather than
        at config load. Cross-block, so it lives here on ``AppConfig`` (spec §5 placement)."""
        if self.personalisation is not None:
            missing = [
                name
                for name, present in (
                    ("normalise", self.normalise is not None),
                    ("features", self.features is not None),
                    ("circuit_types", bool(self.circuit_types)),
                )
                if not present
            ]
            if missing:
                raise ValueError(
                    f"personalisation: requires {missing} — F11 references them (§5)"
                )
        return self

    @model_validator(mode="after")
    def _corners_requires_features_block(self) -> AppConfig:
        """F09 reads ``features``' curvature / ``_corner_indices`` / ``min_xy_finite_fraction``
        machinery — a ``corners:`` block without the ``features:`` block would crash the
        decomposition at first use instead of at config load. Cross-block, so it lives here."""
        if self.corners is not None and self.features is None:
            raise ValueError("corners: requires the features: block — F09 reads its keys (§5)")
        return self

    @model_validator(mode="after")
    def _replay_requires_monitoring_block(self) -> AppConfig:
        """F14's replay engine drives F10 ``monitor_stint``, which reads the whole ``monitoring:``
        block (and, transitively, ``features``/``validate``/``normalise`` — already guarded by
        ``_monitoring_requires_sibling_blocks``). A ``replay:`` block without ``monitoring:`` would
        crash the engine at first call instead of at config load, so make it a build-breaking
        ``CONFIG_001`` here (cross-block, spec §5 placement)."""
        if self.replay is not None and self.monitoring is None:
            raise ValueError("replay: requires monitoring: — F14 drives F10 monitor_stint")
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
