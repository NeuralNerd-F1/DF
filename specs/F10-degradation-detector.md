# Feature F10 — Model 3: Tyre Degradation Detector

> One spec, one session. This document is the single source of truth a session loads
> to build this feature. It must be self-contained: a session should not need the
> 42-page master spec open, only this file plus CLAUDE.md.

**Status:** SPEC WRITTEN
**Master spec sections:** §2.3 (Problem D), §5.2 (5D output line), §5.3 (Model 3 core +
"Degradation Detector Failure Modes" table), §8.2 (module row), §8.4 (`degradation_alerts`,
`laps`, `data_quality_log`), §8.5 (replay endpoint), §10.7 (Screen 6), §11.3 (evaluation),
§12.1 (partner outputs), §13 (Wk 11–12 rows), §14 (risk register degradation row),
§15.1 (Contribution 2), §15.2 (limitations)
**Layer / Week:** ML / Wk 11–12
**Depends on:** F04 (normalisation artefact, read-only), F08 (style-space handle — fingerprint
snapshot only), F03b (live-window slope machinery reuse), F02/F01 (lap flags + tyre age carried
on the stint matrix)

## 1. Purpose (one paragraph)

F10 is Model 3: a **statistical deviation detector, not a separate ML model** (§5.3 — "This is
not a separate ML model — it is a statistical deviation detector operating on the fingerprint
vectors produced by Models 1 and 2"). Within a race stint it measures the cosine distance
between each lap's normalised feature vector and a within-stint baseline fingerprint, smooths
it with a 3-lap rolling mean, and fires a degradation alert when the deviation exceeds a
per-driver / per-circuit-type threshold calibrated from training data — the goal being an
alert **2–5 laps before** the behavioural signal becomes a lap-time signal (§2.3). It is the
system's RQ3 instrument and dissertation Contribution 2.

## 2. Upstream / downstream contracts

- **Consumes:**
  - **F04** → `artefacts/normalisation.json` via `load_normalisation_params` (raises
    `NormalisationError` on missing/corrupt/hash-mismatch) and `apply_normalisation`
    (read-only, never re-fits, never raises; unknown circuit → global params +
    `NORMALISATION_FALLBACK`). F04 §2 names F10 as a consumer: "every model loads this same
    artefact and applies it; **none of them re-fits**." F04 z-scores only **6 of the 28**
    features — F10 layers its own train-only standardiser on top (§4.0-e).
  - **F08** → `StyleSpaceHandle` + `project_unknown(...) -> ProjectionResult` — used **only**
    to produce the displayed `fingerprint_snapshot` (§8.5 replay payload, §10.7 radar). The
    deviation score itself does **not** run through the 5D space — see the reconciliation
    note in §4.0-a. F08 serving never raises; every F08 flag (`SPARSE_PROJECTION`,
    `OUT_OF_DISTRIBUTION`, `UMAP_FALLBACK`, `PROJECTION_FALLBACK`) is carried opaquely inside
    the embedded `ProjectionResult`, never re-interpreted by F10. **Expectation setting:**
    F08's sparse gate is `thresholds.min_clean_laps` (30), so any stint prefix < 30 laps —
    i.e. virtually every monitored stint — returns `SPARSE_PROJECTION` / `LOW_CONFIDENCE` /
    `status=DEGRADED` **by design**; tests must assert the sparse flag, not a clean `OK`.
  - **F03b** → `compute_degradation_slopes(stint: list[StintLap], *, stint_id: int, config,
    log_path) -> DegradationResult` — reused (never reimplemented) for the **live-window**
    F27/F28 slopes (§5.3 "computed on the live stint window"), via the explicit
    frame→`StintLap` adapter in §4.2 step 8 (`StintLap` needs `lap_number: int`,
    `tyre_age: int | None`, `throttle_mean` = F01, `brake_peak_mean` = F06 —
    `app/pipeline/features.py:159`).
  - **F01/F02 (via the stint matrix)** → per-lap `track_status` (F02 owns the green-flag
    interpretation — Filter 1, `LAP_001`, `validate.green_track_status_code`), `is_clean`,
    `has_interpolation` (the `LAP_INTERP` accept-flag), `tyre_age` (possibly
    `DERIVED_AGE`-tagged upstream).
  - **Training matrix (calibration input)** — the F06a-era wide matrix **plus stint
    context columns** (see §3); F10 defines its own `_REQUIRED_COLUMNS` superset because the
    F06a/F08 matrix carries only `driver`/`circuit`/`season`/`split` + `F01..F28`. The
    §4.4 evaluation matrix additionally carries `compound`.
- **Produces:**
  - `artefacts/stint_variation.json` — the §5.3 threshold artefact ("Stored per driver per
    circuit type. Computed from training data."), content-hashed like `normalisation.json`,
    written by `calibrate_stint_variation` and nothing else. Contains the variation stats,
    slope stats, pit-window medians, the **train-only 26-dim feature standardiser**
    (§4.0-e), and the calibration-time `monitoring` knob values.
  - Typed results: `StintVariationFitResult`, `MonitorHandle`, `StintMonitorResult`
    (per-lap `LapDeviation` series + optional `DegradationAlert`), `MonitorHealthResult`,
    `LeadTimeReport` (§11.3 machinery, formally run in F20).
  - MLflow calibration run in experiment `apex-degradation` (own URI key, F08 precedent).
- **Consumed by:**
  - **F13** — serves `StintMonitorResult` fields; `health()` feeds `GET /health` (503 on
    `HEALTH_FAIL`).
  - **F14** — the replay engine drives `monitor_stint` lap-by-lap with growing stint
    prefixes (§8.5 `GET /replay/{session_id}/{driver_id}` → per-lap `deviation_score`,
    `alert_state`, `fingerprint_snapshot`).
  - **F12** — the persisted `degradation_alerts` columns map from `DegradationAlert` as:
    `lap_number ← alert_lap`; `deviation_score`, `alert_confidence`,
    `feature_attribution` direct; `lead_time_actual` is populated **post-race by
    downstream** from the exposed `first_crossing_lap`/`alert_lap`. The remaining alert
    fields (`threshold`, `top_features`, `laps_since_threshold_crossed`,
    `optimal_window_remaining_laps`) are §5.3/§12.1 **payload-only** fields consumed by
    F13/F19, not table columns.
  - **F15** — failsafe harness exercises the §6 failure modes end-to-end.
  - **F19** — Screen 6 gauge/timeline/alert banner render this result.
  - **F20** — runs `lead_time_report` for §11.3 (lead time, TP/FP, compound stratification,
    threshold sensitivity); held-out circuits are touched **only** there and only read-only.
  - **§12.1 partner outputs** — per-lap `deviation_score` and
    `optimal_window_remaining_laps` (int) in the alert payload.

## 3. Inputs & outputs (typed)

### Input — stint frame (serving), calibration matrix, evaluation matrix

`monitor_stint` takes one stint of **F02-accepted, F03-completed** lap rows for a single
driver, ordered by `lap_number` (strictly increasing — this also rules out duplicates), as a
`pd.DataFrame`. Callers own stint slicing; F10 does not police stint membership beyond the
monotonicity check:

| Column group | Columns | Notes |
|---|---|---|
| Features | `F01`..`F28` (RAW, un-normalised) | normalised + standardised internally (§4.2 steps 5–6); `F27`/`F28` columns are **ignored** by the detector (see §4.0-b) |
| Lap context | `lap_number: int`, `tyre_age: float \| null`, `track_status: str \| null`, `is_clean: bool`, `has_interpolation: bool` | eligibility rule §4.2 step 3: a **known non-green** token or `is_clean=False` makes a lap ineligible; a **null** token defers to `is_clean` (F02 Filter 1 already rejects unknown-status laps at ingest — `validate.py:344` "Missing/unknown status is REJECTED, never green" — so a null here can only arrive from a non-F02 path, e.g. F05 uploads, which vouch for laps via `is_clean`) |

Identity (`driver`, `circuit`, `circuit_type`) is passed as keyword arguments —
`circuit_type` is the same caller-supplied string F03a's imputation uses (e.g.
`"high_speed"`); `circuit_type=None` is allowed and routes threshold lookup straight to the
driver-global fallback tier.

`calibrate_stint_variation` takes the wide training matrix with
`_REQUIRED_COLUMNS = (driver, circuit, circuit_type, season, session, stint, lap_number,
split, track_status, is_clean, has_interpolation, tyre_age, F01..F28)`. Missing columns →
raise (`MODEL_TRAINING_FAILED`); at serve time missing columns → typed `INVALID_INPUT`
result.

`lead_time_report` takes an **evaluation matrix** = `_REQUIRED_COLUMNS` **plus `compound`**
(the F01-persisted token; the string `"UNKNOWN"` marks compound-unknown stints, §3.1) and an
optional per-stint `pit_lap` column; when `pit_lap` is absent, **the stint's final lap is
the pit-in ground-truth proxy** (consistent with divergence #10). Malformed/empty input →
typed `INVALID_INPUT` `LeadTimeReport` (never raises).

### Output — result objects (all Pydantic on `BaseResult` unless noted, `extra="forbid"`)

Feature-owned enums live in `app/monitoring/detector.py` (per the `enums.py` docstring:
"Feature-specific enums (e.g. the degradation alert band) live with their owning feature"):

```python
class StintMonitorState(StrEnum):   # per-lap alert_state (§8.5) + stint-level state
    INACTIVE = "INACTIVE"           # stint < activation floor — nothing scored
    BASELINE = "BASELINE"           # lap selected into the baseline fingerprint
    SKIPPED = "SKIPPED"             # lap not scored: ineligible (non-green/unclean) — per-lap only
    PENDING = "PENDING"             # scored, but rolling window not yet filled (no composite score)
    NOMINAL = "NOMINAL"             # composite score at/below threshold
    WATCH = "WATCH"                 # threshold crossed, not yet confirmed ("logged but not alerted")
    ALERT = "ALERT"                 # crossing confirmed alert_confirm_laps consecutive laps
    HALTED = "HALTED"               # MONITOR_ERROR — monitoring halted for this stint

class AlertConfidence(StrEnum):     # §5.3 "confidence of alert (LOW / MEDIUM / HIGH ...)"
    LOW = "LOW"; MEDIUM = "MEDIUM"; HIGH = "HIGH"
```

**Every stint lap gets exactly one `LapDeviation` row** — baseline, skipped, and scored
alike. No lap is silently dropped (constitution rule 1); an eligible candidate-window lap
that was not needed for the baseline (e.g. lap 5 when the extended window selected laps
1, 2, 4) is **scored**, not dropped.

```python
class LapDeviation(BaseModel):                  # one per stint lap (timeline chart row)
    lap_number: int
    state: StintMonitorState                    # BASELINE | SKIPPED | PENDING | NOMINAL | WATCH | ALERT | HALTED
    deviation_raw: float | None                 # 1 − cos(x̂_k, b̂); None for BASELINE/SKIPPED/NaN laps
    deviation_rolling: float | None             # rolling mean; None until window filled
    deviation_score: float | None               # rolling + slope component (the τ comparand)
    slope_component: float                      # slope_weight · slope_z (0.0 when inactive)
    slope_component_active: bool
    threshold: float | None                     # τ for this stint (constant per stint)
    flags: list[ErrorCode] = []                 # e.g. DISTANCE_NAN, LAP_001 (skip reason) on this lap

class DegradationAlert(BaseModel):              # §5.3 alert payload
    alert_lap: int                              # lap the confirmation completed on → degradation_alerts.lap_number
    first_crossing_lap: int                     # start of the confirmed run (lead_time_actual seam)
    laps_since_threshold_crossed: int
    deviation_score: float                      # → degradation_alerts.deviation_score
    threshold: float
    alert_confidence: AlertConfidence           # → degradation_alerts.alert_confidence; capped LOW under DEGRADED_QUALITY
    feature_attribution: dict[str, float]       # 28 keys; sums to deviation_score → degradation_alerts.feature_attribution
    top_features: list[str]                     # attribution_top_k names (Screen 6 banner; payload-only)
    optimal_window_remaining_laps: int | None   # §12.1 pit-window recommendation (payload-only)

class StintMonitorResult(BaseResult):           # never-raises serving result
    driver: str = ""
    circuit: str = ""
    circuit_type: str | None = None
    monitor_state: StintMonitorState = StintMonitorState.INACTIVE   # state of the LAST lap (or INACTIVE/HALTED)
    laps: list[LapDeviation] = []               # one row per input lap; honest partial series on HALTED
    alert: DegradationAlert | None = None       # most recent confirmed alert, else None
    baseline_lap_numbers: list[int] = []
    baseline_extended: bool = False             # 5-lap window was needed
    degraded_quality: bool = False              # all-baseline-laps-interpolated cap
    threshold_source: str = ""                  # "driver_circuit_type" | "driver_global" | "global"
    pit_window_valid: bool = True               # False when crossing knobs differ from calibration (§4.3)
    fingerprint_snapshot: ProjectionResult | None = None   # F08 embed (may itself be DEGRADED/sparse)
    fingerprint_available: bool = False         # snapshot is not None AND snapshot.coords_5d is not None
    n_laps_scored: int = 0                      # laps with a computed distance
    consecutive_nan: int = 0

class StintVariationFitResult(BaseResult):      # calibration output
    artefact_path: str = ""
    content_hash: str = ""
    n_stints_used: int = 0
    n_entries: int = 0                          # (driver, circuit_type) cells written
    n_driver_global: int = 0
    coverage: float = 0.0                       # cells with n_stints ≥ min_calibration_stints
    mlflow_run_id: str | None = None            # None when MLflow degraded (status=DEGRADED + DQ row)

class StintLeadTime(BaseModel):                 # one evaluated stint (Week-11 alert dump row)
    driver: str; circuit: str; circuit_type: str | None
    compound: str                               # F01 token; "UNKNOWN" excluded from stratified views
    n_laps: int
    pit_lap: int                                # ground truth (explicit column or final-lap proxy)
    alert_lap: int | None                       # None = no alert (miss)
    lead_time_laps: int | None                  # pit_lap − alert_lap; negative = alert after pit
    safety_car_affected: bool                   # any non-green lap in the stint (FN characterisation)

class LeadTimeReport(BaseResult):               # §11.3 machinery — read-only, never raises
    n_stints: int = 0
    mean_lead_time_laps: float | None = None    # target 2–5 (§11.3)
    tp_rate: float | None = None                # target > 0.70 (alerted within 5 laps before pit)
    fp_rate: float | None = None                # target < 0.20 (alert more than 5 laps before pit)
    per_stint: list[StintLeadTime] = []         # includes the FN stints with their characteristics
    per_compound: dict[str, dict[str, float]] = {}      # Soft/Medium/Hard/Intermediate → {tp_rate, fp_rate, mean_lead}
    early_alerts_by_driver_circuit: dict[str, int] = {} # "driver/circuit" → early-alert count (§11.3 FP analysis)
    sensitivity: dict[str, dict[str, dict[str, float]]] = {}  # grid_sd → circuit_type → {tp_rate, fp_rate}
    optimal_threshold_per_circuit_type: dict[str, float] = {} # §11.3 "Report optimal threshold per circuit type"

class MonitorHealthResult(BaseResult):          # mirrors F06b/F08 health
    ready: bool = False
    model_version: str = ""
    reason: ErrorCode | None = None
```

`MonitorHandle` is a frozen dataclass (mirrors `StyleSpaceHandle`): the parsed
`stint_variation.json` tables (per-(driver, circuit_type) `{mean, std, n_stints,
median_crossing_to_pit}`, per-driver global, global-all, per-circuit-type + global slope
stats), the **26-dim feature standardiser stats** (`feature_mean`/`feature_std`, §4.0-e),
the artefact's `calibration_config` snapshot, `pit_window_valid: bool` (§4.3),
`normalisation_params: NormalisationParams`, `model_version` (the artefact content hash),
`ready: bool`.

**Errors raised (only at build/load, mirroring F06b/F08):** `DetectorCalibrationError`
(carries `TRAIN_LEAKAGE` or `MODEL_TRAINING_FAILED`) from `calibrate_stint_variation`;
`DetectorLoadError` (carries `MODEL_NOT_FOUND`) from `load_detector`. `monitor_stint`,
`health`, and `lead_time_report` **never raise**.

**Naming guard:** the class name `DegradationResult` is already owned by F03b
(`app/pipeline/features.py:145`) — F10 must not reuse it. The names above are final.

## 4. Method / algorithm

### 4.0 Master-fidelity reconciliations (read first)

- **(a) Which space does the deviation live in?** §5.3's Method row is explicit: "**Cosine
  distance between current lap's normalised feature vector and the within-stint baseline
  fingerprint (computed from first 3 clean laps of stint)**." The deviation core therefore
  operates in the **per-lap feature space** (F04-normalised, then F10-standardised — see
  (e)), not the F08 5D style space. The F08 spec's §2 one-liner ("reads the 5D fingerprint
  as the within-stint baseline coordinate") was a pre-F10 paraphrase and is corrected here:
  a single lap has no 56-dim mean+std aggregate, so a per-lap 5D projection is not defined
  in F08's machinery. F08's contribution to F10 is the **displayed** fingerprint
  (`fingerprint_snapshot` in the §8.5 replay payload; "Fingerprint displayed but no
  degradation score computed" in the `MONITOR_INACTIVE` row; the §10.7 radar), obtained by
  projecting the stint-so-far laps through `project_unknown`.
- **(b) The cosine core excludes F27/F28 (divergence #8, part 1).** The matrix's F27/F28
  columns are per-stint broadcasts computed by F03b over the **whole** stint — using them
  inside the per-lap vector would (i) leak future laps into a leading indicator during
  replay and (ii) double-count them against §5.3's separate "F27 and F28 integration ...
  weighted component". The cosine core therefore uses the **26 features excluding
  `validate.degradation_features` (`[F27, F28]`)**; F27/F28 enter only via the live-window
  weighted component below. This follows the master's own "computed on the live stint
  window" wording.
- **(c) `MONITOR_ERROR` is also the serving backstop.** §5.3 defines it for 3 consecutive
  NaN distances ("return MONITOR_ERROR and halt monitoring for this stint"). F10 reuses it
  for any unforeseen internal serving fault, because the semantics are identical (halt this
  stint's monitoring, typed result, DQ-logged) — the same pattern as F02
  `VALIDATION_ERROR` / F06b `INFERENCE_ERROR`, without minting a new code.
- **(d) Statelessness.** `monitor_stint` is a pure function of the stint prefix it is given.
  Replay (F14) advances lap-by-lap by re-calling with a growing prefix; there is no hidden
  monitor state between calls (O(n²) over a ≤ ~35-lap stint is negligible and keeps the
  never-raises contract trivially testable).
- **(e) Train-only feature standardiser (divergence #7 — the F08-#6 twin).** F04
  z-scores only 6 of the 28 features per circuit; "the other 22 feature keys pass through
  untouched" (`normalise.py:26`) in raw physical units spanning ~4 orders of magnitude
  (F16 gear-hold ms ≤ 10⁴ vs 0–1 fractions). A cosine over that mixed-scale vector — and
  its per-feature attribution — would measure unit scale, not behaviour: the exact defect
  the ratified F08 Gate-2 amendment fixed with a train-only aggregate scaler
  (divergence #6, `00-INDEX.md`). F10 therefore fits a **per-feature standardiser over the
  26 cosine dims** inside `calibrate_stint_variation`, strictly **after** the quarantine +
  tripwire, on training rows only (mean/std per feature, `std` floored at
  `normalise.min_std`); it is persisted **inside `stint_variation.json`** (covered by the
  content hash), carried on `MonitorHandle`, and applied **read-only** at every serve and
  calibration forward pass — one choke point, F04 first, standardiser second. A bundle
  without it (or with wrong width/keys) refuses to load (`MODEL_NOT_FOUND`) — never a
  silent unscaled serve.

### 4.1 Calibration — `calibrate_stint_variation` (train-only fit site)

1. **Validate matrix**: `_REQUIRED_COLUMNS` present, non-empty → else raise
   `DetectorCalibrationError(MODEL_TRAINING_FAILED)`.
2. **Quarantine FIRST** (F04/F06a/F08 pattern): drop `held_out_circuits` (Belgium, Italy)
   and every `split != "train"` row **before any reduction**; then `_assert_no_leakage`
   tripwire — any surviving held-out row raises `DetectorCalibrationError(TRAIN_LEAKAGE)`.
   **No artefact is written on a poisoned run.**
3. **Load F04 params** (read-only) for the pipeline. Do **not** pre-normalise the matrix
   here: Pass B feeds **RAW** rows through the identical §4.2 serving path, so
   `apply_normalisation` and the standardiser each run **exactly once, inside the
   pipeline** — train/serve parity by construction (never double-normalised).
4. **Fit the feature standardiser** (§4.0-e): F04-normalise the quarantined training rows
   once (a throwaway transform for fitting only), take per-feature mean/std over the 26
   cosine dims, floor `std` at `normalise.min_std`. This is the only scaler F10 fits, and
   this step is the only place it is fitted.
5. **Group into stints** by `(driver, season, session, stint)`; keep stints with
   ≥ `features.min_stint_laps` (8) rows.
6. **Pass A — slope stats**: for each kept stint, walk the lap prefixes; **only when a
   prefix has ≥ `features.min_stint_laps` usable laps** (known tyre age + non-NULL F01/F06
   — F03b's own usability rule) call `compute_degradation_slopes` via the §4.2-8 adapter.
   Insufficient prefixes are simply not calls — F03b is never invoked on input it would
   NULL, so calibration produces **zero** `FEATURE_001` spam rows. Pool the computed
   slopes per `circuit_type` (and a global pool) → `{f27: {mean, std}, f28: {mean, std}}`
   (std floored at `normalise.min_std`). Slope inputs are the **RAW** F01/F06 values in
   both calibration and serving (F04 does not touch F01/F06 — its six are
   F09/F11/F13/F14/F19/F20 — so RAW is also the normalised value; pinned to remove all
   ambiguity).
7. **Pass B — deviation stats**: run the §4.2 pipeline (baseline → cosine → rolling →
   slope component, using Pass A stats) over each kept stint. Pool **exactly the laps
   whose `deviation_score` is non-None** (rolling window filled; laps with an inactive
   slope component contribute their `r_k + 0.0`; SKIPPED/NaN laps contribute nothing) per
   `(driver, circuit_type)` → `{mean μ, std σ, n_stints}`; also pool per-driver (all
   circuit types) and global-all tiers from the same quarantined population. σ floored at
   `normalise.min_std`.
8. **Pass C — pit-window stats**: each training stint is scored against **exactly the τ
   the §4.2-9 lookup would resolve for its `(driver, circuit_type)` given the Pass-B
   tables** (`min_calibration_stints` gate and fallback tiers included), using this
   config's `alert_threshold_sd` and `alert_confirm_laps`. Record
   `(last_lap_of_stint − first_crossing_lap)` for stints with a confirmed crossing — the
   stint's final lap is the pit-in proxy (divergence #10). The `(driver, circuit_type)`
   **and** fallback-tier medians all pool those same per-stint gaps. Store as
   `median_crossing_to_pit` (None where no stint crossed).
9. **Write artefact** `artefacts/stint_variation.json`: `{schema_version, config:
   {alert_threshold_sd, alert_confirm_laps, rolling_window, baseline_laps,
   baseline_window_extended, slope_weight, min_stint_laps}, feature_scaler: {feature_id:
   {mean, std}}, entries, driver_global, global, slope_stats, content_hash}` — canonical
   JSON (sorted keys), content-hashed exactly like F04, **no timestamp anywhere in the
   file** (run timestamps live in the MLflow run, F04 precedent). The computation is fully
   deterministic (no RNG, no wall clock) → **byte-identical artefact on re-run**.
10. **MLflow**: log params (the `monitoring:` block), metrics (`n_stints_used`,
    `n_entries`, `coverage`), and the artefact to experiment `apex-degradation`
    (`monitoring.mlflow_tracking_uri` — its OWN key, same sqlite store; F08 precedent). An
    MLflow fault sets `status=DEGRADED` + a DQ row (`code=MODEL_TRAINING_FAILED`,
    `detail` tagged `mlflow`, `fallback_used=True`, F08's exact mechanism —
    `stylespace.py:744`) and leaves `mlflow_run_id=None`; it **never aborts the artefact
    write** and adds no `flags` entry.
11. **No usable stints at all** → raise `DetectorCalibrationError(MODEL_TRAINING_FAILED)`.

### 4.2 Serving — `monitor_stint` (read-only, never raises)

1. **Input gate**: required columns present, frame non-empty, `lap_number` strictly
   increasing (also excludes duplicates) → else typed `INVALID_INPUT` (`status=FAILED`,
   `error_code=INVALID_INPUT`), DQ-logged. Stint membership is the caller's contract
   (§3); F10 does not police it further.
2. **Activation gate** (§5.3): `len(stint) < features.min_stint_laps` (8) →
   `MONITOR_INACTIVE` (`status=DEGRADED`, **operative `error_code=MONITOR_INACTIVE`**,
   `monitor_state=INACTIVE`, empty `laps`). The fingerprint snapshot (step 10) is
   **still attempted** — "Fingerprint displayed but no degradation score computed."
3. **Baseline selection** (§5.3 + failure table): candidates = first `baseline_laps` (3)
   stint laps; a lap is **eligible** iff `is_clean=True` AND `track_status` is not a
   known non-green token (null defers to `is_clean` — §3). If < 3 eligible among the
   first 3 → **extend the candidate window to the first `baseline_window_extended` (5)
   laps** and take the earliest 3 eligible. Still < 3 → `BASELINE_INSUFFICIENT`
   (`status=FAILED`, operative `error_code`), DQ-logged. Every excluded candidate lap is
   DQ-logged with `code=LAP_001` (reusing F02's not-green code — same meaning) and keeps a
   `LapDeviation` row (`state=SKIPPED`, flag `LAP_001`). Selected laps get
   `state=BASELINE`. **Every other stint lap is a scoring candidate** — including an
   eligible candidate-window lap the baseline did not need (lap 5 when the extension
   selected 1, 2, 4): it is scored, never dropped.
4. **Baseline quality** (§5.3 failure table): if **all** selected baseline laps have
   `has_interpolation=True` → accept, set `degraded_quality=True`, flag
   `DEGRADED_QUALITY` (in `flags`, `status=DEGRADED`), and cap any alert confidence at
   `LOW` "regardless of deviation score".
5. **Normalise + standardise** every lap row: F04 `apply_normalisation` with the handle's
   params (unknown circuit → global params, `NORMALISATION_FALLBACK` flag,
   `status=DEGRADED` — F04-owned semantics, propagated honestly), then the handle's
   train-only standardiser over the 26 cosine dims (§4.0-e) — read-only, single choke
   point shared with Pass B.
6. **Deviation core**: `b` = per-feature mean of the 3 baseline laps' standardised 26-dim
   vectors. For every scoring-candidate lap `k`: if ineligible (non-green/unclean) →
   `state=SKIPPED`, flag `LAP_001`, DQ row, no distance. Else `d_k = 1 − cos(x_k, b)`.
   `d_k` NaN/±inf → flag `DISTANCE_NAN` on that row, DQ-log, skip in the rolling mean.
   **Only NaN/±inf distances increment the consecutive-NaN counter; a SKIPPED lap
   freezes an in-progress NaN run — it neither increments nor resets it (symmetric with
   the step-11 WATCH freeze); only a valid distance resets it.**
   `max_consecutive_nan` (3) consecutive NaNs → `MONITOR_ERROR`
   (`status=FAILED`, operative code, `monitor_state=HALTED`), returning the honest
   partial `laps` series up to the halt.
7. **Rolling mean** (§5.3): `r_k` = mean of the last `rolling_window` (3) **valid**
   distances; until then the lap is `state=PENDING` (`deviation_score=None`). SKIPPED
   laps neither extend nor reset a WATCH run — the confirmation counter freezes across
   them. If an **activated stint ends with zero composite scores** (all laps
   PENDING/SKIPPED — e.g. alternating NaNs that never hit 3 consecutive), the result is
   `status=DEGRADED` with the accumulated per-lap flags, `monitor_state=PENDING`, and one
   DQ row — an explicit "monitoring produced nothing" outcome, never a silent NOMINAL.
8. **Slope component** (§5.3 "F27 and F28 integration"): pre-gate once per call — if the
   whole stint cannot produce a slope (no known tyre age anywhere, or fewer than
   `features.min_stint_laps` usable laps in total), write **one** DQ row, set
   `slope_component_active=False` everywhere, and never invoke F03b (avoids its
   per-lap-per-slope `FEATURE_001` rows — `features.py:1113`). Otherwise, for each scored
   lap `k` whose prefix has ≥ `features.min_stint_laps` usable laps, build the adapter
   input `[StintLap(lap_number=int(row.lap_number), tyre_age=int(row.tyre_age) if known
   else None, throttle_mean=row.F01_raw, brake_peak_mean=row.F06_raw)]` and call
   `compute_degradation_slopes(prefix_laps, stint_id=0, ...)` (`stint_id` is a fixed
   sentinel — `monitor_stint` holds exactly one stint). `slope_z_k = (|z(f27)| +
   |z(f28)|) / 2` with z-scores from the handle's per-circuit-type (fallback: global)
   slope stats; **slope stats missing at every tier** (possible when training had no
   usable tyre age) → component `0.0`, `slope_component_active=False`, one DQ row —
   deviation still scored, cosine-only. Composite **`s_k = r_k + slope_weight ·
   slope_z_k`** (divergence #8).
9. **Threshold** (§5.3): lookup order `(driver, circuit_type)` [requires `n_stints ≥
   min_calibration_stints`] → `(driver, global)` → `global` — every fallback hop flags
   `FALLBACK_THRESHOLD`, sets `threshold_source`, DQ-logs (`fallback_used=True`), and
   `status=DEGRADED` ("Report in UI"). **τ = μ + `alert_threshold_sd` (2.0) · σ**, where
   **μ/σ come from the artefact and only the multiplier comes from live config** — the
   master's "Alert threshold ... Computed from training data. Stored per driver per
   circuit type" is satisfied by storing the *stats*; deriving τ from the live multiplier
   is what keeps the §14 "reduce threshold to 1.5SD" contingency a one-line config turn
   (see §4.3 for the mismatch policy).
10. **Fingerprint snapshot**: if a `style_handle` was supplied, call `project_unknown`
    on the **RAW** stint prefix — F08 normalises internally; handing it the
    F04-normalised or standardised frame would double-normalise at the seam (pinned by
    the §8 seam-parity test) — and embed the returned `ProjectionResult` verbatim.
    `fingerprint_available = snapshot is not None and snapshot.coords_5d is not None` —
    defined on the **returned result**, not on exceptions (F08's raw-NN fallback returns
    `coords_5d=None` without raising). A `< thresholds.min_clean_laps` (30) prefix is
    sparse **by design** (§2) — still `fingerprint_available=True` when coords exist. Any
    exception (should be impossible — defence in depth) or no handle →
    `fingerprint_snapshot=None`, `fingerprint_available=False`. Never blocks scoring
    (F07 "never blocks a prediction" precedent).
11. **Alert state machine** (§5.3): crossing = `s_k > τ` (strict). First crossing of a run
    → `WATCH`; "Single-lap threshold crossing is **logged but not alerted**" → DQ row.
    `alert_confirm_laps` (2) **consecutive** scored crossings → `ALERT` fires at lap `k`
    with `first_crossing_lap` = start of the run;
    `laps_since_threshold_crossed = k − first_crossing_lap`. A sub-threshold scored lap
    resets the run to `NOMINAL`. Alert confidence: `s_k ≥ μ + confidence_high_sd·σ` →
    `HIGH`; `≥ μ + confidence_medium_sd·σ` → `MEDIUM`; else `LOW` (divergence #9); capped
    `LOW` under `DEGRADED_QUALITY`. Pit window: `optimal_window_remaining_laps =
    max(0, median_crossing_to_pit − laps_since_threshold_crossed)`; `None` when the
    resolved calibration cell has no median **or** `handle.pit_window_valid` is False
    (divergence #10).
12. **Attribution** (§5.3 "SHAP decomposition of deviation (which features are driving
    it)"): exact additive decomposition via the unit-vector identity
    `1 − cos(x̂, b̂) = ½‖x̂ − b̂‖² = Σᵢ ½(x̂ᵢ − b̂ᵢ)²` in the **standardised** space: each
    cosine feature's contribution is the rolling-window mean of its per-lap
    `½(x̂ᵢ − b̂ᵢ)²` terms; F27/F28 receive their halves of the slope component.
    **`sum(feature_attribution.values()) == deviation_score` exactly** (float tolerance)
    — the mandatory attribution unit test. The detector is not a tree model, so literal
    TreeExplainer SHAP does not apply; this exact decomposition implements the master's
    intent (divergence #11).
13. **Backstop** (§4.0-c): any unforeseen exception → best-effort DQ log →
    `MONITOR_ERROR` result (`status=FAILED`, `monitor_state=HALTED`), never a propagated
    exception.

**`BaseResult` field discipline**: `FAILED` outcomes put the operative code in
`error_code`; scored-but-`DEGRADED` outcomes carry their codes in `flags` with
`error_code=None`; the one non-scoring DEGRADED outcome, `MONITOR_INACTIVE`, keeps its
code in `error_code` (operative-code-on-DEGRADED, F08 precedent). **The §6 table's
placement column is authoritative** wherever this paragraph and an individual row could
be read differently.

### 4.3 Load & health — `load_detector` / `health`

`load_detector` reads `stint_variation.json` (schema + content-hash validated) and
`normalisation.json` (via F04's loader). Missing / corrupt / hash-mismatch /
schema-mismatch artefact, an **empty threshold table** (no entries at any tier), or a
**missing/wrong-width `feature_scaler`** (§4.0-e — never a silent unscaled serve) →
DQ-log then **raise `DetectorLoadError(MODEL_NOT_FOUND)`**; the process must refuse to
serve (F06b/F08 precedent). An **empty `slope_stats`** table does *not* refuse — the
detector serves cosine-only with `slope_component_active=False` + a DQ row at load
(degrade, don't die).

**Config-mismatch policy** (the τ-provenance contract): the artefact stores the
calibration-time knobs (§4.1-9). At load, compare them to live config:
- **Structural knobs** (`rolling_window`, `baseline_laps`, `baseline_window_extended`,
  `slope_weight`, `features.min_stint_laps`) — a mismatch invalidates the Pass-B μ/σ
  themselves → `DetectorLoadError(MODEL_NOT_FOUND)`: recalibrate, the artefact is stale.
- **Crossing knobs** (`alert_threshold_sd`, `alert_confirm_laps`) — μ/σ are unaffected
  (the multiplier is applied at serve), but Pass-C's `median_crossing_to_pit` was
  computed under the old values → load succeeds with `pit_window_valid=False` + one DQ
  row; every alert's `optimal_window_remaining_laps` is `None` until recalibration. This
  keeps §14 contingencies 1–2 one-line config turns without silently serving a stale
  pit-window statistic.

`health(handle)` → `MonitorHealthResult(ready, model_version, reason)`; not-ready reports
`HEALTH_FAIL` with `MODEL_NOT_FOUND` as `reason` (F13 surfaces 503).

### 4.4 Evaluation machinery — `lead_time_report` (shipped here, run in F20)

Implements the §11.3 table in full, read-only (no fit anywhere), over the §3 evaluation
matrix (2023–2024 race stints; pit lap = explicit `pit_lap` column or the final-lap
proxy): **mean alert lead time** (target **2–5 laps before pit**; "Negative lead time
(alert after pit) means system missed the signal" — reported per circuit and compound in
`per_stint`), **per-stint TP rate** (**> 70 % of stints alerted within 5 laps before
pit**) with the FN stints characterised per §11.3 ("were they short stints, unusual
compounds, or safety car affected?" — `n_laps`/`compound`/`safety_car_affected` on each
`StintLeadTime`), **FP rate** (**< 20 % of stints alert more than 5 laps before pit**)
with `early_alerts_by_driver_circuit` ("Report which driver-circuit combinations produce
most early alerts"), **compound stratification** (**Soft, Medium, Hard, Intermediate**;
`"UNKNOWN"`-compound stints excluded from the stratified view only), and the **threshold
sensitivity sweep** at `sensitivity_grid_sd` = **[1, 1.5, 2, 2.5] SD**, stratified per
circuit type, reporting TP/FP per grid point and the arg-best grid point per circuit type
(**"Report optimal threshold per circuit type"**). Malformed input → typed
`INVALID_INPUT` `LeadTimeReport`; never raises.

**Week-11 deliverable** (§13: "Degradation alerts on 2022 validation stints"): after
calibration, run `lead_time_report` read-only over the **`split == "val"`** validation stints
— the `val` split **is** the 2022 validation set by APEX's split convention, so the filter is
on `split`, not a magic `2022` season literal — training circuits only (no held-out contact),
and log the report as an artefact of the MLflow calibration run; its `per_stint` rows are the
alert dump. Held-out circuits (Belgium, Italy) enter **only** via F20's invocation — F10's own
tests use synthetic + training-circuit fixtures only.

## 5. config.yaml keys this feature adds

F10 adds a typed `monitoring:` block (`MonitoringConfig` in `app/contracts/config.py`,
`extra="forbid"`, standing typed-config rule) plus an optional
`monitoring: MonitoringConfig | None = None` field on `AppConfig`.

```yaml
monitoring:                          # F10 Model 3 — degradation detector (§5.3)
  baseline_laps: 3                   # §5.3 — baseline fingerprint = first N clean laps of stint
  baseline_window_extended: 5        # §5.3 failure table — candidate window when <N clean in first N
  rolling_window: 3                  # §5.3 — N-lap rolling mean of cosine distance
  alert_threshold_sd: 2.0            # §5.3 — τ = μ + N·σ of historical intra-stint variation
  alert_confirm_laps: 2              # §5.3 — consecutive laps above τ before the alert fires
  max_consecutive_nan: 3             # §5.3 failure table — N consecutive NaN distances -> MONITOR_ERROR
  slope_weight: 0.25                 # §5.3 weighted F27/F28 component — value master-silent (divergence #8)
  confidence_medium_sd: 2.5          # alert band: s ≥ μ + N·σ -> MEDIUM (master-silent, divergence #9)
  confidence_high_sd: 3.0            # alert band: s ≥ μ + N·σ -> HIGH  (master-silent, divergence #9)
  min_calibration_stints: 3          # cell usable only with ≥ N training stints (master-silent, divergence #12)
  attribution_top_k: 3               # alert banner "feature drivers" count (§10.7)
  sensitivity_grid_sd: [1.0, 1.5, 2.0, 2.5]   # §11.3 threshold sensitivity grid (verbatim)
  variation_artefact_filename: stint_variation.json   # §5.3 /artefacts/stint_variation.json
  mlflow_tracking_uri: sqlite:///artefacts/mlflow.db  # F10's OWN URI (same store, own experiment)
  mlflow_experiment: apex-degradation                 # calibration runs logged here
```

**Referenced, not re-declared** (house rule): `features.min_stint_laps` (8 — §4.2's slope
floor **is** §5.3's activation floor; but see §10 for contingency 3's real blast radius),
`validate.degradation_features` (`[F27, F28]` — the cosine-core exclusion set),
`held_out_circuits`, `normalise.min_std` (σ + standardiser floor), `thresholds.min_clean_laps`
(F08's sparse gate, noted in §2 only), `paths.artefacts_dir` / `paths.data_quality_log`.
The feature standardiser (§4.0-e) needs **no config key** — it is artefact state.

**Validator placement** (F08 precedent): scalar sanity on `MonitoringConfig`
(`baseline_laps ≥ 1`, `baseline_window_extended ≥ baseline_laps`, `rolling_window ≥ 1`,
`alert_confirm_laps ≥ 1`, `max_consecutive_nan ≥ 1`, `alert_threshold_sd > 0`,
`slope_weight ≥ 0`, `min_calibration_stints ≥ 1`, `attribution_top_k ≥ 1`, grid non-empty
and all > 0, and `alert_threshold_sd < confidence_medium_sd < confidence_high_sd`);
cross-block invariants on `AppConfig` (`monitoring` present requires `features`,
`validate`, and `normalise` blocks — it reads their keys). All violations → load-time
`CONFIG_001`.

**No new error code.** All six §5.3 codes are already seeded in
`app/contracts/errors.py:86–92` (`MONITOR_INACTIVE`, `BASELINE_INSUFFICIENT`,
`FALLBACK_THRESHOLD`, `DEGRADED_QUALITY`, `DISTANCE_NAN`, `MONITOR_ERROR`); F10 reuses
`LAP_001` (F02's not-green code, as the baseline-exclusion / mid-stint-skip flag — same
meaning, F08-style reuse), `LAP_PARTIAL` (F02's "tyre age UNKNOWN → F27/F28 NULL" code, as
the slope-inactive marker — the semantically exact fit for "no live-window slope", and the
one code that keeps F03b's `FEATURE_001` count at **zero** for a no-tyre-age stint, §7-11),
`INVALID_INPUT`, `MODEL_NOT_FOUND`, `HEALTH_FAIL`, `NORMALISATION_FALLBACK`, `TRAIN_LEAKAGE`,
`MODEL_TRAINING_FAILED`, `DATA_QUALITY_LOG_UNWRITABLE` (F07/F08 precedent).

## 6. Error codes & failure modes (the failsafe contract)

Master §5.3 failure-table rows are marked ▣; the rest complete the failsafe contract.
Code placement follows the §4.2 discipline: `error_code` on FAILED (and the two
non-scoring DEGRADED) outcomes; `flags` entries on scored-but-degraded outcomes.

| Failure mode | Detection | Error code (placement) | System response | Logged? | User-facing message |
|---|---|---|---|---|---|
| ▣ Stint length < 8 laps | Lap count vs `features.min_stint_laps` | `MONITOR_INACTIVE` (error_code) | `status=DEGRADED`, `monitor_state=INACTIVE`, no scores; fingerprint snapshot still attempted | yes | "Monitoring inactive — stint too short (needs 8 laps)." |
| ▣ Baseline candidate is safety-car / unclean | Eligibility check (§4.2-3) during baseline selection | `LAP_001` (per-lap flag) | Exclude lap (`state=SKIPPED`); if < 3 eligible in first 3 → extend window to first 5 | yes (per excluded lap) | — (internal; surfaced only if it degrades further) |
| ▣ < 3 eligible baseline laps after 5-lap extension | Eligibility count | `BASELINE_INSUFFICIENT` (error_code) | `status=FAILED`, no scores | yes | "Cannot establish a clean baseline for this stint." |
| ▣ Driver variation entry unavailable (new circuit type / unknown driver / thin cell) | Lookup miss in `stint_variation.json` tiers | `FALLBACK_THRESHOLD` (flag) | Fall back `(driver, circuit_type)` → driver-global → global; `threshold_source` set; `status=DEGRADED`; "Report in UI" | yes (`fallback_used=True`) | "Alert threshold from fallback (no history for this driver/circuit type)." |
| ▣ All 3 baseline laps interpolated | `has_interpolation` on all selected baseline laps | `DEGRADED_QUALITY` (flag) | Accept baseline; `status=DEGRADED`; `degraded_quality=True`; alert confidence **capped LOW** regardless of score | yes | "Baseline quality degraded — alert confidence capped at LOW." |
| ▣ Cosine distance NaN | NaN/±inf check on **every** distance | `DISTANCE_NAN` (per-lap flag) | Skip lap in rolling mean; counter++ | yes | — (timeline shows a gap) |
| ▣ 3 consecutive NaN distances | Consecutive-NaN counter vs `max_consecutive_nan` (SKIPPED laps freeze — never increment or reset — the run) | `MONITOR_ERROR` (error_code) | `status=FAILED`, `monitor_state=HALTED`, honest partial lap series returned | yes | "Degradation monitoring halted for this stint (telemetry unusable)." |
| Mid-stint scored lap non-green / unclean | Eligibility re-check on scoring candidates | `LAP_001` (per-lap flag) | `state=SKIPPED`; no distance; freezes (neither increments nor resets) both an in-progress consecutive-NaN run and a WATCH run | yes | — (timeline shows a gap) |
| Activated stint ends with zero composite scores | All rows PENDING/SKIPPED at stint end | accumulated per-lap flags | `status=DEGRADED`, `monitor_state=PENDING` — explicit "nothing scored" outcome | yes (one stint-level row) | "Insufficient usable laps to score this stint." |
| Slope stats missing at every tier / no usable tyre age in stint | Lookup miss / slope pre-gate | `LAP_PARTIAL` (flag) — F02's "tyre age UNKNOWN → F27/F28 NULL"; NOT `FEATURE_001` (F03b is never invoked, so its count stays 0) | Component 0.0, `slope_component_active=False`; cosine-only deviation still served | yes (one row) | — |
| Unforeseen internal serving fault | Backstop `except` at the `monitor_stint` boundary | `MONITOR_ERROR` (error_code) | Best-effort DQ log → typed `FAILED`/`HALTED` result; **never a raised exception** (§4.0-c) | best-effort | "Degradation monitoring halted for this stint (internal error)." |
| Malformed stint frame (missing columns / empty / non-monotonic or duplicate laps) | Input gate | `INVALID_INPUT` (error_code) | `status=FAILED`, nothing scored | yes | "Stint data malformed — monitoring not run." |
| Malformed evaluation matrix in `lead_time_report` | Input gate | `INVALID_INPUT` (error_code) | Typed `FAILED` `LeadTimeReport`; never raises | yes | — (build/eval tooling) |
| Unknown circuit at normalisation | F04 `apply_normalisation` `used_global` | `NORMALISATION_FALLBACK` (flag) | Global params used; `status=DEGRADED` | yes (by F04) | "Circuit not in calibration — global normalisation used." |
| Single-lap threshold crossing (not a failure — §5.3 mandate) | State machine | — (`WATCH` state) | "Logged but not alerted" — the persisted per-lap `WATCH` state IS the auditable record (no error code applies and F10 mints none — divergence #12); no alert object | via the returned per-lap series | Gauge amber; no banner |
| `stint_variation.json` missing / corrupt / hash or schema mismatch / empty thresholds / missing or wrong-width `feature_scaler` | Load-time validation | `MODEL_NOT_FOUND` | DQ-log then **raise** `DetectorLoadError` — refuse to serve | yes | 503 via F13 |
| Structural-knob mismatch (artefact config vs live: `rolling_window`, `baseline_laps`, `baseline_window_extended`, `slope_weight`, `min_stint_laps`) | Load-time comparison (§4.3) | `MODEL_NOT_FOUND` | **Raise** — μ/σ stale, recalibrate | yes | 503 via F13 ("recalibrate") |
| Crossing-knob mismatch (`alert_threshold_sd`, `alert_confirm_laps`) | Load-time comparison (§4.3) | — (`pit_window_valid=False`) | Serve; `optimal_window_remaining_laps=None` on every alert until recalibration | yes (one row at load) | "Pit-window estimate unavailable (threshold re-tuned since calibration)." |
| Detector not loaded at health check | `health()` | `HEALTH_FAIL` (reason `MODEL_NOT_FOUND`) | `ready=False` result | yes | 503 from `GET /health` |
| Held-out/non-train row survives calibration quarantine | `_assert_no_leakage` tripwire | `TRAIN_LEAKAGE` | **Raise**; no artefact written | yes | build log (CI-breaking) |
| Calibration has no usable stints / missing columns / write fault | Guards in `calibrate_stint_variation` | `MODEL_TRAINING_FAILED` | **Raise**; no artefact written | yes | build log |
| MLflow logging fault during calibration | `except` around the MLflow calls | `MODEL_TRAINING_FAILED` (DQ row only, F08 mechanism) | `status=DEGRADED`, `mlflow_run_id=None`; artefact write unaffected | yes (`fallback_used=True`) | build log |
| DQ log sink unwritable | `DataQualityLogError` inside `_DQLedger.record` | `DATA_QUALITY_LOG_UNWRITABLE` (flag) | Never propagates; result flagged | n/a | — |

F08 fault-injection flags carried inside the embedded `ProjectionResult`
(`SPARSE_PROJECTION`, `OUT_OF_DISTRIBUTION`, `UMAP_FALLBACK`, `PROJECTION_FALLBACK`) are
**not** re-raised to F10's level: a fingerprint problem never blocks deviation scoring
(`fingerprint_available=False` at worst).

## 7. Acceptance criteria (each item mechanically testable by /spec-check)

- [ ] Baseline fingerprint = per-feature mean of the **first 3 clean laps** of the stint
      (`monitoring.baseline_laps`); safety-car/unclean laps excluded (each excluded lap →
      `LAP_001`-flagged `SKIPPED` row + DQ row); window extends to the **first 5 laps**
      (`baseline_window_extended`) when < 3 eligible; < 3 after extension →
      `BASELINE_INSUFFICIENT` (`status=FAILED`).
- [ ] Per-lap deviation = **cosine distance** between the lap's F04-normalised **then
      train-only-standardised** vector and the baseline, over the 26 features excluding
      `validate.degradation_features`.
- [ ] The feature standardiser is fitted **only** inside `calibrate_stint_variation`,
      after the quarantine + tripwire; its persisted per-feature `mean` equals the
      hand-recomputed mean of the quarantined, F04-normalised training rows (behavioural
      proof, F08 precedent); `std` floored at `normalise.min_std`; serving applies it
      read-only.
- [ ] Every input lap yields exactly one `LapDeviation` row with a `state` from the §3
      enum — including baseline laps (`BASELINE`), ineligible laps (`SKIPPED`), and an
      eligible-but-unselected candidate-window lap, which is **scored** (the 1,2,4
      baseline geometry test asserts lap 5 is scored and lap 3 is a flagged SKIPPED row).
- [ ] Rolling mean over exactly **`rolling_window` = 3** valid distances; laps before the
      window fills are `PENDING` with `deviation_score=None`; an activated stint ending
      with zero composite scores returns `status=DEGRADED` + `monitor_state=PENDING` +
      a DQ row (never a silent NOMINAL/OK).
- [ ] Threshold: **μ and σ are read from `artefacts/stint_variation.json`** per
      `(driver, circuit_type)` (never recomputed at serve time); `τ = μ +
      alert_threshold_sd (2.0) · σ` derives the multiplier from live config; the
      variation stats are computed from training data only.
- [ ] Activation floor: stints with < **8** rows (`features.min_stint_laps`) return
      `MONITOR_INACTIVE` (operative `error_code`, `status=DEGRADED`) with no scores and
      (when a style handle is supplied) a fingerprint snapshot still attempted.
- [ ] Confirmation: alert fires only after **2 consecutive** scored laps above τ
      (`alert_confirm_laps`); a single crossing yields the `WATCH` state (the persisted
      "logged but not alerted" record — divergence #12) and **no** alert object; a
      sub-threshold scored lap resets the run; a `SKIPPED` lap freezes (neither extends nor
      resets) both the WATCH run and an in-progress consecutive-NaN run.
- [ ] `DegradationAlert` carries the §5.3 payload (`laps_since_threshold_crossed`,
      `feature_attribution` over all 28 keys, `top_features` of size `attribution_top_k`,
      `optimal_window_remaining_laps`, `alert_confidence ∈ {LOW, MEDIUM, HIGH}`), and the
      §8.4 persistence mapping is exactly: `alert_lap→lap_number`; `deviation_score`,
      `alert_confidence`, `feature_attribution` direct; `first_crossing_lap` exposed as
      the `lead_time_actual` seam; remaining fields payload-only.
- [ ] `sum(feature_attribution.values())` equals `deviation_score` within 1e-9 on every
      alert (exact decomposition, computed in the standardised space).
- [ ] F27/F28 enter only via the live-window slope component — the §4.2-8
      frame→`StintLap` adapter feeds RAW `F01`/`F06` + integer-coerced `tyre_age` to
      F03b's `compute_degradation_slopes` (never reimplemented) — weighted by
      `slope_weight`; the matrix's broadcast F27/F28 columns are never read by the
      detector (test: mutating them changes nothing); calibration and serving produce
      identical slopes for the same stint (parity test); F03b is never invoked on a
      prefix it would NULL (assert zero `FEATURE_001` rows from a no-tyre-age stint —
      exactly one F10 DQ row instead).
- [ ] Threshold fallback chain `(driver, circuit_type)` [gated on
      `min_calibration_stints`] → driver-global → global, each hop flagging
      `FALLBACK_THRESHOLD` with `threshold_source` set and `status=DEGRADED`; slope-stats
      miss at every tier → cosine-only scoring with `slope_component_active=False` + one
      DQ row.
- [ ] All-interpolated baseline → `DEGRADED_QUALITY` flag, `status=DEGRADED`, and alert
      confidence capped at `LOW` even when `s_k ≥ μ + confidence_high_sd·σ`.
- [ ] NaN distance → lap skipped + `DISTANCE_NAN` flag; exactly **3 consecutive** NaNs
      (`max_consecutive_nan`) → `MONITOR_ERROR`, `HALTED`, honest partial `laps` series;
      2 consecutive NaNs do **not** halt; a 3-lap safety-car train mid-stint does **not**
      halt (SKIPPED ≠ NaN).
- [ ] Every §6 code path has a unit test that triggers it and asserts the typed result
      and code placement (the six §5.3 codes + `LAP_001`-as-flag, `INVALID_INPUT` (both
      `monitor_stint` and `lead_time_report`), `MODEL_NOT_FOUND`, `HEALTH_FAIL`,
      `NORMALISATION_FALLBACK`, `TRAIN_LEAKAGE`, `MODEL_TRAINING_FAILED`,
      `DATA_QUALITY_LOG_UNWRITABLE`).
- [ ] `monitor_stint` never raises: fault-injection (internals monkeypatched to throw)
      yields a `MONITOR_ERROR` result, not an exception.
- [ ] No `.fit()` touches held-out data: calibration quarantines held-out + non-train rows
      **before any reduction**; a poisoned matrix raises `TRAIN_LEAKAGE` and writes **no
      artefact**; a static scan asserts no `.fit(`/`fit_normaliser`/`groupby`-refit symbol
      in the serving path (`load_detector`/`monitor_stint`/`health`/`lead_time_report`).
- [ ] Calibration is deterministic: two runs on the same matrix produce **byte-identical**
      `stint_variation.json` (no timestamp in the artefact; content hash equal).
- [ ] Calibration is MLflow-logged (params + metrics + artefact) to experiment
      `apex-degradation` via `monitoring.mlflow_tracking_uri`; an MLflow fault →
      `status=DEGRADED` + DQ row (`MODEL_TRAINING_FAILED`, F08 mechanism),
      `mlflow_run_id=None`, artefact still written.
- [ ] `load_detector` raises `DetectorLoadError(MODEL_NOT_FOUND)` after a DQ row on:
      missing/corrupt/hash-mismatched/schema-mismatched artefact, empty threshold table,
      missing or wrong-width `feature_scaler`, or a structural-knob mismatch; a
      crossing-knob mismatch loads with `pit_window_valid=False` and alerts carrying
      `optimal_window_remaining_laps=None`; `health()` on a missing handle reports
      `HEALTH_FAIL` with reason `MODEL_NOT_FOUND`.
- [ ] All thresholds and filenames read from `config.yaml` (`monitoring:` + referenced
      keys); no magic numbers in `app/monitoring/detector.py`; `MonitoringConfig` is
      `extra="forbid"` with the §5 validators (each violation → `CONFIG_001`).
- [ ] `lead_time_report` computes: mean lead time, per-stint TP/FP rates, FN stints
      characterised (`n_laps`, `compound`, `safety_car_affected`),
      `early_alerts_by_driver_circuit`, compound stratification
      (Soft/Medium/Hard/Intermediate; `"UNKNOWN"` excluded from the stratified view
      only), the sensitivity sweep at exactly `sensitivity_grid_sd = [1.0, 1.5, 2.0,
      2.5]` stratified per circuit type, and `optimal_threshold_per_circuit_type` —
      read-only (no `.fit`), never raises, typed `INVALID_INPUT` on malformed input.
- [ ] The Week-11 deliverable exists: the calibration flow runs `lead_time_report` over
      2022 `split=="val"` stints (training circuits only) and logs the report to the
      MLflow calibration run.
- [ ] Every failure path and fallback writes a `data_quality_log` row via the `_DQLedger`
      pattern (`DataQualityLogError` never propagates; result flagged
      `DATA_QUALITY_LOG_UNWRITABLE`).
- [ ] INDEX row updated to DONE only when all of the above pass.

## 8. Test plan

- **Fixtures** (`tests/monitoring/test_detector.py`, mirroring `tests/ml/test_stylespace.py`
  conventions): a synthetic stint generator — a stable base vector + per-lap Gaussian
  jitter, with (i) a controllable drift ramp from lap `j` (degradation signal), (ii)
  injectable safety-car laps (`track_status` token ≠ green), (iii) injectable
  `has_interpolation` flags, (iv) NaN-poisoned feature rows, (v) controllable `tyre_age`
  (known / UNKNOWN) so the slope component activates or not. A small synthetic training
  matrix (2 drivers × 2 circuit types × ≥ `min_calibration_stints` stints) feeds
  `calibrate_stint_variation` to produce a real artefact fixture; a hand-built
  `stint_variation.json` with known `{μ, σ}` and scaler stats gives exact-value threshold
  assertions. An evaluation-matrix fixture adds `compound` (+ one `"UNKNOWN"` stint) and
  `pit_lap`.
- **Happy path**: drift stint → deviation rises, `WATCH` on first crossing, `ALERT` on the
  2nd consecutive crossing; assert `alert_lap`, `first_crossing_lap`,
  `laps_since_threshold_crossed`, band assignment at `μ+2.2σ → LOW`, `μ+2.7σ → MEDIUM`,
  `μ+3.1σ → HIGH`; attribution sums to `deviation_score`; the drifted features top the
  attribution ranking; every input lap has a `LapDeviation` row with the right state
  progression (BASELINE → PENDING → NOMINAL → WATCH → ALERT).
- **Failure fixtures** (one per §6 row): 7-lap stint → `MONITOR_INACTIVE`; SC lap in first
  3 + recovery via extension (baseline = laps 1,2,4, `baseline_extended=True`, lap 3 =
  flagged SKIPPED, lap 5 **scored**); SC laps everywhere in first 5 →
  `BASELINE_INSUFFICIENT`; mid-stint 3-lap SC train → SKIPPED rows, WATCH run frozen not
  reset, **no halt**; unknown `circuit_type=None` → driver-global + `FALLBACK_THRESHOLD`;
  unknown driver → global tier; thin cell (`n_stints < min_calibration_stints`) →
  fallback; all-interpolated baseline → `DEGRADED_QUALITY` + HIGH-magnitude deviation
  still reported `LOW`; NaN×2 → skip only; NaN×3 → `MONITOR_ERROR` halt with partial
  series; alternating NaN/valid so no composite ever forms → `DEGRADED` + `PENDING` (the
  zero-composite outcome); no-tyre-age stint → cosine-only, `slope_component_active=False`,
  exactly one F10 DQ row and zero `FEATURE_001` rows; unknown circuit →
  `NORMALISATION_FALLBACK` flag + `status=DEGRADED`; unwritable `log_path` → result
  flagged `DATA_QUALITY_LOG_UNWRITABLE`, no raise; missing column / empty frame /
  shuffled or duplicated `lap_number` → `INVALID_INPUT`; monkeypatched
  `apply_normalisation` raising → backstop `MONITOR_ERROR`, no exception;
  corrupt/absent/hash-tampered/scaler-less artefact → `DetectorLoadError(MODEL_NOT_FOUND)`;
  artefact with structural-knob mismatch (`rolling_window` changed) → load refuses;
  crossing-knob mismatch (`alert_threshold_sd` changed) → loads, `pit_window_valid=False`,
  alert's `optimal_window_remaining_laps is None`; `health(None)` → `HEALTH_FAIL`;
  malformed evaluation matrix → typed `INVALID_INPUT` `LeadTimeReport`.
- **Leakage & reproducibility**: matrix containing a Belgium row → `TRAIN_LEAKAGE` raise +
  no file on disk; empty post-quarantine matrix → `MODEL_TRAINING_FAILED`; double-run
  byte-identity; static no-`.fit` scan of the serving path; standardiser behavioural proof
  (persisted mean == hand-recomputed quarantined training mean); mixed-scale defence —
  multiply one raw-unit feature column by 10³ in a synthetic training matrix and assert
  the calibrated detector's relative attributions are scale-invariant; calibration/serving
  slope parity on the same stint; mutation test — serving a stint twice with different
  broadcast F27/F28 column values yields identical deviations.
- **Property/edge**: `s_k == τ` exactly → **no** crossing (strict `>`); crossing at the
  final stint lap (confirmation pending → `WATCH`, no alert); alternating
  cross/under/cross → never confirms; the distance sequence **NaN, SKIP, NaN, NaN halts**
  (`MONITOR_ERROR` — the NaN run survives the freeze and reaches 3) while
  **NaN, SKIP, NaN, valid does not halt**; stint of exactly 8 laps activates; laps 1–2 of the
  scored region are `PENDING` (`deviation_score=None`); `slope_component_active=False`
  under UNKNOWN tyre age with deviation still scored; config validators reject
  `confidence_medium_sd ≤ alert_threshold_sd`, `baseline_window_extended < baseline_laps`,
  empty grid (→ `CONFIG_001`).
- **F08 embed**: with a real style handle on a 10-lap prefix → `fingerprint_available=True`
  and the embedded `ProjectionResult` carries `SPARSE_PROJECTION` (sparse-by-design
  assertion, not a clean OK); with a stubbed `project_unknown` returning a DEGRADED
  `coords_5d=None` result (the raw-NN fallback) → `fingerprint_available=False`, snapshot
  still embedded; with `project_unknown` monkeypatched to throw (defence in depth) →
  scoring unaffected, `fingerprint_available=False`; **seam parity** — the embedded
  `fingerprint_snapshot` equals a direct `project_unknown` call on the same RAW stint
  prefix (same coords, same flags), pinning that `monitor_stint` hands F08 raw rows
  (F08 normalises internally) so an accidental hand-off of the standardised frame fails
  loudly here instead of silently double-normalising at the seam.
- **Evaluation**: synthetic stints with known pit laps and injected drift at known lead
  times → `lead_time_report` reproduces the expected mean lead, TP/FP, per-compound split
  (with the `"UNKNOWN"` stint excluded from strata but present in totals), per-circuit-type
  sensitivity rows, and the arg-best `optimal_threshold_per_circuit_type`.

## 9. Definition of done

All §7 criteria MET with passing tests, `/apex-review` clean on the contract checklist,
`ml-auditor` + `failsafe-auditor` run on `app/monitoring/`, mypy clean, INDEX row DONE.

**Limitations to record in the dissertation (and divergences to flag to the supervisor,
continuing the running list — F08 was #6):**

1. **Divergence #7 — train-only feature standardiser (master gap-fill, the F08-#6
   twin).** §5.3 says "normalised feature vector" but F04 normalises only 6 of 28
   features; a mixed-scale cosine would measure unit scale, not behaviour (§4.0-e). F10
   adds a per-feature standardiser fitted on quarantined training rows only, persisted in
   the content-hashed artefact, applied read-only at serve. Fills a gap the master never
   reasoned about; same rationale the supervisor already saw as divergence #6.
2. **Divergence #8 — F27/F28 weighted-component definition.** §5.3 mandates the slopes be
   "incorporated into the deviation score as a weighted component" but is silent on the
   formula and weight. F10 defines `s_k = r_k + slope_weight · mean(|z(f27)|, |z(f28)|)`
   (z vs train slope stats) with `slope_weight: 0.25`, and excludes F27/F28 from the
   cosine core to avoid double-counting and replay look-ahead (§4.0-b).
3. **Divergence #9 — alert-confidence bands.** §5.3 says LOW/MEDIUM/HIGH "based on how far
   above threshold" without boundaries. F10 pins them at `μ+2.5σ` / `μ+3.0σ`
   (config-driven), mirroring the F06b `confidence_high` divergence pattern.
4. **Divergence #10 — pit-window recommendation method.** §5.3 asks for a "pit window
   recommendation based on historical deg patterns"; §12.1 fixes only the field
   (`optimal_window_remaining_laps: int`). F10 uses the calibrated median
   (first-crossing → stint-end) gap per driver/circuit-type, with the stint's final lap
   as the pit-in proxy, suppressed (`None`) whenever the crossing knobs were re-tuned
   after calibration (§4.3). Coarse but honest; refinement is an F20/F11-era option.
5. **Divergence #11 — "SHAP decomposition of deviation".** The detector is not a tree
   model, so TreeExplainer does not apply; F10 ships an exact additive decomposition of
   the cosine distance (`1 − cosθ = ½‖x̂−b̂‖²`) that sums to the reported score. Same
   intent ("which features are driving it"), different mechanism than F07's SHAP.
6. **Divergence #12 — calibration/serving hygiene.** `min_calibration_stints: 3` (a σ
   from one stint is not a distribution), the global-all fallback tier for unknown
   drivers, the mid-stint SKIPPED-lap policy (§4.2-6/7), and the load-time
   config-mismatch policy (§4.3) are master-silent gap-fills, each surfaced via
   flags/DQ rows, never silently. Two code-placement choices fall out of the "no new
   error code" constitution rule: (a) the slope-inactive DQ row reuses **`LAP_PARTIAL`**
   (F02's "tyre age UNKNOWN → F27/F28 NULL"), the semantically exact fit that also keeps
   F03b's `FEATURE_001` count at zero for a no-tyre-age stint; (b) the single-lap `WATCH`
   crossing is recorded as the **persisted per-lap state** in the returned series (the
   auditable "logged but not alerted" record) rather than a separately error-coded
   `data_quality_log` row, because no existing code names a "watch" event and F10 mints
   none.
7. **Limitation — replay validation only** (§15.2 verbatim): "The degradation detector is
   validated on historical replay, not live race conditions. Live performance may differ
   due to session-specific noise not present in clean historical data."
8. **Limitation — pit-lap ground truth is strategic, not physical.** Pit stops happen for
   reasons other than degradation (undercuts, safety cars, weather); §11.3's pit-lap
   ground truth therefore bounds achievable TP/FP. Reported honestly per §11.3's failure-
   analysis columns.

## 10. Risks & mitigations (from master risk register)

- **§14 row (verbatim):** "Degradation detector mean lead time below 2 laps — Likelihood
  Medium, Impact High — Contingency: First: lower confirmation requirement from 2
  consecutive laps to 1 lap. Second: reduce threshold to 1.5SD. Third: add F27 and F28 to
  deviation score earlier (reduce minimum stint from 8 to 6 laps). Report threshold
  sensitivity analysis regardless. — Week 12."
  **How each contingency maps here:** (1) `alert_confirm_laps: 1` and (2)
  `alert_threshold_sd: 1.5` are crossing knobs — genuine one-line config turns; serving
  continues, with only the pit-window estimate suppressed until recalibration (§4.3).
  (3) `features.min_stint_laps: 6` is **not** a free turn: that key is F03b's
  pipeline-wide slope floor, so exercising it means regenerating the feature matrix
  (F27/F28 now populate for 6–7-lap stints, shifting imputation), recalibrating
  `stint_variation.json` (structural knob — load refuses otherwise, by design), and
  accepting that F06/F08 artefacts were trained on the 8-floor matrix (staleness to note
  honestly if invoked). The blast radius is disclosed here so the Week-12 decision is
  informed, not improvised.
- **Hard compound may carry no reliable signal** (§11.3: "Hard compound may show no
  reliable signal. Report this honestly.") → compound stratification is a first-class
  report axis, not an afterthought.
- **Sparse calibration cells** (few training stints for a driver/circuit-type) → the
  fallback-tier chain + `FALLBACK_THRESHOLD` UI surfacing; coverage is an MLflow metric
  so thin calibration is visible at build time, not discovered at serve time.
- **False-positive erosion of trust** (§11.3: "Early alerts reduce engineer trust") →
  the 2-consecutive-lap confirmation is the primary guard; the sensitivity sweep
  quantifies the trade-off before anyone re-tunes `alert_threshold_sd`.
