# Feature F03a — Scalar-Channel Features + Vector Assembly (F01–F18, F22–F24, F26)

> One spec, one session. Self-contained: build this from this file + CLAUDE.md only.

**Status:** SPEC WRITTEN
**Master spec sections:** §4 (Feature Engineering), §4.1 (Feature Computation Contract),
§4.2 (Full Feature Matrix — rows F01–F18, F22–F24, F26), §4.5 (Feature Null Tolerance
Policy), §8.4 (`lap_features` table), Project Plan Wk 3
**Layer / Week:** Data pipeline / Wk 3

## 1. Purpose
Compute the 22 **scalar-channel** lap features — F01–F18 (throttle, braking, speed, gear,
engine) and F22–F24, F26 (throttle/brake/speed interactions and lap-phase) — directly from
one clean, validated lap's telemetry, **and** own the shared feature-matrix infrastructure
every feature plugs into: the 28-key vector contract, the `FeatureComputationError`
type + catch-mark-NULL-log pipeline behaviour, and the 6-NULL tolerance policy (≤6 NULLs →
accept with imputation + `IMPUTED` tag; ≥7 → reject `LAP_NULL_EXCESS`). F03b later fills the
X/Y-derived features (F19–F21, F25) and tyre-age slopes (F27–F28) into the **same** vector
without changing this contract.

## 2. Upstream / downstream contracts
- **Consumes:**
  - F02 → `ValidationResult.clean_laps` — the list of `LapValidationResult` with
    `status=ACCEPTED`. For each, F03a reads `lap_number`, carried `flags`
    (`LAP_INTERP`, `LAP_PARTIAL`), and `null_features` (e.g. `["F27","F28"]` from
    `LAP_PARTIAL`) — those features are forced NULL in assembly, not recomputed.
  - The validated **wide** telemetry frame F02 operated on (post-interpolation; one row per
    sample; columns `time, Speed, Throttle, Brake, RPM, Gear`), sliced to each accepted lap.
  - F01 per-lap session context: **session-maximum RPM** (F17) and **lap distance, m** (F15);
    either absent → the specific NULL trigger fires.
  - A train-only **`FeatureMedians`** lookup (driver × circuit-type → per-feature median) for
    §4.5 imputation; injected, never computed on held-out data (non-negotiable #2).
  - `config.yaml` → new `features:` block + `thresholds.null_ceiling` (6, referenced).
- **Produces:**
  - `ScalarFeatureResult` per lap — the 22 scalar values (`float | None`) + per-feature NULL
    diagnostics (the raw compute output before assembly).
  - `FeatureResult` per lap — the assembled 28-key vector, `null_count`, `imputed_features`,
    `status`, and `error_code` (`LAP_NULL_EXCESS` when rejected). Maps 1:1 to the
    `lap_features` row (§8.4: `feature_vector` JSONB, `null_count`, `imputed_features` TEXT[],
    `corner_type` — left `None` by F03a, set by F03b/F09).
  - One `data_quality_log` row per NULL'd feature, per capped feature, per imputed feature,
    and per rejected lap. **No artefact files. No normalisation** (raw features only — F04 owns
    Z-scoring of F09/F11/F13/F14).
- **Consumed by:**
  - F04 (circuit-aware normalisation) — consumes the assembled matrix.
  - F03b (sibling) — contributes F19–F21, F25, F27, F28 values into F03a's
    `assemble_feature_vector`; reuses `FeatureComputationError` + the NULL/imputation gate.
  - F09 (corner decomposition) re-uses these feature functions per corner sub-vector.

## 3. Inputs & outputs (typed)
- **Input** (per accepted lap):
  `compute_scalar_features(lap: DataFrame, *, lap_number: int, session_max_rpm: float | None,
  lap_distance_m: float | None, config: AppConfig, lap_time_s: float | None = None,
  log_path: Path | None) -> ScalarFeatureResult`, where `lap` is the validated wide frame for
  one lap. **Time base:** F01 persists no per-sample clock (see F02 LAP_004), so the sample
  period for the rate/duration features (F03, F07, F08, F16, F24) is taken from a real
  per-sample time column if present, else derived from `lap_time_s / n_samples` (or the
  frame's `lap_time` column) — the same proxy F02 uses. If neither is available, those
  features return NULL (`time base unavailable`) rather than fabricate a clock.
- **Assembly** (per lap, the shared contract F03b also calls):
  `assemble_feature_vector(values: dict[str, float | None], *, forced_null: list[str],
  flags: list[ErrorCode], medians: FeatureMedians, config, log_path) -> FeatureResult` —
  `values` is the union of every contributor's output keyed by the 28 canonical IDs (missing
  key ⇒ NULL); `forced_null` carries F02's `null_features` (e.g. F27/F28 under `LAP_PARTIAL`).
- **Outputs:**
  - `ScalarFeatureResult(BaseResult) { lap_number:int, features: dict[str,float|None]  # 22
    keys F01–F18,F22,F23,F24,F26, null_features:list[str], capped_features:list[str],
    detail:dict[str,dict], status: OK|DEGRADED, flags:list[ErrorCode] }`.
  - `FeatureResult(BaseResult) { lap_number:int, feature_vector: dict[str,float|None]  # 28
    keys, null_count:int, imputed_features:list[str], corner_type: str|None,
    status: OK|DEGRADED|FAILED, error_code: None|LAP_NULL_EXCESS, flags:list[ErrorCode] }`.
    `null_count` is the count of NULLs **before** imputation; on accept, `feature_vector` has
    those NULLs imputed and `imputed_features` lists them; on reject (≥7 NULL),
    `status=FAILED`, `error_code=LAP_NULL_EXCESS`, `feature_vector` left un-imputed.
- **Exception:** `FeatureComputationError(ErrorCode.FEATURE_001)` — raised by an individual
  feature function when its value is out of the §4.2 range. The driver **catches it per
  feature**, sets that feature NULL, logs `FEATURE_001`, continues. Neither
  `compute_scalar_features` nor `assemble_feature_vector` raises to its caller — every failure
  is caught, coded, and returned in the result object (§4.1 / non-negotiable #1).

## 4. Method / algorithm
All ops vectorised over numpy/pandas — **no per-row Python loops**. Per feature: compute the
raw value, then assert it lies within `features.valid_ranges[Fxx]`; outside → raise
`FeatureComputationError` → caught → NULL + `FEATURE_001`. A documented NULL trigger
(precondition unmet) takes the same NULL + `FEATURE_001` path, `detail[Fxx].reason`
distinguishing `OUT_OF_RANGE` vs `NULL_TRIGGER`. **No `.fit()`/scaler anywhere.**

**Scalar-derived event detection (NOT X/Y):** *braking event* = a contiguous run with
`Brake > features.braking_event_threshold_bar`; *corner* = a local `Speed` minimum with
prominence ≥ `features.corner_min_prominence_kmh`. (F03b's corner-*type* decomposition uses
X/Y curvature — unrelated.) Rate features ("per 50ms") are computed from actual sample `time`
deltas and rescaled to `features.sample_period_ms`, so they are correct at any in-spec rate.

| ID | Name | Domain | Method (summary) | Valid range | NULL trigger |
|----|------|--------|------------------|-------------|--------------|
| F01 | throttle_mean | Throttle | mean(Throttle) | 0–100 | all samples NaN |
| F02 | throttle_std | Throttle | std(Throttle) | 0–50 | std == 0 (constant → sensor fault) |
| F03 | throttle_open_rate | Throttle | mean +ve rate while throttle rising, %/50ms | 0–200 | no throttle-increasing events |
| F04 | throttle_full_fraction | Throttle | frac. samples ≥ `throttle_full_pct` | 0–1 | none |
| F05 | throttle_lift_fraction | Throttle | frac. samples < `throttle_lift_pct` | 0–1 | none |
| F06 | brake_peak_mean | Braking | mean of top-`brake_peak_top_n` brake peaks | 0–200 bar | < `min_braking_events` braking events |
| F07 | brake_duration_mean | Braking | mean braking-event duration (ms) | 50–4000 ms | no braking events |
| F08 | brake_release_rate | Braking | mean brake-pressure decrease on release, bar/50ms | 0–100 | no brake-release events |
| F09 | brake_entry_speed_mean | Braking | mean Speed at first brake application per corner | 0–380 km/h | corner detection failed |
| F10 | brake_overlap_fraction | Braking | frac. samples Throttle>`overlap_throttle_pct` AND Brake>`overlap_brake_pct` | 0–`overlap_cap` (0.25) | none — value > cap → **cap + log**, not NULL |
| F11 | speed_min_mean | Speed | mean of per-corner minimum speeds | 0–200 km/h | corner detection failed |
| F12 | speed_variance | Speed | var(Speed) over lap | 0–8000 | all speed samples identical (sensor fault) |
| F13 | speed_p10 | Speed | `speed_p_low`-th percentile of Speed | 0–150 km/h | none |
| F14 | speed_p90 | Speed | `speed_p_high`-th percentile of Speed | 100–380 km/h | none |
| F15 | gear_change_rate | Gear | gear changes per km = changes / (lap_distance/1000) | 0–15 | lap distance cannot be computed |
| F16 | gear_hold_duration_mean | Gear | mean time held in each gear (ms) | 100–10000 ms | < `min_gear_changes` gear changes |
| F17 | rpm_redline_fraction | Engine | frac. samples RPM > `redline_pct`% of `session_max_rpm` | 0–0.6 | session_max_rpm absent / not deterministic |
| F18 | rpm_std | Engine | std(RPM) | 0–5000 | constant RPM (sensor fault) |
| F22 | throttle_speed_corr | Interaction | Pearson(Throttle, Speed) | -1 to 1 | none |
| F23 | brake_speed_corr | Interaction | Pearson(Brake, Speed delta / deceleration) | -1 to 1 | no braking events |
| F24 | throttle_brake_transition | Interaction | mean time (ms) full-brake-release → first throttle | 0–3000 ms | < `min_transitions` transitions |
| F26 | lap_phase_throttle_delta | Phase | mean throttle 1st 50% of lap − 2nd 50% | -30 to 30 | none |

**Excluded from F03a (→ F03b):** F19, F20, F21 (lateral-G / steering from X/Y), **F25**
(corner_entry_balance — needs corner segmentation), F27, F28 (tyre-age slopes).

**F10 capping (the one non-NULL fallback):** raw overlap > `features.overlap_cap` (0.25)
signals a throttle/brake sensor fault → **clamp** to `overlap_cap`, add `F10` to
`capped_features`, `status=DEGRADED`, log `FEATURE_001` `fallback_used=True
reason="OVERLAP_CAPPED"` (§4.2 row F10).

**Vector assembly + null tolerance (§4.5) — owned here:**
1. Merge contributors into the 28-key `feature_vector`; apply `forced_null` (F02's
   `null_features`). Count NULLs → `null_count`.
2. **`null_count > thresholds.null_ceiling` (>6) → reject:** `status=FAILED`,
   `error_code=LAP_NULL_EXCESS`, log one row, **no imputation** (§4.5: "more than 6 NULL …
   rejected entirely"). Exactly 6 is accepted (boundary).
3. **`null_count ≤ 6` → impute each NULL** from `medians`: driver × circuit-type median if
   present, else global training median; tag each in `imputed_features`, add `IMPUTED` flag,
   `status=DEGRADED`, log one row per imputed feature. Imputation medians are **train-only**.
   If a NULL has **no** train median at all (global table incomplete), the vector cannot be
   made whole → the lap is `status=FAILED`, `error_code=FEATURE_001` (not returned with a
   surviving NULL that would poison F04 / the classifier).
4. Clean lap (0 NULL): `status=OK`, `error_code=None`, empty `imputed_features`.

`overlap_cap` (F10) is tied at config-load to the upper bound of `valid_ranges['F10']`, and the
capped value is range-gated through the same `_checked` path as every feature, so a
misconfigured cap can never silently emit an out-of-range value (§4.2 F10 / non-negotiable #1).

## 5. config.yaml keys this feature adds
`thresholds.null_ceiling` (6) is **referenced** from `Thresholds`, not redefined. F03a adds
the `features:` block + a new typed `FeaturesConfig` sub-model in `app/contracts/config.py`
wired onto `AppConfig` (standing rule: typed sub-model, `extra="forbid"`, no untyped
passthrough; `valid_ranges` validates as `dict[str, tuple[float, float]]` over exactly the 22
scalar IDs). A new `IMPUTED` member is added to `ErrorCode`.
```yaml
# --- F03a scalar-channel feature extraction + assembly (§4.1, §4.2, §4.5) ---
features:
  sample_period_ms: 50            # §4.2 — canonical step for "% per 50ms" rate features (F03, F08)
  throttle_full_pct: 100          # §4.2 F04 — sample counts as full throttle at >= this %
  throttle_lift_pct: 5            # §4.2 F05 — sample counts as a lift below this %
  brake_peak_top_n: 10            # §4.2 F06 — mean of top-N brake-pressure peaks
  min_braking_events: 5           # §4.2 F06/F09 — fewer braking events -> NULL
  braking_event_threshold_bar: 5  # design decision (master leaves "braking event" open) — Brake > this delimits an event
  corner_min_prominence_kmh: 10   # design decision — speed-minimum prominence delimiting a corner (F09/F11)
  overlap_throttle_pct: 5         # §4.2 F10 — throttle > this AND ...
  overlap_brake_pct: 5            # §4.2 F10 — ... brake > this => overlap sample
  overlap_cap: 0.25               # §4.2 F10 — value above -> cap to this and log (sensor fault), not NULL
  speed_p_low: 10                 # §4.2 F13 — low percentile
  speed_p_high: 90                # §4.2 F14 — high percentile
  redline_pct: 97                 # §4.2 F17 — fraction of lap above this % of session-max RPM
  min_gear_changes: 10            # §4.2 F16 — fewer gear changes -> NULL
  min_transitions: 5              # §4.2 F24 — fewer brake->throttle transitions -> NULL
  valid_ranges:                   # §4.2 — value outside [min,max] -> FeatureComputationError -> NULL + FEATURE_001
    F01: [0, 100]
    F02: [0, 50]
    F03: [0, 200]
    F04: [0, 1]
    F05: [0, 1]
    F06: [0, 200]
    F07: [50, 4000]
    F08: [0, 100]
    F09: [0, 380]
    F10: [0, 0.25]
    F11: [0, 200]
    F12: [0, 8000]
    F13: [0, 150]
    F14: [100, 380]
    F15: [0, 15]
    F16: [100, 10000]
    F17: [0, 0.6]
    F18: [0, 5000]
    F22: [-1, 1]
    F23: [-1, 1]
    F24: [0, 3000]
    F26: [-30, 30]
```

## 6. Error codes & failure modes (the failsafe contract)
| Failure mode | Detection | Error code | System response | Logged? | User-facing message |
|--------------|-----------|-----------|-----------------|---------|---------------------|
| Feature value out of §4.2 range | computed value ∉ `valid_ranges[Fxx]` | `FEATURE_001` | feature func raises `FeatureComputationError`; driver catches → feature NULL, `status=DEGRADED` | yes (lap, feature, value, range, reason=`OUT_OF_RANGE`) | "Lap N feature Fxx out of range → NULL." |
| Feature precondition unmet (NULL trigger) | per-feature trigger in §4.2 | `FEATURE_001` | feature NULL, `status=DEGRADED` | yes (lap, feature, reason=`NULL_TRIGGER`) | "Lap N feature Fxx could not be computed → NULL." |
| Overlap > cap (sensor fault) | raw F10 > `overlap_cap` (0.25) | `FEATURE_001` | clamp F10 to cap (not NULL), `F10`→`capped_features`, `status=DEGRADED`, `fallback_used=True` | yes (lap, raw value, reason=`OVERLAP_CAPPED`) | "Lap N: throttle/brake overlap capped (sensor fault)." |
| Unexpected internal error in a feature func | any non-`FeatureComputationError` exception | `FEATURE_001` | caught per feature → NULL (never propagates), reason=`UNEXPECTED` | yes (lap, feature, exception repr) | "Lap N feature Fxx errored → NULL." |
| No time base (no per-sample clock and no `lap_time`) | `dt_ms` unresolvable | `FEATURE_001` | rate/duration features (F03, F07, F08, F16, F24) NULL, reason=`NULL_TRIGGER` | yes (one row per affected feature) | "Lap N feature Fxx could not be computed → NULL." |
| NULL feature has no train median (assembly) | `median_for` returns None for a NULL'd feature | `FEATURE_001` | vector incomplete → lap `status=FAILED`, not accepted | yes (lap, feature) | "Lap N rejected: feature Fxx could not be imputed." |
| ≤6 NULL features → imputed | `0 < null_count ≤ null_ceiling` | `IMPUTED` (flag) | impute each NULL from train-only median; tag; `status=DEGRADED` | yes (one row per imputed feature) | "Lap N: M features imputed." |
| >6 NULL features → reject vector | `null_count > null_ceiling` (>6) | `LAP_NULL_EXCESS` | `status=FAILED`, vector rejected, **no imputation** | yes (lap, null_count) | "Lap N rejected: too many missing features." |

`FEATURE_001`, `LAP_NULL_EXCESS` already exist in `ErrorCode`; **`IMPUTED` is added** by this
feature.

## 7. Acceptance criteria (each item mechanically testable by /spec-check)
- [ ] `ScalarFeatureResult.features` keys are exactly `{F01–F18, F22, F23, F24, F26}` (22);
      F19, F20, F21, F25, F27, F28 are **absent** from it.
- [ ] `FeatureResult.feature_vector` always has exactly the 28 canonical keys F01–F28 (the
      contract F03b plugs into); F03a-uncomputed IDs (F19–F21, F25, F27, F28) appear as keys
      with NULL/imputed values, never missing.
- [ ] Each of the 22 features computes its documented value on a happy-path synthetic lap
      (one asserted expected value per feature, within tolerance).
- [ ] Each documented NULL trigger has a fixture: the feature is `None`, in `null_features`,
      `status=DEGRADED`, with exactly one `FEATURE_001` row `reason=="NULL_TRIGGER"`.
- [ ] An out-of-range value (e.g. F14 = 90 < 100) makes the feature function **raise**
      `FeatureComputationError` (never returns the raw value); the driver catches it, NULLs
      the feature, logs `FEATURE_001` `reason=="OUT_OF_RANGE"`, and the other 21 still compute.
- [ ] F10 raw > 0.25 → `feature_vector["F10"]==0.25`, `"F10" in capped_features`,
      `status=DEGRADED`, one `FEATURE_001` `reason=="OVERLAP_CAPPED"` `fallback_used==True`;
      F10 not in `null_features`.
- [ ] **Null-ceiling boundary:** a vector with exactly 6 NULLs is ACCEPTED, those 6 imputed,
      `imputed_features` has 6 entries, `IMPUTED` flag set; a vector with 7 NULLs is
      `status=FAILED`, `error_code=LAP_NULL_EXCESS`, `imputed_features` empty, no imputation.
- [ ] Imputation uses the driver×circuit-type median when present and the global median
      otherwise (both branches asserted); each imputed feature writes one `data_quality_log`
      row; imputation reads only the injected train-only `medians` (no `.fit()` on input laps).
- [ ] Neither `compute_scalar_features` nor `assemble_feature_vector` raises: a degenerate lap
      (all-NaN channels) returns a coded result (22 NULL → assembly rejects `LAP_NULL_EXCESS`),
      not an exception.
- [ ] One feature failing does not block others (a single-NULL lap returns 21 in-range floats).
- [ ] Every NULL, capped, imputed feature and every rejected lap writes **exactly one**
      `data_quality_log` row (count asserted).
- [ ] No normalisation in F03a: `app/pipeline/features.py` has no `.fit(`, `StandardScaler`,
      or Z-score; outputs raw. (Leakage chokepoint is F04.)
- [ ] No per-row Python loop in `features.py` (vectorised pandas/numpy only).
- [ ] Every threshold, percentile, range bound, event constant, **and the null ceiling (6)**
      is read from `config.yaml`; `features.py` has no numeric literal for any of them.
- [ ] Rate features (F03, F08) give the same value for the same lap at 20 Hz vs 50 Hz, within
      tolerance (rescaling to `sample_period_ms` correct).
- [ ] Every error code (`FEATURE_001` ×{OUT_OF_RANGE, NULL_TRIGGER, OVERLAP_CAPPED,
      UNEXPECTED}, `IMPUTED`, `LAP_NULL_EXCESS`) has a unit test that triggers it.

## 8. Test plan
- **Happy-path fixtures:** a synthetic clean lap with hand-computable channels (trapezoidal
  throttle → known mean/std/full/lift; square braking events → known count/peak/duration;
  speed trace → known percentiles + corner minima; gear staircase over a known lap distance;
  RPM trace → known redline fraction). Assert all 22 features against pre-computed values, and
  a full 28-key `FeatureResult` with `status=OK` when F03b's six are supplied as valid floats.
- **Failure fixtures (one per NULL trigger + each fallback + each gate branch):** constant
  throttle (F02); no rising throttle (F03); <5 braking events (F06/F09); no braking events
  (F07/F23); no release events (F08); `session_max_rpm=None` (F17); `lap_distance_m=None`
  (F15); <10 gear changes (F16); <5 transitions (F24); 0 corners (F09/F11); identical speed
  (F12); constant RPM (F18); out-of-range F14 (OUT_OF_RANGE); raw F10=0.4 (OVERLAP_CAPPED); a
  feature func patched to raise `ValueError` (UNEXPECTED); a 6-NULL vector (IMPUTED accept); a
  7-NULL vector (LAP_NULL_EXCESS reject).
- **Property / edge (pinned both sides):** F04 full at exactly 100 (99.9 not); F05 lift
  strictly `<5` (5 not, 4.9 yes); F06 exactly 5 events (4 → NULL); F16 exactly 10 changes
  (9 → NULL); F24 exactly 5 transitions (4 → NULL); F10 raw==0.25 not capped, 0.2500001
  capped; F14==100 in range, 99.99 OUT_OF_RANGE; **null_count==6 accept+impute, ==7 reject**;
  all-NaN lap → all NULL, no raise.

## 9. Definition of done
All acceptance criteria MET with passing tests; `/apex-review F03a` clean (no silent failure,
no normalisation, config-driven thresholds incl. null ceiling, error-code discipline);
`/spec-check F03a` green; `mypy app` clean; no bare `except` in
`app/pipeline/features.py`. INDEX row `F03a` set to DONE. Limitations for the dissertation:
(1) "braking event" / "corner" thresholds (`braking_event_threshold_bar`,
`corner_min_prominence_kmh`) are design decisions the master leaves open — documented,
config-driven, boundary-tested — affecting F06–F09, F11, F23, F24; (2) imputation quality
depends on the train-only `FeatureMedians` table, which F03a consumes rather than builds.

## 10. Risks & mitigations (from master risk register where relevant)
- **Underspecified event thresholds (Wk-3 risk):** braking/corner constants live in
  `config.yaml`, documented as design decisions and boundary-tested, so recalibration is a
  one-line change.
- **Sampling-rate variation:** rate features computed from real `time` deltas, rescaled to
  `sample_period_ms` — verified by the resample-invariance test.
- **Silent feature corruption:** every NULL (out-of-range or precondition-unmet) is logged
  `FEATURE_001` with a reason; the F10 cap and every imputation are logged with
  `fallback_used`/`IMPUTED`; ≥7 NULL is an explicit `LAP_NULL_EXCESS` rejection — never a
  silent drop. This is the §4.1/§4.5 contract feeding the RQ5 evidence.
- **Normalisation leakage (master §4.3, non-negotiable #2):** mitigated by construction —
  F03a emits **raw** features, contains no `.fit()`, and imputes only from a train-only
  medians table; Z-scoring of F09/F11/F13/F14 is F04's train-only job, asserted absent here.
- **Imputation leakage:** §4.5 medians are computed on training laps only and injected; a
  test asserts F03a never derives medians from the lap(s) under evaluation.
