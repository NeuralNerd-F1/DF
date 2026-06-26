# Feature F03b — Geometry & Degradation Features (F19–F21, F25, F27, F28)

> One spec, one session. Self-contained: build this from this file + CLAUDE.md + the F03a
> spec (the vector contract this plugs into).

**Status:** SPEC WRITTEN
**Master spec sections:** §4.2 (Full Feature Matrix — rows F19–F21, F25, F27, F28), §4.3
(Circuit-Aware Normalisation — F19/F20 are F04's Z-score targets), §4.4 (Corner Type
Decomposition — `CORNER_001` on X/Y unavailable), §4.5 (Null Tolerance — owned by F03a),
Project Plan Wk 3
**Layer / Week:** Data pipeline / Wk 3

## 1. Purpose
Compute the six remaining lap-vector features F03a left to a sibling: the three X/Y-derived
**cornering** features F19 (lateral_g_mean), F20 (lateral_g_peak_mean), F21
(steering_correction_rate); the X/Y-derived **corner-balance** F25 (corner_entry_balance);
and the two **stint-level degradation** slopes F27 (tyre_age_throttle_slope) and F28
(tyre_age_brake_slope). F03b **plugs its six values into F03a's existing 28-key vector** via
`assemble_feature_vector` — it does not own assembly, `FeatureComputationError`, or the 6-NULL
tolerance gate (all F03a's). When the X/Y trace is unavailable the four cornering features are
NULL and the lap **continues** (§4.4: corner decomposition skipped, `CORNER_001` logged, the
other features still computed); rejection happens only if F03a's 6-NULL ceiling is crossed.

## 2. Upstream / downstream contracts
- **Consumes:**
  - F02 → `ValidationResult.clean_laps` (the `LapValidationResult` per accepted lap: its
    `flags`/`null_features`; F02 already forces F27/F28 NULL via `LAP_PARTIAL` when a lap's
    tyre age is UNKNOWN) + the validated wide telemetry frame, sliced per lap. F03b reads the
    **`X`, `Y`** position columns and `Speed`/`Brake` (cornering) and per-lap `tyre_age`,
    `stint` (degradation).
  - F03a → the integration point: `assemble_feature_vector` (unchanged), the
    `FeatureComputationError` type, the shared helpers (`_checked`, `_require`, `_runs`,
    `_DQLedger`, `_corner_indices`), and **F01/F06 per-lap values** (throttle_mean,
    brake_peak_mean) for every lap of a stint — F27/F28 regress these against tyre age.
  - `config.yaml` → the `features:` block (F03b adds its keys + the F19/F20/F21/F25 ranges).
- **Produces:**
  - `GeometryFeatureResult` per lap — `{F19, F20, F21, F25}` values (`float | None`) + NULL
    diagnostics. Merged into `assemble_feature_vector`'s `values`.
  - `DegradationResult` per **stint** — `{F27, F28}` slopes (`float | None`), broadcast to
    every lap of that stint. Merged into each lap's `values`.
  - One `data_quality_log` row per NULL'd feature (`FEATURE_001`) and per X/Y-unavailable lap
    (`CORNER_001`). **No artefact files. No normalisation** (raw features; F04 Z-scores F19/F20).
- **Consumed by:** F04 (normalisation — F19/F20 are its Z-score targets), F03a's assembly (the
  final 28-key vector), F09 (corner-type decomposition reuses the geometry primitives).

## 3. Inputs & outputs (typed)
- **Cornering** (per accepted lap):
  `compute_geometry_features(lap: DataFrame, *, lap_number: int, config: AppConfig,
  log_path: Path | None = None, entity_id: str | None = None) -> GeometryFeatureResult`.
  `lap` is the validated wide frame for one lap (cols `X, Y, Speed, Brake`).
  `GeometryFeatureResult(BaseResult) { lap_number:int, features: dict[str,float|None]  # exactly
  {F19,F20,F21,F25}, null_features:list[str], detail:dict[str,dict], status: OK|DEGRADED }`.
- **Degradation** (per stint):
  `compute_degradation_slopes(stint: list[StintLap], *, stint_id: int, config: AppConfig,
  log_path: Path | None = None) -> DegradationResult`, where
  `StintLap = { lap_number:int, tyre_age:int|None, throttle_mean:float|None  # F01,
  brake_peak_mean:float|None  # F06 }`.
  `DegradationResult(BaseResult) { stint_id:int, f27: float|None, f28: float|None,
  applies_to: list[int]  # lap numbers, null_reason: str|None, status: OK|DEGRADED }`.
  The caller writes `values["F27"]=f27`, `values["F28"]=f28` for every `lap_number` in
  `applies_to` before calling `assemble_feature_vector`.
- **Exception:** reuses F03a's `FeatureComputationError(ErrorCode.FEATURE_001)` — a feature
  out of its §4.2 range raises it (**no clamping**); the driver catches it, NULLs the feature,
  logs `FEATURE_001`, continues. Neither entry point raises for a data fault (non-negotiable #1).

## 4. Method / algorithm
All ops vectorised over numpy/pandas — **no per-row Python loops**. Range-checked features
end with F03a's `_checked(fid, value, cfg)`; out of range → `FeatureComputationError` →
NULL + `FEATURE_001`. F27/F28 are **unconstrained** (no range gate).

**X/Y availability (shared precondition for F19, F20, F21, F25).** If `X`/`Y` are absent or
have no finite samples, the lap's corner geometry cannot be computed: set F19, F20, F21, F25
all NULL, log **one** `CORNER_001` row (§4.4 "corner decomposition skipped … F01–F26 still
computed"), and **continue** — F03b never rejects the lap. Per-feature triggers (when X/Y *is*
present) take the `FEATURE_001` NULL path instead.

**Lateral G (F19–F21).** From the X/Y trace compute curvature
κ = |x'y'' − y'x''| / (x'² + y'²)^1.5 (gradients via `np.gradient`), lateral acceleration
a = v² · κ (speed v in m/s = Speed ÷ 3.6), and lateral G = a / `gravity_m_s2`. Signed lateral
G (for F21) carries the sign of (x'y'' − y'x'').

| ID | Name | Domain | Method (summary) | Valid range | NULL trigger |
|----|------|--------|------------------|-------------|--------------|
| F19 | lateral_g_mean | Cornering | mean of \|lateral G\| over the lap | 0–6 G | X/Y unavailable (→ `CORNER_001`) |
| F20 | lateral_g_peak_mean | Cornering | mean of top-`lateral_g_peak_top_n` distinct lateral-G peaks (prominence ≥ `lateral_g_peak_min_prominence`) | 0–6 G | X/Y unavailable (`CORNER_001`); else < `min_lateral_g_peaks` distinct peaks (`FEATURE_001`) |
| F21 | steering_correction_rate | Cornering | sign-reversals of signed lateral G (ignoring \|G\| < `steering_g_deadband`) per second of lap | 0–5 | X/Y unavailable (`CORNER_001`); else lateral G not computable (`FEATURE_001`) |
| F25 | corner_entry_balance | Interaction | per corner: ratio of brake-on duration in the 1st 50% vs 2nd 50% of the corner (split at the apex); mean over corners | 0–1 | X/Y unavailable (`CORNER_001`); else corner segmentation failed — no corners (`FEATURE_001`) |
| F27 | tyre_age_throttle_slope | Degradation | `np.polyfit(tyre_age, throttle_mean, 1)[0]` across the stint's laps | unconstrained | stint < `min_stint_laps` usable laps, or tyre age UNKNOWN (`FEATURE_001`) |
| F28 | tyre_age_brake_slope | Degradation | `np.polyfit(tyre_age, brake_peak_mean, 1)[0]` across the stint's laps | unconstrained | stint < `min_stint_laps` usable laps, or tyre age UNKNOWN (`FEATURE_001`) |

**Corners for F25** reuse F03a's `_corner_indices` (speed-minima apices). Each corner's window
spans the **bracketing speed maxima** (the entry maximum before the apex to the exit maximum
after — i.e. the full entry→exit arc), split at the apex; the brake-on fraction is the entry
half's brake-on samples over (entry + exit), so trail-braking past the apex lowers the ratio.
Adjacent corners share the inter-corner speed maximum as a half-open boundary, so no sample is
double-counted. F25 is grouped with the X/Y features (its corner segmentation is part of the
geometry pass), so X/Y unavailable → F25 NULL + `CORNER_001`.

**Degradation slopes (F27/F28) — stint-level.** A *usable* stint lap has a **known** tyre age
(consistent with F02 `LAP_PARTIAL`) and a non-NULL feature value (F01 for F27, F06 for F28).
If the stint has ≥ `min_stint_laps` (8) usable laps, the slope is the degree-1 polyfit of the
feature value against tyre age; otherwise F27/F28 are NULL for **every** lap of the stint.
F02 has already forced F27/F28 NULL on any individual lap whose own tyre age is UNKNOWN; F03b
applies the stint-level gate on top.

## 5. config.yaml keys this feature adds
F03b **extends** the existing `features:` block + `FeaturesConfig` (standing rule: typed
sub-model, `extra="forbid"`). The range-coverage validator is widened from the 22 scalar IDs
to the **26 range-bearing IDs** (the 22 + F19, F20, F21, F25); F27/F28 are unconstrained and
carry no range entry. `CORNER_001` already exists in `ErrorCode` (§4.4) — **no new code**.
```yaml
# --- F03b geometry & degradation features (§4.2, §4.3, §4.4) ----------------
  lateral_g_peak_top_n: 5            # §4.2 F20 — mean of top-N lateral-G peaks
  min_lateral_g_peaks: 5             # §4.2 F20 — fewer distinct peaks -> NULL
  lateral_g_peak_min_prominence: 0.2 # design decision — min G prominence for a "distinct" peak (F20)
  steering_g_deadband: 0.1           # design decision — |lateral G| below this ignored for F21 sign-reversals
  gravity_m_s2: 9.81                 # physical constant — lateral accel (m/s^2) -> G (F19/F20/F21)
  min_stint_laps: 8                  # §4.2 F27/F28 — stint with fewer usable laps -> NULL
  valid_ranges:                      # added to the existing map (F27/F28 unconstrained -> absent)
    F19: [0, 6]
    F20: [0, 6]
    F21: [0, 5]
    F25: [0, 1]
```

## 6. Error codes & failure modes (the failsafe contract)
| Failure mode | Detection | Error code | System response | Logged? | User-facing message |
|--------------|-----------|-----------|-----------------|---------|---------------------|
| X/Y trace unavailable | `X`/`Y` absent or no finite samples | `CORNER_001` | F19,F20,F21,F25 NULL; **lap continues** (never rejects) | yes (one row/lap) | "Lap N: X/Y unavailable; cornering features skipped." |
| Feature out of §4.2 range | value ∉ `valid_ranges[Fxx]` | `FEATURE_001` | raise `FeatureComputationError` (no clamp) → NULL | yes (lap, feature, value, range, `OUT_OF_RANGE`) | "Lap N feature Fxx out of range → NULL." |
| F20 too few distinct peaks | < `min_lateral_g_peaks` (X/Y present) | `FEATURE_001` | F20 NULL, `NULL_TRIGGER` | yes | "Lap N feature F20 could not be computed → NULL." |
| F21 lateral G not computable | curvature/G undefined (X/Y present) | `FEATURE_001` | F21 NULL, `NULL_TRIGGER` | yes | "Lap N feature F21 could not be computed → NULL." |
| F25 corner segmentation failed | no corners detected (X/Y present) | `FEATURE_001` | F25 NULL, `NULL_TRIGGER` | yes | "Lap N feature F25 could not be computed → NULL." |
| Stint < 8 usable laps | `< min_stint_laps` usable | `FEATURE_001` | F27,F28 NULL for every lap of the stint | yes (per lap) | "Lap N: stint too short for degradation slope → NULL." |
| Tyre age UNKNOWN | `tyre_age` missing for the stint's laps | `FEATURE_001` | F27,F28 NULL | yes | "Lap N: tyre age unknown → degradation slope NULL." |
| > 6 NULL after F03b's NULLs | F03a `assemble_feature_vector` counts > `null_ceiling` | `LAP_NULL_EXCESS` | lap **rejected by F03a's gate** (e.g. X/Y loss = 4 NULL + F27/F28 NULL = 6; one more → reject) | yes (F03a logs) | "Lap N rejected: too many missing features." |

`CORNER_001`, `FEATURE_001`, `LAP_NULL_EXCESS` all already exist in `ErrorCode`. F03b adds none.

## 7. Acceptance criteria (each item mechanically testable by /spec-check)
- [ ] `GeometryFeatureResult.features` keys are exactly `{F19, F20, F21, F25}`; F03b never
      writes F01–F18, F22–F24, F26, F27, F28 into it.
- [ ] `compute_degradation_slopes` returns `{f27, f28}` for a stint and lists the `applies_to`
      lap numbers; F27/F28 appear in no other result object.
- [ ] Each of the six features computes its documented value on a happy-path fixture (one
      asserted value each: F19/F21/F25 from a synthetic constant-radius arc; F20 from known
      peaks; F27/F28 from a known linear throttle/brake-vs-tyre-age trend).
- [ ] **X/Y unavailable** (cols absent or all-NaN): F19, F20, F21, F25 all `None`, exactly one
      `CORNER_001` row, `status=DEGRADED`, and the result is returned (lap **not** rejected by
      F03b).
- [ ] An out-of-range cornering value (e.g. F19 = 7 G > 6) **raises** `FeatureComputationError`
      (no clamp), is caught → NULL → `FEATURE_001`/`OUT_OF_RANGE`, others still compute.
- [ ] With X/Y present: F20 with < `min_lateral_g_peaks` distinct peaks → NULL/`FEATURE_001`;
      F25 with no detectable corners → NULL/`FEATURE_001`.
- [ ] **Stint < `min_stint_laps`** (e.g. 7 usable laps) → F27 and F28 NULL for every lap of the
      stint, one `FEATURE_001` row per lap; a stint of exactly 8 usable laps computes a slope
      (boundary pinned both sides: 7 → NULL, 8 → value).
- [ ] **Tyre age UNKNOWN** (stint laps' `tyre_age=None`) → F27, F28 NULL.
- [ ] F27/F28 are **not** range-checked (a large slope is returned, never raised/NULLed for
      magnitude).
- [ ] **Integration / combined case:** F03b's six values plug into F03a's
      `assemble_feature_vector` unchanged; a lap that loses X/Y (4 NULL) **and** has F27/F28
      NULL (short stint) sits at the 6-NULL ceiling (imputed, accepted); one additional NULL
      (e.g. a scalar feature) → `LAP_NULL_EXCESS` from F03a's gate.
- [ ] F19, F20 (0–6), F21 (0–5), F25 (0–1) ranges, `min_stint_laps` (8), and the `CORNER_001`
      code are read from `config.yaml`/the `ErrorCode` enum; no magic numbers in the source.
- [ ] No `.fit()`/scaler in F03b (F04 owns Z-scoring of F19/F20); no per-row Python loop.
- [ ] Every error path (`CORNER_001`, `FEATURE_001` via OUT_OF_RANGE and NULL_TRIGGER for each
      feature, and the `LAP_NULL_EXCESS` integration) has a unit test that triggers it.

## 8. Test plan
- **Happy-path fixtures:** a synthetic constant-radius circular arc (X = R cos θ, Y = R sin θ)
  at known speed → analytic lateral G = v²/(R·g) (F19 = that value; F20 = same; F21 from a trace
  with a known number of left/right sign flips). A lap with a braking zone biased to the corner
  entry → known F25 ratio. A stint of 10 laps with `throttle_mean = a − b·tyre_age` → F27 = −b;
  `brake_peak_mean` linear in tyre age → F28 known.
- **Failure fixtures (one per mode):** X/Y columns dropped → `CORNER_001` + 4 NULL, lap
  survives; X/Y all-NaN → same; F19 forced to 7 G via a tight radius → OUT_OF_RANGE; flat speed
  / single corner → F20 < 5 peaks; straight-line X/Y → F25 no corners; a 7-usable-lap stint →
  F27/F28 NULL; stint laps with `tyre_age=None` → F27/F28 NULL.
- **Property / edge (pinned both sides):** stint exactly 8 usable laps → slope computed; 7 →
  NULL. F19 == 6.0 in range; 6.0001 → OUT_OF_RANGE. F20 exactly `min_lateral_g_peaks` peaks →
  computed; one fewer → NULL. **Integration:** assemble a vector with the 4 X/Y NULLs + 2
  degradation NULLs = 6 → accepted+imputed; + 1 more → `LAP_NULL_EXCESS`.

## 9. Definition of done
All acceptance criteria MET with passing tests; `/apex-review F03b` clean (no silent failure,
no normalisation, config-driven thresholds, error-code discipline — `CORNER_001` on X/Y loss,
no lap self-rejection); `/spec-check F03b` green; `mypy app` clean; no bare `except` in the F03b
source. INDEX row `F03b` set to DONE; with it the 28-feature vector is complete and F04 is
unblocked. Limitations for the dissertation: (1) lateral G is derived from the X/Y trace +
speed, not an IMU — accuracy depends on X/Y sampling quality (master risk register: X/Y patchy);
(2) the lateral-G peak prominence and steering deadband (`lateral_g_peak_min_prominence`,
`steering_g_deadband`) are documented, config-driven design decisions; (3) F27/F28 collapse a
stint's degradation into a single linear slope — non-linear deg is out of scope here (the
degradation *detector* is F10).

## 10. Risks & mitigations (from master risk register where relevant)
- **X/Y data patchy (Risk Register / §4.4):** the shared X/Y-availability guard NULLs the four
  cornering features and logs `CORNER_001` while the lap continues — never a crash, never a
  silent zero; F03a's 6-NULL ceiling decides rejection, not F03b.
- **Curvature noise at low speed / near-straight:** lateral G double-differentiates the X/Y
  trace (`np.gradient` of `np.gradient`), which amplifies sampling/quantisation noise on real
  (low-Hz, decimetre-quantised) FastF1 position data — curvature spikes can push F19/F20 out of
  `[0,6]` G and NULL them. By design that is the failsafe path (NULL beats a fabricated value),
  and the 6-NULL tolerance absorbs a few; but if F19/F20/F21 chronically NULL on real laps, a
  smoothing window on X/Y before differentiation is the first tuning lever — to be calibrated
  against real telemetry in F04/evaluation, not against the (perfectly smooth) synthetic fixtures.
- **F20 range `[0,6]` G is copied verbatim from master spec §4.2**, but F20 is a *peak* lateral-G
  mean and the highest-downforce F1 corners can approach/exceed 6 G — those laps would NULL F20.
  The bound is a master-spec contract, not ours to widen here; flag for the spec owner if F20
  NULLs cluster on fast corners during evaluation.
- **Short / single-lap stints:** F27/F28 require ≥ `min_stint_laps` usable laps and known tyre
  age, else NULL — no slope is fabricated from too few points (consistent with F10's stint<8
  `MONITOR_INACTIVE` posture).
- **Normalisation leakage (§4.3, non-negotiable #2):** F03b emits **raw** F19/F20; their
  per-circuit Z-scoring is F04's train-only job, asserted absent here.
