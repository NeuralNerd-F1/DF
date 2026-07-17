# Feature F11 — Car Personalisation Engine (mapping · constraints · confidence)

> One spec, one session. This document is the single source of truth a session loads
> to build this feature. It must be self-contained: a session should not need the
> 42-page master spec open, only this file plus CLAUDE.md.

**Status:** SPEC WRITTEN
**Master spec sections:** §7 (Car Personalisation Engine — §7.1 learned mapping, §7.2 constraint
engine, §7.3 the 10-parameter setup vector, §7.4 unknown-driver onboarding), §6.1 (SHAP "applied
to personalisation engine for every setup recommendation"), §8.2 (module row), §8.4
(`setup_recommendations` table), §8.5 (`GET /drivers/{id}/setup/{circuit_id}`), §10.3 (Screen 2
setup panel), §13 (Wk 13 build row), §14 (risk register — "recommendations too wide" row),
§15.1 (Contribution 3 dependency), §15.2 (limitation — "setup approximated, not measured")
**Layer / Week:** ML / Wk 13
**Depends on:** F08 (`StyleSpaceHandle` — the per-driver **5D fingerprint** input, `known_coords_5d`
+ `project_unknown` + `nearest_neighbours` for §7.4 onboarding), F07 (SHAP machinery —
`TreeExplainer` per regressor), F06a (XGBoost + MLflow + leakage-quarantine training pattern),
F04 (normalisation artefact, read-only — reused only to build the training aggregates), F03a/F03b
(the 28-feature aggregate the setup-target heuristics read)

## 1. Purpose (one paragraph)

F11 is the personalisation engine (§7): it translates a driver's **5D style fingerprint** into a
**10-parameter car-setup recommendation** through three layers that every recommendation must pass
— **Layer 1** a learned mapping (one XGBoost regressor per parameter, input = 5D fingerprint + circuit
type), **Layer 2** a rule-based constraint engine (FIA bounds, physical consistency, the Monaco
override, and a confidence floor — hard limits that override or flag the mapping output), and
**Layer 3** confidence scoring (a per-parameter reliability that gates HIGH/MEDIUM/LOW/WIDE and
turns a low-confidence point estimate into an honest range). It also produces the §7.4
unknown-driver onboarding package (style-match + setup sheet + radar + risk flags + data-quality
summary) and, per §6.1, a SHAP decomposition per recommendation. The setup values are **approximated
from telemetry, never measured** (§15.2) — the system gives a starting direction, not an engineering
spec, and says so on every output. It is a dissertation-Contribution-3 consumer (RQ4 downstream).

## 2. Upstream / downstream contracts

- **Consumes:**
  - **F08** → `StyleSpaceHandle` (`app/ml/stylespace.py`), read-only:
    - `get_style_space(handle) -> StyleSpaceResult` / `handle.known_coords_5d: dict[str, np.ndarray]`
      — the **known-driver 5D fingerprints**, F11's Layer-1 regressor **input** and the calibration
      population. F08 §2 names F11 as the consumer of the "top-3 nearest known drivers" **and** the
      5D fingerprint.
    - `project_unknown(laps, *, circuit, handle, config, log_path) -> ProjectionResult` — the unknown
      driver's 5D coordinate + `nearest_neighbours` (top-3, cosine) for §7.4. Never raises; carries
      its own `SPARSE_PROJECTION` / `OUT_OF_DISTRIBUTION` / `PROJECTION_FALLBACK` flags **opaquely**
      (F11 never re-interprets them; a `coords_5d=None` fallback → onboarding served WIDE, never a
      crash). A `< thresholds.min_clean_laps` (30) prefix is sparse **by design** (§2).
    - `nearest_neighbours(query_5d, known_coords_5d, k) -> list[Neighbour]` — the style-match report.
  - **F04** → `artefacts/normalisation.json` via `load_normalisation_params` / `apply_normalisation`
    (read-only, never re-fits) — used **only** to build the per-driver-per-circuit 28-dim aggregate
    the §4.0-b setup-target heuristics read (the regressor **input** is the F08 5D coordinate, not
    the raw features). F11 fits **no normaliser** and adds no `.fit()` on any F04 param.
  - **F07** → the SHAP pattern (`app/ml/explain.py` `TreeExplainer`, exact) — reused per regressor.
    F07's precedent: **SHAP never blocks a prediction** (fault → `SHAP_UNAVAILABLE`, result still
    returned).
  - **F06a** → the XGBoost + Optuna + MLflow + `_assert_no_leakage` quarantine training pattern
    (`app/ml/classifier.py`) — mirrored, not imported. Seeds from `config.seeds`.
  - **Training aggregates** — the F06a-era wide matrix (`driver`, `circuit`, `circuit_type`, `season`,
    `split`, `F01..F28`) **plus a lap-time / performance column** (§3) so "best-performing stints"
    (§7.1) is definable; F11 declares its own `_REQUIRED_COLUMNS` superset.
- **Produces:**
  - `artefacts/personalisation.pkl` — the champion bundle: **10 fitted XGBoost regressors** (one per
    parameter), their per-parameter **reliability scores** (§4.0-c) and **SHAP explainers**, the
    training **circuit-type vocabulary + FIA-bound table + approximation-heuristic config snapshot**,
    a `content_hash`, and `model_version`. Written by `train_personalisation` and nothing else.
    (Joblib bundle, F08 precedent — not canonical JSON, because it carries fitted sklearn/xgboost
    objects; a sha256 over the pickled bytes gives `model_version`.)
  - Typed results: `SetupRecommendation` (per driver × circuit — the 10-parameter vector + confidence
    vector + constraint flags + SHAP + the mandatory approximation disclaimer),
    `SetupParameter` (one parameter's value/range/confidence/rating/attribution), `OnboardingPackage`
    (§7.4), `PersonalisationFitResult` (training summary), `PersonalisationHealthResult`.
  - MLflow training run in experiment `apex-personalisation` (own URI key, F08/F10 precedent).
- **Consumed by:**
  - **F13** — serves `SetupRecommendation` / `OnboardingPackage`; `health()` feeds `GET /health`
    (503 on `HEALTH_FAIL`).
  - **F12** — the `setup_recommendations` table maps from `SetupRecommendation`: `setup_vector`,
    `confidence_vector`, `constraint_flags` (JSONB), `model_version` (§8.4). `confidence_state` must
    be non-NULL (HIGH/MEDIUM/LOW/WIDE — enforced by the never-NULL confidence contract).
  - **F19** — Screen 2's setup panel renders the 10 animated gauges (range bar for `WIDE`).
  - **F20** — RQ4/§14 evaluation: mapping reliability per parameter, the "5 reliably-inferrable
    parameters" finding; held-out circuits touched **only** there.

## 3. Inputs & outputs (typed)

### Input

- `train_personalisation(matrix, *, style_handle, config, artefact_dir=None, log_path=None)` takes the
  wide training matrix with `_REQUIRED_COLUMNS = (driver, circuit, circuit_type, season, split,
  lap_time_s, F01..F28)` (`lap_time_s` = the F01-persisted clean-lap time, the "best-performing"
  selector — §4.0-a) and the loaded F08 `StyleSpaceHandle` (the 5D fingerprints). Missing columns /
  no usable driver-circuit cells → raise `PersonalisationTrainingError(MODEL_TRAINING_FAILED)`.
- `recommend_setup(*, driver, circuit, circuit_type, handle, style_handle, config, log_path=None) ->
  SetupRecommendation` — serves a **known** driver (5D coord from `style_handle.known_coords_5d`).
  Unknown driver at an unknown circuit → the fallback tiers of §4.2. Never raises.
- `onboard_unknown(laps, *, circuit, circuit_type, handle, style_handle, config, log_path=None) ->
  OnboardingPackage` — the §7.4 package for an uploaded unknown driver (projects via
  `project_unknown`). Never raises.

### Output — result objects (Pydantic on `BaseResult` unless noted, `extra="forbid"`)

Feature-owned enums live in `app/personalisation/engine.py` (per the `enums.py` docstring):

```python
class SetupConfidence(StrEnum):   # per-parameter rating (§7.4 "HIGH / MEDIUM / LOW / WIDE")
    HIGH = "HIGH"; MEDIUM = "MEDIUM"; LOW = "LOW"; WIDE = "WIDE"

class SetupOutputKind(StrEnum):   # §7.3 output-format column
    DEGREES = "DEGREES"; PERCENT = "PERCENT"; INDEX = "INDEX"; INTEGER_LAPS = "INTEGER_LAPS"
    CATEGORY = "CATEGORY"          # Soft/Medium/Stiff (suspension) — ordinal-regressed (divergence #15)
```

```python
class SetupParameter(BaseModel):                 # one of the 10 parameters
    name: str                                    # canonical key, e.g. "front_wing_angle"
    kind: SetupOutputKind
    value: float | str | None                    # point estimate (str for CATEGORY); None only if WIDE-range-only
    value_range: tuple[float, float] | None       # populated iff confidence == WIDE (±wide_range_pct, §7.2)
    category: str | None = None                  # Soft/Medium/Stiff when kind==CATEGORY
    confidence: float                            # [0,1] per-parameter reliability (never NULL — §8.4)
    rating: SetupConfidence
    unit: str                                    # "degrees" | "% front" | "index" | "laps" | "% open" | ""
    fia_min: float | None; fia_max: float | None # the clip bounds actually applied (None = team-defined/N/A)
    constraint_flags: list[ErrorCode] = []       # CONSTRAINT_CLIPPED / CIRCUIT_OVERRIDE_MONACO / WIDE_CONFIDENCE
    shap_top_features: list[str] = []            # fingerprint dims driving this parameter (§6.1)
    shap_available: bool = False

class SetupRecommendation(BaseResult):           # never-raises serving result
    driver: str = ""
    circuit: str = ""
    circuit_type: str | None = None
    parameters: list[SetupParameter] = []        # exactly 10, canonical order (never partial silently)
    consistency_warning: bool = False            # §7.2 front-wing vs brake-bias flag
    monaco_override: bool = False                # §7.2 CIRCUIT_OVERRIDE_MONACO applied
    threshold_source: str = ""                   # "driver_circuit_type" | "driver_global" | "global"
    approximation_disclaimer: str = APPROX_DISCLAIMER  # §15.2 — present on EVERY recommendation
    model_version: str = ""

class Neighbour(BaseModel):                      # re-declared minimal (mirrors F08) for the style-match row
    driver: str; cosine_similarity: float; distance: float

class OnboardingPackage(BaseResult):             # §7.4 — never raises
    setup: SetupRecommendation | None = None     # the setup sheet (may be WIDE-heavy for a sparse unknown)
    style_match: list[Neighbour] = []            # top-3 nearest known drivers (from project_unknown)
    fingerprint_snapshot: ProjectionResult | None = None  # F08 embed (radar coords + risk flags)
    fingerprint_available: bool = False
    risk_flags: list[str] = []                   # OOD fingerprint dims (from ProjectionResult.risk_flags)
    n_laps_used: int = 0
    n_laps_rejected: int = 0
    imputed_features: list[str] = []             # data-quality summary (§7.4)

class PersonalisationFitResult(BaseResult):      # training output
    artefact_path: str = ""
    content_hash: str = ""
    n_train_cells: int = 0                       # (driver, circuit) training rows used
    parameters_fitted: int = 0                   # 10 on success
    reliability: dict[str, float] = {}           # per-parameter CV reliability (the §14 "which 5 are reliable")
    n_reliable: int = 0                          # parameters with reliability ≥ confidence_medium
    mlflow_run_id: str | None = None

class PersonalisationHealthResult(BaseResult):   # mirrors F06b/F08/F10 health
    ready: bool = False
    model_version: str = ""
    reason: ErrorCode | None = None
```

`PersonalisationHandle` is a frozen dataclass (mirrors `StyleSpaceHandle`/`MonitorHandle`): the 10
fitted regressors, their per-parameter reliability + SHAP explainers, the FIA-bound table, the
circuit-type vocabulary, the approximation-heuristic snapshot, `normalisation_params`,
`model_version` (bundle content hash), `ready: bool`.

**Errors raised (only at build/load):** `PersonalisationTrainingError` (carries `TRAIN_LEAKAGE` or
`MODEL_TRAINING_FAILED`) from `train_personalisation`; `PersonalisationLoadError` (`MODEL_NOT_FOUND`)
from `load_personalisation`. `recommend_setup`, `onboard_unknown`, `health` **never raise**.

**Naming guard:** F11's `Neighbour` mirrors F08's field-for-field but is re-declared locally (F11 does
not import F08's internal model), consistent with the module-owns-its-results convention.

## 4. Method / algorithm

### 4.0 Master-fidelity reconciliations (read first — the divergences a build session must not re-litigate)

- **(a) "Best-performing stints" (§7.1 training data) — divergence #14.** The master says training data
  is "best-performing stints per driver per circuit" without defining "best". F11 defines it as the
  **fastest-quartile clean laps** per `(driver, circuit)`: within each cell, keep laps with
  `lap_time_s ≤ Q1` of that cell's clean-lap-time distribution (config `top_quartile_pct: 0.25`),
  requiring ≥ `thresholds.min_clean_laps` (30) clean laps in the cell to qualify (else the cell is
  dropped + `DATA_INSUFFICIENT` DQ row). This is deterministic and telemetry-only (no external result
  data), and gives one **aggregate fingerprint** (F04-normalised 28-dim mean over the kept laps) +
  one **5D coordinate** (from `style_handle.known_coords_5d[driver]`) per cell.
- **(b) Setup-target approximation (§7.1 "approximated from telemetry signatures") — divergence #13.**
  The master gives two examples (brake bias ← longitudinal-G distribution; front wing ← high-speed vs
  slow-corner speed delta) and the §7.3 feature→parameter table, but no formula. F11 defines each of
  the 10 targets as a **documented, monotonic, min-max-normalised linear map of the named §7.3
  fingerprint features** (computed on the cell aggregate, §4.0-a), then affinely rescaled onto the
  parameter's FIA/output range and clipped. The exact per-parameter feature basis + direction is the
  §4.1 table; the coefficients are equal-weight normalised (no fitted magic constants — the *mapping*
  from fingerprint to setup is what Layer 1 learns, the *target* is a transparent heuristic). This is
  the honest core approximation (§15.2) — stated on every output via `approximation_disclaimer`.
- **(c) Per-parameter regression confidence (§7.2 "confidence < 0.5") — divergence #17.** XGBoost
  regression has no native confidence. F11 defines per-parameter **reliability** `r_p ∈ [0,1] =
  max(0, 1 − nRMSE_p)` where `nRMSE_p` is the **train-only, grouped-CV** normalised RMSE of regressor
  `p` (RMSE ÷ target range; `StratifiedGroupKFold` by driver so a driver never spans folds — the same
  leakage guard F06a uses). `r_p` is stored in the bundle. A serve-time prediction's confidence is
  `c_p = r_p × ood_attenuation` where `ood_attenuation ∈ [ood_floor, 1]` shrinks toward `ood_floor`
  as the input 5D fingerprint's distance from the training-fingerprint centroid exceeds
  `ood_sd_threshold` SD (reusing F08's OOD idea; an OOD unknown driver is honestly less certain).
  `c_p < confidence_floor` (0.5) → `WIDE` (§4.0-d). Rating bands: `≥ confidence_high` → HIGH,
  `≥ confidence_medium` → MEDIUM, `≥ confidence_floor` → LOW, else WIDE (config-driven).
- **(d) WIDE range (§7.2 "Range = ±15% of predicted value").** When `c_p < confidence_floor`, `value`
  is set to `None` and `value_range = (v·(1−wide_range_pct), v·(1+wide_range_pct))` with
  `wide_range_pct: 0.15`, **after** FIA clipping (the range is clipped to the legal bounds too), and
  `WIDE_CONFIDENCE` is added to that parameter's `constraint_flags` + a DQ row; `status=DEGRADED`.
- **(e) Categorical parameters (suspension stiffness front/rear = Soft/Medium/Stiff) — divergence #15.**
  The master says "one XGBoost regressor per parameter" but two parameters are categorical. F11 keeps
  one regressor per parameter by **ordinal regression**: encode Soft=0/Medium=1/Stiff=2, regress the
  ordinal target, and at serve **round to the nearest ordinal** → category. `kind=CATEGORY`,
  `category` set, `value` = the ordinal float; FIA clipping is to `[0, 2]`. Confidence/SHAP are
  identical to the numeric path.
- **(f) Circuit-type one-hot (§7.1 "3D one-hot") — divergence #16.** There are **four** circuit types
  (`street, high_speed, technical, mixed`, §8.4 ENUM), so a full one-hot is 4-D; the master's "3D" is
  a **drop-first** encoding. F11 uses **drop-first one-hot over `personalisation.circuit_types`**
  (4 types → 3 dummies, `street` as the reference), giving the master's 3-D input. An unknown
  `circuit_type` (not in the vocabulary) → all-zero dummies + `NORMALISATION_FALLBACK`-style
  `FALLBACK_THRESHOLD` flag, served (degrade, don't die).
- **(g) Statelessness / never-raises.** `recommend_setup` / `onboard_unknown` are pure functions of
  their inputs; a backstop `except` at each boundary returns a typed WIDE/`INFERENCE_ERROR` result,
  never a propagated exception (F06b/F08/F10 precedent).

### 4.1 The 10-parameter setup vector (§7.3 — canonical order, feature basis, bounds)

| # | `name` | Fingerprint basis (§7.3) | `kind` | FIA bound (config) |
|---|---|---|---|---|
| 1 | `front_wing_angle` | F19, F14 | DEGREES | 0–12 |
| 2 | `rear_wing_angle` | F04, F14 | DEGREES | 0–15 |
| 3 | `brake_bias` | F06, F08, F10 | PERCENT | 52–62 |
| 4 | `differential_ramp_entry` | F25, F09 | INDEX | 1–10 |
| 5 | `differential_ramp_exit` | F03, F22 | INDEX | 1–10 |
| 6 | `throttle_map_curve` | F01, F02, F03, F05 | INDEX | 1–9 |
| 7 | `suspension_stiffness_front` | F21, F19 | CATEGORY | 0–2 (Soft/Medium/Stiff) |
| 8 | `suspension_stiffness_rear` | F12, F20 | CATEGORY | 0–2 |
| 9 | `tyre_warmup_laps` | F27 | INTEGER_LAPS | 1–5 |
| 10 | `brake_duct_aperture` | F06, F07, F08 | PERCENT | 30–100 |

The §4.0-b **target heuristic** for parameter `p`: `t_p = clip( fia_min_p + (fia_max_p − fia_min_p) ·
mean_i( minmax_train(f_i) ), fia_min_p, fia_max_p )` over its basis features `f_i` (min-max scaled on
the training aggregates, floored at `normalise.min_std`). Every constant is a config FIA bound; the
"team-defined" bounds default to the output-kind's natural range (INDEX 1–10, CATEGORY 0–2). Bounds
per season live in `personalisation.fia_bounds` (§5).

### 4.2 Serving — `recommend_setup` (read-only, never raises)

1. **Resolve the 5D fingerprint**: `x5 = style_handle.known_coords_5d[driver]`; unknown driver →
   `threshold_source` fallback: the **driver-mean 5D** is unavailable, so route to
   `onboard_unknown`-style projection is the caller's job — `recommend_setup` for an unknown driver
   returns `INVALID_INPUT` (use `onboard_unknown`). Circuit-type unknown → drop-first zeros +
   `FALLBACK_THRESHOLD` (§4.0-f). Build `X = [x5 (5) ‖ circuit_onehot (3)]`.
2. **Layer 1 — predict** each parameter: `v_p = regressor_p.predict(X)`; ordinal params rounded at
   the end (§4.0-e).
3. **Layer 3 — confidence** (before constraints, since the floor gates Layer 2's range): `c_p = r_p ×
   ood_attenuation(x5)`; `rating` band; `WIDE` iff `c_p < confidence_floor`.
4. **Layer 2 — constraints** (hard limits, applied in this order; §7.2):
   a. **Monaco override**: `circuit == monaco_circuit` → set `front_wing_angle` / `rear_wing_angle`
      to their `fia_max`; `monaco_override=True`, `CIRCUIT_OVERRIDE_MONACO` flag + DQ row,
      `status=DEGRADED`. (Applied even under WIDE — a hard override wins.)
   b. **FIA clip**: any `v_p` outside `[fia_min_p, fia_max_p]` → clip; `CONSTRAINT_CLIPPED` flag + DQ
      row (records original + clipped), `status=DEGRADED`.
   c. **Confidence floor → range**: `WIDE` params get `value=None`, `value_range = ±wide_range_pct`
      (clipped), `WIDE_CONFIDENCE` flag + DQ row.
   d. **Physical consistency** (§7.2, checked AFTER clipping, no override): `front_wing_angle >
      consistency_front_wing_deg` (10) AND `brake_bias < consistency_brake_bias_pct` (54) →
      `consistency_warning=True`, `CONSISTENCY_WARNING` flag + DQ row (a **response flag**, never a
      value change — "flag for engineer review").
5. **SHAP** (§6.1): for each parameter, `TreeExplainer(regressor_p)` on `X` → `shap_top_features`
   (top-`shap_top_k` of the 8 input dims mapped to fingerprint-domain names); a SHAP fault →
   `shap_available=False` + `SHAP_UNAVAILABLE` DQ row, **recommendation still returned** (F07
   precedent — never blocks).
6. **Assemble** the 10 `SetupParameter`s (canonical order — always 10, never a silent partial), the
   `approximation_disclaimer`, `model_version`, and the overall status: `OK` only when no flag on any
   parameter and no warning; else `DEGRADED`. Backstop `except` → a 10-param all-`WIDE`
   `INFERENCE_ERROR` result (never raises).

### 4.3 Serving — `onboard_unknown` (§7.4, never raises)

Project the RAW uploaded laps via `project_unknown` (F08 normalises internally — never hand it a
pre-normalised frame). Build the package: **style-match** = `snapshot.nearest_neighbours` (top-3);
**setup sheet** = `recommend_setup` run on the projected `coords_5d` (a `coords_5d=None`
`PROJECTION_FALLBACK` → the whole sheet is `WIDE`, honestly, never a crash); **radar** = the embedded
`ProjectionResult`; **risk flags** = `snapshot.risk_flags`; **data-quality summary** = `n_laps_used`
/ `n_laps_rejected` / `imputed_features` (from the projection + normalisation). `fingerprint_available
= snapshot is not None and snapshot.coords_5d is not None`. Every F08 flag is carried opaquely.

### 4.4 Training — `train_personalisation` (the ONLY fit site) — fail-loud

1. **Validate** `_REQUIRED_COLUMNS` present + non-empty → else `PersonalisationTrainingError(
   MODEL_TRAINING_FAILED)`.
2. **Quarantine FIRST** (F04/F06a/F08/F10 pattern): drop `held_out_circuits` + every `split != "train"`
   row **before any reduction**; `_assert_no_leakage` tripwire → `TRAIN_LEAKAGE` on any survivor,
   **no artefact written**.
3. **Load F04 params** (read-only) to build the per-cell **aggregate** (fastest-quartile clean laps,
   §4.0-a) — the target-heuristic input. Cells with < `min_clean_laps` clean laps dropped + logged.
4. **Build the training table**: one row per `(driver, circuit)` cell → `X = [5D coord ‖ circuit
   one-hot]`, `y_p = target heuristic` (§4.1) for each of the 10 parameters. All targets derived from
   the **quarantined** aggregates only.
5. **Fit** 10 XGBoost regressors (seeds from `config.seeds.xgboost`; `objective=reg:squarederror`;
   Optuna optional, `optuna_trials` from config, sampler seed 42). Compute per-parameter reliability
   `r_p` via `StratifiedGroupKFold`-by-driver CV nRMSE (§4.0-c). No held-out data in any fold.
6. **Fit SHAP explainers** (`TreeExplainer`, exact) per regressor on the training `X`.
7. **Persist** `personalisation.pkl` (joblib): regressors, `r_p`, explainers, FIA-bound table,
   circuit-type vocab, approximation snapshot, `content_hash` (sha256 of the pickled bytes),
   `model_version`. Deterministic given fixed seeds (XGBoost/Optuna 42) — reproducible per §3 of the
   constitution.
8. **MLflow**: log params (the `personalisation:` block), metrics (per-parameter `r_p`, `n_reliable`,
   `n_train_cells`), and the artefact to experiment `apex-personalisation` (own URI key). An MLflow
   fault → `status=DEGRADED` + DQ row (`MODEL_TRAINING_FAILED`, tagged `mlflow`, `fallback_used=True`,
   F08/F10 mechanism), `mlflow_run_id=None`, **never aborts the write**.
9. **No usable cells at all** → `PersonalisationTrainingError(MODEL_TRAINING_FAILED)`.

### 4.5 Load & health — `load_personalisation` / `health`

`load_personalisation` reads `personalisation.pkl` (schema + content-hash validated) and
`normalisation.json` (F04 loader). Missing / corrupt / hash-mismatch / **wrong regressor count**
(≠ 10) / **missing reliability or explainer** → DQ-log then **raise** `PersonalisationLoadError(
MODEL_NOT_FOUND)`; the process refuses to serve (F06b/F08/F10 precedent). A **config-mismatch** on the
FIA-bound table or circuit-type vocab (artefact vs live) → **raise** (`MODEL_NOT_FOUND`; the targets
were built under different bounds — recalibrate). `health(handle)` → not-ready reports `HEALTH_FAIL`
with `MODEL_NOT_FOUND` reason (F13 → 503).

## 5. config.yaml keys this feature adds

F11 adds a typed `personalisation:` block (`PersonalisationConfig` in `app/contracts/config.py`,
`extra="forbid"`, standing typed-config rule) + optional `personalisation: PersonalisationConfig |
None = None` on `AppConfig`, and a top-level `circuit_types` list (referenced by the one-hot and, in
future, F03a imputation).

```yaml
circuit_types: [street, high_speed, technical, mixed]   # §8.4 ENUM; drop-first one-hot ref = street

personalisation:                       # F11 §7 — car personalisation engine
  top_quartile_pct: 0.25               # §7.1 "best-performing stints" = fastest-quartile clean laps (div #14)
  confidence_floor: 0.5                # §7.2 — < this per parameter → WIDE range, not a point estimate
  confidence_medium: 0.65              # rating band: r ≥ this → MEDIUM (master-silent, div #17)
  confidence_high: 0.80                # rating band: r ≥ this → HIGH  (master-silent, div #17)
  wide_range_pct: 0.15                 # §7.2 — WIDE range = ±N of the predicted value
  ood_sd_threshold: 3.0                # §4.0-c — 5D fingerprint > N SD from train centroid → attenuate
  ood_floor: 0.3                       # §4.0-c — confidence attenuation floor for a fully-OOD input
  consistency_front_wing_deg: 10.0     # §7.2 — extreme front wing threshold for the consistency check
  consistency_brake_bias_pct: 54.0     # §7.2 — very-low brake bias threshold
  monaco_circuit: Monaco               # §7.2 — the hard max-downforce override circuit
  shap_top_k: 3                        # §6.1 — top fingerprint drivers per parameter (Screen 2 panel)
  optuna_trials: 30                    # per-regressor HPO budget (0 = defaults, fast tests)
  cv_folds: 5                          # StratifiedGroupKFold-by-driver reliability CV (div #17)
  model_artefact_filename: personalisation.pkl   # under paths.artefacts_dir
  mlflow_tracking_uri: sqlite:///artefacts/mlflow.db   # F11's OWN URI (same store, own experiment)
  mlflow_experiment: apex-personalisation
  fia_bounds:                          # §7.3 per-season legal min/max (2024 defaults; season-keyed)
    front_wing_angle: [0.0, 12.0]
    rear_wing_angle: [0.0, 15.0]
    brake_bias: [52.0, 62.0]
    differential_ramp_entry: [1.0, 10.0]
    differential_ramp_exit: [1.0, 10.0]
    throttle_map_curve: [1.0, 9.0]
    suspension_stiffness_front: [0.0, 2.0]
    suspension_stiffness_rear: [0.0, 2.0]
    tyre_warmup_laps: [1.0, 5.0]
    brake_duct_aperture: [30.0, 100.0]
```

**Referenced, not re-declared** (house rule): `seeds.xgboost` / `seeds.optuna` (reproducibility),
`held_out_circuits`, `thresholds.min_clean_laps` (30 — cell floor), `normalise.min_std` (min-max /
CV floor), F04's `normalise.artefact_filename`, F08's `stylespace` artefacts (via the passed handle),
`paths.artefacts_dir` / `paths.data_quality_log`.

**Validator placement** (F08/F10 precedent): scalar sanity on `PersonalisationConfig`
(`0 < top_quartile_pct ≤ 1`; `0 < confidence_floor < confidence_medium < confidence_high ≤ 1`;
`0 < wide_range_pct < 1`; `ood_sd_threshold > 0`; `0 ≤ ood_floor ≤ 1`; `cv_folds ≥ 2`;
`shap_top_k ≥ 1`; every `fia_bounds` entry `min < max`; the 10 parameter keys present exactly;
filenames non-empty; mlflow keys non-empty). Cross-block on `AppConfig`: `personalisation` present
requires `normalise` + `features` + a non-empty `circuit_types`. All violations → load-time
`CONFIG_001`.

**No new error code.** The four §7 codes are already seeded in `app/contracts/errors.py:99–102`
(`CONSTRAINT_CLIPPED`, `CONSISTENCY_WARNING`, `CIRCUIT_OVERRIDE_MONACO`, `WIDE_CONFIDENCE`); F11 reuses
`INVALID_INPUT`, `MODEL_NOT_FOUND`, `HEALTH_FAIL`, `TRAIN_LEAKAGE`, `MODEL_TRAINING_FAILED`,
`INFERENCE_ERROR` (serving backstop, F06b), `NORMALISATION_FALLBACK`, `SHAP_UNAVAILABLE`,
`FALLBACK_THRESHOLD` (unknown circuit type), `DATA_INSUFFICIENT` (thin cell dropped at train),
`DATA_QUALITY_LOG_UNWRITABLE`.

## 6. Error codes & failure modes (the failsafe contract)

Master §7.2 rows are marked ▣. Placement discipline mirrors F10: operative code in `error_code` on
FAILED outcomes; per-parameter/response codes in `constraint_flags` / `flags` on scored-but-DEGRADED
outcomes; confidence is **never NULL** (§8.4).

| Failure mode | Detection | Error code (placement) | System response | Logged? | User message |
|---|---|---|---|---|---|
| ▣ Parameter outside FIA bounds | Clip check (§4.2-4b) | `CONSTRAINT_CLIPPED` (param flag) | Clip to boundary; `status=DEGRADED` | yes (orig+clipped) | "Value clipped to legal bound." |
| ▣ Front wing >10° AND brake bias <54% | Consistency check (§4.2-4d) | `CONSISTENCY_WARNING` (response flag) | Flag only, **no override**; `status=DEGRADED` | yes | "Front-wing / brake-bias combination flagged for engineer review." |
| ▣ Monaco | `circuit == monaco_circuit` | `CIRCUIT_OVERRIDE_MONACO` (param flag) | Front+rear wing → max legal; `status=DEGRADED` | yes | "Monaco: maximum downforce override applied." |
| ▣ Regression confidence < 0.5 (a parameter) | `c_p < confidence_floor` (§4.0-c/d) | `WIDE_CONFIDENCE` (param flag) | Return range ±15% (clipped), `value=None`; `status=DEGRADED` | yes | "Low confidence — a range is shown, not a point value." |
| Unknown circuit type | Not in `circuit_types` vocab | `FALLBACK_THRESHOLD` (flag) | Zero dummies used; `status=DEGRADED` | yes | "Circuit type unknown — generic mapping used." |
| Unknown driver passed to `recommend_setup` | Driver ∉ `known_coords_5d` | `INVALID_INPUT` (error_code) | `status=FAILED`, use `onboard_unknown`; nothing scored | yes | "Unknown driver — use onboarding." |
| Sparse / OOD / fallback projection (onboarding) | F08 flags in the embedded `ProjectionResult` | — (opaque, carried) | Setup sheet served WIDE-heavy; `fingerprint_available=False` at worst; never blocks | yes (by F08) | "Limited data — recommendations are wide." |
| SHAP raises for a parameter | `except` around `TreeExplainer` | `SHAP_UNAVAILABLE` (flag) | `shap_available=False`; recommendation still returned (F07) | yes | — (SHAP panel hidden) |
| Unforeseen internal serving fault | Backstop `except` at the boundary | `INFERENCE_ERROR` (error_code) | Best-effort DQ log → 10-param all-`WIDE` `FAILED` result; **never raises** | best-effort | "Recommendation unavailable — internal error." |
| Malformed serve input (missing driver/circuit) | Input gate | `INVALID_INPUT` (error_code) | `status=FAILED`, nothing scored | yes | "Request malformed." |
| `personalisation.pkl` missing / corrupt / hash mismatch / wrong regressor count / missing reliability|explainer | Load-time validation | `MODEL_NOT_FOUND` | DQ-log then **raise** `PersonalisationLoadError` — refuse to serve | yes | 503 via F13 |
| FIA-bound / circuit-vocab mismatch (artefact vs live) | Load-time comparison | `MODEL_NOT_FOUND` | **Raise** — targets stale, recalibrate | yes | 503 via F13 ("recalibrate") |
| Detector not loaded at health check | `health()` | `HEALTH_FAIL` (reason `MODEL_NOT_FOUND`) | `ready=False` | yes | 503 from `GET /health` |
| Thin `(driver, circuit)` cell at train | < `min_clean_laps` clean laps | `DATA_INSUFFICIENT` (flag) | Cell dropped from training; `status=DEGRADED` | yes | build log |
| Held-out / non-train row survives quarantine | `_assert_no_leakage` tripwire | `TRAIN_LEAKAGE` | **Raise**; no artefact written | yes | build log (CI-breaking) |
| No usable cells / missing columns / write fault | Guards in `train_personalisation` | `MODEL_TRAINING_FAILED` | **Raise**; no artefact written | yes | build log |
| MLflow logging fault during training | `except` around the MLflow calls | `MODEL_TRAINING_FAILED` (DQ row only) | `status=DEGRADED`, `mlflow_run_id=None`; artefact write unaffected | yes | build log |
| DQ log sink unwritable | `DataQualityLogError` in `_DQLedger.record` | `DATA_QUALITY_LOG_UNWRITABLE` (flag) | Never propagates; result flagged | n/a | — |

## 7. Acceptance criteria (each mechanically testable by /spec-check)

- [ ] **Ten parameters, canonical order, always present.** `recommend_setup` returns exactly 10
      `SetupParameter`s in the §4.1 order, each with a non-NULL `confidence` and a `rating` ∈
      {HIGH, MEDIUM, LOW, WIDE}; a partial result is never returned silently.
- [ ] **Layer 1 — one regressor per parameter.** Training fits exactly 10 XGBoost regressors; input is
      the **5D fingerprint ‖ drop-first circuit one-hot** (8 dims); ordinal params (suspension) are
      regressed then rounded to the nearest category (divergence #15).
- [ ] **Setup targets are the documented telemetry heuristic (divergence #13)**, computed on the
      fastest-quartile clean-lap aggregates only (divergence #14); the `approximation_disclaimer`
      (§15.2) is present on **every** `SetupRecommendation`.
- [ ] **Layer 2 — FIA clip.** A predicted value outside `fia_bounds[p]` is clipped to the boundary
      with `CONSTRAINT_CLIPPED` on that parameter + a DQ row recording original and clipped;
      `status=DEGRADED`.
- [ ] **Layer 2 — Monaco override.** `circuit == monaco_circuit` sets front+rear wing to their
      `fia_max`, `monaco_override=True`, `CIRCUIT_OVERRIDE_MONACO` flag + DQ row — regardless of the
      mapping output.
- [ ] **Layer 2 — consistency warning.** front wing > 10 AND brake bias < 54 → `consistency_warning`
      + `CONSISTENCY_WARNING` flag + DQ row, and **no value is changed** (flag only).
- [ ] **Layer 3 — confidence + WIDE.** A parameter with `c_p < confidence_floor` (0.5) returns
      `value=None`, `value_range = predicted ± wide_range_pct` (clipped to FIA bounds), `rating=WIDE`,
      `WIDE_CONFIDENCE` flag + DQ row; the rating bands map at `confidence_medium` / `confidence_high`.
- [ ] **Confidence is never NULL** on any parameter of any returned result (§8.4), including the
      backstop `INFERENCE_ERROR` path (all-`WIDE`).
- [ ] **Reliability drives the §14 finding.** `PersonalisationFitResult.reliability` has a value per
      parameter and `n_reliable` counts those ≥ `confidence_medium`; the CV is
      `StratifiedGroupKFold`-by-driver (no driver spans folds).
- [ ] **SHAP per recommendation (§6.1).** Each parameter carries `shap_top_features` of size
      `shap_top_k` and `shap_available=True` on the happy path; a SHAP fault → `shap_available=False`
      + `SHAP_UNAVAILABLE`, recommendation still returned (never blocked).
- [ ] **§7.4 onboarding.** `onboard_unknown` returns a `style_match` of top-3 neighbours, an embedded
      `fingerprint_snapshot`, `risk_flags`, a data-quality summary (`n_laps_used`/`n_laps_rejected`/
      `imputed_features`), and a setup sheet; a `coords_5d=None` projection → a WIDE-heavy sheet,
      never a crash; `fingerprint_available` is defined on the returned result.
- [ ] **Unknown circuit type** → drop-first zeros + `FALLBACK_THRESHOLD`, served DEGRADED (not a
      crash); **unknown driver** to `recommend_setup` → `INVALID_INPUT`.
- [ ] **Never raises.** `recommend_setup` / `onboard_unknown` under fault injection (a regressor /
      `project_unknown` / SHAP monkeypatched to throw) return a typed result, not an exception.
- [ ] **No `.fit()` on held-out data.** Training quarantines held-out + non-train rows **before any
      reduction**; a poisoned matrix raises `TRAIN_LEAKAGE` and writes **no artefact**; a static scan
      asserts no `.fit(` in the serving path (`load_personalisation` / `recommend_setup` /
      `onboard_unknown` / `health`).
- [ ] **Reproducibility.** Two training runs on the same matrix + seeds produce an artefact with equal
      per-parameter reliability (to 1e-9) and equal `content_hash`; all seeds from `config.seeds`.
- [ ] **MLflow-logged** to `apex-personalisation`; an MLflow fault → `status=DEGRADED` + DQ row
      (`MODEL_TRAINING_FAILED`), `mlflow_run_id=None`, artefact still written.
- [ ] **Load refuses** (`MODEL_NOT_FOUND`, after a DQ row) on missing / corrupt / hash-mismatch /
      wrong-regressor-count / missing-reliability|explainer / FIA-or-vocab-mismatch; `health(None)` →
      `HEALTH_FAIL` with reason `MODEL_NOT_FOUND`.
- [ ] **Every §6 code path** has a test that triggers it and asserts the typed result + code
      placement (the four §7 codes + `INVALID_INPUT`, `MODEL_NOT_FOUND`, `HEALTH_FAIL`,
      `SHAP_UNAVAILABLE`, `FALLBACK_THRESHOLD`, `DATA_INSUFFICIENT`, `INFERENCE_ERROR`, `TRAIN_LEAKAGE`,
      `MODEL_TRAINING_FAILED`, `DATA_QUALITY_LOG_UNWRITABLE`).
- [ ] **All thresholds / bounds / filenames from `config.yaml`**; no magic numbers in
      `app/personalisation/engine.py`; `PersonalisationConfig` is `extra="forbid"` with the §5
      validators (each violation → `CONFIG_001`).
- [ ] INDEX row updated to DONE only when all of the above pass.

## 8. Test plan

- **Fixtures** (`tests/personalisation/`, mirroring `tests/ml/` conventions): a synthetic training
  matrix (≥ `min_clean_laps` clean laps for several `(driver, circuit)` cells across all four circuit
  types, held-out Belgium/Italy present with `split="test"`, a `lap_time_s` column with a controllable
  fastest-quartile), plus a real F08 `StyleSpaceHandle` built from the same drivers (reuse the F08
  build fixture). A module-scoped `trained` fixture runs `train_personalisation` ONCE. A hand-built
  `PersonalisationHandle` with known regressors (stub predictors returning fixed values) + known
  reliability for exact-value constraint/confidence assertions.
- **Happy path**: known driver × non-Monaco circuit → 10 parameters, ratings assigned, SHAP present,
  disclaimer present, `status=OK` when nothing is clipped/flagged.
- **Constraint fixtures** (one per §7.2 row): a stub regressor over-predicting front wing → clipped +
  `CONSTRAINT_CLIPPED`; Monaco → wings at max + `CIRCUIT_OVERRIDE_MONACO`; front-wing-high +
  brake-bias-low → `CONSISTENCY_WARNING` with values unchanged; a low-reliability parameter →
  `WIDE_CONFIDENCE` range = ±15% (exact), `value=None`, clipped to bounds.
- **Confidence/OOD**: a fingerprint far from the training centroid → attenuated confidence → more
  WIDE parameters than an in-distribution one; rating band boundaries at `confidence_medium` /
  `confidence_high` (hand-built reliability).
- **Onboarding**: real handle on a 10-lap unknown prefix → `fingerprint_available=True`, style_match
  size 3, SPARSE flag carried; stubbed `project_unknown` returning `coords_5d=None` → WIDE-heavy
  sheet, `fingerprint_available=False`, no crash; `project_unknown` throwing → onboarding still
  returns (never blocks).
- **Never-raises**: monkeypatch a regressor / SHAP / `project_unknown` to throw → typed
  `INFERENCE_ERROR` / degraded result, no exception; confidence still non-NULL.
- **Leakage & reproducibility**: Belgium row → `TRAIN_LEAKAGE` raise + no file; empty post-quarantine
  → `MODEL_TRAINING_FAILED`; two runs → equal reliability + equal `content_hash`; static no-`.fit`
  scan of the serving path; behavioural proof that targets were computed on quarantined training rows
  only.
- **Load & health**: corrupt/absent/hash-tampered/wrong-count/reliability-less/vocab-mismatch bundle →
  `PersonalisationLoadError(MODEL_NOT_FOUND)`; `health(None)` → `HEALTH_FAIL`.
- **Config validators**: reject `confidence_medium ≤ confidence_floor`, `wide_range_pct ≥ 1`, a
  missing FIA-bound key, `cv_folds < 2`, empty `circuit_types` (→ `CONFIG_001`).
- **Evaluation (F20 machinery, shipped-not-run)**: the reliability table + `n_reliable` reproduce the
  §14 "which 5 parameters are reliably inferrable" finding on synthetic data with known learnable vs
  noise targets.

## 9. Definition of done

All §7 criteria MET with passing tests, `/apex-review` clean on the contract checklist, `ml-auditor`
+ `failsafe-auditor` run on `app/personalisation/`, mypy clean, INDEX row DONE.

**Limitations to record in the dissertation (and divergences to flag to the supervisor — the running
list continues; F10 ended at #12):**

1. **Divergence #13 — setup-target approximation.** §7.1 says "approximated from telemetry signatures"
   with two examples; F11 defines all 10 targets as documented equal-weight normalised maps of the
   §7.3 fingerprint features onto the FIA range. The honest core limitation (§15.2): the system learns
   a *fingerprint → approximated-setup* mapping, not a *fingerprint → measured-setup* one.
2. **Divergence #14 — "best-performing stints".** Defined as the fastest-quartile clean laps per
   `(driver, circuit)` (config `top_quartile_pct`), telemetry-only.
3. **Divergence #15 — categorical parameters.** Suspension stiffness (Soft/Medium/Stiff) kept as "one
   regressor per parameter" via ordinal regression + round-to-nearest.
4. **Divergence #16 — circuit-type one-hot.** Four circuit types → drop-first (3-D) encoding, matching
   the master's "3D one-hot".
5. **Divergence #17 — regression confidence.** No native XGBoost confidence; defined as
   `1 − CV-nRMSE` per parameter, attenuated by input OOD-ness, with config bands.
6. **Limitation — 5 reliably-inferrable parameters (§14 risk).** Differential ramp and suspension
   stiffness are expected to be weakly inferrable from telemetry alone; F11 **reports this honestly**
   via the reliability table (does not silently narrow scope), and the confidence layer surfaces it as
   WIDE — this is an honest research contribution, not a hidden failure.

## 10. Risks & mitigations (from master risk register)

- **§14 row (verbatim):** "Personalisation engine recommendations too wide (all LOW confidence) —
  Likelihood Medium, Impact Medium — Contingency: Report as finding: certain setup parameters
  (differential ramp, suspension stiffness) are not reliably inferrable from telemetry alone. Narrow
  scope to the 5 parameters with highest mapping confidence. This is an honest research contribution.
  — Week 13 / Ongoing / 20."
  **How it maps here:** the per-parameter reliability table + WIDE rating make the trade-off explicit
  and measurable; narrowing to the top-5 is a **report/UI decision on top of the same artefact** (all
  10 are always computed and returned with honest confidence — the engine never hides a parameter),
  so the contingency is a one-line consumer filter, not a rebuild.
- **Approximation is not measurement** (§15.2) → the `approximation_disclaimer` is a required field on
  every recommendation; the dissertation states it in the limitations section.
- **OOD unknown drivers** (feeder-series projection outside the F1 distribution) → the OOD attenuation
  + carried `risk_flags` surface adaptation risk rather than over-confident numbers.
- **SHAP / projection faults** never block a recommendation (F07/F08 precedent) — the engine degrades
  (WIDE / no-SHAP) but always returns a typed, confidence-bearing result.
