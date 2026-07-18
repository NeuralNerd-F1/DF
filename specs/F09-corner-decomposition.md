# Feature F09 — Corner-Type Decomposition + Sub-Fingerprints (ablation)

> One spec, one session. This document is the single source of truth a session loads
> to build this feature. It must be self-contained: a session should not need the
> 42-page master spec open, only this file plus CLAUDE.md.

**Status:** SPEC WRITTEN
**Master spec sections:** §4.4 (Corner Type Decomposition — the three radius bands + the
`CORNER_001` failure), §5.1 (Model 1 classifier — the ablation's comparand), §8.4
(`lap_features.corner_type` ENUM), §10.3 (Screen 2 — the Slow/Medium/Fast corner tabs +
sub-fingerprint radar), §11.1 (classifier evaluation — the ablation is reported alongside),
§13 (Wk 9 row: "corner sub-fingerprints for all drivers · ablation result"), §14 (risk
register — "X/Y position data unavailable for corner decomposition"), §15 (dissertation:
"does corner-type information improve the classifier?")
**Layer / Week:** ML / Wk 9
**Depends on:** F03a (`_corner_indices` speed-minima apices, the telemetry channels + `_DQLedger`),
F03b (the curvature primitive `_signed_lateral_g` / `_xy_available` — X/Y → κ → lateral G),
F06a (`train_classifier` / the CV machinery the ablation reuses), F02/F01 (accepted clean-lap
telemetry frames). **F04-free** (divergence #23): the 5 corner sub-features are new derived
quantities F04 never scaled, so there is no `normalisation.json` entry to apply — see §4.0-f.

## 1. Purpose (one paragraph)

F09 segments each lap into **three corner-type categories** (§4.4) — **slow** (radius < 100 m),
**medium** (100–200 m), **high-speed** (> 200 m) — using the **curvature** already derived from the
X/Y trace by F03b, and computes a **feature sub-vector per corner type**. Aggregated per driver,
these are the **corner sub-fingerprints** that Screen 2's Slow/Medium/Fast tabs render (§10.3) and
that let the system "identify where driver differences are most pronounced" (§4.4). F09's headline
research deliverable is the **ablation** (§13/§15): does adding corner-type information to the lap
vector **improve the classifier's driver separation**? It is a research/analysis feature — it serves
sub-fingerprints and reports the ablation finding, and — per the failsafe contract — when the X/Y
trace is unavailable it **skips decomposition, logs `CORNER_001`, and the pipeline continues**
(the lap's F01–F26 are unaffected).

## 2. Upstream / downstream contracts

- **Consumes:**
  - **F03b** → the curvature primitive. `_signed_lateral_g(x, y, speed, cfg)` (`features.py:840`)
    computes **κ = |x'y'' − y'x''| / (x'² + y'²)^1.5** (via `np.gradient`) and returns the signed
    lateral G; `_xy_available(lap, cfg)` (`features.py:823`) is the shared X/Y precondition (≥ 3
    finite paired samples AND finite-fraction ≥ `features.min_xy_finite_fraction`). F09 reuses the
    **same κ** — the corner **radius r = 1/κ** — so a corner's classification is consistent with
    F03b's lateral-G units (divergence #22).
  - **F03a** → `_corner_indices(speed, cfg)` (`features.py:290`) = the speed-minimum **apices**, and
    the corner-window logic (the speed maxima bracketing each apex, `features.py:923`) — F09's corner
    segmentation is **exactly** F03b/F25's, so a "corner" means the same thing everywhere. Also the
    per-lap telemetry channels (`X`, `Y`, `Speed`, `Brake`, `Throttle`), `_DQLedger`, `_numeric`,
    `_finite`.
  - **F04** → NOT consumed (divergence #23). The original plan z-scored the aggregate via
    `normalisation.json`, but the 5 corner sub-features (apex speed, lateral G, throttle, brake,
    reversal rate) are **new derived quantities F04 never saw** — `normalisation.json` holds no
    params for them, so there is nothing to apply. The ablation scores via XGBoost (scale-invariant)
    and the Screen-2 radar is min-max-scaled per axis at render (F19), so no consumer needs the
    aggregate pre-scaled. F09 stores the raw sub-vector only; there is no `NORMALISATION_FALLBACK`
    path.
  - **F06a** → `train_classifier` / the CV machinery (`app/ml/classifier.py`) — the ablation reuses
    the **StratifiedGroupKFold-by-circuit** CV + `_base_xgb_params` (seed `config.seeds.xgboost`,
    `n_jobs=1`) to score **base vs expanded** feature matrices (§4.4-c). Leakage-quarantined
    identically (held-out circuits + non-train split dropped before any fit).
  - **Training matrix** — the F06a-era wide matrix (`driver`, `circuit`, `season`, `split`,
    `F01..F28`) is the ablation **base**; F09 builds the **expanded** matrix by appending the
    per-lap corner sub-features (§4.1). The per-lap decomposition itself needs the **telemetry
    frames** (X/Y etc.), so `decompose_lap` runs at the feature stage, not on the aggregated matrix.
- **Produces:**
  - Typed per-lap `CornerDecompositionResult` — the three corner-type sub-vectors (`float | None`),
    per-type corner counts, and the `CORNER_001` skip flag. Never raises.
  - `artefacts/corner_subfingerprints.json` — per-`(driver, corner_type)` **sub-fingerprint**
    aggregates (mean sub-vector + n_corners + n_laps), content-hashed + deterministic like F04/F10
    (the Screen-2 radar source), written by `build_sub_fingerprints` and nothing else.
  - `AblationReport` (typed) — base vs expanded **CV macro-F1**, the **delta**, per-driver deltas,
    and the honest **finding** (`IMPROVES` / `NEUTRAL` / `DEGRADES` by a config margin). Logged to
    MLflow experiment `apex-ablation` (own URI key, F08/F10/F11 precedent); formally run in F20.
  - Typed `SubFingerprintFitResult`, `CornerHealthResult`.
- **Consumed by:**
  - **F19** — Screen 2's Slow/Medium/Fast corner tabs re-animate the radar from the sub-fingerprints.
  - **F20** — runs the ablation over held-out circuits (read-only) and reports the §11.1 finding.
  - **F08** (optional, future) — a corner-decomposed style space is an F20-era extension; F09 ships
    the sub-fingerprints, not a second UMAP.

## 3. Inputs & outputs (typed)

Feature-owned enum lives in `app/ml/corners.py` (per the `enums.py` docstring):

```python
class CornerType(StrEnum):        # §4.4 — the three radius bands + the master's §8.4 ENUM tokens
    SLOW = "SLOW"                 # apex radius < corners.radius_slow_max (100 m)
    MEDIUM = "MEDIUM"            # radius_slow_max ≤ radius < corners.radius_medium_max (200 m)
    HIGH_SPEED = "HIGH_SPEED"    # radius ≥ radius_medium_max
```

### Input — per-lap telemetry frame / training matrix

`decompose_lap` takes ONE accepted clean-lap telemetry frame (the same `X`/`Y`/`Speed`/`Brake`/
`Throttle` frame F03b consumes) + `lap_number`/`driver`/`circuit` identity kwargs.
`build_sub_fingerprints` / `run_ablation` take the wide training matrix (F06a's `_REQUIRED_COLUMNS`)
**plus a per-lap decomposition** — because the sub-features are per-lap, the caller supplies the
decomposition alongside the matrix (§4.1), or `build_sub_fingerprints` is handed the telemetry frames
keyed by lap. `run_ablation` reuses the classifier matrix; missing columns → typed
`MODEL_TRAINING_FAILED` (build) / `INVALID_INPUT` (serve-shaped) result.

### Output — result objects (Pydantic on `BaseResult` unless noted, `extra="forbid"`)

```python
# The per-corner-type sub-feature set (§4.1, divergence #19): the corner-scoped features that carry
# a driver's cornering signature — computed over ONLY that corner type's corners.
CORNER_SUB_FEATURES = ("apex_speed_mean", "lateral_g_mean", "throttle_mean",
                       "brake_mean", "steering_reversal_rate")   # 5 per corner type

class CornerSubVector(BaseModel):                  # one corner type's sub-features for one lap
    corner_type: CornerType
    n_corners: int                                 # corners of this type in the lap
    features: dict[str, float | None]              # exactly CORNER_SUB_FEATURES; None if n_corners==0

class CornerDecompositionResult(BaseResult):       # per lap — never raises
    lap_number: int
    sub_vectors: dict[str, CornerSubVector] = {}   # keyed by CornerType; always all 3 keys
    xy_available: bool = False
    n_corners_total: int = 0
    # status: OK (decomposed), DEGRADED (some type had 0 corners), FAILED never — a skip is DEGRADED
    # error_code: CORNER_001 when xy_available is False (operative-code-on-DEGRADED, F08/F10 precedent)

class SubFingerprint(BaseModel):                   # per (driver, corner_type) aggregate
    driver: str
    corner_type: CornerType
    features: dict[str, float | None]              # mean sub-vector over the driver's corners of type
    n_corners: int
    n_laps: int

class SubFingerprintFitResult(BaseResult):         # build output
    artefact_path: str = ""
    content_hash: str = ""
    n_drivers: int = 0
    n_entries: int = 0                             # (driver, corner_type) cells written
    coverage: dict[str, float] = {}               # corner_type → fraction of drivers with the cell
    mlflow_run_id: str | None = None

class AblationReport(BaseResult):                  # §4.4-c — read-only, never raises (run in F20)
    base_macro_f1: float | None = None
    expanded_macro_f1: float | None = None
    delta: float | None = None                     # expanded − base
    per_driver_delta: dict[str, float] = {}
    finding: str = ""                              # "IMPROVES" | "NEUTRAL" | "DEGRADES" (§4.4-c margin)
    n_base_features: int = 0                        # 28
    n_expanded_features: int = 0                    # 28 + 3·len(CORNER_SUB_FEATURES)
    mlflow_run_id: str | None = None

class CornerHealthResult(BaseResult):              # mirrors F06b/F08/F10/F11 health
    ready: bool = False
    model_version: str = ""
    reason: ErrorCode | None = None
```

`CornerHandle` is a frozen dataclass (mirrors the loaded-artefact handles): the parsed
`corner_subfingerprints.json` (per-`(driver, corner_type)` cells), `sub_features`,
`model_version` (artefact content hash), `ready`. **F04-free** — no `normalisation_params`
(divergence #23).

**Errors raised (only at build/load):** `SubFingerprintBuildError` (`TRAIN_LEAKAGE` /
`MODEL_TRAINING_FAILED`) from `build_sub_fingerprints`; `CornerLoadError` (`MODEL_NOT_FOUND`) from
`load_sub_fingerprints`. `decompose_lap`, `run_ablation`, `health` **never raise**.

**Naming guard:** `corner_type` is F03a's existing `lap_features` column concept (§8.4); F09's
`CornerType` enum names the same tokens. The class names above are final.

## 4. Method / algorithm

### 4.0 Master-fidelity reconciliations (read first)

- **(a) Corner classification is per-CORNER by apex radius — divergence #18.** §4.4 says "laps are
  segmented into three corner-type categories using curvature." F09 classifies each **corner** (from
  `_corner_indices`), not each sample: a corner's **radius `r = 1 / κ at the apex`** — the apex (the
  speed-minimum sample) is the corner's defining tightest point, and a mid-window central-difference
  κ is **robust** to the boundary/window transition artefacts that dominate `max(κ)` (a fall-back to
  `1/max(κ over the window)` covers a non-finite apex κ). `r < radius_slow_max` → SLOW,
  `< radius_medium_max` → MEDIUM, else HIGH_SPEED. Per-sample classification would shatter a single
  corner across bands; per-corner keeps a corner whole (and matches the master's whole-corner
  examples). A corner with no real curvature at the apex (a straight mis-flagged as an apex) is
  excluded.
- **(b) Corner windows are F03b/F25's exact segmentation.** F09 reuses `_corner_indices` + the speed-
  maxima bracketing (`features.py:923`) so "a corner" is identical to F25's. Each corner's window is
  half-open `[left_max, right_max)` (no sample double-counted); the sub-features are computed over the
  union of a corner type's windows.
- **(c) The ablation is a self-contained CV comparison — divergence #21.** §4.4/§15 ask "does
  corner-type information improve the classifier?" F09 answers with the **same CV protocol as F06a**
  (`StratifiedGroupKFold`-by-circuit, seed 42, `_base_xgb_params`) scoring two matrices: **base** =
  the 28 `FEATURE_IDS`; **expanded** = base ‖ the 3 corner sub-vectors (3·5 = 15 extra columns,
  per-lap, imputed like any feature). It reports **mean CV macro-F1** for each, the **delta**, and a
  `finding` (`IMPROVES` if `delta > ablation_margin`, `DEGRADES` if `< −ablation_margin`, else
  `NEUTRAL`). It does **not** retrain/replace the F06 champion — it is a research report (F20 runs it
  over held-out; held-out never touches a `.fit` outside that read-only evaluation). This keeps the
  ablation honest and cheap without perturbing the shipped classifier.
- **(d) `CORNER_001` is the skip, not a failure (§4.4).** X/Y unavailable (`_xy_available` False) →
  decomposition skipped, `status=DEGRADED`, `error_code=CORNER_001`, all three sub-vectors NULL, the
  lap **continues**. This is F03b's exact behaviour for the geometry features — F09 adds no new
  rejection. A per-lap DQ row (`CORNER_001`) is written (reusing F03b's code — same meaning).
- **(e) Statelessness / never-raises.** `decompose_lap` is a pure function of the lap frame;
  `run_ablation`/`health` never raise (typed result on any fault, F06b/F08/F10/F11 precedent).
- **(f) Sub-fingerprint scale — F04-free (divergences #20 + #23).** The sub-features mix units
  (km/h apex speed, G, 0–1 fractions, rate), but F09 stores them **raw**, not F04-normalised: the
  5 corner sub-features are new derived quantities F04 never scaled (`normalisation.json` holds no
  params for them), the ablation scores via scale-invariant XGBoost, and the Screen-2 radar is
  min-max scaled per axis at render (F19's concern). A future cosine/UMAP consumer would scale at
  its own boundary. There is **no `NORMALISATION_FALLBACK` path** in F09.

### 4.1 Per-lap decomposition — `decompose_lap`

1. **X/Y gate**: `_xy_available(lap, config)` False → return `CornerDecompositionResult(status=DEGRADED,
   error_code=CORNER_001, xy_available=False, sub_vectors={all 3 → CornerSubVector(n_corners=0,
   features=all-None)})` + one `CORNER_001` DQ row. Continue (the lap's F01–F26 are F03's, untouched).
2. **Curvature + corners**: κ per sample via F03b's formula; `corners = _corner_indices(speed, cfg)`;
   per corner, window `[left_max, right_max)`, `r = 1 / κ[apex]` (fall back to `1/max(κ[window])` if
   the apex κ is non-finite; skip the corner if neither is a real curvature).
3. **Classify** each corner → `CornerType` by `r` vs `radius_slow_max` / `radius_medium_max`.
4. **Per-type sub-features** (§4.1 `CORNER_SUB_FEATURES`), over the union of that type's windows:
   `apex_speed_mean` (mean of the apex-sample speeds), `lateral_g_mean` (mean |signed G|),
   `throttle_mean`, `brake_mean` (mean over the windows), `steering_reversal_rate` (F21's sign-
   reversal count over the windows ÷ their wall-time). A type with **0 corners** → that
   `CornerSubVector` has `features` all-None, `n_corners=0`; the result is `DEGRADED` (a lap rarely
   has all three types). A per-feature compute fault → that key NULL + a `FEATURE_001` DQ row (F03
   precedent), never a raised exception.
5. **Assemble**: all three `CornerType` keys always present; `status=OK` iff all three types had ≥ 1
   corner AND no NULL sub-feature, else `DEGRADED` (never `FAILED`).

### 4.2 Sub-fingerprints — `build_sub_fingerprints` (the aggregate writer)

Group the per-lap decompositions by `(driver, corner_type)`; **quarantine held-out + non-train rows
first** (F04/F06a/F08/F10/F11 pattern; `_assert_no_leakage` tripwire → `TRAIN_LEAKAGE`, no artefact);
per cell take the corner-count-weighted **mean sub-vector** over the driver's corners of that type
(raw sub-features, §4.0-f), require ≥ `min_corners_per_type` corners else the cell is **dropped +
`DATA_INSUFFICIENT`** and the build result goes `DEGRADED` (surfaced on the typed result, not only
in the DQ log). Write `corner_subfingerprints.json` — canonical JSON (sorted keys, no timestamp,
embedded `content_hash`, byte-identical on re-run, F04/F10 precedent). MLflow-log (degrade-don't-die).
No usable cells → `SubFingerprintBuildError(MODEL_TRAINING_FAILED)`.

### 4.3 Ablation — `run_ablation` (§4.4-c, read-only, never raises)

Quarantine (held-out + non-train dropped before any fit). Build **base** (28 `FEATURE_IDS`) and
**expanded** (base ‖ the 15 per-lap corner sub-features) matrices; a corner type absent from a lap
leaves its columns NaN and XGBoost consumes NaN natively (no imputation). Score both with the **F06a
CV** (`StratifiedGroupKFold(cv_folds, seed=42)`, macro-F1). `delta = expanded − base`; `finding` by `ablation_margin`. Log to `apex-ablation`. Malformed
input → typed `INVALID_INPUT` `AblationReport`. **Held-out circuits enter only via F20's invocation.**

### 4.4 Load & health — `load_sub_fingerprints` / `health`

`load_sub_fingerprints` reads `corner_subfingerprints.json` (schema + content-hash validated) and
`normalisation.json` (F04 loader); missing / corrupt / hash- or schema-mismatch / **empty cell
table** → DQ-log then **raise** `CornerLoadError(MODEL_NOT_FOUND)`. `health(handle)` → not-ready
reports `HEALTH_FAIL` with `MODEL_NOT_FOUND` reason (F13 → 503).

## 5. config.yaml keys this feature adds

F09 adds a typed `corners:` block (`CornersConfig` in `app/contracts/config.py`, `extra="forbid"`,
standing rule) + optional `corners: CornersConfig | None = None` on `AppConfig`.

```yaml
corners:                             # F09 §4.4 — corner-type decomposition + ablation
  radius_slow_max: 100.0             # §4.4 — apex radius (m) below this → SLOW corner
  radius_medium_max: 200.0           # §4.4 — below this → MEDIUM; at/above → HIGH_SPEED
  min_corners_per_type: 5            # sub-fingerprint cell needs ≥ N corners of the type (div #20)
  ablation_margin: 0.01              # macro-F1 delta band for IMPROVES/NEUTRAL/DEGRADES (div #21)
  cv_folds: 5                        # ablation CV (mirrors classifier.cv_folds)
  subfingerprint_artefact_filename: corner_subfingerprints.json
  mlflow_tracking_uri: sqlite:///artefacts/mlflow.db   # F09's OWN URI (same store, own experiment)
  mlflow_experiment: apex-ablation
```

**Referenced, not re-declared** (house rule): `features.min_xy_finite_fraction` + the F03b geometry
keys (`gravity_m_s2`, `steering_g_deadband`, the `_corner_indices` machinery), `held_out_circuits`,
`seeds.xgboost` (ablation CV), `paths.artefacts_dir` / `paths.data_quality_log`. (F09 is F04-free —
it references **no** `normalise.*` key; divergence #23.)

**Validator placement** (F08/F10/F11 precedent): scalar sanity on `CornersConfig`
(`0 < radius_slow_max < radius_medium_max`; `min_corners_per_type ≥ 1`; `0 < ablation_margin < 1`;
`cv_folds ≥ 2`; filenames + mlflow keys non-empty); cross-block on `AppConfig` (`corners` present
requires `features` — the curvature / `_corner_indices` keys F09 reads). All violations →
load-time `CONFIG_001`.

**No new error code.** `CORNER_001` is already seeded (`app/contracts/errors.py:59`, "curvature / X-Y
unavailable → skip decomposition"); F09 reuses `FEATURE_001` (per-feature NULL), `DATA_INSUFFICIENT`
(thin cell), `INVALID_INPUT`, `MODEL_NOT_FOUND`, `HEALTH_FAIL`, `TRAIN_LEAKAGE`,
`MODEL_TRAINING_FAILED`, `DATA_QUALITY_LOG_UNWRITABLE`.

## 6. Error codes & failure modes (the failsafe contract)

Master §4.4 row is marked ▣. Placement follows the house discipline (operative code in `error_code`
on the non-scoring DEGRADED outcome `CORNER_001`; `flags` on scored-but-degraded).

| Failure mode | Detection | Error code (placement) | System response | Logged? | User-facing message |
|---|---|---|---|---|---|
| ▣ X/Y trace unavailable | `_xy_available` False | `CORNER_001` (error_code) | Skip decomposition; `status=DEGRADED`; all 3 sub-vectors NULL; lap continues (F01–F26 intact) | yes | "Corner decomposition skipped — position data unavailable." |
| A corner type has 0 corners in a lap | Per-type corner count | — (`flags` empty; NULL sub-vector) | That `CornerSubVector` NULL; `status=DEGRADED` | no (normal) | — (that tab shows "no data for this lap") |
| A corner sub-feature cannot be computed | Per-feature guard | `FEATURE_001` (per-feature flag) | That key NULL; lap continues | yes | — (internal) |
| Thin `(driver, corner_type)` cell at build | < `min_corners_per_type` | `DATA_INSUFFICIENT` (flag) | Cell NULL / dropped; `status=DEGRADED` | yes | build log |
| Malformed lap frame (missing X/Y columns) | Same `_xy_available` gate | `CORNER_001` (error_code) | Treated as X/Y-unavailable skip | yes | as above |
| Malformed ablation matrix (missing feature columns) | Input gate | `INVALID_INPUT` (error_code) | Typed `AblationReport` `FAILED`; never raises | yes | — (build/eval tooling) |
| `corner_subfingerprints.json` missing / corrupt / hash or schema mismatch / empty cells | Load-time validation | `MODEL_NOT_FOUND` | DQ-log then **raise** `CornerLoadError` — refuse to serve | yes | 503 via F13 |
| Detector not loaded at health check | `health()` | `HEALTH_FAIL` (reason `MODEL_NOT_FOUND`) | `ready=False` | yes | 503 from `GET /health` |
| Held-out/non-train row survives quarantine | `_assert_no_leakage` tripwire | `TRAIN_LEAKAGE` | **Raise**; no artefact | yes | build log (CI-breaking) |
| No usable cells / missing columns / write fault | Guards in `build_sub_fingerprints` | `MODEL_TRAINING_FAILED` | **Raise**; no artefact | yes | build log |
| MLflow logging fault (build / ablation) | `except` around the MLflow calls | `MODEL_TRAINING_FAILED` (DQ row only, F08 mechanism) | `status=DEGRADED`, `mlflow_run_id=None`; artefact unaffected | yes | build log |
| Unforeseen `decompose_lap` / `run_ablation` fault | Backstop `except` at the boundary | `CORNER_001` (decompose) / `INVALID_INPUT` (ablation) | Best-effort DQ log → typed result; **never a raised exception** | best-effort | — |
| DQ log sink unwritable | `DataQualityLogError` in `_DQLedger.record` | `DATA_QUALITY_LOG_UNWRITABLE` (flag) | Never propagates; result flagged | n/a | — |

## 7. Acceptance criteria (each mechanically testable by /spec-check)

- [ ] **Three corner types by apex radius** (§4.4): each corner classified by `r = 1/κ[apex]` —
      `< radius_slow_max` (100) → SLOW, `< radius_medium_max` (200) → MEDIUM, else HIGH_SPEED; a
      straight (no real apex curvature) is not a corner. Corner segmentation is F03b/F25's
      `_corner_indices` + speed-maxima windows (a parity test: the same corners F25 uses).
- [ ] **Every lap yields all three `CornerType` sub-vectors** (never a silent missing key); a type
      with 0 corners → NULL sub-vector (`n_corners=0`), `status=DEGRADED`.
- [ ] **Sub-features = `CORNER_SUB_FEATURES`** computed over ONLY that corner type's windows (a lap
      with corners only in one band leaves the other two NULL — behavioural proof).
- [ ] **X/Y unavailable → `CORNER_001`** (operative `error_code`, `status=DEGRADED`), all 3 NULL, one
      DQ row, lap continues; `decompose_lap` never raises (fault injection → `CORNER_001` result).
- [ ] **Sub-fingerprints**: per-`(driver, corner_type)` corner-count-weighted mean sub-vector; a cell
      with < `min_corners_per_type` corners → dropped + `DATA_INSUFFICIENT` + the build result goes
      `DEGRADED` (surfaced on the typed `SubFingerprintFitResult`, not only the DQ log); artefact
      byte-identical on re-run (no timestamp; content hash equal); raw sub-features, F04-free
      (divergence #23 — no `NORMALISATION_FALLBACK` path).
- [ ] **Leakage**: `build_sub_fingerprints` / `run_ablation` quarantine held-out + non-train **before
      any reduction/fit**; a poisoned matrix raises `TRAIN_LEAKAGE` with no artefact; a static scan
      asserts no `.fit(` in the serving path (`decompose_lap`/`load_sub_fingerprints`/`health`).
- [ ] **Ablation** (§4.4-c): `run_ablation` scores base (28) vs expanded (28 + 15) with the F06a CV
      (`StratifiedGroupKFold`, seed 42); reports `base_macro_f1`, `expanded_macro_f1`, `delta`,
      `per_driver_delta`, and a `finding` ∈ {IMPROVES, NEUTRAL, DEGRADES} by `ablation_margin`;
      read-only (no `.fit` on held-out), never raises, typed `INVALID_INPUT` on malformed input;
      MLflow-logged to `apex-ablation`.
- [ ] **Reproducibility**: two ablation runs on the same matrix + seeds → equal macro-F1 (1e-9);
      all seeds from `config.seeds`.
- [ ] **Load refuses** (`MODEL_NOT_FOUND` after a DQ row) on missing / corrupt / hash- or
      schema-mismatch / empty-cell-table; `health(None)` → `HEALTH_FAIL` with reason `MODEL_NOT_FOUND`.
- [ ] **Every §6 code path** has a test that triggers it and asserts the typed result + code
      placement (`CORNER_001`, `FEATURE_001`, `DATA_INSUFFICIENT`, `INVALID_INPUT`, `MODEL_NOT_FOUND`,
      `HEALTH_FAIL`, `TRAIN_LEAKAGE`, `MODEL_TRAINING_FAILED`, `DATA_QUALITY_LOG_UNWRITABLE`).
- [ ] **All thresholds/filenames from `config.yaml`** (`corners:` + referenced keys); no magic numbers
      in `app/ml/corners.py`; `CornersConfig` `extra="forbid"` with the §5 validators (→ `CONFIG_001`).
- [ ] INDEX row updated to DONE only when all of the above pass.

## 8. Test plan

- **Fixtures** (`tests/ml/test_corners.py`, mirroring `tests/ml/` conventions): a synthetic lap frame
  generator producing an X/Y trace with **controllable corner geometry** — a hairpin (r < 100),
  a medium sweep (100–200), a fast kink (> 200), and a straight — so the classifier bands and the
  per-type sub-features are exactly assertable; plus X/Y-absent and X/Y-sparse frames. A small
  training matrix (reusing `tests/ml/conftest.build_matrix`) + per-lap decompositions for the
  sub-fingerprint + ablation fixtures; a hand-built `corner_subfingerprints.json` for exact load tests.
- **Happy path**: a lap with one corner of each type → three non-NULL sub-vectors, correct band per
  corner, `apex_speed`/`lateral_g` monotone with radius; `status=OK`.
- **Failure fixtures** (one per §6 row): X/Y-absent → `CORNER_001` skip (all NULL, lap continues);
  X/Y-sparse (< `min_xy_finite_fraction`) → `CORNER_001`; a lap with only slow corners → medium/high
  NULL, `DEGRADED`; a poisoned build matrix (Belgium) → `TRAIN_LEAKAGE` + no file; thin cell →
  `DATA_INSUFFICIENT`; empty post-quarantine → `MODEL_TRAINING_FAILED`; unwritable log → flagged, no
  raise; corrupt/absent/hash-tampered artefact → `CornerLoadError`; `health(None)` → `HEALTH_FAIL`;
  malformed ablation matrix → typed `INVALID_INPUT` `AblationReport`; monkeypatched internals raising
  → `decompose_lap` returns `CORNER_001`, `run_ablation` returns `INVALID_INPUT`, no exception.
- **Leakage & reproducibility**: matrix with a Belgium row → `TRAIN_LEAKAGE` raise + no file on disk;
  double-run byte-identical `corner_subfingerprints.json`; static no-`.fit` scan of the serving path;
  ablation macro-F1 equal across two runs (1e-9); sub-fingerprint built on quarantined training rows
  only (behavioural proof).
- **Corner-segmentation parity**: the corners F09 classifies are exactly `_corner_indices` + the F25
  windows (assert against F03b on the same lap — F09 never re-segments).
- **Ablation**: a matrix where corner sub-features carry real driver signal → `delta > ablation_margin`
  → `IMPROVES`; a matrix where they are pure noise → `NEUTRAL`; per-driver deltas populated.

## 9. Definition of done

All §7 criteria MET with passing tests, `/apex-review` clean on the contract checklist, `ml-auditor`
+ `failsafe-auditor` run on `app/ml/corners.py`, mypy clean, INDEX row DONE.

**Limitations to record in the dissertation (divergences to flag to the supervisor — the running
list continues; F11 ended at #17):**

1. **Divergence #18 — per-corner (not per-sample) classification.** A corner is classified whole by
   its apex radius `1/max(κ)`; per-sample banding would fragment a corner. Matches the master's
   whole-corner examples.
2. **Divergence #19 — the corner sub-feature set.** §4.4 says "a feature sub-vector" without naming
   the features; F09 defines the 5 corner-scoped `CORNER_SUB_FEATURES` (apex speed, lateral G,
   throttle, brake, steering-reversal rate) — the features that carry a cornering signature.
3. **Divergence #20 — sub-fingerprint aggregation + `min_corners_per_type`.** Per-`(driver,
   corner_type)` mean over the F04-normalised frame; a cell needs ≥ 5 corners (a mean over one corner
   is not a fingerprint).
4. **Divergence #21 — ablation method.** A self-contained F06a-CV comparison of base vs expanded
   matrices reporting a macro-F1 delta + finding; it does NOT retrain the shipped champion. Cheap,
   honest, non-perturbing. The expanded matrix is **not imputed** — a lap with no corners of a type
   leaves those columns NaN and XGBoost consumes NaN natively (the original spec's "median-imputed"
   step is dropped; a test exercises the NaN-bearing production shape).
5. **Divergence #22 — radius units.** `r = 1/κ` reuses F03b's curvature (validated to keep lateral G
   in 0–6 G), so `r` is in the same length unit as F03b's X/Y; the 100 m / 200 m thresholds are
   config-tunable if a session's X/Y scale differs. The classification is internally consistent with
   F03b regardless.
6. **Divergence #23 — F04-free aggregate (supersedes the spec's original F04-normalised plan).** The
   5 corner sub-features are new derived quantities F04 never scaled, so `normalisation.json` holds no
   params for them — there is nothing to apply. The ablation scores via scale-invariant XGBoost and
   the Screen-2 radar min-max-scales per axis at render (F19), so no consumer needs a pre-scaled
   aggregate. F09 drops the F04 dependency and the `NORMALISATION_FALLBACK` path entirely, stores the
   raw sub-vector, and `CornerHandle` carries no `normalisation_params`; the cross-block validator
   requires only `features`.
7. **Limitation — corner decomposition depends on X/Y quality** (§14): sessions without a usable X/Y
   trace get no sub-fingerprints (`CORNER_001`), reported honestly per session; the ablation is run on
   the subset with X/Y.

## 10. Risks & mitigations (from master risk register)

- **§14 row (verbatim):** "X/Y position data unavailable for corner decomposition — Likelihood Medium,
  Impact Low — Corner decomposition skipped for affected sessions. Features F19–F21 and F25 set to
  NULL. — Week 9." **How it maps here:** `CORNER_001` skip is the whole failure path; F19–F21/F25 NULL
  is F03b's existing behaviour (F09 adds no new NULL); the ablation runs on the X/Y-available subset
  and reports the coverage honestly.
- **Corner decomposition may NOT improve the classifier** (§15 asks the question openly). The ablation
  is designed to report a **NEUTRAL or DEGRADES** finding as a legitimate research result, not a
  failure — the `finding` field + per-driver deltas make "it doesn't help" a first-class, publishable
  outcome, consistent with the dissertation's honest-findings ethos.
- **Sparse corner types on some circuits** (a street circuit may have few high-speed corners) →
  `min_corners_per_type` + `DATA_INSUFFICIENT` surfacing + per-corner-type coverage as a build metric,
  so thin sub-fingerprints are visible at build time, not discovered at serve.
