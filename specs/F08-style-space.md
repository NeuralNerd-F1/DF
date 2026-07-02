# Feature F08 — Driver Style Space (UMAP · PCA fallback · unknown-driver projection)

> One spec, one session. Self-contained: build this from this file + CLAUDE.md only.
> F08 is **Model 2**. It projects each driver's aggregate 28-feature distribution into a **continuous
> style space** — a **5D fingerprint coordinate** (consumed by F10/F11) and a **2D visualisation
> projection** (Screen 5). The space is fit on **normalised training laps only**; **unknown drivers
> project in by a forward pass** (`transform()`, no retrain). Every projection carries an explicit
> confidence gate (sparse / out-of-distribution) and a **fail-loud fallback chain** (UMAP → PCA →
> raw-space nearest-neighbour). Like F07, F08 introduces **no new error code** — the four §5.2 codes
> and the `OUT_OF_DISTRIBUTION` `ConfidenceState` are already seeded.

**Status:** SPEC WRITTEN
**Master spec sections:** §5.2 (Model 2 — Driver Style Space: algorithm, 5D/2D output, unknown-driver
projection, model persistence, stability validation, teammate proximity), §5.2 *Model 2 Failure Modes
& Responses* (the four codes), §3.4 (Unknown-Driver upload edge case → project into style space with
uncertainty bounds + nearest-neighbour matches), §7.4 (Unknown-Driver Onboarding Output: style-match
report, risk flags), §8.2 (module `app/ml/stylespace.py`: "PCA fallback if UMAP fails; out-of-distribution
check on every projection"), §8.4 (`fingerprints.fingerprint_vector`; `drivers`: 8 known + unlimited
unknown), §8.5 (`GET /stylespace`, `POST /fingerprint/unknown`, `GET /compare` `compatibility_delta`),
§11.2 (Style-space evaluation: unknown-driver NN 6/8, UMAP Procrustes variance < 0.05, intra-team <
inter-team), §11.4 (Failsafe test: UMAP out-of-distribution → warning, dashed marker, no crash),
Risk Register row *"UMAP projection unstable for unknown drivers"*, §15.2 (limitation: UMAP is
stochastic).
**Layer / Week:** ML / Wk 8
**Depends on:** **F04** (`artefacts/normalisation.json` — the leakage chokepoint; F08 consumes it
**read-only** via `load_normalisation_params` / `apply_normalisation`, never refits). The real
`normalisation.json` first materialised in the **F06a era (DONE)**, so the artefact exists by the time
F08 is built. F08 also consumes the same **wide training feature matrix** F06a trains on
(`driver`/`circuit`/`season`/`split` + the 28 `FEATURE_IDS`) — F08 does **not** depend on the F06
classifier itself (INDEX: `F08 … depends on F04`), only on the shared assembled matrix + `FEATURE_IDS`.

## 1. Purpose
Every APEX driver must have a **coordinate in one universal style space** so drivers can be compared,
degradation measured against a fingerprint (F10), and cars personalised (F11) — and so a **driver never
seen in training can be placed in the same space without retraining** (the master's *Contribution 3 —
Cross-Series Style Transfer*, RQ4). F08 aggregates each driver's clean **normalised** training laps to
a 56-dim distribution (mean + std of the 28 features), fits **UMAP** to a **5D fingerprint** embedding
and a **2D visualisation** embedding, and persists both plus a **PCA fallback** artefact. An unknown
driver is feature-extracted → normalised with the **same saved params** → aggregated → **projected by a
forward pass** (`umap.transform`), gated for **sparsity** (`SPARSE_PROJECTION`, <30 laps) and
**out-of-distribution** (`OUT_OF_DISTRIBUTION`, Mahalanobis > 3 SD on any principal component), and
returned with its **top-3 nearest known drivers** (cosine, 5D). Per the failsafe contract, a corrupt/
missing UMAP model degrades to **PCA** (`UMAP_FALLBACK`) and a failing `transform()` degrades to
**raw-space cosine nearest-neighbour** (`PROJECTION_FALLBACK`) — the projection **never crashes and
never silently returns a wrong coordinate**.

## 2. Upstream / downstream contracts
- **Consumes (the seams):**
  - **F04 `artefacts/normalisation.json`** via `load_normalisation_params` (fail-loud read + hash verify)
    and `apply_normalisation` (pure, per-vector, never-refit transform). F08 applies the **exact same
    per-circuit centre/scale** the classifier saw; an unknown circuit → global params
    (`NORMALISATION_FALLBACK`), exactly as F06b. **No `.fit` on normalisation, ever** (non-negotiable #2).
  - **The wide training feature matrix** (`pd.DataFrame`) — columns `driver`, `circuit`, `season`,
    `split`, and the 28 `FEATURE_IDS` (`F01…F28`), one row per clean lap. Same shape F06a's
    `train_classifier` consumes; the caller (a training script / F20) assembles it. F08 quarantines the
    **held-out circuits (Belgium, Italy) and non-`train` split rows BEFORE any `.fit`** and a fail-loud
    `_assert_no_leakage` tripwire aborts the build if any survive.
  - `FEATURE_IDS` (28, from `app/pipeline/features.py`); `config.seeds.numpy` (42),
    `config.held_out_circuits`, `config.thresholds.min_clean_laps` (30 — the sparse floor).
- **Produces:**
  - `artefacts/umap_model.pkl` — a joblib bundle of the **two fitted UMAP models** (`umap_5d`,
    `umap_2d`) + build metadata (seeds, n_neighbors, feature order). A **sidecar** `umap_model.pkl.sha256`
    holds the digest verified at load (corruption/tamper detection → `UMAP_FALLBACK`).
  - `artefacts/pca_fallback.pkl` — a joblib bundle of the fitted **`PCA(n_components=5)`** (the 5D/2D
    fallback engine **and** the OOD principal-component basis), the **per-PC training std** (for the
    Mahalanobis/PC check), the **known-driver 56-dim aggregate matrix** (for the raw-NN fallback), and the
    **per-feature known-driver mean/std** (for `risk_flags`). Deterministic (PCA is not stochastic).
  - `StyleSpaceBuildResult` — offline build summary: known-driver `coords_5d` / `coords_2d`, artefact
    paths + hash, `n_drivers`, `n_features` (56), effective `n_neighbors`, `stability`, `proximity`.
  - `StyleSpaceHandle` — the loaded, immutable serving bundle (both UMAP models + PCA + known coords +
    OOD stats + normalisation params + `engine` + `ready`), built once at startup (mirrors F06b
    `ModelHandle`).
  - `ProjectionResult` — one unknown-driver forward pass: `coords_5d`, `coords_2d`,
    `nearest_neighbours` (top-3), `confidence_state`, sparse/OOD flags, `projection_engine`, `risk_flags`.
  - `StyleSpaceResult` — all known drivers' `coords_5d` / `coords_2d` for `GET /stylespace`.
  - `StabilityReport` (10-seed Procrustes variance) and `TeammateProximityResult` (intra- vs inter-team
    5D distance) — **dissertation-evidence functions** (§5.2 "Reported in dissertation"; §11.2). "Failure
    of these tests is a meaningful research finding, not a system failure" — they return a `passes` bool
    and **never raise / never block**.
- **Consumed by:**
  - **F10** (degradation) — reads the **5D fingerprint** as the within-stint baseline coordinate
    (`INDEX`: "F08 feeds F10"); F10 declares a dependency on F08.
  - **F11** (personalisation, §7.4) — reads the **top-3 nearest known drivers** (style-match report),
    the **5D coordinate**, and the **`risk_flags`** (feature dims outside the known-driver distribution).
  - **F13** — serves `GET /stylespace` (`{drivers:[{id,code,coords_2d,coords_5d,season}], unknown_drivers}`),
    `POST /fingerprint/unknown` (**200** with `SPARSE_PROJECTION` warning if <30 laps; the **<5-lap → 400**
    hard reject and the **OOD-warning → dashed-marker** UX are F13's HTTP mapping, not F08's — F08 always
    computes + flags), and `GET /compare/{id1}/{id2}` **`compatibility_delta`** (5D-distance-derived);
    consumes F08's `health()` for `GET /health`.
  - **F16 / F18** — Screen 5 (Three.js 3D scatter over 3 of the 5 UMAP dims + D3 2D fallback); the
    animated unknown-driver "white sphere" projection uses `coords_2d`/`coords_5d` + `nearest_neighbours`.
  - **F20** — runs the formal §11.2 evaluation (unknown-driver leave-one-out NN accuracy, UMAP stability,
    teammate proximity) on the **held-out circuits**; F08 provides the functions, F20 is the only place
    Belgium/Italy are evaluated.

## 3. Inputs & outputs (typed)
All ops vectorised (pandas 3.x / numpy 2.x). The **serving** functions (`load_style_space`,
`project_unknown`, `get_style_space`, `health`, `teammate_proximity`) perform **no `.fit(` /
`.fit_transform(`** (static-scan asserted, as F06b/F07). There are **two fit sites**:
`build_style_space` (the production style-space model) and `stability_report` (a **dissertation-only**
UMAP re-fit for the §11.2 Procrustes evidence). **Both** are train-only and guarded by
`_assert_no_leakage`, and **neither refits normalisation** — the static scan asserts no `fit_normaliser`
/ normalisation refit *anywhere* in the module, and no `.fit(` in the five serving bodies above.

- **Build the style space (offline, once after features are assembled):**
  `build_style_space(laps: pd.DataFrame, *, config: AppConfig, artefact_dir: str | Path | None = None,
  log_path: str | Path | None = None) -> StyleSpaceBuildResult`. Steps: validate columns → quarantine
  held-out + non-`train` → `_assert_no_leakage` → apply F04 normalisation (loaded params, **no refit**) →
  aggregate per driver to 56-dim → **fit the train-only aggregate `StandardScaler` and standardise the
  aggregate matrix (§4 step 6, the §9.7 gap-fill)** → fit `umap_5d`, `umap_2d`, `pca_fallback` on the
  scaled matrix (all seeded) → compute the OOD stats, known coords, stability + proximity → write both
  artefacts deterministically (scaler persisted in the PCA bundle; its fingerprint in the UMAP bundle) →
  **log the build run to MLflow** (params + metrics + artefacts; non-negotiable #3). **Raises** `StyleSpaceBuildError`
  (fail-loud, after logging) on: missing columns, empty training pool, leakage tripwire (`TRAIN_LEAKAGE`),
  fewer than `umap_n_components_5d + 1` drivers, or a UMAP/PCA fit exception (`MODEL_TRAINING_FAILED`).
- **`StyleSpaceBuildResult(BaseResult)`:**
  ```
  { umap_artefact_path: str, umap_sha256: str, pca_artefact_path: str,
    drivers: list[str], coords_5d: dict[str, list[float]], coords_2d: dict[str, list[float]],
    n_drivers: int, n_features: int,             # 56 = 28 means + 28 stds
    effective_n_neighbors: int,                  # min(config, n_drivers - 1); logged if clamped
    stability: StabilityReport | None, proximity: TeammateProximityResult | None,
    mlflow_run_id: str | None,
    status, error_code, flags, logged }
  ```
- **Load the style space (startup, once) — fail-loud only when NO engine is usable:**
  `load_style_space(*, config: AppConfig, artefact_dir=None, log_path=None) -> StyleSpaceHandle`.
  Reads `umap_model.pkl` + verifies its sidecar hash; a **corrupt/missing UMAP** logs `UMAP_FALLBACK`
  and sets `engine="pca"` (PCA becomes primary) — **not** a hard failure. Then loads `pca_fallback.pkl`
  and `normalisation.json`; if the **PCA fallback OR normalisation is also missing/corrupt** there is no
  usable projection engine → **raise** `StyleSpaceLoadError` (`MODEL_NOT_FOUND`), after logging (mirrors
  F06b `load_model`: refuse to serve rather than project from nothing). Returns a **ready** handle
  otherwise (`engine ∈ {"umap","pca"}`).
- **`StyleSpaceHandle`** (frozen dataclass): `{ umap_5d, umap_2d, pca, scaler (the train-only aggregate
  standardiser — REQUIRED, applied read-only at every forward pass), drivers: list[str],
  known_coords_5d: dict[str, np.ndarray], known_aggregate: np.ndarray (n_drivers × 56, stored RAW),
  pc_std: np.ndarray (per-PC training std, for OOD), feature_mean: np.ndarray, feature_std: np.ndarray
  (per-feature known-driver stats, for risk_flags), normalisation_params: NormalisationParams,
  engine: "umap" | "pca", model_version: str, ready: bool }`.
- **Project one unknown driver (per call, read-only, never raises):**
  `project_unknown(laps: pd.DataFrame, *, circuit: str, handle: StyleSpaceHandle, config: AppConfig,
  log_path=None) -> ProjectionResult`. `laps` = the unknown's **raw** clean lap-feature rows (the 28
  `FEATURE_IDS`; a `circuit` column may override the `circuit` arg per lap). Normalises each lap with the
  handle's params (`apply_normalisation`), aggregates → 56-dim, runs the gates + projection + NN. `laps`
  with **zero rows** → `PROJECTION_FALLBACK` coded result (no aggregate possible), never a raise.
- **`ProjectionResult(BaseResult)`:**
  ```
  { coords_5d: list[float] | None,   # None only on PROJECTION_FALLBACK (raw-NN, no reliable coord)
    coords_2d: list[float] | None,
    nearest_neighbours: list[Neighbour],   # top-k by cosine; in 5D normally, raw-56D on PROJECTION_FALLBACK
    confidence_state: ConfidenceState,     # HIGH (in-dist, >=30 laps) | LOW_CONFIDENCE (sparse OR raw-NN fallback) | OUT_OF_DISTRIBUTION; decoupled from status, always populated
    sparse: bool, ood: bool, ood_max_pc_sd: float,
    projection_engine: "umap" | "pca" | "raw_nn",
    risk_flags: list[str],                 # feature IDs whose aggregate is > ood_sd_threshold SD outside the known-driver dist (§7.4)
    n_laps_used: int, laps_rejected: int,
    status, error_code, flags, logged }
  ```
  `Neighbour` = `{ driver: str, cosine_similarity: float, distance: float }`.
- **Known-driver style space (serving `GET /stylespace`):**
  `get_style_space(handle: StyleSpaceHandle) -> StyleSpaceResult`. Pure read of the handle's known coords.
- **`StyleSpaceResult(BaseResult)`:** `{ drivers: list[str], coords_5d: dict[str, list[float]],
  coords_2d: dict[str, list[float]], engine: str }`.
- **Readiness (F13 `GET /health`):** `health(handle: StyleSpaceHandle | None) -> StyleSpaceHealthResult`
  — not-ready → `HEALTH_FAIL` (→ 503) with `MODEL_NOT_FOUND` reason; ready → OK + `model_version` +
  `engine` (mirrors F06b `health`).
- **Stability (dissertation, §5.2/§11.2) — never raises:**
  `stability_report(laps: pd.DataFrame, *, config: AppConfig, log_path=None) -> StabilityReport`.
  **Quarantines held-out + non-`train` rows and runs `_assert_no_leakage` before aggregating** (identical
  to `build_style_space` steps 2–3), applies the **loaded** F04 params (**no refit**), then fits `umap_5d`
  `stability_n_runs` (10) times over the resulting training aggregates with a **fixed seed sequence**
  (`seeds.numpy + i`), Procrustes-aligns each run to run 0 (`scipy.spatial.procrustes`), and reports the
  mean aligned disparity. This is a **second (dissertation-only) fit site** — it fits UMAP but never
  refits normalisation, and its leakage guard is identical to the build's (a poisoned frame is excluded /
  degrades, never fitted on — see the never-raises backstop in §4).
- **`StabilityReport(BaseResult)`:** `{ procrustes_variance: float | None, per_run_disparity: list[float],
  n_runs: int, n_drivers: int, passes: bool (variance < stability_variance_threshold), status, error_code,
  flags, logged }`.
- **Teammate proximity (dissertation, §5.2/§11.2) — never raises:**
  `teammate_proximity(handle_or_coords, teams: dict[str, str], *, config: AppConfig) ->
  TeammateProximityResult`. Mean intra-team vs mean inter-team Euclidean distance in **5D**.
- **`TeammateProximityResult(BaseResult)`:** `{ mean_intra_team: float | None, mean_inter_team: float |
  None, passes: bool (intra < inter), teams_evaluated: int, status, error_code, flags, logged }`.
- **Pure helpers (unit-testable in isolation):** `aggregate_driver(feature_rows: pd.DataFrame) ->
  np.ndarray` (56-dim, `[mean(F01..F28), std(F01..F28)]`, population std `ddof=0`);
  `nearest_neighbours(query_5d, known_coords_5d, k) -> list[Neighbour]` (cosine);
  `ood_check(aggregate_56, pca, pc_std, threshold, min_std, *, scaler) -> tuple[bool, float]` (max
  per-PC |z|; standardises the raw aggregate through the train-only `scaler` before the PC transform).
- **Exceptions:** serving paths **never raise** — every fault is a coded result / a not-ready-then-
  fallback handle. Only `build_style_space` (`StyleSpaceBuildError`) and `load_style_space`
  (`StyleSpaceLoadError`, only when no engine is usable) raise, and only **after** logging.

## 4. Method / algorithm
**Aggregation (the 56-dim representation, §5.2 "8 drivers × 56 values").** For a set of clean lap
feature rows, `aggregate_driver` computes, per feature `F01..F28`, the **mean** and the **population std
(`ddof=0`)** across laps → a 56-vector `[mean(F01), …, mean(F28), std(F01), …, std(F28)]`. `ddof=0`
matches F04's `normalise.ddof` convention and makes a **single lap** yield `std=0.0` (finite, not NaN) so
the sparse path never produces a NaN coordinate. NULL feature cells are ignored per feature: a feature
with **at least one** finite value uses `nanmean`/`nanstd`, while an **all-NULL** feature is
**explicitly** set to `mean=0.0, std=0.0` via an `np.where(all_null_col, 0.0, nanmean(...))` guard —
bare `nanmean`/`nanstd` of an all-NaN slice returns **NaN** (not 0.0), so the all-null case is handled
explicitly and never left to NaN (matches F04's centre-only fallback intent). Any non-finite value
surviving in the assembled 56-dim aggregate is caught by the hard guard in `project_unknown` (step 2b),
so a NaN never reaches the OOD gate or `transform` (where `NaN > threshold` would silently evaluate
`False` and ship a NaN coordinate). `_AGGREGATE_DDOF = 0` is a **module constant** (like `normalise.py`'s
`_CENTRE_ONLY_STD`), not a tunable, so it is excluded from the config-literal scan.

**`build_style_space` (offline, the production fit site).**
1. Require columns `{driver, circuit, season, split} ∪ FEATURE_IDS`; missing →
   `StyleSpaceBuildError(MODEL_TRAINING_FAILED)` (logged).
2. `train = laps[(split=="train") & ~circuit.isin(held_out_circuits)]`; then `_assert_no_leakage(train)`
   (raises `TRAIN_LEAKAGE` if any held-out circuit survived — **no artefact written on a poisoned run**,
   exactly like F04). Empty `train` → `StyleSpaceBuildError` (`MODEL_TRAINING_FAILED`).
3. **Normalise** the frame with F04's **loaded** params (`load_normalisation_params` +
   `apply_normalisation` per lap, vectorised assembly as F06a's `_apply_to_frame`) — the six
   circuit-dependent features Z-scored per circuit; **no refit**.
4. `aggregate_driver` per `driver` (sorted, deterministic label order like F06a `_label_space`) → the
   `n_drivers × 56` matrix `A`.
5. Guard **n_drivers**: UMAP needs `n_components < n_samples` and `n_neighbors < n_samples`. Set
   `effective_n_neighbors = min(umap_n_neighbors, n_drivers - 1)` (logged if clamped). If
   `n_drivers < umap_n_components_5d + 1` → `StyleSpaceBuildError` (`MODEL_TRAINING_FAILED`) — cannot fit
   a 5D embedding of that few points.
6. **Scale, then fit** (the §9.7 gap-fill). The 56-dim aggregate mixes the six F04 z-scored features
   (~unit scale) with 22 raw-physical-unit features spanning ~4 orders of magnitude (F16 gear-hold ms up
   to 10⁴, F12 speed variance up to 8·10³ vs 0–1 fractions and [−1,1] correlations), so a euclidean
   UMAP/PCA fit on the raw matrix would be dominated by the largest-magnitude columns rather than
   driving style. Therefore: `scaler = StandardScaler().fit(A)` (**train-only by construction** — the
   leakage tripwire already ran; `scale_` floored at `normalise.min_std` so a (near-)constant column
   cannot explode the transform), `A_scaled = scaler.transform(A)`; then fit (all with
   `random_state=seeds.numpy`, `n_neighbors=effective_n_neighbors`, `min_dist=umap_min_dist`,
   `metric=umap_metric`): `umap_5d = UMAP(n_components=umap_n_components_5d)`,
   `umap_2d = UMAP(n_components=umap_n_components_2d)`, `pca = PCA(n_components=pca_n_components,
   svd_solver="full", random_state=seeds.numpy)` — **all on `A_scaled`**. Any fit exception (including a
   NaN reaching the scaler fit) → `StyleSpaceBuildError` (`MODEL_TRAINING_FAILED`), logged. The scaler is
   **unconditional** (no config toggle — a toggle would create a scaled/unscaled artefact ambiguity).
7. **OOD basis:** `pc_scores = pca.transform(A_scaled)`; `pc_std = pc_scores.std(axis=0, ddof=0)` (per-PC
   training spread — the between-driver variance the Mahalanobis/PC check standardises against; lives in
   the **scaled** space, like the PCA it standardises). `feature_mean`/`feature_std` = per-column
   mean/std of the **RAW** `A` (for `risk_flags`, which z-scores per feature itself and stays
   interpretable in physical units; `_raw_nn` standardises through the handle's scaler at
   transform time — see step 5 of `project_unknown`).
8. **Known coords:** via the shared scaler-aware forward pass (`_project_aggregate`) on the RAW
   aggregates — scaling happens inside the single choke point, identically for build, load, and serving.
9. **Persist:** joblib-dump `umap_model.pkl` (both UMAP models + feature order + seeds + the
   **`scaler_sha256` cross-bundle fingerprint** — ties the UMAP to the scaler it was fitted against, so
   a partial rebuild is detected at load) and its sidecar `.sha256`; joblib-dump `pca_fallback.pkl`
   (pca + **scaler** + pc_std + **RAW** A + feature stats + drivers). The aggregates are persisted RAW
   (single source of truth); scaling is applied at transform time. PCA + scaler are deterministic; UMAP
   is **not byte-identical across runs** (documented, §9).
10. Compute `stability_report` + `teammate_proximity` (best-effort; failures degrade, do not abort the
    build). **Log to MLflow** (F08's **own** `stylespace.mlflow_tracking_uri` + `stylespace.mlflow_experiment`
    — F08 does **not** read the optional `classifier` block, preserving the §2 independence claim; reading
    `config.classifier.mlflow_tracking_uri` would `AttributeError` when that block is absent): params
    (seeds, n_neighbors, n_components, metric, n_drivers), metrics (`procrustes_variance`,
    `teammate_proximity_pass`, `n_drivers`), artefacts (both pkls). An MLflow-logging fault **degrades**
    the build result (flag), it does not abort the artefact write (the pkls are the deliverable).

**`load_style_space` (startup).** Read `umap_model.pkl`; recompute sha256 and compare to the sidecar.
Missing file / read error / hash mismatch / **n_components mismatch vs the current config** (stale
bundle after a config bump would silently serve wrong-length coordinates) → log `UMAP_FALLBACK`,
`engine="pca"`, `umap_5d=umap_2d=None`.
Load `pca_fallback.pkl` + `normalisation.json`; **either** missing/corrupt → log + raise
`StyleSpaceLoadError(MODEL_NOT_FOUND)` (no usable engine). The PCA's `n_components_` must equal
`pca_n_components` — a stale-vs-config bundle would silently truncate PCA-engine coordinates
(`p[:n5]` of a shorter transform) → log + raise `StyleSpaceLoadError(MODEL_NOT_FOUND)`. The
**aggregate scaler is REQUIRED, with fitted stats**: a pca
bundle lacking a 56-dim scaler with `mean_`/`scale_` (pre-scaler artefact, or a scaler-shaped object
without the stats the fingerprint needs) → log + raise `StyleSpaceLoadError(MODEL_NOT_FOUND)`
with an explicit `AGGREGATE_SCALER_ABSENT` detail — **never** a silent skip-scaling load (that would
project raw aggregates through models fitted in the scaled space) and never an untyped
`AttributeError`. **Cross-bundle skew check:** if the
UMAP bundle's `scaler_sha256` does not match the loaded scaler's fingerprint (partial rebuild /
pre-scaler UMAP), log `UMAP_FALLBACK` (`CROSS_BUNDLE_SCALER_MISMATCH`) and degrade to the PCA engine —
per-file sidecar hashes cannot detect cross-file inconsistency. Coord reconstruction then runs through
the shared scaler-aware forward pass; a loaded-but-unusable UMAP (its `transform` raises) degrades to
PCA (`UMAP_FALLBACK`), and if the PCA retry also fails → `StyleSpaceLoadError(MODEL_NOT_FOUND)`.
`model_version = "sha256:" + digest[:16]` of the
governing engine's bytes (UMAP if present, else PCA), mirroring F06b `_resolve_model_version`.

**`project_unknown` (per call) — never raises.**
1. Normalise each lap (`apply_normalisation`, handle's params); an unknown circuit → global params +
   `NORMALISATION_FALLBACK` flag (carried, `status=DEGRADED`). Zero clean laps → `PROJECTION_FALLBACK`
   coded result immediately.
2. `agg = aggregate_driver(normalised laps)` → 56-dim. `n = n_laps_used`.
2b. **Non-finite guard (defence-in-depth):** if `agg` holds any non-finite value (should be impossible —
   `apply_normalisation` NULLs non-finite inputs and the aggregation is all-null-safe) → log + return a
   coded `PROJECTION_FALLBACK` result with **empty neighbours** — a non-finite query cannot be ranked
   (an `inf` slips past the zero-norm guard and yields NaN cosines), so no ranking is fabricated and
   NaN never feeds the gates below.
3. **Sparse gate (§5.2 FM1):** `n < thresholds.min_clean_laps` (30) → `sparse=True`, flag
   `SPARSE_PROJECTION`, `confidence_state=LOW_CONFIDENCE`, `status=DEGRADED` (projection still computed —
   "compute projection but flag confidence as LOW").
4. **OOD gate (§5.2 FM2):** `(ood, ood_max_pc_sd) = ood_check(agg, pca, pc_std, ood_sd_threshold,
   min_std, scaler=handle.scaler)` — the check standardises the raw aggregate through the handle's
   scaler (the PC basis lives in the scaled space), then max over PCs of `|pc_score_i / pc_std_i|`;
   `> ood_sd_threshold` (3) → `ood=True`,
   `error_code=OUT_OF_DISTRIBUTION`, `confidence_state=OUT_OF_DISTRIBUTION`, `status=DEGRADED`
   ("projection computed but marked unreliable").
5. **Project (§5.2 FM4):** `x = scaler.transform(agg.reshape(1, -1))` (inside `_project_aggregate` —
   the single scaler-aware choke point). `try:` if `engine=="umap"` →
   `coords_5d = umap_5d.transform(x)[0]`, `coords_2d = umap_2d.transform(x)[0]`; else (`engine=="pca"`,
   i.e. `UMAP_FALLBACK` active) → `p = pca.transform(x)[0]`, `coords_5d = p[:5]`, `coords_2d = p[:2]` and
   carry the `UMAP_FALLBACK` flag + `status=DEGRADED`. **On ANY exception** → catch, log
   `PROJECTION_FALLBACK`, `projection_engine="raw_nn"`, `coords_5d=coords_2d=None`, and compute
   `nearest_neighbours` by **cosine in the scaler-standardised 56-dim aggregate space** (§5.2 FM4's
   "nearest-neighbour in raw feature space using cosine distance", with the §9.7 gap-fill applied to
   this path too: unscaled cosine is dominated by the largest-magnitude physical-unit columns, and the
   ranking is this path's entire output — query and `known_aggregate` are standardised identically
   through the handle's read-only scaler). `status=DEGRADED`. When a step-4 OOD
   was already detected, `PROJECTION_FALLBACK` becomes the **operative `error_code`** (no coordinate was
   produced) and `OUT_OF_DISTRIBUTION` is retained on `flags` with `ood=True` (step 8 precedence).
6. **Nearest neighbours (§7.4):** on a successful projection, `nearest_neighbours(coords_5d,
   known_coords_5d, nearest_neighbours_k)` — top-k by **cosine similarity** in 5D.
7. **Risk flags (§7.4):** feature IDs `Fnn` where `|agg_mean[nn] − feature_mean[nn]| /
   max(feature_std[nn], min_std) > ood_sd_threshold` (the unknown's per-feature mean lies outside the
   known-driver distribution) — the **same `min_std` floor** `ood_check` applies to `pc_std`, so a
   near-constant feature across the 8 known drivers (`feature_std ≈ 0`) can neither spuriously flag
   (`inf`) nor silently drop (`0/0 = NaN`); "potential adaptation challenges flagged explicitly".
8. **`confidence_state` (driver-level, decoupled from `status` — as F06b keeps HIGH/MEDIUM even on a
   `NORMALISATION_FALLBACK` DEGRADE, inference.py 427–433):** `OUT_OF_DISTRIBUTION` (most severe) >
   `LOW_CONFIDENCE` (sparse **or** the raw-NN `PROJECTION_FALLBACK` path — no reliable coordinate) >
   `HIGH` (in-distribution, ≥30 laps). An engine (`UMAP_FALLBACK`) or `NORMALISATION_FALLBACK` degrade
   **alone** does not lower `confidence_state` (a clean driver projected via the PCA engine is still
   `HIGH`); it only sets `status=DEGRADED`. `confidence_state` is therefore **always populated**.
   **`error_code` (single operative code; the rest de-duped onto `flags`, as F06b `_dedup`):**
   `PROJECTION_FALLBACK` (no coordinate) > `OUT_OF_DISTRIBUTION` > `None`. **`status`:** `OK` only when
   none of {sparse, ood, `UMAP_FALLBACK` engine, `PROJECTION_FALLBACK`, `NORMALISATION_FALLBACK`} holds;
   otherwise `DEGRADED`. F08 never emits `FAILED` from `project_unknown`. **Never-raises backstop:** the
   whole body runs under a top-level `try/except` (mirroring F06b `predict` 462–477 / F07
   `explain_prediction`) — any unforeseen fault → best-effort log + a coded `PROJECTION_FALLBACK` result,
   never a propagated exception. `stability_report` and `teammate_proximity` carry the same backstop
   (which also catches their internal `_assert_no_leakage` tripwire → `DEGRADED`, never a raise).

**`ood_check` (pure).** `z = pca.transform(scaler.transform(agg[None, :]))[0] / pc_std` (guard `pc_std`
≥ `min_std` to avoid div-by-zero, reusing F04's `min_std` idea via a local floor; the raw aggregate is
standardised through the train-only scaler because the PC basis + `pc_std` live in the scaled space);
return `(max|z| > threshold, max|z|)`. The **PCA fallback artefact doubles as the OOD detector** — one
artefact, two jobs.

**`nearest_neighbours` (pure).** Cosine similarity `sim = a·b / (‖a‖‖b‖)`; a zero-norm vector (all-zero
coord) → similarity 0.0 (never a div-by-zero / NaN neighbour). Return the top-`k` known drivers by `sim`
desc as `Neighbour(driver, cosine_similarity, distance=1−sim)`.

**`stability_report` (§5.2 "10 seeds + Procrustes", §11.2) — a dissertation-only fit site.**
**Quarantine** held-out + non-`train` rows and run `_assert_no_leakage` (identical to `build_style_space`
steps 1–2), **normalise** with the loaded F04 params (**no refit**), aggregate the training frame once,
**standardise it with its own in-memory train-only aggregate scaler** (same §4 step 6 recipe, never
persisted — so the Procrustes evidence measures the geometry the production model actually ships);
then fit `umap_5d` `stability_n_runs` times with `random_state = seeds.numpy + i` (a **fixed,
reproducible** seed sequence — this is the one place the seed is deliberately varied, and it is varied
deterministically, non-negotiable #3). Align each run's `n_drivers × 5` coord matrix to run 0 with
`scipy.spatial.procrustes` (returns a disparity ∈ [0, 1]); `procrustes_variance` = mean disparity over
runs 1..n; `passes = variance < stability_variance_threshold` (0.05). A fit/alignment fault degrades
(`status=DEGRADED`, `procrustes_variance=None`) — never raises, never blocks (a stability failure is a
"research finding, not a system failure", §5.2).

**`teammate_proximity` (§5.2, §11.2).** Over the known 5D coords + a `driver → team` map: `mean_intra`
= mean pairwise Euclidean distance within teams, `mean_inter` = mean pairwise across teams;
`passes = mean_intra < mean_inter`. Teams with a single driver contribute no intra pair. No comparable
pair → `None` (never a fabricated 0.0). Never raises.

**status contract:** clean in-distribution projection on the UMAP engine → `OK`; sparse / OOD /
`UMAP_FALLBACK` engine / `PROJECTION_FALLBACK` / `NORMALISATION_FALLBACK` → `DEGRADED`; F08 serving never
emits `FAILED`. `build_style_space` → `OK` when both artefacts written cleanly and stability passes,
else `DEGRADED` (e.g. stability variance ≥ `stability_variance_threshold`, `n_neighbors` clamped) with
`error_code=None` — these are research findings / adaptations, **not** registered error conditions
(consistent with §6); hard build failures **raise** `StyleSpaceBuildError` before a result is built.

## 5. config.yaml keys this feature adds
F08 adds a new typed `stylespace:` block (a `StyleSpaceConfig` sub-model on `AppConfig`,
`extra="forbid"`, per the standing typed-config rule). It **reuses** the four pre-seeded §5.2 error
codes (`SPARSE_PROJECTION`, `OUT_OF_DISTRIBUTION`, `UMAP_FALLBACK`, `PROJECTION_FALLBACK`), the
`OUT_OF_DISTRIBUTION` `ConfidenceState`, the two ML-training-integrity codes (`TRAIN_LEAKAGE`,
`MODEL_TRAINING_FAILED`) and `MODEL_NOT_FOUND` / `HEALTH_FAIL` / `NORMALISATION_FALLBACK` — **no new
error code** (as F07). The sparse floor is `thresholds.min_clean_laps` (30) — **not** re-declared here.
```yaml
stylespace:
  umap_n_components_5d: 5           # §5.2 — the fingerprint coordinate dimensionality
  umap_n_components_2d: 2           # §5.2 — the visualisation projection dimensionality
  umap_n_neighbors: 7              # §5.2 — UMAP neighbourhood; capped to n_drivers-1 at build (8 drivers -> 7)
  umap_min_dist: 0.1              # UMAP layout tightness (design decision; umap-learn default)
  umap_metric: euclidean         # distance on the 56-dim aggregate (design decision)
  pca_n_components: 5             # PCA fallback dims (first 2 == 2D fallback); also the OOD PC basis (§5.2)
  ood_sd_threshold: 3.0          # §5.2 — |z| on ANY principal component above this -> OUT_OF_DISTRIBUTION
  nearest_neighbours_k: 3        # §7.4 — top-N nearest known drivers (cosine, 5D)
  umap_model_filename: umap_model.pkl    # §5.2 — under paths.artefacts_dir (+ .sha256 sidecar)
  pca_model_filename: pca_fallback.pkl   # §5.2 — under paths.artefacts_dir
  stability_n_runs: 10           # §5.2/§11.2 — UMAP runs (fixed seed sequence) for Procrustes stability
  stability_variance_threshold: 0.05  # §11.2 — Procrustes-aligned variance target (dissertation)
  mlflow_tracking_uri: sqlite:///artefacts/mlflow.db  # #3 — F08's OWN URI (same store, own experiment); NOT read from the classifier block
  mlflow_experiment: apex-stylespace  # #3 — build run logged here
```
**Validator placement (on `StyleSpaceConfig`):** `umap_n_components_2d ≥ 1`; `umap_n_components_5d >
umap_n_components_2d`; `pca_n_components ≥ umap_n_components_5d` (so PCA can serve both fallbacks);
`umap_n_neighbors ≥ 2`; `0 < umap_min_dist < 1`; `umap_metric` non-empty; `ood_sd_threshold > 0`;
`nearest_neighbours_k ≥ 1`; both filenames non-empty and **distinct**; `stability_n_runs ≥ 2`;
`stability_variance_threshold > 0`; `mlflow_tracking_uri` and `mlflow_experiment` non-empty. A mis-edit
is a load-time
`CONFIG_001`. (`n_drivers`-dependent bounds — `n_neighbors < n_drivers`, `n_components_5d < n_drivers` —
are **data-dependent** and enforced fail-loud at **build** time, not in the static config validator.)

**Build-env note.** `umap-learn` is **not yet installed** (verified this session; `scipy 1.18.0`,
`numba 0.65.1`, `llvmlite 0.47.0`, `scikit-learn 1.9.0`, `numpy 2.4.6` are already present from F07). Per
the F06a/F07 precedent, pin the newest release with a Python-3.14 wheel (the venv is Python 3.14) —
`umap-learn` + its `pynndescent` transitive dep. `numba` already caps `numpy<2.5` (satisfied at 2.4.6),
so **no numpy movement is expected**; record the resolved versions in `requirements.lock` and back-note
any transitive stack movement, exactly as F04/F06a/F07 did.

## 6. Error codes & failure modes (the failsafe contract)
The four master §5.2 modes are the spine; F08 adds **no new error code**. Serving **never raises** and
**never returns a wrong coordinate silently** — every fallback is coded, logged, and surfaced.

| Failure mode | Detection | Error code / flag | System response | Logged? | User-facing message |
|--------------|-----------|-------------------|-----------------|---------|---------------------|
| Unknown-driver telemetry too sparse (<30 laps) | lap count before projection (`n < thresholds.min_clean_laps`) | `SPARSE_PROJECTION` (flag), `status=DEGRADED`, `confidence_state=LOW_CONFIDENCE`, `sparse=True` | **compute the projection**, flag confidence LOW (UI shows a wider uncertainty ellipse) | yes | "Projection based on limited data (N laps); confidence is low." |
| Unknown-driver aggregate outside training distribution | Mahalanobis / PC check: `max_i |pc_i / pc_std_i| > ood_sd_threshold` (3 SD) | `OUT_OF_DISTRIBUTION` (code), `status=DEGRADED`, `confidence_state=OUT_OF_DISTRIBUTION`, `ood=True` | projection computed but **marked unreliable** (UI shows a dashed-outline marker) | yes | "This driver falls outside the known style distribution; projection is approximate." |
| UMAP model file corrupt or missing / fitted against a different aggregate scaler (partial rebuild) / loaded but its `transform` raises at coord reconstruction | sha256 sidecar hash verify + `scaler_sha256` cross-bundle fingerprint check + guarded reconstruction at load | `UMAP_FALLBACK` (flag), `engine="pca"`, `status=DEGRADED` on each PCA-engine projection | **fall back to PCA** (2-component viz / 5-component fingerprint); PCA is always maintained as backup | yes | none (monitoring; UI unaffected — style space still functional) |
| `transform()` fails for an unknown driver | try/except around the `transform()` call | `PROJECTION_FALLBACK` (operative code even if OOD co-occurs; OOD → `flags`), `projection_engine="raw_nn"`, `coords=None`, `confidence_state=LOW_CONFIDENCE`, `status=DEGRADED` | **fall back to nearest-neighbour cosine in the scaler-standardised 56-dim aggregate space** (§9.7 applies here too — unscaled cosine is magnitude-dominated); return the matches, no coordinate. The backstop preserves the already-established `sparse`/`used_global`/lap-count values — the result never contradicts the DQ rows written upstream | yes | "Could not place this driver on the map; showing closest known drivers instead." |
| Unknown circuit at normalisation | `apply_normalisation` `used_global` | `NORMALISATION_FALLBACK` (flag, F04-owned), `status=DEGRADED` | normalise with global params (carried from F04) | yes | none (monitoring) |
| Build: held-out circuit reached the fit | `_assert_no_leakage` tripwire after the exclusion filter | `TRAIN_LEAKAGE` (raise `StyleSpaceBuildError`) | **abort the build; no artefact written** | yes | build-time only |
| Build: missing required columns / empty training pool / too few drivers / UMAP·PCA fit throws | column + count/shape check + guarded fit | `MODEL_TRAINING_FAILED` (raise `StyleSpaceBuildError`) | abort the build, logged | yes | build-time only |
| Load: no usable projection engine (PCA **and**/or normalisation missing/corrupt; pca bundle lacking the 56-dim aggregate scaler — pre-scaler artefact; PCA coord-reconstruction retry fails) | file/hash/bundle-schema checks + guarded reconstruction at load | `MODEL_NOT_FOUND` (raise `StyleSpaceLoadError`) → F13 `HEALTH_FAIL` (503) | refuse to serve (fail-loud, mirrors F06b — **never** a silent unscaled load) | yes | "Style space unavailable." (503) |
| DQ-log sink unwritable during any of the above | ledger `failed` counter | `DATA_QUALITY_LOG_UNWRITABLE` (flag) | surface on `flags`, never crash the never-raises path | n/a | none |
| Stability variance ≥ threshold / intra ≥ inter | Procrustes / proximity result | *(not a system error)* `passes=False`, `status=DEGRADED`, `error_code=None` | **report as a research finding** (§5.2); UMAP retained for known drivers, PCA available as primary for unknowns | MLflow metrics at build (`procrustes_variance`, `stability_passes`, `teammate_proximity_pass`) + the typed result — a legitimate research finding is **not** a `data_quality_log` event (that log is for data-quality faults, and repurposing an error code for it would break code discipline) | dissertation only |

The **User-facing message** column is illustrative copy F13/F16 render; tests assert
`error_code`/`flags`/`status`/`confidence_state`/field values, not prose. Stability / proximity failure
is explicitly **not** a system failure (§5.2, §11.2) — it is a `passes=False` finding.

## 7. Acceptance criteria (each item mechanically testable by /spec-check)
- [ ] **5D fingerprint + 2D viz from UMAP:** `build_style_space` fits **two** `umap.UMAP` models
      (`n_components == stylespace.umap_n_components_5d` and `== umap_n_components_2d`) plus one
      `sklearn.decomposition.PCA` (`n_components == pca_n_components`); `coords_5d[d]` has length 5 and
      `coords_2d[d]` length 2 for every known driver `d`.
- [ ] **56-dim aggregate:** `aggregate_driver` returns a length-56 vector `[mean(F01..F28),
      std(F01..F28)]` with **population std (`ddof=0`)**; a single-lap input yields finite values with the
      std half all-zero (no NaN).
- [ ] **Train-only fit / held-out never touches `.fit` (non-negotiable #2):** **both** fit sites —
      `build_style_space` and `stability_report` — exclude `held_out_circuits` + non-`train` rows and run
      `_assert_no_leakage` before any fit. The quarantine mask filters held-out circuits regardless of
      their `split` label, so the tripwire is **defence-in-depth** (it can only fire if the filter itself
      is broken); its raise/no-artefact/DQ-row path is therefore exercised by **simulating a broken
      filter** (monkeypatched tripwire → `StyleSpaceBuildError(TRAIN_LEAKAGE)`, **no artefact written**;
      `stability_report` degrades via its backstop, never fitting), and the train-only reach is proven
      **behaviourally**: the persisted scaler's `mean_` equals the hand-recomputed quarantined training
      aggregate, so held-out rows provably never entered any fit.
- [ ] **Normalisation reused, never refit:** `build_style_space`, `project_unknown`, and
      `stability_report` obtain centre/scale via `load_normalisation_params` / `apply_normalisation`, with
      **no `fit_normaliser` / normalisation refit anywhere** in the module; a static source scan finds
      **no `.fit(` / `.fit_transform(`** in the five serving bodies `project_unknown`, `load_style_space`,
      `get_style_space`, `health`, `teammate_proximity`. The **two** UMAP/PCA fit sites are
      `build_style_space` and `stability_report` (dissertation-only), and both are train-only + leakage-guarded.
- [ ] **Unknown-driver forward pass (§5.2):** `project_unknown` on a clean ≥30-lap in-distribution input
      returns `coords_5d`/`coords_2d` via `umap.transform` (no refit), `confidence_state=HIGH`,
      `status=OK`, `nearest_neighbours` of length `nearest_neighbours_k` ordered by cosine desc.
- [ ] **Sparse gate (§5.2 FM1):** `n_laps_used < thresholds.min_clean_laps` (30) → `sparse=True`, flag
      `SPARSE_PROJECTION`, `confidence_state=LOW_CONFIDENCE`, `status=DEGRADED`, **coords still present**
      (projection computed). Boundary asserted at 29 vs 30 laps.
- [ ] **OOD gate (§5.2 FM2):** an aggregate driven `> ood_sd_threshold` (3) SD on a principal component
      → `ood=True`, `error_code=OUT_OF_DISTRIBUTION`, `confidence_state=OUT_OF_DISTRIBUTION`,
      `status=DEGRADED`, coords still present; an in-distribution aggregate → `ood=False`. Both sides of the
      threshold asserted (stub/handcrafted `pca`+`pc_std`).
- [ ] **UMAP→PCA fallback (§5.2 FM3):** a corrupt/missing `umap_model.pkl` (hash mismatch) →
      `load_style_space` returns a **ready** handle with `engine=="pca"` and logs `UMAP_FALLBACK` (no
      raise); a projection on it carries the `UMAP_FALLBACK` flag, `status=DEGRADED`, and yields coords
      from PCA (`coords_2d == pca first 2 comps`).
- [ ] **transform→raw-NN fallback (§5.2 FM4):** a handle whose `transform` raises → `project_unknown`
      returns `PROJECTION_FALLBACK`, `projection_engine=="raw_nn"`, `coords_5d is None`,
      `nearest_neighbours` computed by **cosine in the scaler-standardised 56-dim aggregate space**
      (a rebalancing scaler demonstrably reshapes the ranking — unscaled cosine would not),
      `status=DEGRADED`, **no exception**.
      With a co-occurring OOD, `error_code == PROJECTION_FALLBACK` and `OUT_OF_DISTRIBUTION` is on `flags`.
- [ ] **Never a NaN coordinate / `confidence_state` always set:** an unknown that is all-NULL on one
      feature aggregates to a **finite** 56-dim (that feature `mean=std=0.0`, not NaN); a forced non-finite
      aggregate → step-2b `PROJECTION_FALLBACK` (never a NaN coord, never a silent OOD pass).
      `confidence_state` is **non-null on every returned `ProjectionResult`** — including the
      `UMAP_FALLBACK`-engine path (`HIGH` when clean/in-dist) and the `PROJECTION_FALLBACK` path
      (`LOW_CONFIDENCE`) — decoupled from `status`.
- [ ] **No usable engine → fail-loud:** `load_style_space` with PCA fallback **and/or** normalisation
      missing/corrupt raises `StyleSpaceLoadError(MODEL_NOT_FOUND)` after logging; `health(None)` →
      `HEALTH_FAIL` + `MODEL_NOT_FOUND` reason.
- [ ] **Nearest neighbours (§7.4):** `nearest_neighbours` returns the top-`k` known drivers by cosine
      similarity in 5D; a zero-norm query → similarity `0.0`, no NaN / no div-by-zero.
- [ ] **Risk flags (§7.4):** a feature whose unknown-aggregate mean is `> ood_sd_threshold` SD outside
      the known-driver per-feature distribution appears in `risk_flags`; an in-range feature does not.
- [ ] **Reproducibility (#3):** `build_style_space` sets `random_state = seeds.numpy` on both UMAP models
      and the PCA; re-running the build produces **identical `pca_fallback.pkl`** (PCA deterministic) and
      **UMAP coordinates equal within a documented tolerance** (UMAP not byte-identical — see §9). The
      MLflow build run logs seeds + params + both artefacts.
- [ ] **Stability report (§5.2/§11.2):** `stability_report` **excludes held-out + non-`train` rows and
      runs `_assert_no_leakage` before fitting** (a poisoned frame degrades via the backstop, never fits on
      Belgium/Italy), then runs `stability_n_runs` UMAP fits with the **fixed seed sequence** `seeds.numpy
      + i`, returns `procrustes_variance` (mean `scipy.spatial.procrustes` disparity ∈ [0,1]) and `passes =
      variance < stability_variance_threshold`; identical runs → variance `0.0`; **never raises**.
- [ ] **Teammate proximity (§5.2/§11.2):** `teammate_proximity` returns `mean_intra_team`,
      `mean_inter_team`, `passes = intra < inter`; single-driver teams contribute no intra pair; no
      comparable pair → `None` (never fabricated 0.0); never raises.
- [ ] **n_neighbors / n_components adapt to n_drivers:** with `n_drivers=8`, `effective_n_neighbors ==
      min(umap_n_neighbors, 7)` (logged if clamped); `n_drivers < umap_n_components_5d + 1` →
      `StyleSpaceBuildError(MODEL_TRAINING_FAILED)`.
- [ ] **Config-driven:** the module reads every `stylespace.*` key + `thresholds.min_clean_laps`; a
      literal scan finds no `5`/`2`/`7`/`3.0`/`0.05`/`30` threshold literal in the module body
      (`_AGGREGATE_DDOF = 0` is a named structural constant, excluded).
- [ ] **Config validator:** `AppConfig` load raises when `umap_n_components_5d <= umap_n_components_2d`,
      `pca_n_components < umap_n_components_5d`, `umap_n_neighbors < 2`, `umap_min_dist ∉ (0,1)`,
      `ood_sd_threshold <= 0`, `nearest_neighbours_k < 1`, the two filenames equal, `stability_n_runs < 2`,
      `stability_variance_threshold <= 0`, or `mlflow_tracking_uri` / `mlflow_experiment` empty.
- [ ] **Logged + DQ row:** for each logged mode (`SPARSE_PROJECTION`, `OUT_OF_DISTRIBUTION`,
      `UMAP_FALLBACK`, `PROJECTION_FALLBACK`, `NORMALISATION_FALLBACK`, `TRAIN_LEAKAGE`,
      `MODEL_TRAINING_FAILED`, `MODEL_NOT_FOUND`) `result.logged`/the raise is preceded by a
      `data_quality_log` row with the matching code (asserted against a tmp log).
- [ ] **Aggregate scaler — train-only fit, read-only serving (§4 step 6 / §9.7):** the persisted
      scaler's `mean_` equals the hand-recomputed mean of the leakage-quarantined training aggregate
      (held-out rows provably absent from its fit); serving calls `scaler.transform` only, never
      `fit`/`fit_transform` (recording-stub asserted); `scale_ ≥ normalise.min_std` everywhere (a
      zero-variance feature builds and projects finite); `ood_check` demonstrably standardises its
      input through the scaler (a shifted stub scaler flips the OOD verdict).
- [ ] **Scaler artefact contract:** `pca_fallback.pkl` carries the scaler (`n_features_in_ == 56`,
      fitted `mean_`/`scale_` present); the UMAP bundle carries its `scaler_sha256` fingerprint.
      Loading a bundle **without** the scaler (or with a wrong-width or stats-less scaler) →
      `StyleSpaceLoadError(MODEL_NOT_FOUND)` with an
      `AGGREGATE_SCALER_ABSENT` detail — never a silent unscaled load, never an untyped
      `AttributeError`. A fingerprint mismatch
      (partial rebuild) → `UMAP_FALLBACK` degrade to the PCA engine. **Dimension checks vs config:**
      a umap bundle whose `n_components` mismatch the config → `UMAP_FALLBACK` degrade; a pca bundle
      whose `n_components_` mismatch → `MODEL_NOT_FOUND` (a stale bundle never silently serves
      wrong-length or truncated coordinates).
- [ ] **Build/load coord parity:** `load_style_space`-reconstructed known + per-season coords equal the
      build's emitted coords within tolerance — the scaler is provably applied identically at every
      `_project_all` site. The load-time reconstruction degrade chain is pinned: a valid-hash UMAP whose
      `transform` raises → PCA engine + `UMAP_FALLBACK` logged; PCA retry also failing →
      `StyleSpaceLoadError(MODEL_NOT_FOUND)`.
- [ ] **Honest projection accounting:** the sparse gate counts **signal-carrying** laps (boundary
      asserted with mixed all-NULL rows: 29-of-31 sparse vs 30-of-32 not), and every fallback result —
      including the transform-failure raw-NN path and the never-raises backstop — reports the
      best-effort `n_laps_used`/`laps_rejected`, never `(raw row count, 0)` after laps were rejected;
      a pre-processing fault reports `(0, n_input)`, never claiming unprocessed rows were "used". The
      backstop also preserves the already-established `sparse`/`used_global` flags, so the result
      never contradicts the `SPARSE_PROJECTION`/`NORMALISATION_FALLBACK` DQ rows written upstream.
- [ ] Every error code / flag above has a unit test that triggers it and asserts the result.

## 8. Test plan
- **Fixtures (`tests/ml/conftest.py`, extend):** a synthetic wide training matrix — ≥ (`n_components_5d`
  + a margin) drivers × several circuits × ≥30 clean laps each, with `split`/`season`/`circuit`/`driver`
  columns and the 28 `FEATURE_IDS`, plus a materialised `normalisation.json` (reuse F04/F06a's fixture)
  → `build_style_space` under tmp → a genuine `StyleSpaceHandle` via `load_style_space`, so one
  end-to-end `project_unknown` runs on a real forward pass. Add a `StubHandle` (hand-set `pca`/`pc_std`/
  `known_coords_5d`/`known_aggregate`) to pin the OOD, NN, sparse, and fallback boundaries
  deterministically, and a `raising` UMAP stub for `PROJECTION_FALLBACK`.
- **Happy path (real build):** ≥30-lap in-distribution unknown → 5D+2D coords, `HIGH`, `status=OK`,
  `k` neighbours ordered, empty `risk_flags`.
- **Boundary fixtures (via StubHandle):** 29 vs 30 laps (sparse); PC z-score 2.9 vs 3.1 (OOD);
  in-range vs out-of-range feature (risk_flags).
- **Fallback chain:** corrupt `umap_model.pkl` (flip a byte / bad sidecar) → `UMAP_FALLBACK`, PCA engine;
  `transform` stub raises → `PROJECTION_FALLBACK` raw-NN; PCA **and** normalisation absent →
  `StyleSpaceLoadError(MODEL_NOT_FOUND)`.
- **Leakage / build:** Belgium/Italy in the `train` split → `TRAIN_LEAKAGE` raise + **no artefact**;
  empty pool / `n_drivers < n_components_5d+1` → `MODEL_TRAINING_FAILED`.
- **Purity:** `aggregate_driver`, `nearest_neighbours`, `ood_check` tested as pure functions with
  handcrafted arrays (exact 56-dim, exact cosine order, exact max-PC z).
- **Stability / proximity (pure-ish):** identical seeds→variance 0.0; divergent aggregates→higher;
  intra<inter passes, intra≥inter `passes=False` (not an error); single-driver team; no comparable pair
  → `None`. `scipy.spatial.procrustes` disparity ∈ [0,1].
- **Reproducibility:** two builds → byte-identical `pca_fallback.pkl` (scaler included); UMAP coords
  equal within tolerance.
- **Aggregate scaler:** persisted scaler + fingerprint in the bundles; hand-recomputed train-only
  `mean_`; build/load coord parity; zero-variance feature (floored `scale_`, finite projection);
  pre-scaler / wrong-width bundle → fail-loud `MODEL_NOT_FOUND`; fingerprint mismatch → PCA degrade;
  recording stub proves serving transforms but never fits; shifted stub proves `ood_check` scales;
  a raising scaler lands in the typed `PROJECTION_FALLBACK` backstop.
- **Load reconstruction chain:** UMAP absent but PCA good → working `pca` handle; valid-hash UMAP whose
  `transform` raises → PCA degrade + `UMAP_FALLBACK`; both engines failing reconstruction →
  `MODEL_NOT_FOUND`.
- **Accounting:** signal-count sparse boundary with mixed all-NULL rows; fallback results report
  `n_laps_used`/`laps_rejected` honestly.
- **Static scans:** no `.fit(` / `.fit_transform(` in the five serving bodies (`project_unknown`,
  `load_style_space`, `get_style_space`, `health`, `teammate_proximity`); no `fit_normaliser` /
  normalisation refit **anywhere** in the module; the inverse whole-module scan (module minus the two
  sanctioned fit sites contains no fit call); both fit sites contain the aggregate-scaler step; a
  held-out-poisoned frame is caught by
  `_assert_no_leakage` in **both** fit sites (`build_style_space` raises, `stability_report` degrades);
  threshold-literal scan for `5`/`2`/`7`/`3.0`/`0.05`/`30`; `requirements.lock` is BOM-free.
- **Config validator:** assert `AppConfig` rejects each bad `stylespace:` value above.

## 9. Definition of done
All acceptance criteria MET with passing tests; **ml-auditor PASS** (no refit at project time; loaded
normalisation reused, never recomputed; held-out laps never enter any `.fit`; **both** `.fit` sites
(`build_style_space` + the dissertation-only `stability_report`) are train-only and leakage-guarded; the
sparse/OOD gates are enforced, never bypassed; fallbacks degrade,
never inflate or fabricate a coordinate); **failsafe-auditor PASS** (all four §5.2 modes + the build/load
fail-loud paths handled, logged, surfaced; a projection is never silently wrong and the service refuses
to start with no usable engine); `/apex-review F08` clean (no silent failure — every fallback is coded +
logged, never a bare `except: pass`; config-driven thresholds; error-code discipline / reuse; no magic
numbers); `/spec-check F08` green; `mypy app` clean. INDEX F08 row → **DONE**.
**Limitations for the dissertation:** (1) **Small-sample manifold** — the master fixes the UMAP training
matrix at **8 drivers × 56** (one point per driver); UMAP/PCA on ~8 points is statistically fragile,
`n_neighbors` is capped at `n_drivers − 1`, and 5D from 8 points may fall back to random init. This is
*why* the master mandates the PCA fallback + the 10-seed stability test; report the observed stability
variance and, per the risk register, **switch to PCA as the primary unknown-driver projection if
Procrustes variance > 0.05**. (2) **UMAP is stochastic** (§15.2) — with a fixed `random_state`
coordinates are reproducible within numerical tolerance but `umap_model.pkl` is **not byte-identical**
across runs (unlike F04's `normalisation.json`); the reproducibility contract (#3) is met by the fixed
seed + the MLflow-logged build config, and the sidecar hash is **corruption detection**, not a cross-run
identity guarantee. `pca_fallback.pkl` **is** deterministic. (3) **UMAP 5D vs 2D are independent
embeddings** — the 2D is not a projection of the 5D (a UMAP property); only the **PCA fallback**'s 2D is
the first two components of its 5D. (4) **OOD uses between-driver variance** (8-point PC std), which is
the right axis for "is this driver stylistically outside the knowns?" but is a small-sample estimate;
the master's "Mahalanobis > 3 SD on any principal component" is implemented as the per-PC max, with the
full Mahalanobis distance reported alongside. (5) **The <5-lap hard reject (HTTP 400) and the OOD dashed-
marker are F13's mapping**, not F08's — F08 always computes + flags (precedent: F07's 206). (6)
**Stability & teammate-proximity are dissertation evidence** produced here but formally evaluated in
**F20** on the held-out circuits; a failure of either is a research finding, not a system failure (§5.2).
(7) **Train-only aggregate scaler — a master gap-fill (supervisor divergence #6).** The master §5.2
specifies Model 2's input in one sentence ("mean and std of each of the 28 features … 8 drivers × 56
values") and is **silent** on the scale of that aggregate; §4.3's normalisation covers only the six
circuit-dependent per-lap features. Taken literally, the 56-dim matrix mixes ~unit-scale z-scored
columns with raw physical units spanning ~4 orders of magnitude (F16 gear-hold ms ≤10⁴, F12 speed
variance ≤8·10³, F18 RPM std ≤5·10³ vs 0–1 fractions and [−1,1] correlations), so a euclidean UMAP/PCA
fit would measure variation of the large-unit features, not driving style — corrupting the continuous
style space RQ4/Contribution-3 depend on and every downstream consumer (F10/F11/F18/F20). F08 therefore
standardises the aggregate with a **train-only `StandardScaler`** (fit inside the leakage-guarded build,
persisted in the PCA bundle, applied read-only at every forward pass — the exact §4.3/F04 contract
shape, one stage up; `scale_` floored at `normalise.min_std`). This **fills a gap the master never
reasoned about rather than contradicting a mandate** (the §5.2 "no refit" sentence concerns the UMAP
model itself); it is nonetheless flagged to the supervisor as divergence #6, alongside the F06a/F06b
divergences. Coordinates changed relative to a raw-space fit; no consumer existed at the time
(F10/F11/F13/F16/F18 unbuilt, no fitted artefact committed), so the change is migration-free.

## 10. Risks & mitigations (from master risk register where relevant)
- **UMAP projection unstable for unknown drivers (risk register, prob Low / impact Medium):** the PCA
  fallback is **always maintained** and doubles as the OOD detector; the 10-seed Procrustes stability
  test quantifies instability, and if variance > `stability_variance_threshold` (0.05) the documented
  mitigation is to **make PCA the primary projection for unknown drivers** while retaining UMAP for the
  known-driver style space. ml-auditor + the stability test gate this.
- **A wrong coordinate ships silently (§5.2 headline "out-of-distribution check on every projection"):**
  every projection runs the OOD gate and returns a coded `OUT_OF_DISTRIBUTION` (never a confident wrong
  placement); a `transform()` fault degrades to raw-NN with `coords=None` (never a fabricated point);
  failsafe-auditor gates this.
- **Normalisation leakage via the style-space fit (non-negotiable #2):** the fit consumes the F04
  **training-fit** params (read-only) and aggregates **train-circuit** laps only; `_assert_no_leakage`
  aborts on any held-out survivor and no artefact is written; the serving path never calls `.fit`.
- **Serving with no model (fail-loud):** `load_style_space` refuses to start (`MODEL_NOT_FOUND` → 503)
  when neither UMAP nor PCA (nor normalisation) is usable — mirrors F06b; the health check surfaces it.
- **Reproducibility of a stochastic model (#3):** fixed `random_state=seeds.numpy`, a fixed stability
  seed sequence, MLflow-logged build config, and a deterministic PCA fallback — the stochasticity is
  bounded, quantified, and reported, not hidden.
- **Dependency migration (build env):** `umap-learn` (+ `pynndescent`) not yet installed; pin the newest
  Python-3.14 wheel per the F06a/F07 precedent, record in `requirements.lock`, and back-note any
  transitive stack movement (no numpy movement expected — `numba` already pins `numpy<2.5`).
```
