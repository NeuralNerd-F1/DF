# APEX Feature Index & Build Order

One feature = one spec = one focused session. Build in dependency order: never start a
feature whose upstream prerequisites are not `DONE`. Status values:
`TODO -> SPEC WRITTEN -> IN PROGRESS -> DONE`.

Generate each spec with `/feature-spec <id> <area>`, build it with `/plan-feature <id>`
then implement, gate it with `/spec-check <id>` and `/apex-review <id>`.

## Layer 0 — Foundation contracts (do first, tiny, unblocks everything)
| ID  | Feature | Depends on | Wk | Status |
|-----|---------|-----------|----|--------|
| F00 | Error-code Enum, result objects, `config.yaml`, repo skeleton, Docker Compose | — | 1 | DONE (2026-06-25) |

## Layer 1 — Data pipeline
| ID  | Feature | Depends on | Wk | Status |
|-----|---------|-----------|----|--------|
| F01 | Data ingestion (FastF1/OpenF1/Ergast, caching, circuit breakers) | F00 | 1 | DONE (2026-06-26) |
| F02 | Lap filter chain / validation contract (9 filters, error codes) | F00, F01 | 2 | DONE (2026-06-26) |
| F03a | Scalar-channel features (F01–F18, F22–F24, F26) + vector assembly, `FeatureComputationError`, 6-NULL tolerance | F00, F02 | 3 | DONE (2026-06-26) |
| F03b | X/Y cornering (F19–F21), corner-balance (F25), tyre-age slopes (F27–F28) — plug into F03a's vector | F00, F02, F03a | 3 | DONE (2026-06-26) |
| F04 | Circuit-aware normalisation (train-only, saved artefact) | F03a, F03b | 4 | DONE (2026-06-26) |
| F05 | Manual CSV upload pipeline + edge cases (UPLOAD_001–004) | F02, F03 | 16 | TODO |

## Layer 2 — Machine learning
| ID  | Feature | Depends on | Wk | Status |
|-----|---------|-----------|----|--------|
| F06 | Model 1 — XGBoost classifier + confidence gate + calibration + Optuna + MLflow | F04 | 5–6 | DONE (F06a 2026-06-30, F06b 2026-07-01) |
| F07 | SHAP explainability (beeswarm, force, radar, NL templates) | F06 | 7 | DONE (2026-07-01) |
| F08 | Model 2 — UMAP style space + PCA fallback + unknown-driver projection | F04 | 8 | DONE (2026-07-02) |
| F09 | Corner-type decomposition + sub-fingerprints (ablation) | F03a, F03b | 9 | DONE (2026-07-18) |
| F10 | Model 3 — Degradation detector (deviation, rolling mean, alert threshold) | F04, F08 | 11–12 | DONE (2026-07-17) |
| F11 | Car personalisation engine (3 layers: mapping, constraints, confidence) | F07, F08 | 13 | DONE (2026-07-17) |

## Layer 3 — Platform
| ID  | Feature | Depends on | Wk | Status |
|-----|---------|-----------|----|--------|
| F12 | PostgreSQL schema + Alembic migrations | F00 | 15 | TODO |
| F13 | FastAPI endpoints + Pydantic contracts + health check | F06–F11, F12 | 15–16 | TODO |
| F14 | Redis cache + Celery queue + replay engine | F13 | 16 | TODO |
| F15 | Failsafe evaluation harness (RQ5 — 7 tests) | F13, F14 | 14 | TODO |

## Layer 4 — Frontend
| ID  | Feature | Depends on | Wk | Status |
|-----|---------|-----------|----|--------|
| F16 | Design system + Screen 1 Command Centre + Screen 2 Driver Profile | F13 | 17 | TODO |
| F17 | Screen 3 Ingestion/Prediction + Screen 4 Comparison | F13, F16 | 18 | TODO |
| F18 | Screen 5 Style Space (Three.js + D3 2D fallback) | F16 | 18 | TODO |
| F19 | Screen 6 Historical Replay Monitor | F14, F16 | 19 | TODO |

## Layer 5 — Evaluation & writing
| ID  | Feature | Depends on | Wk | Status |
|-----|---------|-----------|----|--------|
| F20 | Evaluation framework (classifier, stability, degradation, failsafe) | F06–F15 | 20 | TODO |
| F21 | Dissertation: Intro, Related Work, Design, Implementation | all | 21 | TODO |
| F22 | Dissertation: Evaluation, Discussion, Conclusion + submit | F20 | 22 | TODO |

## Dependency notes
- F04 is the leakage chokepoint: F06, F08, F10, F11 all consume its saved artefact.
  Get F04's train-only contract right before anything downstream.
- F08 (style space) feeds F10 (degradation) and F11 (personalisation).
- F13 cannot be finished until the models it serves (F06–F11) are DONE; stub the
  Pydantic schemas earlier (Wk 15) but wire real outputs after.
- The held-out circuits (Belgium, Italy) are quarantined from F00 onward; F20 is the
  only place they are evaluated.

## Amendments
- **F09 spec — 2026-07-17 (feature/F09-corner-decomposition):** `specs/F09-corner-decomposition.md`
  written (SPEC WRITTEN). Corner-type decomposition (§4.4): each lap's corners are segmented into
  **slow (<100 m) / medium (100–200 m) / high-speed (>200 m)** by **apex radius `r = 1/max(κ)`**,
  reusing F03b's curvature primitive (`_signed_lateral_g`) + F03a's `_corner_indices` corner windows
  (so a "corner" is identical to F25's). Each corner type yields a 5-feature sub-vector
  (`apex_speed`, `lateral_g`, `throttle`, `brake`, `steering_reversal_rate`), aggregated per
  `(driver, corner_type)` into the Screen-2 **sub-fingerprints** (`corner_subfingerprints.json`,
  content-hashed/timestamp-free). Headline deliverable = the **ablation** (§13/§15 "does corner-type
  info improve the classifier?"): a self-contained F06a-CV comparison of **base (28) vs expanded
  (28 + 15)** matrices → macro-F1 delta + `IMPROVES/NEUTRAL/DEGRADES` finding (NEUTRAL/DEGRADES is a
  legitimate result), logged to `apex-ablation`; does NOT retrain the champion. **No new error code**
  — `CORNER_001` is pre-seeded (`errors.py:59`) for the X/Y-unavailable **skip** (F03b's exact
  behaviour, lap continues); reuses `FEATURE_001`/`DATA_INSUFFICIENT`/`TRAIN_LEAKAGE`/etc.
  **Divergences #18–#22** (per-corner classification; the 5 sub-features; `min_corners_per_type`;
  self-contained ablation; radius units tied to F03b's curvature). Target `app/ml/corners.py`;
  depends on F03a/F03b (geometry) + F06a (CV) + F04 (aggregate scale). See §4.0/§9.
- **F09 build — 2026-07-18 (feature/F09-corner-decomposition): DONE.** `app/ml/corners.py` +
  `tests/ml/test_corners.py` (36 tests; full suite **650 green**, mypy + ruff clean; ml-auditor +
  failsafe-auditor folded). Two build-time refinements to the spec: (a) apex radius is `r = 1/κ[apex]`
  (the speed-minimum sample's central-difference κ, robust to the boundary artefacts that dominate
  `1/max(κ)`), fallback to `1/max(κ[window])`; (b) **divergence #23 — F09 is F04-free**: the 5 corner
  sub-features are new derived quantities F04 never scaled (`normalisation.json` has no params for
  them), the ablation scores via scale-invariant XGBoost, and F19 min-max-scales the radar at render —
  so the `NORMALISATION_FALLBACK` path was dropped, `CornerHandle` carries no `normalisation_params`,
  and the cross-block validator requires only `features`. Thin cells now surface `DEGRADED` +
  `DATA_INSUFFICIENT` on the typed result; the expanded ablation matrix is NaN-native (no imputation).
  **Divergences #18–#23.**
- **F11 spec — 2026-07-17 (feature/F11-personalisation):** `specs/F11-personalisation.md` written
  (SPEC WRITTEN). Model 4 (personalisation, §7) is a **3-layer engine**: Layer 1 learned mapping (one
  XGBoost regressor per setup parameter; input = F08 **5D fingerprint** ‖ drop-first circuit one-hot),
  Layer 2 constraint engine (FIA-bound clip → `CONSTRAINT_CLIPPED`; Monaco max-downforce override →
  `CIRCUIT_OVERRIDE_MONACO`; front-wing>10°+brake<54% → `CONSISTENCY_WARNING` flag-only; confidence
  floor <0.5 → `WIDE_CONFIDENCE` ±15 % range), Layer 3 confidence scoring (per-parameter reliability =
  `1 − CV-nRMSE`, OOD-attenuated, never-NULL). Produces the **10-parameter setup vector** (§7.3), a
  SHAP decomposition per recommendation (§6.1, F07 reuse), and the §7.4 unknown-driver onboarding
  package (style-match top-3 + radar + risk flags + data-quality summary, F08 `project_unknown`
  reuse). **No new error code** — the four §7 codes were pre-seeded in `errors.py:99–102`; reuses
  `INVALID_INPUT`/`MODEL_NOT_FOUND`/`SHAP_UNAVAILABLE`/`FALLBACK_THRESHOLD`/`INFERENCE_ERROR`/etc.
  Champion bundle `artefacts/personalisation.pkl` (10 regressors + reliability + explainers), MLflow
  experiment `apex-personalisation`, F06a leakage-quarantine + reproducibility discipline.
  **Divergences #13–#17** (setup-target approximation heuristic; "best-performing" = fastest-quartile
  clean laps; categorical suspension via ordinal regression; drop-first circuit one-hot;
  regression-confidence = `1 − CV-nRMSE`) + the §14 "only ~5 parameters reliably inferrable" finding
  reported honestly (never silently narrowed). Depends on F08 (5D fingerprints) + F07 (SHAP) + F06a
  (training pattern) + F04 (aggregates). See `specs/F11-personalisation.md` §4.0/§9. **Next: build.**
- **F10 BUILT — 2026-07-17 (feature/F10-degradation-detector):** `app/monitoring/detector.py` implements
  Model 3 in full — `calibrate_stint_variation` (the one train-only fit site: quarantine → tripwire →
  26-dim standardiser → Pass A slope stats / Pass B μ-σ / Pass C pit-window medians → deterministic
  content-hashed artefact → MLflow), `load_detector` (schema/hash/empty-thresholds/scaler-width/
  structural-knob refusals; crossing-knob mismatch → `pit_window_valid=False`), `monitor_stint`
  (never-raises; baseline → 26-dim cosine → 3-lap rolling → live-window F27/F28 slope → strict-`>`
  2-consecutive alert → exact 28-key attribution), `health`, and `lead_time_report` (§11.3). A shared
  pure `_score_stint_core` gives calibration/serving parity by construction. **72 tests** in
  `tests/monitoring/` + 5 `MonitoringConfig` tests; full suite green (**~554**), `mypy app` + `ruff`
  clean. `ml-auditor` + `failsafe-auditor` run on `app/monitoring/`; all high/medium findings fixed +
  test-covered (calibration write-fault → typed `MODEL_TRAINING_FAILED`; NaN-safe baseline `b_vec`;
  `_standardise_lap` floors at `normalise.min_std` not the module constant; never-raises pre-try escape
  on a `None` handle; per-candidate `LAP_001` rows on the `BASELINE_INSUFFICIENT` path). Two
  code-placement refinements folded into the spec (§5/§6/§7 + divergence #12): the slope-inactive DQ row
  reuses **`LAP_PARTIAL`** (keeps F03b's `FEATURE_001` count at zero, §7-11), and a single-lap `WATCH`
  crossing is the **persisted per-lap state**, not a mis-coded DQ row. `/spec-check F10` → DONE.
  See `specs/F10-degradation-detector.md`.
- **F10 spec — 2026-07-03 (apex-F10-degradation-detector):** `specs/F10-degradation-detector.md` written
  (SPEC WRITTEN), hardened by a 4-lens adversarial workflow (master-fidelity re-read of the PDF,
  leakage/reproducibility, failsafe, upstream-seams — 33 findings triaged, 2 blockers + all majors fixed).
  **Model 3 is a statistical deviation detector, not an ML model** (§5.3): per-lap cosine distance of the
  F04-normalised **then train-only-standardised** 26-feature vector (F27/F28 excluded — broadcast columns
  would double-count and leak future laps into replay) vs a first-3-clean-laps baseline (window extends
  to 5), 3-lap rolling mean, τ = μ + 2σ per driver per **circuit type** from
  `artefacts/stint_variation.json` (train-only, `_assert_no_leakage`-guarded, content-hashed,
  deterministic — **no timestamp in the artefact**), 2-consecutive-lap confirmation, F27/F28 live-window
  slopes as the §5.3 weighted component (F03b `compute_degradation_slopes` reused via a `StintLap`
  adapter, never invoked on prefixes it would NULL — no `FEATURE_001` flood). Alert payload: exact
  cosine-decomposition attribution (`1−cosθ = ½‖x̂−b̂‖²`, sums to the score), LOW/MEDIUM/HIGH band,
  pit-window median. **No new error code** — the six §5.3 codes were pre-seeded in `errors.py`; reuses
  `LAP_001` (baseline-exclusion/skip flag), `INVALID_INPUT`, `MODEL_NOT_FOUND`, `TRAIN_LEAKAGE`, etc.
  Review blocker fixed: F04 z-scores only 6/28 features, so a raw cosine would measure unit scale — F10
  fits a **train-only 26-dim standardiser** persisted inside the content-hashed artefact
  (**divergence #7**, the F08-#6 twin); further divergences **#8–#12** (slope formula/weight 0.25,
  confidence bands 2.5σ/3.0σ, pit-window method + stint-end proxy, SHAP-as-exact-decomposition,
  hygiene knobs + the structural-vs-crossing config-mismatch load policy). Depends on F04 + F08 — the
  deviation does **not** run in the 5D space (F08 §2's "5D baseline coordinate" paraphrase corrected:
  F08 supplies only the displayed `fingerprint_snapshot`, sparse-by-design under 30 laps). Ships
  `lead_time_report` (§11.3 in full, incl. per-circuit-type optimal threshold + FN/FP characterisation;
  Week-11 deliverable = read-only run over 2022 val-split stints), formally evaluated in F20.
  See `specs/F10-degradation-detector.md` §4.0/§9.
- **F08 aggregate scaler (Gate-2 fix) — 2026-07-02 (feature/F08-style-space):** a CONFIRMED code-review
  finding was fixed on the branch before merge: F04 z-scores only **6 of the 28** features, so the 56-dim
  driver aggregate mixed ~unit-scale columns with raw physical units spanning ~4 orders of magnitude
  (F16 gear-hold ms ≤10⁴, F12 speed variance ≤8·10³ vs 0–1 fractions), letting the largest-magnitude
  features dominate the euclidean UMAP/PCA fit — the style space would have measured unit scale, not
  style. The master is **silent** on aggregate scale (verified against the PDF: §5.2's "no refit"
  concerns the UMAP model; §4.3 is per-lap only), so F08 adds a **train-only aggregate `StandardScaler`**
  (fit inside the leakage-guarded build after the tripwire; `scale_` floored at `normalise.min_std`;
  persisted in `pca_fallback.pkl`; `scaler_sha256` cross-bundle fingerprint in `umap_model.pkl`; applied
  read-only at every forward pass via the single `_project_aggregate`/`ood_check` choke points;
  `stability_report` uses its own in-memory equivalent). Fail-loud: a pre-scaler/wrong-width bundle →
  `MODEL_NOT_FOUND` (never a silent unscaled load); a fingerprint mismatch → `UMAP_FALLBACK` degrade.
  **Master gap-fill, flagged to the supervisor as divergence #6** (spec §9.7). Same pass also pinned the
  previously-untested load-reconstruction degrade chain, the sparse-gate **signal-count** boundary, fixed
  fallback lap accounting (`n_laps_used`/`laps_rejected` honest on every path), added the
  `requirements.lock` BOM regression scan, and hardened the fit-site scan to a whole-module inverse scan.
  No config/error-code changes; aggregates persist RAW; `_risk_flags` deliberately stays in raw space
  (it z-scores per feature itself); the `PROJECTION_FALLBACK` raw-NN ranking standardises query + known
  rows through the same scaler (unscaled cosine is magnitude-dominated — §9.7 applies there too, a
  `/code-review high` catch). The same review added load-time dimension checks vs config (stale umap
  bundle → `UMAP_FALLBACK`; stale pca bundle → `MODEL_NOT_FOUND`), a stats-less-scaler typed refusal,
  backstop preservation of `sparse`/`used_global` + honest pre-fault accounting `(0, n_input)`, and
  empty (never NaN) neighbours on a non-finite query. Coordinates changed vs the raw-space fit; no
  consumer existed (F10/F11/F13/F16/F18
  unbuilt, no fitted artefact committed) — migration-free. See `specs/F08-style-space.md` §4/§6/§7/§9.7.
- **F08 spec — 2026-07-01 (apex-F08-style-space):** `specs/F08-style-space.md` written (SPEC WRITTEN),
  hardened by a 4-lens adversarial workflow (master-fidelity — 0 findings; leakage/reproducibility;
  failsafe; house-convention). **Model 2**: aggregate each driver's clean **normalised training** laps to
  a **56-dim** distribution (mean+std of the 28 features), fit **UMAP → 5D fingerprint + 2D viz** and a
  **PCA fallback**; unknown drivers project by a **forward pass** (`umap.transform`, no retrain) gated for
  sparsity (`SPARSE_PROJECTION`, <30 laps → reuses `thresholds.min_clean_laps`) and out-of-distribution
  (`OUT_OF_DISTRIBUTION`, Mahalanobis > 3 SD on any PC). **Fail-loud fallback chain:** UMAP corrupt/missing
  → PCA (`UMAP_FALLBACK`); `transform()` throws → raw-56D cosine NN (`PROJECTION_FALLBACK`); no usable
  engine → `MODEL_NOT_FOUND` (503). Depends on **F04** only (consumes `normalisation.json` read-only +
  the shared F06a training matrix; **not** the F06 classifier). Like F07, **no new error code** — the four
  §5.2 codes + the `OUT_OF_DISTRIBUTION` `ConfidenceState` are pre-seeded; reuses `TRAIN_LEAKAGE` /
  `MODEL_TRAINING_FAILED` for the two fit sites (`build_style_space` + the dissertation-only
  `stability_report`, both train-only + `_assert_no_leakage`-guarded). Adds a typed `stylespace:` block +
  its own MLflow URI (no coupling to the optional `classifier` block). Mirrors F06b's load/handle/health +
  never-raises serving. **Feeds F10** (5D fingerprint) **and F11** (nearest-neighbours / style-match /
  risk flags); stability (10-seed Procrustes < 0.05) + teammate-proximity functions are produced here,
  formally evaluated in **F20**. **Stack:** adds `umap-learn` (+`pynndescent`); no numpy movement expected
  (`numba` already caps `numpy<2.5`). Two limitations flagged for the supervisor: (1) the master fixes the
  UMAP matrix at **8 points** (small-sample → PCA-primary fallback if stability > 0.05); (2) UMAP is
  **stochastic** — coordinates reproducible within tolerance via fixed seed but `umap_model.pkl` is **not**
  byte-identical (PCA fallback is). See `specs/F08-style-space.md` §9/§10.
- **F07 DONE — 2026-07-01 (feature/F07-shap-explainability):** `app/ml/explain.py` — TreeExplainer
  (exact) over the F06 champion's **base XGBoost** (`calibrated_classifiers_[0].estimator.estimator`);
  local force + 6-domain radar + top-2 NL, global beeswarm artefact, and multi-season Jensen-Shannon
  stability. Explainability **never blocks a prediction** (SHAP fault → `SHAP_UNAVAILABLE`); reuses
  the pre-seeded `SHAP_UNAVAILABLE`/`SHAP_CONSISTENCY_WARN` codes — **no new code**. Adds a typed
  `shap:` block (`ShapConfig`). **Stack movement:** installs **SHAP 0.52**, whose `numba` dep caps
  `numpy<2.5`, so the pin moved **numpy 2.5.0 → 2.4.6** (cp314 wheel; full F01–F06 suite re-run green
  on 2.4.6 before any F07 code) — ratified like F04/F06a; back-noted in CLAUDE.md + `pyproject`.
  Gates: failsafe-auditor + ml-auditor + `/code-review high` all run, findings fixed with regression
  tests (build-global class-order keyed off `explainer.drivers`; zero-contribution domain never named;
  all-NaN season → `None` not 0.0; explainer/handle driver-order guard). See
  `specs/F07-shap-explainability.md`.
- **F06 split — 2026-06-27 (apex-F06a-classifier-training):** F06 (XGBoost classifier) is split
  into **F06a** (training / Optuna ≥200 trials / `CalibratedClassifierCV` / MLflow champion
  registration / held-out evaluation) and **F06b** (inference / confidence gates / serving the
  `PredictionResult` / startup health). **Seam = the registered champion model:** F06a produces +
  registers it; F06b loads it and never retrains. F06a is also where the real
  `artefacts/normalisation.json` first materialises (F04 shipped machinery only). F06a adds two new
  error codes — `TRAIN_LEAKAGE` (build-breaking leakage in a CV/calibration fold) and
  `MODEL_TRAINING_FAILED` (trial-scope + run-scope). The five runtime gate codes
  (`LOW_CONFIDENCE`/`AMBIGUOUS`/`INVALID_INPUT`/`PERF_WARN`/`MODEL_NOT_FOUND`) stay with F06b. The
  original combined spec remains at `specs/F06-driver-classifier.md`; F06a is
  `specs/F06a-classifier-training.md`. ml-auditor is the F06a gate (CV-fold leakage). See
  `specs/F06a-classifier-training.md`. **F06a DONE + merged 2026-06-30.**
- **F06b spec — 2026-06-30 (apex-F06a-classifier-training):** `specs/F06b-classifier-inference.md`
  written (SPEC WRITTEN), critiqued by a 3-lens adversarial workflow. F06b loads the champion +
  `normalisation.json` read-only (joblib↔joblib seam), applies the gates, returns a typed
  `PredictionResult`; **never refits**. Adds **one** new error code `INFERENCE_ERROR` (unexpected-
  internal backstop, mirrors F02 `VALIDATION_ERROR`); reuses the five §5.1 codes + `HEALTH_FAIL`.
  Adds config `classifier.confidence_high` (HIGH/MEDIUM split — master pins only the 0.65 LOW gate),
  `latency_warn_ms`, and `labels_filename`. **Two open divergences flagged for the supervisor:**
  (1) the champion is **28-feature**, not §5.1's 34-dim (circuit-type + season one-hot dropped;
  circuit context enters via F04 normalisation) — **inherited from F06a** (back-noted in F06a §9);
  (2) `confidence_high` HIGH/MEDIUM boundary is undefined by the master. See
  `specs/F06b-classifier-inference.md` §3/§9.
- **F06a labels.json seam-completion — 2026-06-30 (feature/F06b-classifier-inference):** F06a's
  champion is fit on **integer** labels (`classes_ == [0..n-1]`), so the index→driver-name map was
  unavailable at the seam. F06a's `train_classifier` now also writes `artefacts/labels.json` (the
  sorted driver names) next to `classifier.pkl` (+ `labels_artefact_path` on the result), and F06b's
  `load_model` cross-checks it against the champion's class count, refusing to serve (fail-loud
  `MODEL_NOT_FOUND`) on any mismatch. Additive seam-completion (precedent: F01 `track_status`),
  built on the F06b branch.

- **F03 split — 2026-06-26 (feature/F03a-scalar-features):** the 28-feature extraction is
  split into **F03a** (the 22 scalar-channel features F01–F18, F22–F24, F26 computable
  directly from validated telemetry, plus the shared infrastructure: the 28-key vector
  contract, `FeatureComputationError` + catch-mark-NULL-log behaviour, and the §4.5 6-NULL
  tolerance / imputation / `LAP_NULL_EXCESS` gate) and **F03b** (the X/Y-derived cornering
  features F19–F21, the corner-balance F25, and the tyre-age slopes F27–F28, which plug into
  F03a's vector without changing it). F03a adds a new `IMPUTED` error code. Downstream
  consumers (F04, F09) now depend on both. See `specs/F03a-scalar-features.md`.

- **F01 — 2026-06-26 (feature/F01-track-status):** additive change — raw per-lap
  `track_status` (FastF1 `Laps.TrackStatus`) is now persisted to the Parquet cache and
  added to the F01→F02 column contract; new `TRACK_STATUS_UNAVAILABLE` error code. See
  `specs/F01-data-ingestion.md` §11.
- **F02 — 2026-06-26 (feature/F02-lap-filtering):** DONE. Added validator
  (`app/pipeline/validate.py`), `LapStatus` enum, `ValidateConfig` block, and two F02
  error codes from review hardening: `ORPHAN_SAMPLES` (null-lap samples dropped + logged)
  and `VALIDATION_ERROR` (malformed-frame / unexpected-internal backstop, kept distinct
  from per-lap data codes). `ValidateConfig` enforces `Speed ∈ required_channels`.

- **F04 — 2026-06-26 (feature/F04-circuit-aware-normalisation):** DONE. Authored the missing
  spec (`specs/F04-circuit-aware-normalisation.md`) and shipped the **leakage-chokepoint
  machinery**: `app/pipeline/normalise.py` (one writer `fit_normaliser`, two readers
  `load_normalisation_params` / `apply_normalisation` — readers provably contain no
  `.fit`/`groupby`), the typed `NormaliseConfig` block + `NORMALISED_FEATURE_IDS` tuple
  (F09/F11/F13/F14/F19/F20), and the deterministic content-hashed artefact. Reuses the existing
  `NORMALISATION_FALLBACK` code (no new code). **No real `artefacts/normalisation.json` is
  committed** — there is no training matrix until the F06 era; this session ships the machinery
  only. The per-circuit fit floor reuses `thresholds.min_clean_laps` (30), defined per
  driver-circuit in §3.2 but applied per-circuit here (documented design decision).
