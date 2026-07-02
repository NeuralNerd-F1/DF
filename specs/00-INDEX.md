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
| F09 | Corner-type decomposition + sub-fingerprints (ablation) | F03a, F03b | 9 | TODO |
| F10 | Model 3 — Degradation detector (deviation, rolling mean, alert threshold) | F04, F08 | 11–12 | TODO |
| F11 | Car personalisation engine (3 layers: mapping, constraints, confidence) | F07, F08 | 13 | TODO |

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
