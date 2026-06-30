# Feature F06a — Model 1: Driver Classifier — Training / Tuning / Calibration / MLflow

> One spec, one session. Self-contained: build this from this file + CLAUDE.md only.
> F06a is the **offline training half** of F06. It produces and registers the champion
> model; the inference/serving half (gates, `PredictionResult`, latency, startup health)
> is **F06b** and is out of scope here. **Seam between the two = the registered champion
> model** (see INDEX amendment + the F06 split plan).

**Status:** SPEC WRITTEN
**Master spec sections:** §5.1 (Model 1 — classifier: algorithm, split, CV, tuning, tracking,
seeds, target, baselines), §11.1 (classifier evaluation: macro-F1, per-driver F1, confusion
matrix, calibration curve, baseline comparison, normalisation ablation), §9 tech-stack rows
(XGBoost 2.x, Optuna 3.x SQLite backend, MLflow 2.x), Project Plan Wk 5–6, Risk Register row 1
**Layer / Week:** ML / Wk 5–6
**Depends on:** F04 (normalisation machinery + the leakage chokepoint), F03a/F03b (the 28-feature
matrix). Splits the existing `specs/F06-driver-classifier.md` into F06a (this) + F06b (serving).

## 1. Purpose
F06a trains the 8-class driver-fingerprint classifier and registers the **champion** model that
every downstream consumer depends on. It assembles the F03 feature matrix, **produces the real
`artefacts/normalisation.json` for the first time** (F04 shipped the machinery only), runs the
Optuna hyperparameter search (≥200 trials) under grouped cross-validation, calibrates the
champion's probabilities on a disjoint hold-out, logs every trial to MLflow, and measures
macro-F1 on the fully-withheld circuits (Belgium, Italy). Its single most important contract is
**no normalisation leakage** — held-out circuits and the calibration split must never touch a
`.fit()`/CV fold — because if they do, every accuracy number F06–F11 reports is silently
inflated. F06a returns a typed `ClassifierTrainingResult` and never serves a prediction (that is
F06b).

## 2. Upstream / downstream contracts
- **Consumes:**
  - The assembled wide feature matrix from **F03a/F03b** — one row per accepted lap: the 28
    canonical keys (`F01..F28`), plus `driver` (label), `circuit`, `season`, and a `split`
    column (`train`/`val`/`test`). Belgium and Italy rows carry `circuit ∈ held_out_circuits`.
  - **F04** normalisation machinery, used exactly as F04 defines it: `fit_normaliser` (the ONLY
    writer; train circuits only), then `load_normalisation_params` + `apply_normalisation`
    (read-only). F06a contains **no scaler fit of its own** beyond this single F04 call.
  - `config.yaml` → new `classifier:` block + referenced `seeds`, `held_out_circuits`, and
    `thresholds.confidence_gate` / `thresholds.entropy_ambiguous_bits` (referenced for the
    metadata stamped onto the champion, **applied** only in F06b).
- **Produces:**
  - **`artefacts/normalisation.json`** — the real artefact, produced here for the first time by
    F04's `fit_normaliser` on the actual training laps (F04 DoD note: "produced when F06 first
    fits on real laps").
  - **Champion model** registered in the **MLflow** model registry **and** persisted to
    `artefacts/<classifier.model_artefact_filename>` (the on-disk file F06b loads at startup).
  - `ClassifierTrainingResult` (run summary, below), plus the MLflow run/trial ledger
    (params + macro-F1 + per-class F1 + confusion matrix + calibration curve + model artefact
    per the champion run; params + objective per trial; one run per baseline).
- **Consumed by:** **F06b** (loads the champion + applies the gates), **F07** (SHAP TreeExplainer
  on this exact champion), **F08/F10/F11** (consume the same `normalisation.json`), **F20**
  (re-runs held-out evaluation as the RQ1 evidence).

## 3. Inputs & outputs (typed)
- **Train entry point (the only writer of model/artefacts):**
  `train_classifier(laps: DataFrame, *, config: AppConfig, artefact_dir: Path | None = None,
  mlflow_tracking_uri: str | None = None, log_path: Path | None = None) ->
  ClassifierTrainingResult`, where `laps` is the wide matrix (`F01..F28`, `driver`, `circuit`,
  `season`, `split`). Default `artefact_dir = paths.artefacts_dir`.
- **Outputs:**
  - `ClassifierTrainingResult(BaseResult)`:
  { mlflow_run_id: str, model_uri: str, model_artefact_path: str,
  champion_params: dict[str, float | int | str], n_trials: int, n_trials_failed: int,
  cv_strategy: str,                       # "stratified_group_kfold_by_circuit"
  cv_folds: int,
  train_circuits: list[str], excluded_circuits: list[str],   # == held_out_circuits
  calibration_method: str, calibration_holdout_frac: float,
  held_out_macro_f1: float,               # Belgium + Italy
  held_out_per_driver_f1: dict[str, float],   # 8 drivers
  confusion_matrix_path: str, calibration_curve_path: str,
  baseline_macro_f1: dict[str, float],    # majority / logistic_regression / random_forest
  beats_all_baselines: bool,
  normalisation_ablation_macro_f1: float, # train-circuit F1 with normalisation OFF (§11.1)
  target_macro_f1: float, target_met: bool,        # held_out_macro_f1 >= target
  per_driver_floor: float, per_driver_floor_met: bool,
  seeds: dict[str, int], normalisation_content_hash: str,
  status, error_code, flags, logged }
  - `TrialRecord(BaseModel) { trial_number: int, params: dict, macro_f1: float | None,
    state: str }` — one per Optuna trial (logged to MLflow; failed/pruned trials carry
    `macro_f1=None`).
- **Exception:** `ClassifierTrainingError(Exception)` carrying an `ErrorCode`, raised **only** on
  the build-breaking fail-loud paths (leakage detected; no valid champion produced; registration
  failed). A degraded-but-successful run (target missed, a baseline not beaten, individual trials
  failed) **returns** a `ClassifierTrainingResult` with `status=DEGRADED` + a coded `flags` entry
  — it does not raise.

## 4. Method / algorithm
All ops vectorised (pandas 3.x / numpy 2.x — no per-row Python loops). Every seed is read from
`config.seeds` (`numpy=42`, `xgboost=42`, `optuna=42`); a rerun reproduces the same champion
metrics bit-for-bit.

1. **Split & quarantine (before anything fits).** Partition `laps` by the `split` column.
   Belgium and Italy (`config.held_out_circuits`) are the held-out **test** set and are set aside
   **first**; they enter only at step 8, transformed through *loaded* params. A `_assert_no_leakage`
   guard re-checks that no held-out circuit is present in the train/val/calibration partitions and
   **raises `ClassifierTrainingError`** (`TRAIN_LEAKAGE`) if one is — mirrors F04's fail-loud
   pattern; **no model is written on a poisoned run.**
2. **Produce normalisation params (F04, once).** Call `fit_normaliser(train_laps, config=...)`
   → writes the real `artefacts/normalisation.json` (train circuits only; F04's own
   `_assert_no_leakage` is the second guard). Then `load_normalisation_params(...)` and
   `apply_normalisation(...)` every row of every split. **F06a never calls `.fit` on a scaler
   itself and never re-normalises held-out data with held-out statistics.**
3. **Carve a disjoint calibration hold-out.** From the training rows reserve
   `classifier.calibration_holdout_frac` (stratified by `driver`) as a calibration set that is
   **excluded from the Optuna fit folds**. Calibration data and fit data are provably disjoint
   (asserted) so calibration does not leak into training.
4. **Baselines (logged first, Wk 5).** Train and log to MLflow, on the same normalised features:
   **majority-class**, **logistic regression**, **random forest (default params)**. Record each
   model's macro-F1.
5. **Optuna Bayesian search (Wk 6).** `optuna` study, **TPE sampler seeded 42**, **SQLite storage
   backend** (`optuna_storage` — trial persistence across restarts, §9). **Minimum
   `classifier.optuna_trials` (200) trials.** Each trial trains XGBoost (2.x, `seed=42`,
   8-class `multi:softprob`) under **grouped K-fold CV** —
   `StratifiedGroupKFold(n_splits=cv_folds)` with `y = driver` and `groups = circuit`, so each
   circuit lands wholly in a single fold while the driver mix stays balanced; objective =
   **macro-averaged F1 on the validation fold**. Every trial logs params + objective to MLflow as
   a `TrialRecord`. A trial that raises or yields a NaN objective is **pruned, logged**
   (`MODEL_TRAINING_FAILED` at trial scope, `n_trials_failed += 1`), and the study continues — it
   is never silently dropped.

   > **Design note — CV splitter choice (ml-auditor will probe this).** §5.1's wording is
   > internally contradictory: "stratified 5-fold, stratified by driver × circuit, **to prevent
   > circuit leakage**." *Stratifying* by a driver×circuit composite (a single `StratifiedKFold`
   > label) puts a proportional slice of **every** driver×circuit combination into **every** fold,
   > so the same circuit recurs across train and validation folds — it balances representation but
   > does **not** prevent circuit leakage. Because the Optuna search optimises against the CV
   > validation-fold macro-F1 (step 5), a circuit-contaminated CV signal would steer all ≥200
   > trials of champion selection on an inflated objective, *before* the held-out set is ever seen
   > (step 8). To honour §5.1's **stated intent** ("to prevent circuit leakage") we implement the
   > splitter that actually achieves it: **`StratifiedGroupKFold` grouped by circuit, stratified by
   > driver** — each circuit sits in exactly one fold, so CV macro-F1 is an honest
   > unseen-training-circuit estimate and the search optimises against a leakage-clean objective.
   > **This is a deliberate divergence from §5.1's literal composite wording**, recorded here and
   > in §9; it is **pending confirmation with the supervisor**. If the supervisor prefers the
   > literal composite `StratifiedKFold`-by-driver×circuit, this flips to that splitter and the
   > resulting circuit-familiarity inflation is reported as an F20/RQ5 finding rather than silently
   > shipped. The held-out Belgium/Italy quarantine (step 8) remains the final generalisation test
   > regardless of the CV splitter.
6. **Champion fit + calibration.** Refit XGBoost on the full fit-set with the best params, then
   wrap in **`CalibratedClassifierCV`** (`method = classifier.calibration_method`, fit on the
   disjoint calibration hold-out via `cv="prefit"` so calibration never re-sees the fit data).
7. **Register champion.** Log the champion run (params, macro-F1, per-class F1, confusion matrix,
   calibration curve, model artefact) and **register it as champion** in the MLflow registry; also
   persist the calibrated model to `artefacts/<model_artefact_filename>` for F06b. If no trial
   produced a valid model, **raise `ClassifierTrainingError`** (`MODEL_TRAINING_FAILED`).
8. **Held-out evaluation (§11.1).** Predict Belgium + Italy (normalised via *loaded* params) and
   compute **macro-F1** (target `classifier.target_macro_f1` = 0.78), **per-driver F1** (floor
   `classifier.per_driver_floor` = 0.70), the **confusion matrix** heatmap, and the **calibration
   curve**. Compute the **normalisation ablation** (train-circuit macro-F1 with normalisation off)
   as a reported number. Missing the target or a per-driver floor sets `status=DEGRADED` + a flag
   and triggers the §10 contingency reporting — it is a research finding, **not** a raise.

**Leakage is the headline contract (ml-auditor gate).** Three independent guards: (a) held-out
rows set aside before any fit and re-checked by `_assert_no_leakage`; (b) F04's own fail-loud
`fit_normaliser`; (c) calibration/fit disjointness assertion. A static source scan asserts F06a's
held-out evaluation path calls `apply_normalisation` (never `fit_normaliser`/`.fit`).

## 5. config.yaml keys this feature adds
A new `classifier:` block + a typed `ClassifierConfig` sub-model in `app/contracts/config.py`
wired onto `AppConfig` (standing rule: typed sub-model, `extra="forbid"`, no untyped passthrough —
see [[typed-config-submodels-in-config-py]]). The **gate** constants
(`confidence_gate=0.65`, `entropy_ambiguous_bits=2.9`, `null_ceiling=6`) already live under
`thresholds:` and are **referenced** for champion metadata, **not** re-declared (they are
*applied* in F06b). `seeds` and `held_out_circuits` are referenced, never redefined.
```yaml
# --- F06a classifier training (offline; produces + registers the champion) ---
classifier:
  cv_folds: 5                        # §5.1 K-fold count
  cv_strategy: stratified_group_kfold_by_circuit  # StratifiedGroupKFold (see §4 design note)
  cv_stratify_by: driver             # class balance across folds
  cv_group_by: circuit               # each circuit held wholly in ONE fold (true leakage guard)
  optuna_trials: 200                 # §5.1 minimum trial count (do not ship fewer)
  optuna_objective: macro_f1         # §5.1 macro-averaged F1 on the validation fold
  optuna_storage: sqlite:///artefacts/optuna_classifier.db  # §9 persistence across restarts
  calibration_method: isotonic       # §5.1 CalibratedClassifierCV method (isotonic|sigmoid)
  calibration_holdout_frac: 0.2      # disjoint calibration split (excluded from fit folds)
  baselines: [majority, logistic_regression, random_forest]  # §5.1 / §11.1 comparators
  target_macro_f1: 0.78              # §5.1 / §11.1 held-out (Belgium, Italy) target
  per_driver_floor: 0.70             # §11.1 per-driver F1 floor (below = research finding)
  mlflow_experiment: apex-classifier # MLflow experiment name
  mlflow_registered_model: apex-driver-classifier  # champion registry name
  model_artefact_filename: classifier.pkl  # calibrated champion under paths.artefacts_dir (F06b loads)
```

## 6. Error codes & failure modes (the failsafe contract)
F06a is an **offline training** stage; its failures are build-breaking integrity violations
(fail loud) or reported quality findings (degraded result), not per-prediction runtime states.
The five **runtime gate codes** (`LOW_CONFIDENCE`, `AMBIGUOUS`, `INVALID_INPUT`, `PERF_WARN`,
`MODEL_NOT_FOUND`) are **owned by F06b** and are *not* exercised here.

| Failure mode | Detection | Error code | System response | Logged? | User-facing message |
|--------------|-----------|-----------|-----------------|---------|---------------------|
| Held-out circuit (Belgium/Italy) in train/val/calibration | `_assert_no_leakage` after split | `TRAIN_LEAKAGE` *(new)* | **raise `ClassifierTrainingError`; no model or artefact written** | n/a (build-breaking; surfaced to caller) | "Training aborted: held-out circuit present in training input." |
| Calibration split overlaps fit split | disjointness assertion | `TRAIN_LEAKAGE` *(new)* | **raise `ClassifierTrainingError`; no champion registered** | n/a (raised) | "Training aborted: calibration set leaks into training set." |
| An Optuna trial raises / NaN objective | per-trial try + NaN check | `MODEL_TRAINING_FAILED` *(new)* | prune trial, `n_trials_failed += 1`, **study continues** | yes (trial number + reason; MLflow trial state=FAIL) | none (internal) |
| No valid champion (every trial failed) or registration fails | `n_valid_trials == 0` / MLflow error | `MODEL_TRAINING_FAILED` *(new)* | **raise `ClassifierTrainingError`; no champion** | yes (run-scope DQ row) | "Classifier training produced no valid model." |
| Held-out macro-F1 < `target_macro_f1` (0.78) | post-eval compare | `MODEL_TRAINING_FAILED` *(flag, run continues)* | `status=DEGRADED`; flag set; champion still registered; confusion analysis + §10 contingency reported | yes (run-scope; target + achieved) | none (research finding, reported in dissertation) |
| A driver's held-out F1 < `per_driver_floor` (0.70) | per-driver compare | `MODEL_TRAINING_FAILED` *(flag)* | `status=DEGRADED`; flag set; champion still registered; named in report | yes (driver + F1) | none (research finding) |
| XGBoost does not beat all 3 baselines | `beats_all_baselines == False` | `MODEL_TRAINING_FAILED` *(flag)* | `status=DEGRADED`; flag set; baseline-comparison analysis reported | yes (per-baseline macro-F1) | none (research finding) |

`TRAIN_LEAKAGE` and `MODEL_TRAINING_FAILED` are **new** codes this feature adds to
`app/contracts/errors.py` (precedent: F02 added `ORPHAN_SAMPLES`/`VALIDATION_ERROR` during
hardening). `MODEL_TRAINING_FAILED` is carried at **two scopes** — a logged per-trial row
(study continues) **and** the returned result's `error_code`/`flags` on a degraded-but-successful
run, or the raised exception when zero trials succeed — analogous to F04 carrying
`NORMALISATION_FALLBACK` at both row and result scope. The **User-facing message** column is
illustrative API copy (F13 would render it); the contract callers branch on is the `error_code`.

## 7. Acceptance criteria (each item mechanically testable by /spec-check)
- [ ] **Leakage guard (primary):** `_assert_no_leakage` **raises `ClassifierTrainingError`**
      (`TRAIN_LEAKAGE`) when Belgium/Italy appear in any train/val/calibration partition; **no
      model or artefact is written** on that run.
- [ ] **Held-out never fitted:** Belgium/Italy enter only the final evaluation and are transformed
      via `apply_normalisation` with *loaded* params; a **static source scan** asserts the
      held-out evaluation path contains no `fit_normaliser`/`.fit(`.
- [ ] **Normalisation is F04's, produced once:** F06a calls `fit_normaliser` exactly once (train
      circuits), then only `load_normalisation_params`/`apply_normalisation`; the resulting
      `normalisation.json` `content_hash` is recorded on the result.
- [ ] **Calibration disjoint:** the calibration hold-out and the Optuna fit folds share **zero
      rows** (asserted); `CalibratedClassifierCV` calibrates a `FrozenEstimator`-wrapped (prefit)
      base model on the disjoint set (sklearn ≥1.6 replacement for `cv="prefit"`).
- [ ] **CV strategy:** cross-validation uses **`StratifiedGroupKFold` (`cv_folds=5`) grouped by
      `circuit`, stratified by `driver`**, such that **no `circuit` appears in both the train and
      validation index of any fold**; `cv_strategy == "stratified_group_kfold_by_circuit"` on the
      result. A **fold-membership test** inspects each fold's train/val indices and asserts the
      circuit sets are disjoint.
- [ ] **Trial floor:** the study runs **≥ `optuna_trials` (200)** non-failed trials; a run that
      cannot reach the floor is a failure, not a silent short study.
- [ ] **Trial-failure handling:** a trial that raises/NaN is pruned, `n_trials_failed` increments,
      it is logged, and the **study continues** (asserted via an injected failing trial).
- [ ] **Baselines logged:** all three baselines (majority, logistic regression, random forest) are
      trained on the same features and logged to MLflow; `baseline_macro_f1` has all three keys;
      `beats_all_baselines` is computed.
- [ ] **Champion registered:** the calibrated champion is registered in the MLflow registry
      (`model_uri` set) **and** written to `artefacts/<model_artefact_filename>` (the file F06b
      loads).
- [ ] **MLflow completeness:** the champion run logs params, macro-F1, per-class F1, confusion
      matrix, calibration curve, and the model artefact.
- [ ] **Held-out evaluation:** `held_out_macro_f1` and `held_out_per_driver_f1` (8 keys) are
      measured on Belgium+Italy; `target_met` ⇔ `held_out_macro_f1 >= target_macro_f1` (0.78);
      `per_driver_floor_met` ⇔ all 8 ≥ 0.70.
- [ ] **Normalisation ablation:** `normalisation_ablation_macro_f1` (train-circuit F1, normalisation
      off) is reported (§11.1 key result).
- [ ] **Degraded, not crashed:** a run that misses the target / a per-driver floor / a baseline
      returns `status=DEGRADED` + the `MODEL_TRAINING_FAILED` flag and **still registers a
      champion** (does not raise).
- [ ] **Reproducibility:** all seeds read from `config.seeds`; training twice on the same input
      yields identical champion macro-F1 and identical `champion_params`.
- [ ] **No magic numbers:** every constant (folds, trial count, fractions, targets, filenames,
      stratify/group keys) is read from `config.yaml`; a source scan asserts `classifier.py` has no
      numeric/string literal for any of them.
- [ ] Every error code above (`TRAIN_LEAKAGE`, `MODEL_TRAINING_FAILED`) has a unit test that
      triggers it and asserts the result/exception.
- [ ] **Scope fence:** F06a contains no inference path — no `PredictionResult`, no gate
      application (`confidence_gate`/`entropy`/`null_ceiling`); a source scan confirms those are
      deferred to F06b.

## 8. Test plan
Tests run **fast** — XGBoost/Optuna are exercised with a tiny synthetic matrix, `optuna_trials`
overridden to a small N via the config fixture, and `cv_folds` reduced where a criterion does not
pin the real value. The **200-trial / 0.78-target** acceptance items assert the *wiring reads the
config value*, not that a real 0.78 is hit on synthetic data.
- **Fixtures (`tests/ml/conftest.py`):** a `classifier_config` fixture (real repo config with
  `paths.artefacts_dir` + `paths.data_quality_log` under `tmp_path`, MLflow + Optuna tracking on
  **`sqlite://` stores** under `tmp_path` — the model registry is unsupported on `file://` stores,
  so sqlite is mandatory, not a choice — small `optuna_trials`); a `training_matrix(...)` builder
  producing the wide matrix (`F01..F28`, `driver`, `circuit`, `season`, `split`) with knobs for
  driver/circuit/season counts, laps-per-combo, separable-vs-overlapping driver signal, held-out
  rows, and an injectable failing-trial hook.
- **Happy path:** a separable synthetic matrix → a champion is registered + written to disk, a real
  `normalisation.json` is produced, `ClassifierTrainingResult` is `OK`, all MLflow artefacts logged.
- **Failure / boundary fixtures (one per §6 row + each boundary in §7):** held-out-in-train →
  raises `TRAIN_LEAKAGE` (no files written); calibration/fit overlap → raises `TRAIN_LEAKAGE`;
  **fold-membership disjointness** (no circuit in both train and val of any fold, asserted on the
  real `StratifiedGroupKFold` splitter); injected failing trial → pruned + counted + study
  continues; all-trials-fail → raises `MODEL_TRAINING_FAILED`; missing target / per-driver floor /
  un-beaten baseline → `DEGRADED` + flag + champion still registered; reproducibility (train twice
  → identical champion macro-F1 + params); the static source scans (held-out path has no `.fit`;
  one `fit_normaliser` call; no-magic-number; no inference path).

## 9. Definition of done
All acceptance criteria MET with passing tests; **ml-auditor PASS** on leakage (held-out + CV
folds + calibration disjointness) + seeding + calibration + baseline logging — this is the gate
that matters for F06a; `/apex-review F06a` clean (no silent failure, no normalisation leakage,
config-driven thresholds, error-code discipline, no bare `except`); `/spec-check F06a` green;
`mypy app` clean. INDEX F06 row updated (F06a DONE; F06b TODO). **Limitations for the
dissertation:** (1) F06a serves nothing — gates and `PredictionResult` are F06b. (2) This is where
the real `artefacts/normalisation.json` first materialises (F04 shipped machinery only); its
content_hash is recorded for downstream reproducibility. (3) Setup-related labels are not used;
this model only identifies drivers. (4) If the held-out target (0.78) is not met, the §10
contingency is reported honestly with confusion analysis rather than tuned-to-fit. (5)
**CV-splitter divergence from the master spec:** §5.1 literally says "stratified by driver×circuit
to prevent circuit leakage"; that composite does not hold circuits out, so F06a implements
`StratifiedGroupKFold` grouped by circuit to satisfy §5.1's stated anti-leakage intent. This is a
**deliberate, documented divergence pending confirmation with the supervisor**; if the literal
composite is preferred, the splitter flips and the resulting leakage is reported as an F20/RQ5
finding. (6) **Dependency-stack migration (ratified by F06a, mirrors F04).** The constitution pins
XGBoost 2.x / Optuna 3.x / MLflow 2.x; on Python 3.14 + pandas 3.x that is unbuildable (full
`mlflow` caps `pandas<3` and would downgrade the F04-ratified stack). F06a pins the newest releases
with 3.14 wheels — `xgboost==3.3.0`, `optuna==4.9.0`, `scikit-learn==1.9.0`, **`mlflow-skinny==3.14.0`**
(the lightweight tracking+registry client, same `mlflow.*` API, no bundled server/pandas) — on a
**sqlite** tracking store. MLflow model logging uses `serialization_format="cloudpickle"` (skinny
omits the skops default). (7) **`CalibratedClassifierCV` API:** scikit-learn ≥1.6 removed
`cv="prefit"`; F06a wraps the fitted base model in `FrozenEstimator` and fits the calibrator on the
disjoint hold-out — identical leakage-safe behaviour to the old prefit mode. (8) Input-schema
divergence (28 vs 34): §5.1 specifies a 34-dim input (28 features + circuit-type one-hot + season
one-hot); F06a trains a 28-feature champion, encoding circuit context via circuit-aware
normalisation (F04) rather than as a model input, with season unused. Deliberate, leakage-aware
design decision (keeps circuit identity out of the model); adding circuit-type + season as inputs
is held in reserve as risk-register contingency (1) if held-out macro-F1 falls below target.
Flagged to supervisor alongside the CV-splitter divergence.

## 10. Risks & mitigations (Risk Register row 1)

- **Normalisation leakage re-entering through CV folds (non-negotiable #2):** the headline
  F06a risk. Mitigated by three guards: (i) the held-out set is set aside before any `.fit()`,
  asserted by `_assert_no_leakage`; (ii) F04's fail-loud `fit_normaliser` / `apply_normalisation`
  split, with the held-out path only ever calling `apply_normalisation`; (iii) the
  `CalibratedClassifierCV` calibration set is asserted disjoint from the training folds. A static
  source scan confirms no downstream path recomputes a scaler. The ml-auditor is the gate.

- **Macro-F1 < 0.78 on held-out circuits (Belgium, Italy):** contingencies are applied and
  reported verbatim; metrics are never tuned to fit the target. In order:
  (1) **First** — add frequency-domain features (FFT on throttle and speed traces).
  (2) **Second** — reduce classification from 8 drivers to 6, dropping PER and SAI (the
  most-confused midfield pair per the master spec).
  (3) **Third** — relax the target to 0.72 and report honestly, with the confusion matrix and a
  threshold-sensitivity analysis.
  Systematic teammate confusion (e.g. VER–PER as Red Bull teammates, per §11.1) is reported as a
  *finding*, not a failure, regardless of whether the target is met.

- **Non-reproducible champion:** all three seeds read from `config.seeds` (numpy=42, xgboost=42,
  optuna=42); the Optuna TPE sampler is seeded; SQLite storage persists trials across restarts; a
  reproducibility test asserts identical champion metrics and params across two runs.

- **Under-building the search:** the easy-to-skimp items — ≥200 Optuna trials, the CV strategy,
  held-out-only evaluation — are each pinned by an explicit acceptance criterion and test, so they
  cannot be quietly dropped.

- **Silent trial failure:** a trial that raises or yields a NaN objective is pruned, counted
  (`n_trials_failed`), and logged (`MODEL_TRAINING_FAILED`) — never dropped silently. Zero valid
  trials escalates to a fail-loud error rather than registering an empty champion.