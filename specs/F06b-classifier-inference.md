# Feature F06b — Model 1: Driver Classifier — Inference / Confidence Gates / Serving

> One spec, one session. Self-contained: build this from this file + CLAUDE.md only.
> F06b is the **inference half** of F06. It LOADS the champion F06a registered and serves a
> gated `PredictionResult`; it **never trains, tunes, or re-fits anything**. **Seam between the
> two = the registered champion model** (F06a produces it; F06b loads it).

**Status:** SPEC WRITTEN
**Master spec sections:** §5.1 (Model 1 — inference: confidence gate, entropy/AMBIGUOUS, NULL
ceiling, model-not-found, latency; the Model 1 Failure-Modes table), §8.4 (`predictions` table —
`confidence_state` ENUM, "No NULL confidence"), §8.5 (`POST /predict` response contract + `GET
/health`), §11.4 (NULL-tolerance failsafe test), Project Plan Wk 5–6
**Layer / Week:** ML / Wk 5–6
**Depends on:** **F06a** (the registered champion + the real `artefacts/normalisation.json`),
F04 (`load_normalisation_params` / `apply_normalisation`). Splits the original
`specs/F06-driver-classifier.md` into F06a (training, DONE) + F06b (this).

## 1. Purpose
Given a 28-feature lap vector + its circuit, F06b loads the F06a champion once at startup and
returns a **calibrated probability distribution over all 8 drivers** plus an explicit
**confidence state** — never a single forced label. Below the confidence gate it returns an
uncertainty state (`LOW_CONFIDENCE` / `AMBIGUOUS`) with the top-3 candidates instead of a guess;
a vector it cannot process (>6 NULLs) is rejected before the model is ever called; an unexpected
internal fault is caught and returned as a coded result (never a 500); a missing champion is a
fail-loud startup error. F06b is the inference contract the F13 `POST /predict` endpoint
serialises — it owns the gating logic, not the HTTP layer. It adds **one** new error code,
`INFERENCE_ERROR` (the unexpected-internal backstop, mirroring F02's `VALIDATION_ERROR`); the five
§5.1 codes and four confidence states already exist.

## 2. Upstream / downstream contracts
- **Consumes (the seam — all produced by F06a):**
  - The **champion** — loaded from `artefacts/<classifier.model_artefact_filename>`
    (`classifier.pkl`). **Serializer pinned across the seam: F06a writes it with `joblib.dump`,
    F06b reads it with `joblib.load`** (the MLflow-registry copy is cloudpickle and is a separate
    path; F06b's disk load is joblib↔joblib). It is the calibrated
    `CalibratedClassifierCV(FrozenEstimator(XGBoost))`. Loaded **once** at startup, shared
    read-only across requests.
  - `artefacts/normalisation.json` via F04 `load_normalisation_params` (once) +
    `apply_normalisation` (per vector). F06b **never** calls `fit_normaliser` / `.fit`.
  - **`artefacts/<classifier.labels_filename>` (`labels.json`)** — the index→driver-name map.
    F06a's champion is fit on **integer** labels, so `champion.classes_` is `[0..n-1]`, **not**
    driver names; F06a (seam-completion, this branch) writes `labels.json` = the sorted driver-name
    list it integer-encoded against, so `labels[i]` is the name of class `i`. **`load_model`
    cross-checks `len(labels) == len(champion.classes_)` and `classes_ == [0..n-1]` and fails loud
    (`ModelLoadError`/`MODEL_NOT_FOUND`, logged) on any mismatch** — a stale/wrong `labels.json`
    refuses to serve rather than silently mislabel. `drivers = labels`.
  - **`model_version`** — derived from the **MLflow registered-model version** of
    `classifier.mlflow_registered_model` when a registry connection is available; otherwise a
    deterministic fallback = `sha256` of the champion artefact bytes (so the `predictions.
    model_version` column is always populated, even loading the bare pkl offline).
- **Produces:** `PredictionResult` (below) — the typed object every caller branches on; a
  `ModelHandle` (loaded artefacts); and a `health()` readiness check F13's `GET /health` calls.
- **Consumed by:** **F13** — `POST /predict` serialises `PredictionResult` (mapping
  `top_prediction` → the §8.5 `predicted_driver` field / §8.4 `predicted_driver_id`; `None` below
  the gate → null `predicted_driver`; `confidence`/`top_3`/`shap_*` map 1:1). `GET /health` calls
  `health()`. **F07** (SHAP fills `shap_values`/`shap_available` — F06b leaves them `None`/`False`).
  **F10** (degradation reuses the same champion + gating).

## 3. Inputs & outputs (typed)
- **Load (startup, once):**
  `load_model(*, config: AppConfig, artefact_dir: Path | None = None, log_path: Path | None = None)
  -> ModelHandle`. A missing/corrupt champion, `normalisation.json`, or `labels.json` — or a
  labels/class-count or feature-count mismatch — **raises** `ModelLoadError` (`MODEL_NOT_FOUND`)
  **after** writing a `MODEL_NOT_FOUND` `data_quality_log` row (§5.1 "Log MODEL_NOT_FOUND"); the
  process must refuse to serve.
- **Predict (per call, read-only, never raises):**
  `predict(feature_vector: dict[str, float | None], *, circuit: str, handle: ModelHandle,
  config: AppConfig, log_path: Path | None = None) -> PredictionResult`.
- **`ModelHandle`** (loaded, immutable): `{ model, normalisation_params: NormalisationParams,
  drivers: list[str], model_version: str, ready: bool }`.
- **`PredictionResult(BaseResult)`:**
  ```
  { probabilities: dict[str, float],          # over ALL 8 drivers, sums to 1.0; {} on INVALID_INPUT/INFERENCE_ERROR
    top_prediction: str | None,               # None below the gate / on a rejected input
    confidence: float,                        # max probability; 0.0 (never NULL) on rejected/error paths
    confidence_state: ConfidenceState | None, # HIGH|MEDIUM|LOW_CONFIDENCE|AMBIGUOUS; None ONLY when status=FAILED (model not run)
    top_3: list[tuple[str, float]],           # (driver, prob) desc; [] on INVALID_INPUT/INFERENCE_ERROR
    shap_values: dict[str, float] | None,     # None here — F07 fills it
    shap_available: bool,                     # False here — F07 sets it
    latency_ms: float,
    model_version: str,
    null_features: list[str],                 # populated only on INVALID_INPUT
    status, error_code, flags, logged }
  ```
  **`confidence_state` is non-NULL on every PREDICTED path** (HIGH/MEDIUM/LOW_CONFIDENCE/AMBIGUOUS)
  and is `None` **only** when `status=FAILED` and the model was not run (`INVALID_INPUT`,
  `INFERENCE_ERROR`) — those results are never persisted to the `predictions` table, so §8.4's
  "No NULL confidence" (a constraint on persisted rows) holds. `confidence` itself is never NULL.
- **Exceptions:** `ModelLoadError` (`MODEL_NOT_FOUND`) at load/startup only. `predict` **never
  raises**.

**Input-schema divergence from §5.1 (inherited from F06a — flag for the supervisor).** §5.1 lists
the model Input as "28-feature lap vector, circuit type encoding (3 dims), season (one-hot 3)" —
a 34-dim input. **F06a trained a 28-feature champion** (`_xy` uses only `F01..F28`; `circuit`
drives circuit-aware normalisation, `season` is unused as a feature). F06b therefore feeds
`predict_proba` exactly the 28 normalised features; `circuit` is a normalisation selector, not a
model column, and there is no `season` parameter. This is a **deliberate, documented divergence**:
circuit-aware normalisation (F04) already encodes circuit context, and the dissertation reports
that choice. An acceptance criterion (`champion.n_features_in_ == 28`) pins the seam so a future
34-dim champion can't silently mismatch.

## 4. Method / algorithm
All ops vectorised (pandas 3.x / numpy 2.x). The module is **read-only at inference**: a static
scan asserts no `.fit(` / `fit_normaliser` / `optuna` / training symbol.

**`load_model` (startup, once):**
1. Resolve `artefact_dir / classifier.model_artefact_filename`; absent/unreadable → log a
   `MODEL_NOT_FOUND` DQ row, then **raise `ModelLoadError`** (`MODEL_NOT_FOUND`). Else `joblib.load`
   the calibrated champion.
2. `load_normalisation_params(artefact_dir / normalise.artefact_filename)`; its `NormalisationError`
   (missing/corrupt) is logged + wrapped to `ModelLoadError` (`MODEL_NOT_FOUND`) — without params
   the model cannot be served un-normalised.
3. Load `artefact_dir / classifier.labels_filename` (`labels.json`) → `drivers`. **Fail-loud guards
   (all → `ModelLoadError`/`MODEL_NOT_FOUND`, logged):** labels file absent/unreadable;
   `len(drivers) != len(champion.classes_)`; `champion.classes_ != [0..n-1]`;
   `champion.n_features_in_ != len(FEATURE_IDS)` (28). A stale/mismatched labels file therefore
   refuses to serve. Then record `model_version` (§2), `ready=True`.

**`health() -> HealthResult`:** returns `ready` + the carried `model_version`; when the champion/
params are not loaded it reports **`HEALTH_FAIL`** (the existing "critical component down →503"
code) with `MODEL_NOT_FOUND` as the attached reason — this is what F13's `GET /health` surfaces as
503.

**`predict` (per call) — gate order is fixed and total:**
1. **NULL gate FIRST (model not called):** count NULLs among the 28 features. If
   `> thresholds.null_ceiling` (6) → return a **rejected** result: `status=FAILED`,
   `error_code=INVALID_INPUT`, `null_features=[...]`, `top_prediction=None`,
   `confidence_state=None`, `confidence=0.0`, `probabilities={}`, `top_3=[]`, one DQ row,
   `logged=True`. The model is **not** invoked. (≤6 NULLs are imputed exactly as F03/F04 define —
   no new imputation here.) *Defence-in-depth:* F03 already rejects >6 NULLs at extraction with
   `LAP_NULL_EXCESS`; `INVALID_INPUT` is the inference-layer re-check for vectors arriving by other
   paths (uploads, direct calls). §11.4's boundary test asserts `LAP_NULL_EXCESS` at the F03 layer
   and `INVALID_INPUT` at the F06b seam.
2. **Normalise:** `apply_normalisation(feature_vector, circuit=circuit, params=handle.
   normalisation_params, config=config, log_path=log_path)` — read-only; an unknown circuit
   degrades to global params (`NORMALISATION_FALLBACK` flag, `status=DEGRADED`), never raises.
3. **Probabilities (guarded):** wrap `model.predict_proba(x)` in try/except. On success →
   `probabilities = {drivers[i]: p_i}` (keys are `champion.classes_` **in order**, never re-derived
   from request data), `confidence = max(p)`, `top_3` = 3 highest desc. On ANY exception (inf/NaN,
   feature mismatch) → catch, log, return `error_code=INFERENCE_ERROR`, `status=FAILED`,
   `confidence_state=None`, `confidence=0.0`, `probabilities={}`, `top_3=[]` — **never re-raise**
   (the §8.5 "never 500 on model output" guarantee).
4. **Entropy gate:** `H = -Σ p·log2(p)`. If `H > thresholds.entropy_ambiguous_bits` (2.9) →
   `confidence_state=AMBIGUOUS`, `top_prediction=None`, `status=DEGRADED` (input likely not F1).
   **Checked before the confidence gate** — a vector that is both near-uniform and low-max-prob
   resolves to `AMBIGUOUS`.
5. **Confidence gate:** else if `confidence < thresholds.confidence_gate` (0.65) →
   `LOW_CONFIDENCE`, `top_prediction=None`, `status=DEGRADED`, return `top_3` (never a forced label).
6. **Accepted band:** else `top_prediction = drivers[argmax(p)]`, `status=OK`; `confidence_state =
   HIGH` if `confidence >= classifier.confidence_high` (0.85) else `MEDIUM`.
7. **Latency:** stamp `latency_ms`; if `> classifier.latency_warn_ms` (200) add `PERF_WARN` to
   `flags` + one DQ row (latency does not by itself degrade `status`).

**status ↔ confidence_state contract:** HIGH/MEDIUM → `OK`; LOW_CONFIDENCE/AMBIGUOUS → `DEGRADED`;
INVALID_INPUT/INFERENCE_ERROR → `FAILED` (model not run / failed); `NORMALISATION_FALLBACK` overlays
`DEGRADED` on an otherwise-OK predicted path. `confidence` is **never NULL** on any path.

**confidence_state naming vs §8.4.** §8.4 abbreviates the ENUM as `HIGH/MEDIUM/LOW/AMBIGUOUS`; the
canonical member (and the value F12's Alembic ENUM + F13 persist) is **`LOW_CONFIDENCE`** — "LOW" is
the master's shorthand. Documented divergence; no per-layer remapping.

## 5. config.yaml keys this feature adds
F06b adds **three** keys to the existing `classifier:` block (typed onto `ClassifierConfig`); the
gate constants it *applies* (`confidence_gate=0.65`, `entropy_ambiguous_bits=2.9`,
`null_ceiling=6`) already live under `thresholds:` and are **referenced, never re-declared**. It
also adds **one error code** `INFERENCE_ERROR` to `app/contracts/errors.py` (+ the canonical-set
test).
```yaml
classifier:
  # ... F06a training keys unchanged ...
  labels_filename: labels.json  # F06b seam — sorted driver-name list (index->name for classes_)
  confidence_high: 0.85       # max-prob >= this -> HIGH else MEDIUM (HIGH/MEDIUM split; §8.4 enum,
                              #   master pins only the 0.65 LOW gate -> documented design decision)
  latency_warn_ms: 200        # prediction latency over this -> PERF_WARN flag (§5.1)
```
**Validator placement:** the intra-model checks (`0 < confidence_high <= 1.0`, `latency_warn_ms > 0`,
non-empty `labels_filename`) live on `ClassifierConfig`; the **cross-field** check
`thresholds.confidence_gate < confidence_high` lives on **`AppConfig`** (the sub-model has
`extra="forbid"` and cannot see its sibling `thresholds`). A mis-edit is a load-time `CONFIG_001`.

## 6. Error codes & failure modes (the failsafe contract)
Inference **never raises** (every fault is a coded `PredictionResult`); only startup model-load
raises. Callers branch on `error_code` / `confidence_state` / `status`.

| Failure mode | Detection | Code / state | System response | Logged? | User-facing message |
|--------------|-----------|--------------|-----------------|---------|---------------------|
| Max prob < 0.65 | gate every prediction | `LOW_CONFIDENCE`, `status=DEGRADED` | `top_prediction=None`, return top-3 + full dist | yes | "Insufficient telemetry signal for confident identification. Top candidates shown." |
| Near-uniform dist | Shannon entropy > 2.9 bits | `AMBIGUOUS`, `status=DEGRADED` | flag likely non-F1, `top_prediction=None` | yes | "Input data does not match F1 telemetry patterns. Check data source." |
| Champion / params missing at startup | file existence + load | raise `MODEL_NOT_FOUND`; `health()`→`HEALTH_FAIL` | refuse to start; F13 `GET /health` 503, reason `MODEL_NOT_FOUND` | **yes** (startup DQ row before raise) | 503 with reason `MODEL_NOT_FOUND` |
| Feature vector > 6 NULLs | NULL count before inference | `INVALID_INPUT`, `status=FAILED`, `confidence_state=None` | **do not call model**; `null_features=[...]`, `confidence=0.0` | yes | "Feature extraction incomplete. N features could not be computed: [list]." |
| Unexpected internal inference fault | try/except around `predict_proba` | `INFERENCE_ERROR` *(new)*, `status=FAILED`, `confidence_state=None` | catch, return coded result, never re-raise | yes | "Prediction failed internally; no result produced." |
| Prediction > 200 ms | per-call timing | `PERF_WARN` (flag) | return result + `PERF_WARN` in `flags` | yes | none (monitoring only) |
| Unknown circuit at normalise | `circuit ∉ per_circuit` | `NORMALISATION_FALLBACK` (flag), `status=DEGRADED` | global params, still predicts | yes (via F04) | "Circuit X not in training set; global normalisation applied." |

`INFERENCE_ERROR` is the **one new** code (mirrors F02's `VALIDATION_ERROR` backstop). The other
five (`LOW_CONFIDENCE`, `AMBIGUOUS`, `MODEL_NOT_FOUND`, `INVALID_INPUT`, `PERF_WARN`), `HEALTH_FAIL`,
and `NORMALISATION_FALLBACK` already exist. Confidence states from `ConfidenceState`
(`OUT_OF_DISTRIBUTION` is F08's, not used here). The **User-facing message** column is illustrative
copy F13 renders; tests assert `error_code` / `confidence_state` / `status`, not prose.

## 7. Acceptance criteria (each item mechanically testable by /spec-check)
- [ ] Output is a calibrated probability vector over **all 8 drivers** summing to 1.0 (±1e-6) plus
      `top_3` on every PREDICTED path (HIGH/MEDIUM/LOW_CONFIDENCE/AMBIGUOUS); not just a label.
- [ ] **Label-order fidelity:** `drivers` is loaded from `labels.json` (the sorted F06a names);
      `probabilities` keys equal `drivers` in `classes_` order and `top_prediction ==
      drivers[argmax(predict_proba)]` — tested with a stub whose `predict_proba` peaks on a known
      column, asserting the correct driver name (guards silent mis-labelling).
- [ ] **Stale/mismatched labels refuse to serve:** a `labels.json` whose length ≠ the champion's
      class count (or absent) makes `load_model` **raise `ModelLoadError` (`MODEL_NOT_FOUND`)** and
      log a `LABELS_MISMATCH`/`LABELS_ABSENT` DQ row — never serve mislabelled predictions.
- [ ] **28-feature seam:** `champion.n_features_in_ == 28`; `predict` feeds exactly the 28
      normalised features (no season/circuit-type columns).
- [ ] **No retrain / no fit:** static source scan — no `.fit(`, `fit_normaliser`, `optuna`,
      `CalibratedClassifierCV(` in the module; F06b only `load_*` + `predict` + `health`.
- [ ] **Normalisation loaded, not fit:** `predict` calls `apply_normalisation(params=handle.
      normalisation_params, config=config, ...)`; any circuit transformed, never re-fit.
- [ ] **Confidence-gate boundary:** max-prob **0.649 → `LOW_CONFIDENCE`, `top_prediction is None`**;
      **0.651 → not LOW** (asserted both sides of 0.65).
- [ ] **Entropy-gate boundary + precedence:** near-uniform (~12.5% each) → `AMBIGUOUS`; peaked → not;
      a vector that is BOTH `H>2.9` AND max-prob<0.65 → `AMBIGUOUS` (entropy wins, LOW branch not
      reached).
- [ ] **HIGH/MEDIUM split:** max-prob `0.85 → HIGH`, `0.84 → MEDIUM` (both above the LOW gate).
- [ ] **NULL ceiling:** **6 NULLs → predicted**, **7 NULLs → `INVALID_INPUT` with the model NOT
      called** (asserted via a spy on `predict_proba`); `null_features` lists the missing ids.
- [ ] **`confidence` never NULL:** float on every path (0.0 on INVALID_INPUT/INFERENCE_ERROR);
      `confidence_state` set on every PREDICTED path and `None` only when `status=FAILED`.
- [ ] **status per path:** HIGH/MEDIUM→`OK`; LOW_CONFIDENCE/AMBIGUOUS→`DEGRADED`;
      INVALID_INPUT/INFERENCE_ERROR→`FAILED`; unknown circuit overlays `DEGRADED`.
- [ ] **Never raises at predict:** a garbage/NaN vector and a stub model whose `predict_proba`
      raises both return a coded `PredictionResult` (`INFERENCE_ERROR`), never an exception.
- [ ] **NORMALISATION_FALLBACK:** an unknown circuit → `NORMALISATION_FALLBACK` in `flags`,
      `status=DEGRADED`, a valid prediction still returned, `confidence` non-NULL, `logged=True`.
- [ ] **Logged + DQ row:** for each logged mode (LOW_CONFIDENCE, AMBIGUOUS, INVALID_INPUT,
      INFERENCE_ERROR, PERF_WARN, NORMALISATION_FALLBACK) `result.logged is True` AND a
      `data_quality_log` row with the matching `error_code` is written (asserted against tmp log).
- [ ] **Model-not-found:** absent/corrupt `classifier.pkl` OR `normalisation.json` → `load_model`
      **raises `ModelLoadError` (`MODEL_NOT_FOUND`)** AND writes a `MODEL_NOT_FOUND` DQ row;
      `health()` reports `HEALTH_FAIL`/not-ready. After a clean load, `health()` reports ready.
- [ ] **model_version:** non-empty on the result, tracing to the F06a champion (MLflow version or
      artefact hash); deterministic for a given champion.
- [ ] **Latency flag (both sides):** timed cost 201 ms → `PERF_WARN` present; 199/200 ms → absent
      (and latency alone does not set `status=DEGRADED`).
- [ ] **SHAP deferred:** `shap_values is None`, `shap_available is False`.
- [ ] **Config-driven:** the code reads `thresholds.{confidence_gate,entropy_ambiguous_bits,
      null_ceiling}` and `classifier.{confidence_high,latency_warn_ms}`; a literal scan finds no
      `0.65`/`2.9`/`0.85` float literal in the module body (integer thresholds checked by
      reference, not literal-absence, to avoid index/range false positives).
- [ ] **Config validator:** `AppConfig` load raises when `confidence_high <= confidence_gate`,
      `confidence_high > 1.0`, or `latency_warn_ms <= 0`.
- [ ] Every error code / state above has a unit test that triggers it and asserts the result.

## 8. Test plan
- **Fixtures (`tests/ml/conftest.py`, extend):** reuse F06a's module-scoped `trained` run — it
  already writes a real `classifier.pkl` + `normalisation.json` under tmp — as the loadable
  champion, so F06b tests load a genuine model via `load_model`. Add a `StubModel` exposing
  `classes_` + a `predict_proba` returning a caller-supplied distribution, to pin gate boundaries
  deterministically (independent of the real model's output) AND to inject a raising
  `predict_proba` for the INFERENCE_ERROR path; a `proba_vector(max_prob=, entropy=)` builder; and a
  `predict_proba` spy to assert the model is NOT called on the >6-NULL path.
- **Happy path:** clean vector via the real champion → HIGH or MEDIUM, distribution sums to 1.0,
  `top_3` ordered, `shap_available is False`, `status=OK`, `model_version` non-empty.
- **Boundary fixtures (one per §7 boundary, via StubModel):** 0.649/0.651 (LOW gate), 0.84/0.85
  (HIGH/MEDIUM), entropy just over/under 2.9, the both-gates case (→AMBIGUOUS), 6/7 NULLs
  (INVALID_INPUT + not-called spy), latency 199/200/201 ms (monkeypatched timer).
- **Label-order:** StubModel with a known peak column → assert the right driver name + keys order.
- **Never-raises / INFERENCE_ERROR:** StubModel `predict_proba` raises → coded result, no exception;
  garbage/NaN and all-NULL vectors → coded results; `confidence` non-NULL on all.
- **Fail-loud load:** missing `classifier.pkl` → `ModelLoadError` + DQ row; missing
  `normalisation.json` → `ModelLoadError`; `health()` HEALTH_FAIL in both; ready after clean load.
- **Config validator:** assert `AppConfig` rejects the three bad `confidence_high`/`latency_warn_ms`
  cases.
- **Static scans:** no `.fit(`/`fit_normaliser`/`optuna` in the module; float-literal scan for
  0.65/2.9/0.85.

## 9. Definition of done
All acceptance criteria MET with passing tests; **ml-auditor PASS** (no refit at inference; loaded
normalisation; gates present on every output path; calibrated probabilities; correct label order);
`/apex-review F06b` clean (no silent failure — never 500 / never NULL confidence; config-driven
thresholds; error-code discipline; no bare `except` beyond the deliberate logged `INFERENCE_ERROR`
catch-all); `/spec-check F06b` green; `mypy app` clean. INDEX F06 row → **F06a DONE; F06b DONE**.
**Limitations for the dissertation:** (1) SHAP is wired by F07; F06b returns `shap_values=None`
/ `shap_available=False`. (2) HIGH/MEDIUM split uses a new `confidence_high` (0.85) — master pins
only the 0.65 LOW gate, so the boundary is a documented, config-driven decision to calibrate
against the reliability curve. (3) **Input-schema divergence:** the champion is **28-feature**, not
§5.1's 34-dim (circuit-type + season one-hot); circuit context enters via F04 normalisation,
season is unused. **Inherited from F06a** — should be back-noted in F06a's limitations too;
**flag for the supervisor** alongside F06a's CV-splitter divergence. (4) F06b is the inference
*logic*; the HTTP endpoint and its 400/200/503 + `predicted_driver` field mapping and `GET /health`
wiring are F13 — F06b provides the `PredictionResult` + `health()` they consume. (5) Single-lap
synchronous prediction only; batch SHAP / async is F07/F14. (6) `confidence_state` "LOW" in §8.4 is
the enum member `LOW_CONFIDENCE` (shorthand reconciliation; F12 Alembic ENUM matches the member).

## 10. Risks & mitigations (from master where relevant)
- **A forced label below the gate (headline §5.1 risk):** every accepted-label path is guarded by
  the entropy gate then the 0.65 gate; below either, `top_prediction is None` + an explicit state.
  Boundary tests pin 0.649/0.651 and the both-gates precedence.
- **Silent 500 / NULL confidence (§8.5/§8.4):** `predict` never raises — the `predict_proba`
  call is wrapped to a coded `INFERENCE_ERROR`; `confidence` is never NULL and `confidence_state`
  is non-NULL on every predicted path (None only on FAILED, never-persisted results). Startup is the
  only raise (`MODEL_NOT_FOUND`), surfaced via `health()`→`HEALTH_FAIL`→503.
- **Silent mis-labelling:** `probabilities`/`top_prediction` are keyed strictly to
  `champion.classes_` order, asserted by a peaked-stub test — a transposed zip is caught.
- **Accidental retrain / scaler refit at serve time (non-negotiable #2):** load read-only; static
  scan forbids `.fit`/`fit_normaliser`/`optuna`; F06a's held-out generalisation is preserved.
- **Seam drift (serializer / version / feature count):** joblib↔joblib pinned; `model_version`
  sourced from the MLflow version or artefact hash; `n_features_in_ == 28` asserted at load so a
  retrained 34-dim champion fails fast rather than mis-predicting.
- **HIGH/MEDIUM threshold unspecified by master:** config `confidence_high`, documented, tunable
  from the calibration curve; reported in the dissertation.
