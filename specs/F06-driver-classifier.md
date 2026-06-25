# Feature F06 — Model 1: Driver Fingerprint Classifier (XGBoost)

**Status:** SPEC WRITTEN
**Master spec sections:** §5.1, §11.1, Risk Register row 1
**Layer / Week:** ML / Wk 5–6

## 1. Purpose
Given a 28-feature lap vector, predict which of the 8 known drivers produced it, as a
calibrated probability distribution over all 8 — never a single forced label. Below the
confidence gate the model returns an explicit uncertainty state instead of a guess.

## 2. Upstream / downstream contracts
- **Consumes:** F04 normalised feature matrix + `artefacts/normalisation.json` (params
  fit on training circuits ONLY).
- **Produces:** champion model in MLflow registry + `PredictionResult`.
- **Consumed by:** F07 (SHAP), F13 (`POST /predict`), and the degradation detector
  baseline (F10).

## 3. Inputs & outputs (typed)
- Input: 28-feature lap vector + circuit-type encoding (3D) + season (one-hot 3).
- Output:
  `PredictionResult { probabilities: dict[driver,float], top_prediction: str|None,
  confidence: float, confidence_state: HIGH|MEDIUM|LOW_CONFIDENCE|AMBIGUOUS,
  top_3: list, shap_values: dict|None, shap_available: bool, latency_ms: float }`.

## 4. Method
- XGBoost (gradient-boosted trees), wrapped in `CalibratedClassifierCV` for probability
  calibration. 8-class.
- Split: train on 8 drivers × 10 circuits × 3 seasons with **Belgium and Italy fully
  withheld**. Stratified 5-fold CV by driver×circuit (prevents circuit leakage).
- Tuning: Optuna Bayesian search, **min 200 trials**, objective = macro-averaged F1 on
  the validation fold. All trials logged to MLflow (params, macro-F1, per-class F1,
  confusion matrix, artefact); best registered as champion.
- Baselines trained and logged for comparison: majority-class, logistic regression,
  random forest (default params), all on the same features.
- Inference gating (every call):
  - max probability < 0.65 -> `LOW_CONFIDENCE`, return top-3 candidates.
  - Shannon entropy of the distribution > 2.9 bits -> `AMBIGUOUS` (input likely not F1
    telemetry).
  - feature vector with > 6 NULLs -> do not call the model; return `INVALID_INPUT`.

## 5. config.yaml keys this feature adds
```yaml
classifier:
  confidence_gate: 0.65          # §5.1 max-prob floor
  entropy_ambiguous_bits: 2.9    # §5.1 near-uniform -> AMBIGUOUS
  null_ceiling: 6                # >6 NULLs -> INVALID_INPUT (shared w/ F03)
  optuna_trials: 200
  target_macro_f1: 0.78          # held-out (Belgium, Italy)
  latency_warn_ms: 200
  held_out_circuits: [Belgium, Italy]
seeds: { numpy: 42, xgboost: 42, optuna: 42 }
```

## 6. Error codes & failure modes
| Failure mode | Detection | State / code | System response | User message |
|---|---|---|---|---|
| Max prob < 0.65 | gate on every prediction | `LOW_CONFIDENCE` | Return top-3 + probs, no forced label | "Insufficient telemetry signal for confident identification. Top candidates shown." |
| Near-uniform dist | Shannon entropy > 2.9 bits | `AMBIGUOUS` | Flag input as likely non-F1 | "Input data does not match F1 telemetry patterns." |
| Model file missing at startup | health check | `MODEL_NOT_FOUND` | Refuse to start; health endpoint 503 | 503 with reason |
| Vector has > 6 NULLs | null count pre-inference | `INVALID_INPUT` | Do not call model; list null features | "Feature extraction incomplete. N features could not be computed: [...]" |
| Prediction > 200ms | timing middleware | `PERF_WARN` (log) | Return result + latency flag header | none |

## 7. Acceptance criteria
- [ ] Output is a calibrated probability vector over all 8 drivers plus top-3, not just
      a label.
- [ ] `normalisation.json` is loaded and applied; **no `.fit()` on val/test/upload data**;
      Belgium and Italy never enter any fit or CV fold.
- [ ] CV is stratified by driver×circuit.
- [ ] Optuna runs ≥ 200 trials; champion registered in MLflow; all 3 baselines logged.
- [ ] max-prob < 0.65 returns `LOW_CONFIDENCE` with top-3 (asserted at the boundary,
      e.g. 0.649 vs 0.651).
- [ ] entropy > 2.9 bits returns `AMBIGUOUS`.
- [ ] > 6 NULLs returns `INVALID_INPUT` without invoking the model.
- [ ] All seeds read from config; rerun reproduces the same champion metrics.
- [ ] Reported macro-F1 is measured on held-out Belgium+Italy (target > 0.78; see risk
      contingency if not met).

## 8. Test plan
- Happy path: synthetic clean vector -> HIGH/MEDIUM state with a valid distribution
  summing to 1.
- Gate boundary: craft distributions with max prob 0.649 and 0.651 -> LOW_CONFIDENCE vs
  not.
- Entropy: near-uniform (~12.5% each) vector -> AMBIGUOUS.
- NULL boundary: vectors with 6 and 7 NULLs -> accepted vs INVALID_INPUT.
- Leakage guard test: assert the fitted scaler/CV never saw held-out circuits (inspect
  the fit inputs).
- Reproducibility: train twice with fixed seeds -> identical champion macro-F1.

## 9. Definition of done
All criteria MET with passing tests; ml-auditor PASS on leakage + seeding + calibration
+ gates; `/spec-check F06` verdict DONE. Limitation to record: setup-related labels are
not used here; this model only identifies drivers.

## 10. Risks & mitigations (Risk Register row 1)
- Macro-F1 < 0.78 on held-out circuits -> (1) add frequency-domain features (FFT on
  throttle and speed traces); (2) reduce 8 -> 6 drivers (drop PER, SAI as most-confused
  midfield pair); (3) relax target to 0.72 and report honestly with confusion analysis.
  Threshold-sensitivity and confusion matrix reported regardless.
