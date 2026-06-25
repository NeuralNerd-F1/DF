# APEX Feature Index & Build Order

One feature = one spec = one focused session. Build in dependency order: never start a
feature whose upstream prerequisites are not `DONE`. Status values:
`TODO -> SPEC WRITTEN -> IN PROGRESS -> DONE`.

Generate each spec with `/feature-spec <id> <area>`, build it with `/plan-feature <id>`
then implement, gate it with `/spec-check <id>` and `/apex-review <id>`.

## Layer 0 — Foundation contracts (do first, tiny, unblocks everything)
| ID  | Feature | Depends on | Wk | Status |
|-----|---------|-----------|----|--------|
| F00 | Error-code Enum, result objects, `config.yaml`, repo skeleton, Docker Compose | — | 1 | TODO |

## Layer 1 — Data pipeline
| ID  | Feature | Depends on | Wk | Status |
|-----|---------|-----------|----|--------|
| F01 | Data ingestion (FastF1/OpenF1/Ergast, caching, circuit breakers) | F00 | 1 | TODO |
| F02 | Lap filter chain / validation contract (9 filters, error codes) | F00, F01 | 2 | TODO |
| F03 | Feature extraction — 28 features, NULL tolerance | F00, F02 | 3 | TODO |
| F04 | Circuit-aware normalisation (train-only, saved artefact) | F03 | 4 | TODO |
| F05 | Manual CSV upload pipeline + edge cases (UPLOAD_001–004) | F02, F03 | 16 | TODO |

## Layer 2 — Machine learning
| ID  | Feature | Depends on | Wk | Status |
|-----|---------|-----------|----|--------|
| F06 | Model 1 — XGBoost classifier + confidence gate + calibration + Optuna + MLflow | F04 | 5–6 | TODO |
| F07 | SHAP explainability (beeswarm, force, radar, NL templates) | F06 | 7 | TODO |
| F08 | Model 2 — UMAP style space + PCA fallback + unknown-driver projection | F04 | 8 | TODO |
| F09 | Corner-type decomposition + sub-fingerprints (ablation) | F03 | 9 | TODO |
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
