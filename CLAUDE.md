# APEX — Project Constitution

> This file is read at the start of every Claude Code session. Keep it short and
> always-true. Anything feature-specific lives in `specs/`, not here.

## What APEX is
A production-grade F1 driver behavioural profiling platform. Driver telemetry ->
28-feature lap vectors -> driver fingerprint -> classifier, style space,
degradation detection, and car personalisation. The whole system is built around
one principle: **no component fails silently.**

## The five non-negotiables (these override convenience, always)
1. **No silent failure.** Every failure path returns a typed result with an
   explicit error code (e.g. `LAP_003`, `DATA_001`). Never `except: pass`. Never
   drop a lap, row, or prediction without logging it to `data_quality_log`.
2. **No normalisation leakage.** Normalisation/scaling params are computed on the
   training split ONLY, saved to `artefacts/normalisation.json`, and loaded for
   val/test/upload. Recomputing scalers on held-out data is a build-breaking bug.
   Held-out circuits (Belgium, Italy) never touch any `.fit()` call.
3. **Reproducibility.** All seeds fixed in `config.yaml` (numpy=42, xgboost=42,
   optuna sampler=42). Every training run logged to MLflow with params + metrics +
   artefact. A run that can't be reproduced from the repo is not done.
4. **Confidence gates are mandatory.** Model outputs below threshold return an
   explicit uncertainty state (`LOW_CONFIDENCE`, `AMBIGUOUS`, `OUT_OF_DISTRIBUTION`),
   never a forced answer. Classifier gate: max prob < 0.65.
5. **Every feature/model output is unit-tested, including its failure path.** A
   feature returning out-of-range raises `FeatureComputationError`; the test suite
   asserts both the happy path and the NULL/error path.

## Tech stack (versions pinned in pyproject.toml / package.json)
- Backend: Python 3.11, FastAPI 0.111+, Pydantic v2, SQLAlchemy 2.0 + asyncpg,
  Alembic, Celery 5.x.
- Data/ML: pandas 2.x, numpy 1.x (all ops vectorised — no per-row Python loops),
  XGBoost 2.x + CalibratedClassifierCV, Optuna 3.x, umap-learn 0.5.x (PCA fallback),
  SHAP 0.45+ (TreeExplainer, exact), MLflow 2.x, FastF1 3.x.
- Stores: PostgreSQL 15 (Supabase), Redis 7.x (Upstash). Cache is an optimisation,
  never a hard dependency — the system must run with Redis down.
- Frontend: Next.js 14 (App Router), TypeScript strict (no implicit any), D3.js 7,
  Three.js r128 (D3 2D fallback), Zustand 4, Tailwind 3. Dark theme only.

## Repo map
```
app/pipeline/    ingest.py validate.py features.py normalise.py
app/ml/          classifier.py stylespace.py
app/monitoring/  detector.py
app/personalisation/ engine.py
app/api/         routes + Pydantic schemas
app/cache/ app/tasks/
artefacts/       normalisation.json, umap_model.pkl, pca_fallback.pkl, models
tests/           mirrors app/ ; one test module per source module
specs/           one spec per feature + 00-INDEX.md (build order + status)
config.yaml      seeds, thresholds, paths — single source of truth for constants
```

## Commands
- Install: `poetry install`
- Test: `poetry run pytest -q` (a feature is not done until its tests pass)
- Lint: `poetry run ruff check . --fix`
- Types: `poetry run mypy app`
- Migrations: `poetry run alembic revision --autogenerate -m "..."` then `alembic upgrade head`
- Run API: `poetry run uvicorn app.api.main:app --reload --port 8001`
- Frontend: `cd frontend && npm run dev`

## Conventions
- No magic numbers in code. Thresholds (0.65 gate, 30-lap minimum, 6-NULL ceiling,
  20Hz floor, 115% lap-time bound) live in `config.yaml` and are imported.
- Error codes are an Enum in `app/contracts/errors.py`. Strings are never hardcoded.
- Pipeline stages return Pydantic result objects (`ValidationResult`,
  `FeatureResult`, `PredictionResult`), never bare tuples/dicts.
- Tests use fixtures of synthetic passing AND failing laps — every error code has a
  test that triggers it.
- Conventional commits: `feat(features): add F19 lateral-g extraction`.

## How we work (see specs/00-INDEX.md and APEX_BUILD_GUIDE.md)
One feature = one spec in `specs/` = one focused session. Plan mode before writing
code. `/spec-check` and `/apex-review` before every commit. `/clear` between features.
