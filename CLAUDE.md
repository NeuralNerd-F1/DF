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
- Backend: Python 3.12+, FastAPI 0.111+, Pydantic v2, SQLAlchemy 2.0 + asyncpg,
  Alembic, Celery 5.x.
- Data/ML: pandas 3.x, numpy 2.x (all ops vectorised — no per-row Python loops),
  XGBoost 3.x + CalibratedClassifierCV (scikit-learn 1.x), Optuna 4.x, umap-learn 0.5.x
  (PCA fallback), SHAP 0.45+ (TreeExplainer, exact), mlflow-skinny 3.x, FastF1 3.x.
- Stores: PostgreSQL 15 (Supabase), Redis 7.x (Upstash). Cache is an optimisation,
  never a hard dependency — the system must run with Redis down.
- Frontend: Next.js 14 (App Router), TypeScript strict (no implicit any), D3.js 7,
  Three.js r128 (D3 2D fallback), Zustand 4, Tailwind 3. Dark theme only.

> **Stack note (migration ratified by F04).** The runtime is **pandas 3.x — copy-on-write is
> the default** and **numpy 2.x — 1.x type aliases removed, stricter casting** — on **Python
> 3.12+**. The floor moved off 3.11 because `numpy 2.5.0` declares `Requires-Python >=3.12`;
> the pinned `numpy`/`pandas` are the only wheels with Python-3.14 support, so installing the
> deps transitively migrated the whole stack. Write new feature code against pandas-3.x /
> numpy-2.x semantics, not 1.x/2.x idioms. F01–F03b were authored across that transition and
> are test-covered, so they are NOT reopened — but the ground shifted under already-merged
> code: any odd pandas copy-on-write or numpy-alias behaviour downstream traces to here.

> **Stack note (ML deps ratified by F06a).** The ML stack is pinned at the newest releases with
> Python-3.14 wheels — **XGBoost 3.3, Optuna 4.9, scikit-learn 1.9, mlflow-skinny 3.14** — not the
> 2.x/3.x/2.x the line above originally named. Full `mlflow` caps `pandas<3`, which would downgrade
> the F04-ratified pandas-3.x stack, so APEX uses **mlflow-skinny** (the lightweight tracking +
> registry client, same `mlflow.*` API) on a **sqlite** store — the model registry is unsupported
> on `file://` stores. scikit-learn ≥1.6 removed `CalibratedClassifierCV(cv="prefit")`, so prefit
> calibration wraps the base model in `FrozenEstimator`. SHAP / umap-learn (F07 / F08) are not
> installed yet and will likely migrate similarly when those features land. See
> `specs/F06a-classifier-training.md` §8/§9 for the full rationale.

> **Stack note (SHAP + numpy pin moved by F07).** F07 installs **SHAP 0.52** (TreeExplainer, exact)
> — the newest release with a Python-3.14 wheel. Its `numba` dependency (numba 0.65.1) caps
> `numpy<2.5`, so the pin moved **numpy 2.5.0 → 2.4.6** (a cp314 wheel exists, still Python 3.14).
> This is the same kind of transitive migration F04 (pandas/numpy) and F06a (mlflow-skinny) already
> set; the full F01–F06 suite (351 tests) was re-run **green on numpy 2.4.6 before** any F07 code
> was written. Write against numpy-2.4.x — behaviour is unchanged vs 2.5 for APEX's usage (both are
> numpy-2.x / copy-on-write). See `specs/F07-shap-explainability.md` §5/§9.

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
