# Feature F13 — FastAPI Endpoints + Pydantic Contracts + Health Check

> One spec, one session. Self-contained: build this from this file + CLAUDE.md only.

**Status:** DONE
**Master spec sections:** §8.5 (the 14 API endpoints + response contracts + error states), §8.1/§8.3
(every component has a health check; DB breaker → 503), §9 (FastAPI 0.111+, Pydantic v2 for all
request/response validation — invalid input rejected at the schema layer), §11 Failsafe tests
(DB-unavailability → 503 `DB_UNAVAILABLE` no partial writes; `/predict` never 500; 50-concurrent
p95 < 500ms; cache-down graceful), Project Plan Wk 15, Risk Register (database unavailability,
concurrent requests)
**Layer / Week:** Platform / Wk 15
**Depends on:** F06 (F06b inference), F07 (SHAP), F08 (style space), F09 (corner sub-fingerprints —
via F07 radar only), F10 (degradation), F11 (personalisation), F12 (persistence) — **all DONE**

## 1. Purpose
F13 is APEX's **HTTP serving layer**: a FastAPI application that wires the already-built,
never-raising model handles (F06b classifier · F07 SHAP · F08 style space · F10 degradation · F11
personalisation) and the F12 persistence layer into the master-§8.5 REST surface, with Pydantic v2
request/response contracts and a **real** `/health` that reports true component status (503 if any
critical component is down). F13 **serves** — it never refits a model, and it computes live from
request bodies or reads committed rows from the DB; it never fabricates data. The synchronous
serving surface (`/predict`, `/fingerprint/unknown`, the driver reads, `/stylespace`, `/health`,
`/metrics`) is implemented in full; the async/cache/replay/upload endpoints get their Pydantic
schemas here but their live engines land in F14 (Redis + Celery + replay) and F05 (manual upload).

## 2. Upstream / downstream contracts
- **Consumes (all read-only, never refit):**
  - **F06b** `load_model → ModelHandle`, `predict(...) → PredictionResult`, `health(handle) → HealthResult`.
  - **F07** `build_explainer(handle) → ShapExplainer`, `explain_prediction(...) → ExplanationResult`,
    `season_stability(...) → SeasonStabilityResult`, `aggregate_domains`/`describe_style` (pure). No F07
    health fn — SHAP degrades to `SHAP_UNAVAILABLE`.
  - **F08** `load_style_space → StyleSpaceHandle`, `project_unknown(laps,...) → ProjectionResult`,
    `get_style_space(handle) → StyleSpaceResult`, `health(handle) → StyleSpaceHealthResult`.
  - **F10** `load_detector → MonitorHandle`, `health(handle) → MonitorHealthResult`. (Live
    `monitor_stint`/replay is F14's use; F13 only reports F10 health.)
  - **F11** `load_personalisation → PersonalisationHandle`, `recommend_setup(...) → SetupRecommendation`,
    `onboard_unknown(...) → OnboardingPackage`, `health(handle) → PersonalisationHealthResult`.
  - **F12** `make_engine` / `make_session_factory` / `session_scope` / `make_db_breaker` /
    `check_database → DbHealthResult` / `repository.persist|write_quality_log_row` /
    `DatabaseUnavailableError` / `DbIntegrityError`.
  - `config.yaml` → new `api:` block; the model artefact paths come from the existing per-model config
    blocks (`classifier`/`stylespace`/`monitoring`/`personalisation`) — not re-declared here.
- **Produces:**
  - `app/api/` package: `main.py` (`create_app()` factory + lifespan + exception handlers + CORS),
    `schemas.py` (all §8.5 request/response Pydantic models + the `ErrorResponse` envelope),
    `deps.py` (dependency-injected handles + DB session + settings), `routes/` (one module per
    endpoint group), `metrics.py` (Prometheus registry).
  - Three new error codes: `NOT_FOUND`, `FORBIDDEN`, `API_ERROR` (typed HTTP-layer bodies).
  - **No new model artefact.** F13 serves existing artefacts / DB rows; it persists nothing except an
    optional best-effort `predictions`-row write on `/predict` (via the F12 repository).
- **Consumed by:** F14 (adds Redis cache wrapping the reads, Celery async ingestion, the replay
  engine — fills the 4 deferred routes), F05 (manual upload route), F16–F19 (the Next.js frontend
  calls this API), F15/F20 (the RQ5 failsafe harness fires the concurrency + DB-outage tests here).

## 3. Inputs & outputs (typed)
F13's I/O is HTTP: **request models** validate input at the schema layer (invalid → 422), **response
models** mirror the §8.5 contracts, and one **error envelope** carries every typed failure.

- **Error envelope** (every non-2xx typed body): `ErrorResponse { error_code: ErrorCode, detail: str,
  flags: list[ErrorCode] = [] }`.
- **Request models** (examples): `PredictRequest { feature_vector: dict[str, float | None], circuit: str }`;
  `UnknownFingerprintRequest { laps: list[dict], circuit: str, circuit_type: str | None = None }`.
- **Response models** (mirror §8.5 verbatim — full list in §4.1). Each is built by mapping the internal
  never-raising `*Result` object to the HTTP shape; e.g.
  `PredictResponse { predicted_driver: str | None, confidence: float, confidence_state: ConfidenceState | None,
  top_3: list[TopK], shap_values: dict[str, float] | None, shap_available: bool, model_version: str }`.
- **Health output:** `HealthResponse { status: Literal["healthy","unhealthy"], models_loaded:
  dict[str, bool], db_connected: bool, cache_connected: bool | None, version: str }` — HTTP **503**
  when `status == "unhealthy"` (any critical component down), else 200.
- **App factory:** `create_app(config: AppConfig | None = None) -> FastAPI` — builds the app, wires the
  lifespan (loads handles + DB engine once), registers routes + exception handlers + CORS.

## 4. Method / algorithm

### 4.0 Design notes (the decisions a builder must not re-litigate)
- **(a) Endpoint scope split** (**primary divergence #28**, ratified from the INDEX Wk15/Wk16 line).
  F13 = the **synchronous serving surface + real `/health` + all Pydantic schemas**; F14 = Redis
  cache + Celery async ingestion + replay engine; F05 = manual upload. **F13 fully implements** (real,
  tested): `/health`, `/predict`, `/fingerprint/unknown`, `/drivers`, `/drivers/{id}/fingerprint`,
  `/drivers/{id}/setup/{circuit_id}`, `/drivers/{id}/stability`, `/compare/{id1}/{id2}`, `/stylespace`,
  `/metrics`. **F13 defines schemas for but defers the live engine of** (route returns **503 +
  `HEALTH_FAIL`** "backing service not yet available", so the OpenAPI surface is complete and honest,
  never faked): `POST /ingest/session`, `GET /ingest/status/{id}`, `POST /ingest/upload`,
  `GET /replay/{session_id}/{driver_id}`.
- **(b) Handles are dependency-injected, loaded once in the lifespan** (`app/api/deps.py`).
  `create_app`'s lifespan calls each `load_*` **best-effort**: a load failure (`ModelLoadError` etc. —
  and note **no champion artefacts are committed**, so a fresh checkout WILL fail to load) is caught
  and logged, the handle stored as `None`, and the app still boots so `/health` can report the outage
  as 503 rather than crashing on startup (§8.1). Every route reads its handle via `Depends`; **tests
  override the dependencies** (`app.dependency_overrides`) with fake handles — this is the only way to
  test endpoint logic without committed artefacts.
- **(c) Never-500 on model output** (§8.5 `/predict`). The internal `predict`/`explain_prediction`/
  `project_unknown`/`recommend_setup` never raise; their typed uncertainty states
  (`LOW_CONFIDENCE`/`AMBIGUOUS`/`SPARSE_PROJECTION`/`OUT_OF_DISTRIBUTION`/`WIDE_CONFIDENCE`) map to a
  **200 body**, never an HTTP 5xx. A global exception handler is the backstop for genuine bugs on ANY
  route → typed **500 `API_ERROR`** (no raw stack trace ever leaves the process; "no unhandled
  exceptions", §11).
- **(d) DB criticality is per-endpoint.** DB-backed **reads** (`/drivers`, `/drivers/{id}/fingerprint`,
  `/compare`, `/stability`) and any **write** go through F12 `session_scope`; DB down → **503
  `DB_UNAVAILABLE`**, rollback, no partial write. **Stateless compute** endpoints (`/predict`,
  `/fingerprint/unknown`, `/stylespace`, `/setup`) do NOT need the DB — they stay 200 even when the DB
  is down. **Build divergence #30 — `/predict` is stateless (no `predictions`-row write).** The spec
  first cast `/predict`'s `predictions`-row persistence as an optional best-effort write; the build
  drops it because it is structurally infeasible without fabricating data: the F12 `predictions` row
  requires a NOT-NULL `lap_id` (FK→`laps`) and `predicted_driver_id` (FK→`drivers`), and a live
  `/predict` receives only an ad-hoc `{feature_vector, circuit}` with no persisted lap or driver
  entity — inventing those rows would violate §1/§4.0(e) ("F13 never fabricates data"). The
  `predictions` table is populated by the F14 batch/replay path (where laps ARE persisted). The
  `predict_persist` config knob is therefore not shipped.
- **(e) Data source per endpoint.** `/predict` + `/fingerprint/unknown` compute **live** from the
  request body (fully functional in F13). `/stylespace` + `/setup` serve from the **loaded handles**
  (F08 known-driver coords; F11 known-driver regressors). `/drivers` + `/drivers/{id}/fingerprint` +
  `/compare` + `/stability` read the F12 **`fingerprints` table** — empty in a fresh deploy, so they
  correctly return the §8.5 empty/404 responses until a batch job populates it (that population job is
  **out of F13 scope** — F14/F20). No fabrication anywhere.
- **(f) HTTP ↔ ErrorCode mapping via exception handlers.** Typed exceptions/results map to status +
  `ErrorResponse`: `DatabaseUnavailableError`→503 `DB_UNAVAILABLE`; `DbIntegrityError`→409 `DB_INTEGRITY`;
  a `NotFoundError` (thin F13 helper)→404 `NOT_FOUND`; Pydantic `RequestValidationError`→422; the
  `>6-NULL`/`<5-laps` input gates→400 `INVALID_INPUT`; anything else→500 `API_ERROR`. Codes are the
  `ErrorCode` enum — never hard-coded strings (house convention).
- **(g) Reproducibility of the stack.** FastAPI 0.138.1 / starlette 1.3.1 / uvicorn 0.49.0 are present
  in the venv but **absent from `pyproject.toml`** — F13 pins them EXACTLY (non-negotiable #3) and adds
  **httpx** (test client dep). Same "feature adds the dep it first imports" pattern as F01/F06a/F07.
- **(h) `/metrics` is Prometheus, internal-only** (§8.5 — 403 if external). Gated by
  `api.metrics_internal_only`; an external caller (non-loopback / missing internal token) → **403
  `FORBIDDEN`**. Exposes request count, latency histogram, model-confidence distribution, breaker state.
- **(i) Concurrency is by construction** (§11 — 50 concurrent, consistent, p95 < 500ms). Handles are
  **frozen dataclasses** (read-only, no shared mutable state); each request gets its own
  `AsyncSession` via `session_scope`; handlers are pure over the injected handles → same input = same
  output. F13 asserts *completion + consistency*; the p95 < 500ms number is formally measured in **F20**.

### 4.1 The 14 endpoints (method · path · source · response model · status contract, from §8.5)

| Method · Path | F13? | Source | Response model (fields per §8.5) | Status contract |
|---|---|---|---|---|
| `GET /health` | ✅ real | F12 `check_database` + each `health()` + cache | `HealthResponse{status,models_loaded,db_connected,cache_connected,version}` | 503 if any critical down |
| `GET /drivers` | ✅ real | DB `fingerprints` | `list[DriverSummary{id,code,radar_values}]` | 200 always (empty if none) |
| `GET /drivers/{id}/fingerprint` | ✅ real | DB `fingerprints` (+F07 NL/stability) | `FingerprintResponse{vector_5d,shap_radar,nl_description,stability}` | 404 unknown · 206 partial seasons |
| `GET /drivers/{id}/setup/{circuit_id}` | ✅ real | F11 `recommend_setup` | `SetupResponse{parameters[10],confidence_ratings,constraint_flags}` | 404 no data · 200 `WIDE_CONFIDENCE` if limited |
| `GET /drivers/{id}/stability` | ✅ real | F07 `season_stability` (per-season SHAP from DB) | `StabilityResponse{js_scores,year_on_year}` | 206 if <3 seasons |
| `POST /predict` | ✅ real | F06b `predict` + F07 `explain_prediction` | `PredictResponse{predicted_driver,confidence,confidence_state,top_3,shap_values,shap_available}` | 400 invalid · 200 `LOW_CONFIDENCE` · never 500 |
| `POST /fingerprint/unknown` | ✅ real | F08 `project_unknown` + F11 `onboard_unknown` | `UnknownFingerprintResponse{fingerprint_id,fingerprint_vector,radar_values,style_description,nearest_neighbours,risk_flags,data_quality}` | 200 `SPARSE_PROJECTION` if <30 laps · 400 if <5 |
| `GET /compare/{id1}/{id2}` | ✅ real | DB `fingerprints` + F07 radar delta | `CompareResponse{driver_1_radar,driver_2_radar,delta_shap,most_distinctive_features,compatibility_delta}` | 404 if either unknown |
| `GET /stylespace` | ✅ real | F08 `get_style_space` | `StyleSpaceResponse{drivers[{id,code,coords_2d,coords_5d,season}],unknown_drivers}` | 200 always (empty if none) |
| `GET /metrics` | ✅ real | Prometheus registry | text/plain Prometheus | 403 if external |
| `POST /ingest/session` | 🔶 schema-only | — (F14 Celery) | `IngestSessionResponse{session_id,laps_accepted,laps_rejected,rejection_summary,quality_report}` | 503 `HEALTH_FAIL` until F14 |
| `GET /ingest/status/{session_id}` | 🔶 schema-only | — (F14) | `IngestStatusResponse{status,progress,error_detail}` | 503 `HEALTH_FAIL` until F14 |
| `POST /ingest/upload` | 🔶 schema-only | — (F05) | `UploadResponse{upload_id,laps_accepted,laps_rejected,cleaning_report,warnings}` | 503 `HEALTH_FAIL` until F05 |
| `GET /replay/{session_id}/{driver_id}` | 🔶 schema-only | — (F14 replay) | `ReplayResponse{laps[{lap_number,feature_vector,deviation_score,alert_state,fingerprint_snapshot}]}` | 503 `HEALTH_FAIL` until F14 |

### 4.2 `/health` criticality (§8.1/§8.3)
`status = "healthy"` **iff** `db_connected` (F12 `check_database` → `DbHealthResult.ok`) AND every model
in `models_loaded` is `True` (each handle's `health().ready`). **Cache is NOT critical** (constitution:
"cache is an optimisation, never a hard dependency") — `cache_connected` is reported (`None`/`False` in
F13, since Redis is F14) but never forces 503. Any critical failure → `status="unhealthy"` → **HTTP 503**.

## 5. config.yaml keys this feature adds
```yaml
# --- F13 API serving layer (FastAPI + Pydantic + health, §8.5/§8.3/§9) ----------------------------
# Wires the DONE model handles + the F12 DB into the master-§8.5 REST surface. Model artefact paths
# come from the per-model blocks (classifier/stylespace/monitoring/personalisation) — not re-declared.
# The DSN stays env-only (F12). Cache is F14: cache_connected is reported but never critical.
api:
  version: "0.1.0"                 # /health version string (single source for the API version)
  cors_origins: ["http://localhost:3000"]  # §8.5 frontend origins (F16 Next.js dev server)
  docs_enabled: true               # serve OpenAPI /docs + /redoc
  metrics_internal_only: true      # §8.5 — /metrics returns 403 to external callers
  latency_slo_ms: 500              # §11 p95 target (surfaced as a PERF_WARN log/metric; formal in F20)
  request_timeout_s: 10            # per-request wall-clock ceiling (declared; enforced in F14/F20)
  min_projection_laps: 5           # §8.5 — /fingerprint/unknown floor: fewer laps cannot be projected (400)
  max_request_laps: 5000           # §11 DoS guard — reject oversized /fingerprint/unknown payloads (400)
  max_feature_keys: 128            # §11 DoS guard — reject oversized /predict feature vectors (400)
  # allow_credentials: false       # §8.5 CORS — optional, defaults false (F13 has no cookies/auth)
```
**As-built note.** `predict_persist` is NOT shipped (build divergence #30 — `/predict` is stateless;
see §4.0(d)). `min_projection_laps` replaces the earlier hard-coded `<5` gate so the floor is
config-driven (no magic numbers, §7). `max_request_laps`/`max_feature_keys` are §11 DoS guards added
by the security review (reject oversized payloads at 400, logged). `allow_credentials` defaults false
(F13 has no auth); a validator forbids `"*"` in `cors_origins` when it is true. `request_timeout_s` is
validated but its HTTP-layer enforcement is deferred to F14/F20 (like `latency_slo_ms`).
New typed `ApiConfig` (in `app/contracts/config.py`, standing rule — feature config lives in
`config.py`, `extra="forbid"`), wired as `AppConfig.api: ApiConfig | None = None`. Validators:
non-empty `version`; `cors_origins` a list; `latency_slo_ms`/`request_timeout_s` > 0; booleans typed.

## 6. Error codes & failure modes (the failsafe contract)

| Failure mode | Detection | Error code | System response | Logged? | User-facing message |
|--------------|-----------|-----------|-----------------|---------|---------------------|
| Invalid request body (missing/typed field) | Pydantic `RequestValidationError` at the schema layer | (Pydantic) | **422** + FastAPI validation detail (rejected before business logic, §9) | no | "Invalid request: <field errors>" |
| Input gate: >6 NULL features / <5 laps | F06b `INVALID_INPUT` / F08 `<5` guard | `INVALID_INPUT` (reuse) | **400** + `ErrorResponse` | yes | "Input rejected: too many missing features." |
| Resource not found (driver / combination) | DB read returns none | `NOT_FOUND` *(new, div #29)* | **404** + `ErrorResponse` | no | "Driver not found." |
| DB unreachable / breaker open on a DB-backed route | `DatabaseUnavailableError` from `session_scope` | `DB_UNAVAILABLE` (reuse) | **503** + `ErrorResponse`; rollback, **no partial write** (§11) | yes | "Database temporarily unavailable." |
| Integrity violation on a write | `DbIntegrityError` from the repository | `DB_INTEGRITY` (reuse) | **409** + `ErrorResponse` | yes → `data_quality_log` | "Record rejected: integrity constraint." |
| Any critical component down at `/health` | `check_database` not-ok OR a handle not ready | `HEALTH_FAIL` (reuse) | **503** `HealthResponse{status:"unhealthy"}` | — | "Service unhealthy: <component>." |
| Model output below gate (never an error) | F06b/F08/F11 typed uncertainty state | `LOW_CONFIDENCE`/`AMBIGUOUS`/`SPARSE_PROJECTION`/`OUT_OF_DISTRIBUTION`/`WIDE_CONFIDENCE` (reuse) | **200** with the state in the body — **never 500** (§8.5) | (per model) | (state-specific, e.g. "Low confidence.") |
| SHAP fault on `/predict` | F07 `SHAP_UNAVAILABLE` | `SHAP_UNAVAILABLE` (reuse) | **200** with `shap_available=false` (prediction preserved) | yes | (prediction returned; explanation omitted) |
| `/metrics` called externally | non-loopback / missing internal token | `FORBIDDEN` *(new, div #29)* | **403** + `ErrorResponse` | no | "Metrics endpoint is internal only." |
| Deferred endpoint (ingest/upload/replay) | route reached before F14/F05 wires it | `HEALTH_FAIL` (reuse) | **503** + `ErrorResponse` "backing service not yet available" | no | "This endpoint is not available yet." |
| Unexpected internal fault on any route | uncaught exception in a handler | `API_ERROR` *(new, div #29)* | **500** + `ErrorResponse` (no raw stack; "no unhandled exceptions", §11) | yes | "Internal error." |
| Model artefact missing at startup | `load_*` raises `ModelLoadError` in lifespan | `MODEL_NOT_FOUND` (reuse) | app boots; handle=None; `/health` → 503 (fail-loud, observable) | yes | (surfaced via `/health`) |

## 7. Acceptance criteria (each item must be mechanically testable by /spec-check)
- [ ] `create_app()` builds a FastAPI app that boots even when model artefacts are absent (handles load
      best-effort → `None`, app does not crash).
- [ ] `GET /health` returns 200 `healthy` only when DB is connected+migrated AND every model handle is
      ready; **503 `unhealthy`** when the DB is down OR any handle is not loaded; cache-down alone stays 200.
- [ ] `POST /predict` returns **200** with `confidence_state` for a valid vector, **200 `LOW_CONFIDENCE`/
      `AMBIGUOUS`** for weak/ambiguous input, **400 `INVALID_INPUT`** for >6 NULL, **422** for a malformed
      body, and **200 `shap_available=false`** on a SHAP fault — and **never a 5xx on model output**.
- [ ] `POST /fingerprint/unknown` returns 200 `SPARSE_PROJECTION` (<30 laps), 400 (<5 laps), and flags
      `OUT_OF_DISTRIBUTION` in the body when OOD.
- [ ] `GET /drivers` and `GET /stylespace` return **200 with empty arrays** when there is no data.
- [ ] `GET /drivers/{id}/fingerprint`, `GET /compare/{id1}/{id2}` return **404 `NOT_FOUND`** for an
      unknown driver; `/fingerprint` returns **206** on partial seasons.
- [ ] `GET /drivers/{id}/setup/{circuit_id}` returns 10 parameters (200), 404 when no data, and a
      `WIDE_CONFIDENCE` rating in the body when reliability is low.
- [ ] `GET /drivers/{id}/stability` returns **206** when fewer than 3 seasons are available.
- [ ] `GET /metrics` returns Prometheus text to an internal caller and **403 `FORBIDDEN`** to an external one.
- [ ] The 4 deferred endpoints return **503 `HEALTH_FAIL`** but are present in the OpenAPI schema with
      their response models.
- [ ] A DB-backed read with the DB down returns **503 `DB_UNAVAILABLE`** and performs no partial write.
- [ ] An unhandled exception in any handler returns a typed **500 `API_ERROR`** (no raw stack trace).
- [ ] 50 concurrent `POST /predict` requests all complete with **consistent** bodies (same input → same output).
- [ ] Every error code above has a test that triggers it and asserts the status + `ErrorResponse.error_code`.
- [ ] No `.fit()` / no refit anywhere in `app/api/` (F13 only serves). Stated explicitly.
- [ ] All thresholds/knobs read from `config.yaml` `api:`; no magic numbers; DSN stays env-only.
- [ ] `mypy app` clean, `ruff` clean; FastAPI/httpx pinned exactly in `pyproject.toml`.

## 8. Test plan
- **Client:** add `httpx` (dev). Endpoint tests use `fastapi.testclient.TestClient` (sync, simple) with
  `app.dependency_overrides` injecting **fake handles** (a `ModelHandle`/`StyleSpaceHandle`/… stub, or a
  fake that returns a canned `*Result`) and an **aiosqlite** session factory (reuse the F12 `tests/db`
  fixtures). The 50-concurrent test uses `httpx.AsyncClient(transport=ASGITransport(app=app))`.
- **Happy-path fixtures:** a fake classifier handle returning a high-confidence `PredictionResult`; a
  migrated in-memory DB seeded with one `fingerprints` row; a fake style handle with 2 known drivers.
- **Failure fixtures (one per code/mode):** `INVALID_INPUT` (7-NULL vector); `NOT_FOUND` (unknown
  driver id); `DB_UNAVAILABLE` (breaker pre-opened on the injected session dep); `DB_INTEGRITY` (duplicate
  write); `HEALTH_FAIL` (`/health` with a `None` handle and/or down DB); `FORBIDDEN` (`/metrics` with an
  external client host); `API_ERROR` (a route whose injected dep raises); `SHAP_UNAVAILABLE` (fake
  explainer `ready=False`); every `LOW_CONFIDENCE`/`AMBIGUOUS`/`SPARSE_PROJECTION`/`OUT_OF_DISTRIBUTION`/
  `WIDE_CONFIDENCE` state via a fake `*Result` carrying it → assert 200 + body state.
- **Property/edge:** empty DB → empty arrays / 404; `/predict` idempotence (same body twice → identical
  response); the deferred routes present in `app.openapi()` but 503 at call; `/predict` with DB down →
  still 200 (stateless compute — it needs no DB; see divergence #30).
- **Concurrency:** 50 concurrent `/predict` → all 200, bodies identical for identical input.

## 9. Definition of done
All acceptance criteria MET with passing tests (offline — no Postgres, no Redis, no committed
artefacts; fakes via dependency-overrides), `/apex-review F13` clean on the contract checklist,
`mypy app` + `ruff` clean, INDEX row → DONE. `failsafe-auditor` run on `app/api/` (RQ5: `/predict`
never 500, DB-down → 503 no partial write, no unhandled exceptions). Record in the dissertation
limitations: (1) the DB-backed read endpoints (`/drivers`, `/fingerprint`, `/compare`, `/stability`)
return empty/404 until a **fingerprint-population batch job** (F14/F20) writes the `fingerprints`
table — F13 serves what exists, it does not populate; (2) the 4 async/upload/replay endpoints are
schema-only in F13 (503) — their engines are F14/F05; (3) p95 < 500ms is asserted for *completion +
consistency* in F13 and formally *timed* in F20; (4) Redis cache is F14 — F13 runs cache-less
(`cache_connected` reported, never critical); (5) `/predict` is stateless — no `predictions`-row
write (divergence #30), and `request_timeout_s` is declared but not yet HTTP-enforced (F14/F20).

**Build outcome (2026-07-21).** BUILT + DONE. `app/api/` (`main` · `deps` · `errors` · `schemas` ·
`metrics` + `routes/{health,predict,drivers,style,metrics,deferred}`), the `api:` config block +
`ApiConfig`, and 3 new codes (`NOT_FOUND`/`FORBIDDEN`/`API_ERROR`). `failsafe-auditor` ran on
`app/api/` (RQ5 evidence): confirmed `/predict` never 5xx on model output, no `.fit()`, no bare
excepts, stateless routes provably DB-independent, `session_scope` rollback-before-raise (no partial
write), global handler leaks no stack trace, best-effort startup degrade always logged. Findings
fixed: DB_UNAVAILABLE + DB_INTEGRITY now write a `data_quality_log` row (§6 "Logged? yes"); the
`<min_projection_laps` gate logs; a genuine `INFERENCE_ERROR` surfaces its distinct code (not
collapsed to `HEALTH_FAIL`); `/metrics` no longer trusts a spoofed `X-Forwarded-For` from a direct
external peer; the `<5` magic number is now config-driven. Divergences #28 (endpoint split), #29
(3 codes), #30 (`/predict` stateless). `ruff` + `mypy app` clean.

`/apex-review F13` run (all 7 contract items PASS/addressed): re-ran `failsafe-auditor` on the
post-fix `app/api/` (0 HIGH; 4 prior fixes verified landed). Review follow-ups applied: added the
spec-§8 regression test **stateless routes stay 200 with the DB down** (guards the no-`get_db_session`
structural guarantee), added a `DB_INTEGRITY` dq-log-row assertion, hoisted `drivers.py`'s `else 3`
stability fallback to a named constant, and documented the intentional two-level startup logging in
`_try_load`.

**Code review (high) + security review — 2026-07-21.** Two independent review subagents ran over the
F13 diff. No HIGH code-review findings. **Fixed with tests:** (code-review M1) the three "driver
radar" endpoints (`/drivers`, `/drivers/{id}/fingerprint`, `/compare`) used three *different* and
partly non-deterministic season/circuit selection rules (oldest vs latest vs arbitrary) — a
reproducibility bug vs §11/§4.0(i); all three now select the **latest (season, circuit)
deterministically** (`ORDER BY season, circuit_id`) + a cross-endpoint consistency test; (security
HIGH-1) the Prometheus middleware labelled metrics with the **raw** request path (path params +
unmatched 404s) → unbounded label cardinality → unauthenticated memory-exhaustion DoS; now labels the
**matched route template**; (security MED-2) added `max_request_laps`/`max_feature_keys` DoS guards
(reject oversized `/fingerprint/unknown` and `/predict` payloads at 400, logged); (security MED-4)
CORS `allow_credentials` now defaults **false** + a validator forbids `"*"`-origin-with-credentials;
(security LOW-5) `/openapi.json` is now gated with `docs_enabled` (was always served); (code-review
L1) `describe_style` is guarded on a missing `shap:` block (mirrors `_stability_of`); (code-review
M2/L3) added `/stability` 200+404 tests and a `/setup` all-parameters test.

**Deployment-dependent preconditions (documented, out of F13 scope — spec §4.0 frames F13 as an
internal serving layer fronted by F14/ingress):** (security MED-3) no per-request auth on data
endpoints — an ingress/F14 auth layer is a hard precondition before public exposure; the `/metrics`
gate is IP-trust (no token) — a shared-secret token is an optional future hardening; (security LOW-6)
`request_timeout_s` HTTP enforcement is F14/F20; (code-review L2) `/fingerprint/unknown` ships the F08
projection + neighbours only — `radar_values`/`style_description` enrichment via F11 `onboard_unknown`
is deferred; (code-review L4) genuine catch-all 500s aren't Prometheus-metered (they exit via the
outer ServerErrorMiddleware); (code-review L6) `vector_5d` relies on the F14/F20 batch writer using
coordinate-sortable JSON keys.

## 10. Risks & mitigations (from master risk register where relevant)
- **No model artefacts are committed** → handles fail-loud at load; every model-serving endpoint would
  503 in a fresh checkout. → **Mitigation:** lifespan loads best-effort (app boots, `/health` reports
  the outage as 503 — correct fail-loud); all endpoint-logic tests inject **fake handles** via
  `dependency_overrides`; a marked "real-artefacts" smoke test is skipped when artefacts are absent
  (mirrors F12's `@pytest.mark.postgres`). This is the top risk and shapes the whole test strategy.
- **`/predict` must never 500 (§8.5/§11).** → The internal `predict`/`explain_prediction` never raise;
  their uncertainty states map to 200 bodies; a global exception handler is the backstop; the optional
  DB persist is best-effort (never fails the response). Unit-tested for each state.
- **DB unavailability → 503, no partial write (§11).** → Reuse F12's `session_scope` guarantees
  (rollback on any fault) + the `DatabaseUnavailableError → 503 DB_UNAVAILABLE` exception handler;
  tested with a pre-opened breaker.
- **Concurrent requests: races / inconsistent output (§11).** → Handles are frozen (read-only) dataclasses,
  per-request `AsyncSession`, pure handlers — same input → same output by construction; tested with 50
  concurrent `/predict`. (p95 timing is F20.)
- **Normalisation leakage (master register).** N/A — F13 refits nothing and loads F04 params read-only
  through the handles. Stated so `/apex-review` ticks it off.
- **Stack not pinned** (FastAPI/uvicorn ad-hoc in the venv, httpx absent). → Pin FastAPI 0.138.1 /
  starlette 1.3.1 / uvicorn 0.49.0 exactly + add httpx (dev) — reproducibility #3; back-note like the
  earlier transitive migrations.
- **Endpoint-scope ambiguity (Wk15 "stub" vs Wk16 "full").** → Resolved decisively in §4.0(a) and
  flagged as divergence #28; the 4 deferred endpoints are honest 503s with real schemas, not fakes.
