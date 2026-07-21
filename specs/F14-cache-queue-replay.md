# Feature F14 — Redis Cache + Celery Queue + Replay Engine

> One spec, one session. Self-contained: build this from this file + CLAUDE.md only.

**Status:** DONE
**Master spec sections:** §1.3 (scope — historical replay replaces live integration), §5.3 (Model 3 —
the replay's per-lap engine), §6.1 (batch SHAP as a Celery task; single-lap SHAP stays synchronous),
§8.1 (cache + queue are named components with health checks), §8.2 (`Cache` module `app/cache/redis.py`
+ `Task Queue` module `app/tasks/celery.py` + their failsafe behaviour), §8.3 (Redis circuit breaker —
timeout 2s / 1 retry / 1s backoff / open after 2 failures in 30s → cache-miss fallback, writes dropped,
log `CACHE_DEGRADED`), §8.4 (`predictions` / `fingerprints` schema — the batch-path write targets),
§8.5 (`/ingest/session` 202-async, `/ingest/status`, `/replay`, `/health` `cache_connected`, `/metrics`
cache-hit-rate), §9 (Redis via Upstash — 1-hour TTL, "cache is a performance optimisation, not a
dependency"; Celery + Redis broker; synchronous fallback for single-lap requests), §10.7 (Screen 6
Historical Race Replay Monitor), §11.4 (RQ5 failsafe: cache-unavailability resilience + concurrent
request handling), §12.1 (replay `deviation_score`-per-lap as a partner-system contract), §13 (Wk 14
cache/concurrent tests, Wk 16 replay engine)
**Layer / Week:** Platform / Wk 16
**Depends on:** **F13** (the FastAPI serving layer + the 4 deferred route stubs + `AppState`/`deps.py`
best-effort load pattern + Pydantic schemas) — **DONE**. Consumes (all read-only, refits nothing):
**F01** (`CircuitBreaker`/`BreakerConfig` pattern), **F04** (`normalisation.json`), **F06b** (`predict`),
**F07** (`explain_prediction` / `season_stability` for the batch fingerprint enrichment), **F08**
(`project_unknown` / `get_style_space` / aggregate build for `fingerprints`), **F10** (`load_detector` /
`monitor_stint` — the replay engine), **F12** (`session_scope` / `repository.persist` / `DbCircuitBreaker`
/ `Prediction` + `Fingerprint` + `Lap` ORM models / typed DB exceptions).

## 1. Purpose
F14 completes APEX's **platform layer**: a **Redis cache** that transparently accelerates the read
endpoints (and is *never* a hard dependency — the system runs identically with Redis down), a **Celery
task queue** that runs the heavy **session-ingestion batch** asynchronously (the batch path that finally
populates the `predictions` and `fingerprints` tables F13 left empty), and a **replay engine** that
streams a persisted driver-session lap-by-lap through the F10 degradation detector to power Screen 6's
Historical Race Replay Monitor. F14 fills the three async/replay routes F13 stubbed as `503`
(`POST /ingest/session`, `GET /ingest/status/{id}`, `GET /replay/{session_id}/{driver_id}`) with real
engines, and wires cache liveness + hit-rate into `/health` and `/metrics`. Like F13 it **serves and
orchestrates already-trained artefacts — it never calls `.fit()`**.

## 2. Upstream / downstream contracts
- **Consumes (all read-only, never refit):**
  - **F13** — `create_app()` / `AppState` (`app/api/deps.py`) / the best-effort `_try_load` startup
    pattern / the `app/api/routes/` registration in `main.py` / `service_unavailable` + the registered
    exception handlers (`DatabaseUnavailableError→503`, `DbIntegrityError→409`, `Exception→500 API_ERROR`)
    / all §8.5 Pydantic schemas incl. the 4 deferred response models.
  - **F12** — `session_scope(session_factory, db_breaker)`, `repository.persist(session, obj)`,
    `repository.insert_lap_features(...)`, `repository.write_quality_log_row(...)`,
    `make_db_breaker`/`DbCircuitBreaker.run`, `Lap`/`Prediction`/`Fingerprint`/`Driver`/`Circuit` ORM
    models, `DatabaseUnavailableError`/`DbIntegrityError`.
  - **F10** — `load_detector(config,...) → MonitorHandle`, `monitor_stint(stint: DataFrame, *, driver,
    circuit, circuit_type, handle, config, style_handle=None) → StintMonitorResult` (**never raises**,
    **stateless** — "F14 replays by re-calling with a growing prefix"). Surfaces `StintMonitorState`
    (`INACTIVE/BASELINE/SKIPPED/PENDING/NOMINAL/WATCH/ALERT/HALTED`), `LapDeviation`, `DegradationAlert`.
  - **F06b** — `load_model → ModelHandle`, `predict(feature_vector, *, circuit, handle, config) →
    PredictionResult` (**never raises**) → the `predictions`-row source.
  - **F08** — `load_style_space → StyleSpaceHandle`, `project_unknown(...) → ProjectionResult`,
    `get_style_space`, plus the driver-aggregate build for `fingerprints.vector` (5D). **`build_style_space`
    is a fit site and is NOT called by F14** (serving only).
  - **F07** — `explain_prediction` / `season_stability` for the optional `fingerprints.shap_global` /
    `radar_values` / `stability_score` enrichment. SHAP never blocks (`SHAP_UNAVAILABLE`).
  - **F01** — the `CircuitBreaker` / `call_with_breaker` / `BreakerConfig` shape (the Redis breaker mirrors
    the async `DbCircuitBreaker.run`).
  - `config.yaml` → new `cache:`, `celery:`, `replay:` blocks; the Redis/Celery DSNs are **env-only**
    (`APEX_REDIS_URL`, `APEX_CELERY_BROKER_URL`, `APEX_CELERY_RESULT_BACKEND`) — never committed (§9).
- **Produces:**
  - `app/cache/redis.py` — `RedisCache` (async client wrapper + its own circuit breaker + cache-aside
    `get_or_compute` + namespaced key builder), a typed `CacheResult`, and the `CACHE_*` graceful-miss
    contract. A `FakeCache`/`fakeredis`-backed test double.
  - `app/tasks/celery.py` — the Celery app factory (broker+backend = Redis; `task_always_eager` in
    tests) + `ingest_session_task` (the async pipeline→persist batch) + `batch_shap_task` (§6.1) + the
    job status/progress helpers.
  - `app/replay/engine.py` — `replay_stint(session_id, driver_id, *, session, handles, config) →
    ReplayResult` (read-only; persisted laps → growing-prefix `monitor_stint` → per-lap trace).
  - `app/api/routes/ingest.py` (`POST /ingest/session` 202, `GET /ingest/status/{id}`) and
    `app/api/routes/replay.py` (`GET /replay/{session_id}/{driver_id}`) — replace the F13 deferred stubs.
    `POST /ingest/upload` **stays an F05 503 stub** (out of F14 scope).
  - New request schema `IngestSessionRequest` (`app/api/schemas.py`); real `cache_connected` at `/health`
    + cache-hit-rate / queue-depth / breaker-state metrics at `/metrics`.
  - **Three new error codes:** `QUEUE_UNAVAILABLE`, `INGEST_TASK_FAILED`, `REPLAY_PARTIAL`.
  - **No new model artefact.** F14 writes *DB rows* (`laps`/`lap_features`/`predictions`/`fingerprints`/
    `degradation_alerts`) via the F12 repository; it produces no `.pkl`/`.json` artefact.
- **Consumed by:** **F19** (Screen 6 replay monitor calls `/replay` + drives client-side playback),
  **F15 / F20** (the RQ5 harness fires the cache-unavailability + 50-concurrent tests here; F20 also uses
  the ingestion batch to populate the eval DB), **F16–F18** (benefit from the cached read endpoints).

## 3. Inputs & outputs (typed)
F14's surface is (a) HTTP routes and (b) a Celery task. All DTOs extend F13's `ApiModel`
(`extra="forbid"`; a stray field → 422).

- **`IngestSessionRequest`** *(new)* `{ year: int, event: str, session: str = "R" }` — the FastF1
  session coordinates to ingest (`event` = GP name/round; `session` = R/Q/FP…). Validators: `year` in a
  sane range; non-empty `event`; `session` in an allowed set.
- **`IngestSessionResponse`** *(exists, F13)* `{ session_id, laps_accepted, laps_rejected,
  rejection_summary, quality_report }` — returned **202** at enqueue with `session_id` set and the count
  fields at their *pending* placeholders (see §4.0-b, divergence #31).
- **`IngestStatusResponse`** *(exists)* `{ status: PENDING|PROCESSING|COMPLETE|FAILED, progress: float,
  error_detail: str|None }`. `404` if `session_id` unknown.
- **`ReplayResponse`** *(exists)* `{ laps: list[ReplayLap] }`, `ReplayLap { lap_number, feature_vector,
  deviation_score, alert_state, fingerprint_snapshot }`. `200` full · `206` partial · `404` unknown.
- **`CacheResult`** *(new, internal — not an HTTP body)* `{ value: Any|None, hit: bool,
  state: ErrorCode|None }` where `state ∈ {None (hit/clean miss), CACHE_MISS, CACHE_UNAVAILABLE,
  CACHE_DEGRADED}`. Never raises.
- **`ReplayResult`** *(new, internal)* mirrors `ReplayResponse` + `partial: bool` + `flags: list[ErrorCode]`
  (`BaseResult`-derived). Maps to the HTTP body in the route.
- **`HealthResponse.cache_connected`** *(exists)* — F14 makes it **real** (`True` iff the Redis breaker is
  closed and a `PING` succeeds; `False` when down). **Never critical** — it never forces 503.

## 4. Method / algorithm

### 4.0 Design notes (the decisions a builder must not re-litigate)
- **(a) The cache is non-critical, by construction (§9 / constitution #non-negotiable).** Every cache
  call goes through `RedisCache`, which owns its **own circuit breaker** (§8.3 policy: timeout 2s, 1
  retry, 1s backoff, opens after 2 failures in 30s). A cache **read** that misses / times out / hits an
  open breaker returns a `CacheResult(hit=False, state=CACHE_MISS|CACHE_UNAVAILABLE|CACHE_DEGRADED)` and
  the caller falls through to the real source — **the request still succeeds with a 200**. A cache
  **write** with the breaker open is **silently dropped** (`CACHE_DEGRADED`, logged once). `RedisCache`
  **never raises** into a handler. `cache_connected` is reported at `/health` but is *never* a critical
  component. This is the RQ5 "cache unavailability resilience" guarantee (§11.4) made structural.
- **(b) `/ingest/session` is 202-async; the terminal report is polled (divergence #31).** The master
  §8.5 lists `/ingest/session` with *both* a full-result body **and** "202 Accepted — processing is
  async." For a fire-and-forget queue you can't return real counts before the job runs, so: `POST`
  enqueues `ingest_session_task` and returns **202** with `session_id` (= the Celery task id) and honest
  *pending placeholders* (`laps_accepted=0`, …, `quality_report={"state":"PENDING"}`) — **no fabricated
  counts** (§1). `GET /ingest/status` returns `{status, progress, error_detail}` throughout **and, once
  `COMPLETE`, the terminal report in its `result` field** — the real `laps_accepted`/`rejection_summary`/
  `quality_report` (the §8.5 count fields). So the master's response contract is genuinely fulfilled,
  just deferred to the poll. (`result` is an optional field added to `IngestStatusResponse` — the F13
  schema stays backwards-compatible.) The counts also live durably in the persisted `laps` +
  `data_quality_log` rows, but no client needs to reconstruct them from there.
- **(c) The Celery ingestion task is THE batch writer — replay is read-only (divergence #36; resolves
  F13 #30).** The master ties DB population to the async ingestion path (§8.2 "large session ingestion";
  classifier produces `predictions`; fingerprint aggregate feeds `fingerprints`), and gives `/replay` a
  **read-only** contract (returns a `laps` array; §8.5). So: **`ingest_session_task`** runs the pipeline
  and writes `laps`/`lap_features`/`predictions`/`fingerprints` (now the `predictions` FKs `lap_id` +
  `predicted_driver_id` resolve to persisted rows — the exact blocker that made F13's `/predict`
  stateless). **The replay engine writes nothing** — it reads persisted laps and recomputes the
  deviation trace deterministically per call. This makes F13's DB-backed reads (`/drivers`,
  `/fingerprint`, `/compare`, `/stylespace`, `/setup`) light up once a session is ingested.
- **(d) Replay drives F10 by growing prefix (§5.3 / §10.7).** `monitor_stint` is pure over the prefix it
  is given. The replay engine loads the (session, driver)'s persisted clean+unclean laps ordered by
  `lap_number`, reconstructs the `monitor_stint` input DataFrame (required columns: `lap_number`,
  `tyre_age`, `track_status`, `is_clean`, `has_interpolation`, + the 28 `FEATURE_IDS`), and for lap *k*
  in `1..N` calls `monitor_stint(frame.iloc[:k])`, emitting that lap's `LapDeviation` →
  `ReplayLap{deviation_score, alert_state=state.value, fingerprint_snapshot}`. Playback (Play/Pause/1×/5×/
  10×/scrub) is **client-side** (Zustand `replay position`, §9) — there is **no websocket** (§13 "Live
  WebSocket replaced with replay engine"). The endpoint returns the whole `laps` array in one pull.
- **(e) Worker/broker-down degradation is typed, never silent (§8.2).** Master: "synchronous fallback
  for single-lap requests only; batch requests queued until worker recovers." `/predict` (the single-lap
  request) is already synchronous (F13) and needs no queue — the fallback rule holds by construction. A
  **batch** `POST /ingest/session` when the broker cannot be reached (cannot even enqueue) returns a
  typed **503 `QUEUE_UNAVAILABLE`** (logged) — the request is refused loudly, not dropped. When the
  broker is up but no worker is consuming, the job simply sits `PENDING` (master's "queued until worker
  recovers") — honest and observable via `/ingest/status`.
- **(f) F14 refits nothing.** All artefacts (`normalisation.json`, `classifier.pkl`, `umap_model.pkl`/
  `pca_fallback.pkl`, `stint_variation.json`) load read-only through the F06b/F08/F10 handles. No
  `.fit()` anywhere in `app/cache/`, `app/tasks/`, `app/replay/`. Stated so `/apex-review` ticks it off.
- **(g) The whole unit suite runs offline (mirrors F12's aiosqlite pattern).** Cache tests use a
  `fakeredis`/dict-backed `FakeCache`; Celery tests run with `task_always_eager=True` +
  `task_eager_propagates` (the task body runs in-process, synchronously, no broker/worker); DB writes go
  to aiosqlite via the F12 `tests/db` fixtures. A `@pytest.mark.redis` and a `@pytest.mark.celery`
  integration test exercise the real Redis/worker only when `APEX_REDIS_URL`/`APEX_CELERY_BROKER_URL`
  are set — skipped by default. **No Redis, no worker, no committed artefacts required to test.**
- **(h) Cache keys are namespaced + versioned (divergence #33; master gives no key scheme).** Key =
  `f"{cache.key_prefix}:v{api.version}:{namespace}:{ident}"` — e.g.
  `apex:v0.1.0:driver_profile:{driver_id}`, `apex:v0.1.0:stylespace`,
  `apex:v0.1.0:predict:{sha256(feature_vector||circuit)}`. The `v{api.version}` segment invalidates the
  whole cache on an API version bump. Master gives no invalidation-on-write rule (TTL-only) — F14
  additionally **invalidates the affected `driver_profile:{id}` keys** after a batch fingerprint write
  (divergence #37) so a re-ingest is not shadowed by a stale profile for up to an hour.
- **(i) Stack additions (reproducibility #3; same "feature adds the dep it imports" pattern as
  F01/F06a/F07/F12/F13).** F14 pins **`redis` (async, `redis.asyncio`)** and **`celery` 5.x** (+
  `fakeredis` as a dev dep). `redis-py` is pure-Python (cp314 ✓). Celery pulls `kombu`/`billiard`/`vine`;
  **if `billiard` lacks a cp314 wheel** (the F12-psycopg risk), the contingency is documented in §10 (run
  the worker under a compatible interpreter / use the `solo`/threads pool / the offline suite is
  broker-independent via eager mode). Exact pins are chosen at build and back-noted in CLAUDE.md.

### 4.1 The three F14 components (path · engine · source · failsafe)

| Component | Path / entrypoint | Engine | Source | Failsafe contract |
|---|---|---|---|---|
| **Cache** | `app/cache/redis.py` `RedisCache.get_or_compute(key, ttl, compute)` | cache-aside + own breaker | Redis (Upstash) | miss/down → `CACHE_*` → compute from source, **200**; writes dropped when open |
| **Queue** | `POST /ingest/session` → `ingest_session_task` | Celery (Redis broker) | FastF1 → F01–F04 → F06b/F08 → DB | broker down → **503 `QUEUE_UNAVAILABLE`**; task fault → status `FAILED` + `INGEST_TASK_FAILED` |
| **Replay** | `GET /replay/{session_id}/{driver_id}` → `replay_stint` | growing-prefix `monitor_stint` | DB `laps`/`lap_features` (read-only) | unknown → **404 `NOT_FOUND`**; partial → **206 `REPLAY_PARTIAL`**; DB down → **503 `DB_UNAVAILABLE`** |

### 4.2 `ingest_session_task` (the async batch — §8.2/§8.4)
Runs in a worker (or eagerly in tests). Deterministic, idempotent per `session_id`. Steps, each updating
job `progress` in the result backend:
1. **Ingest** (F01) the FastF1 session from cache/source → raw laps (breaker-guarded; `DATA_001`/
   `DATA_SOURCE_UNAVAILABLE`/`STALE_DATA` on source trouble).
2. **Validate** (F02) → clean/unclean laps + per-lap `LapStatus` (`LAP_*` rows → `data_quality_log`).
3. **Features** (F03a/F03b) → the 28-key vector per lap (`FEATURE_001`/`LAP_NULL_EXCESS`/`IMPUTED`).
4. **Normalise** (F04) read-only via `normalisation.json` (`NORMALISATION_FALLBACK` for unknown circuit).
5. **Persist** each lap (`repository.persist(Lap)`, `insert_lap_features`) inside `session_scope`;
   count `laps_accepted`/`laps_rejected`; build `rejection_summary` (per-code counts).
6. **Predict** (F06b `predict`) per clean lap → `repository.persist(Prediction)` (`lap_id` +
   `predicted_driver_id` now resolvable). `LOW_CONFIDENCE`/`AMBIGUOUS` are stored states, not failures.
7. **Fingerprint** — aggregate the driver's clean normalised laps → F08 5D `vector` + F07 `radar_values`/
   `shap_global` + `season_stability` `stability_score` → `repository.persist(Fingerprint)`
   (UNIQUE(driver_id, season, circuit_id) → upsert-or-skip; a duplicate → `DbIntegrityError` handled).
8. **Invalidate** the affected `driver_profile` cache keys (§4.0-h) and store the terminal
   `IngestSessionResponse` in the result backend; set status `COMPLETE`.
Any uncaught fault → status `FAILED`, `error_detail` set, `INGEST_TASK_FAILED` logged to
`data_quality_log` — **the job never dies silently** (§1). A `DB_UNAVAILABLE` mid-batch rolls back the
open `session_scope` (no partial write) and fails the job with that code.

### 4.3 `/health` + `/metrics` (§8.5)
`/health` gains a **real** `cache_connected` (breaker-closed + `PING`); still **healthy iff DB + every
model handle ready** (cache excluded). `/metrics` adds `apex_cache_hits_total`/`apex_cache_misses_total`
(hit rate), `apex_cache_breaker_state`, `apex_ingest_jobs{status=…}` (queue depth by state) — labelled by
the **matched route template only** (the F13 security fix — no unbounded cardinality).

## 5. config.yaml keys this feature adds
```yaml
# --- F14 Redis cache (an optimisation, never a hard dependency; §8.2/§8.3/§9) ---------------------
# The cache accelerates the read endpoints and the model-output path. Redis is NEVER critical: a miss,
# timeout, or open breaker degrades to a graceful CACHE_MISS and the request serves from source (200).
# The DSN is a SECRET, sourced from url_env_var — never committed here (§9). The breaker mirrors the
# master §8.3 Redis policy VERBATIM. The offline test suite uses a FakeCache and never reads this DSN.
cache:
  enabled: true                     # false → RedisCache is a no-op (always CACHE_MISS); system unaffected
  url_env_var: APEX_REDIS_URL       # env var holding the Redis DSN (e.g. rediss://…@upstash)
  key_prefix: apex                  # §4.0-h namespace root; full key = {prefix}:v{api.version}:{ns}:{id}
  driver_profile_ttl_s: 3600        # §9 — 1-hour driver-profile TTL (the one master-named TTL)
  model_output_ttl_s: 3600          # /predict model-output cache TTL (master-silent → mirror 1h, div #34)
  normalisation_ttl_s: 3600         # normalisation-param cache TTL (master-silent → mirror 1h, div #34)
  stylespace_ttl_s: 3600            # style-space read TTL (master-silent → mirror 1h, div #34)
  breaker: { timeout_s: 2, retries: 1, backoff_s: [1], open_after: 2, window_s: 30 }  # §8.3 verbatim

# --- F14 Celery task queue (async ML batch jobs; §8.2/§9) -----------------------------------------
# Redis is broker AND result backend (§9). The heavy async job is session ingestion; batch SHAP (§6.1)
# is the other task. Broker/backend DSNs are env-only. Tests run task_always_eager (no broker/worker).
# Celery retry/timeout constants are master-SILENT (§8.3 numbers are for external HTTP deps) → chosen
# here (divergence #35).
celery:
  broker_url_env_var: APEX_CELERY_BROKER_URL       # Redis broker DSN (env-only)
  result_backend_env_var: APEX_CELERY_RESULT_BACKEND  # Redis result backend DSN (env-only)
  task_default_queue: apex
  worker_concurrency: 4             # worker pool size (design decision)
  task_soft_time_limit_s: 600       # soft per-task cap → graceful stop (design decision)
  task_time_limit_s: 900            # hard per-task cap → kill (design decision)
  task_max_retries: 3               # ingest-task retries on a transient fault (div #35)
  task_retry_backoff_s: [5, 15, 30] # backoff between task retries (div #35)
  result_ttl_s: 86400               # result-backend retention for the terminal IngestSessionResponse
  always_eager: false               # tests set true (synchronous, in-process, no broker)

# --- F14 replay engine (Screen 6 Historical Race Replay Monitor; §5.3/§10.7/§12.1) ----------------
# Read-only: reads persisted (session, driver) laps and streams a growing prefix through F10
# monitor_stint. All Model-3 constants (baseline_laps, rolling_window, alert_threshold_sd, min_stint_laps,
# alert_confirm_laps, …) live under `monitoring:` and are referenced, never re-declared. Playback speed
# is client-side (Zustand) — not server config.
replay:
  max_replay_laps: 200              # DoS guard on the returned laps array (mirrors api.max_request_laps)
```
DSNs are **env-only** (`.env.example` documents `APEX_REDIS_URL` / `APEX_CELERY_BROKER_URL` /
`APEX_CELERY_RESULT_BACKEND`). New typed `CacheConfig` / `CeleryConfig` / `ReplayConfig` live in
`app/contracts/config.py` (standing rule — feature config in `config.py`, `extra="forbid"`), each
embedding the shared `BreakerConfig` where relevant, wired as `AppConfig.cache/celery/replay: … | None`.
Validators: TTLs > 0; `key_prefix` non-empty; `worker_concurrency`/`task_*_time_limit_s` > 0;
`task_retry_backoff_s` length ≥ `task_max_retries` or documented shorter-repeats-last. `api.request_timeout_s`
HTTP enforcement remains **F20** (perf harness), not F14 — F14 supplies the cache that meets the p95 budget.

## 6. Error codes & failure modes (the failsafe contract)

| Failure mode | Detection | Error code | System response | Logged? | User-facing message |
|--------------|-----------|-----------|-----------------|---------|---------------------|
| Cache read miss / Redis down / breaker open | `RedisCache.get` → key absent / timeout / open | `CACHE_MISS` (+ `CACHE_UNAVAILABLE`) *(reuse, pre-seeded)* | graceful miss → compute from source → **200** | yes → `CACHE_UNAVAILABLE` once per outage | (transparent; slightly slower) |
| Cache write while breaker open | `RedisCache.set` with breaker open | `CACHE_DEGRADED` *(reuse)* | write **silently dropped**, request unaffected → **200** | yes (once) | (transparent) |
| Redis down for a whole prediction run (RQ5) | Redis disabled; 100 consecutive `/predict` | `CACHE_UNAVAILABLE` *(reuse)* | zero failed predictions; < 200ms avg increase (§11.4) | yes (throughout) | (transparent) |
| Celery broker/worker unreachable at enqueue (batch) | broker ping / `apply_async` raises | `QUEUE_UNAVAILABLE` *(new)* | **503** + `ErrorResponse`; batch refused loudly (never dropped) | yes | "Ingestion queue temporarily unavailable." |
| Async ingestion task fault | uncaught exception in `ingest_session_task` | `INGEST_TASK_FAILED` *(new)* | job status → **`FAILED`** + `error_detail`; `/ingest/status` **200** | yes → `data_quality_log` | "Ingestion failed: <detail>." |
| Unknown `session_id` on `/ingest/status` or `/replay` | result-backend / DB lookup misses | `NOT_FOUND` *(reuse, F13)* | **404** + `ErrorResponse` | no | "Session not found." |
| Unknown session **or** driver on `/replay` | DB read returns no laps | `NOT_FOUND` *(reuse)* | **404** + `ErrorResponse` | no | "Session or driver not found." |
| Partial replay data (some laps missing/unscored) | fewer persisted laps than session count / gaps | `REPLAY_PARTIAL` *(new)* | **206** + `laps` (flag set); no fabrication of missing laps | no | "Partial replay data returned." |
| DB unreachable on `/replay` or a batch write | `DatabaseUnavailableError` from `session_scope` | `DB_UNAVAILABLE` *(reuse, F12)* | **503**; rollback; **no partial write** (§11.4) | yes | "Database temporarily unavailable." |
| Integrity violation on a batch write (dup fingerprint) | `DbIntegrityError` from `persist` | `DB_INTEGRITY` *(reuse)* | row skipped/upserted; job continues; **409** if surfaced at HTTP | yes → `data_quality_log` | (internal) |
| Model-3 states surfaced during replay | `monitor_stint` typed states | `MONITOR_INACTIVE`/`BASELINE_INSUFFICIENT`/`FALLBACK_THRESHOLD`/`DEGRADED_QUALITY`/`DISTANCE_NAN`/`MONITOR_ERROR` *(reuse, F10)* | **200/206** with per-lap `alert_state`; never a 5xx | (per F10) | (state-specific) |
| Source unavailable during ingestion, cache present | F01 breaker open + cache hit | `STALE_DATA` *(reuse)* | ingest proceeds on cached telemetry, flagged | yes | (flagged in `quality_report`) |
| Cache poisoned / unreadable value | deserialise fault on a cached blob | `CACHE_MISS` *(reuse)* | treat as miss → recompute; optionally evict key | yes | (transparent) |

## 7. Acceptance criteria (each item must be mechanically testable by /spec-check)
- [ ] `RedisCache.get_or_compute` returns the cached value on a hit and **computes from source on a
      miss**; a **down Redis / open breaker never raises** and yields a `CACHE_MISS`/`CACHE_UNAVAILABLE`/
      `CACHE_DEGRADED` `CacheResult` while still returning the computed value.
- [ ] With the cache disabled/down, **100 consecutive `/predict` calls all return 200** (zero failures)
      and `CACHE_UNAVAILABLE` is logged — the §11.4 cache-unavailability guarantee (perf bound formally
      timed in F20).
- [ ] The Redis breaker opens after **2 failures in 30s** and short-circuits reads to a graceful miss
      (config-driven, no magic numbers).
- [ ] `POST /ingest/session` returns **202** with a `session_id` and pending-placeholder counts, and
      enqueues `ingest_session_task`; when the broker is unreachable it returns **503 `QUEUE_UNAVAILABLE`**.
- [ ] `ingest_session_task` (eager) runs F01→F04→F06b→F08 and **persists `laps` + `lap_features` +
      `predictions` + `fingerprints`**, sets progress and terminal `COMPLETE` status; a mid-task fault
      sets **`FAILED` + `INGEST_TASK_FAILED`** (logged) and **rolls back with no partial write** on a DB
      outage.
- [ ] `GET /ingest/status/{id}` returns `{status ∈ PENDING/PROCESSING/COMPLETE/FAILED, progress,
      error_detail}`; **404 `NOT_FOUND`** for an unknown id.
- [ ] `GET /replay/{session_id}/{driver_id}` returns **200** with one `ReplayLap` per persisted lap whose
      `deviation_score`/`alert_state`/`fingerprint_snapshot` come from `monitor_stint` on the growing
      prefix; **404 `NOT_FOUND`** for an unknown session/driver; **206 `REPLAY_PARTIAL`** on partial data;
      **503 `DB_UNAVAILABLE`** when the DB is down (no partial write).
- [ ] The replay per-lap trace is **deterministic** and equals lap-by-lap `monitor_stint` on the same
      prefixes (growing-prefix parity) — same session → identical `laps` array.
- [ ] After an ingestion completes, F13's DB-backed reads (`GET /drivers`, `/drivers/{id}/fingerprint`,
      `/compare`, `/stylespace`, `/setup`) return the **persisted** rows (no longer empty).
- [ ] `/health` reports a **real** `cache_connected` and stays **200 when only the cache is down**;
      `/metrics` exposes cache hit-rate + queue-depth + breaker-state.
- [ ] Every error code above has a unit test that triggers it and asserts the typed result/status.
- [ ] **No `.fit()` / no refit anywhere** in `app/cache/`, `app/tasks/`, `app/replay/` (F14 serves only).
- [ ] All thresholds/TTLs/breaker knobs read from `config.yaml` (`cache:`/`celery:`/`replay:`/`monitoring:`);
      **no magic numbers**; Redis/Celery DSNs stay **env-only**.
- [ ] Whole suite runs **offline** (FakeCache + Celery eager + aiosqlite, no committed artefacts);
      `mypy app` clean, `ruff` clean; `redis`/`celery` pinned exactly in `pyproject.toml`.

## 8. Test plan
- **Cache (`tests/cache/`):** `FakeCache` (dict-backed) + `fakeredis` for the async path. Cases: hit
  returns cached; miss computes + stores; TTL honoured; a raising/timing-out backend → graceful
  `CACHE_UNAVAILABLE` (never raises); breaker opens after 2 failures/30s then short-circuits reads;
  write-with-open-breaker → dropped + `CACHE_DEGRADED`; key builder namespaces + versions; poisoned value
  → treated as miss; `enabled:false` → always miss.
- **Queue (`tests/tasks/`):** `always_eager=True`. Happy path: a fake FastF1 session (reuse the F01/F02
  synthetic-lap fixtures) → task persists laps/predictions/fingerprints into an aiosqlite DB; assert row
  counts + `rejection_summary` + terminal `COMPLETE`. Failure fixtures: a pipeline fault →
  `FAILED`+`INGEST_TASK_FAILED`+dq-log row; a DB outage mid-batch (pre-opened breaker) → rollback, no
  partial write; a duplicate fingerprint → `DbIntegrityError` handled, job continues; broker-unreachable
  at the route → `QUEUE_UNAVAILABLE`.
- **Replay (`tests/replay/`):** seed an aiosqlite DB with a persisted stint (clean + a couple of unclean
  laps). Assert: one `ReplayLap` per lap; growing-prefix parity vs direct `monitor_stint` calls;
  `alert_state` transitions (`NOMINAL`→`WATCH`→`ALERT`) on a synthetic degrading stint; 404 unknown
  session/driver; 206 + `REPLAY_PARTIAL` when laps are missing; 503 when the DB breaker is open; a
  `MONITOR_INACTIVE` short stint surfaces the state (200) not an error; `max_replay_laps` truncation
  logged (no silent cap).
- **API (`tests/api/`):** extend F13's `dependency_overrides` + `TestClient` fixtures with a
  `FakeCache`/eager-Celery/`ReplayResult` override. `POST /ingest/session` → 202 + schema; `/ingest/status`
  PENDING→COMPLETE→FAILED + 404; `/replay` 200/206/404/503; the **RQ5 cache-resilience** test (Redis
  down → 100 `/predict` all 200, `CACHE_UNAVAILABLE` logged); a cache-hit path asserts the source
  compute is **not** re-invoked. Reuse the F13 50-concurrent `/predict` consistency test (now cache-fronted).
- **Config (`tests/contracts/`):** `CacheConfig`/`CeleryConfig`/`ReplayConfig` validators (bad TTL,
  empty prefix, missing env var name); DSN never read from YAML.
- **Property/edge:** empty session (0 laps) → 202 then `COMPLETE` with 0/0 counts (not an error);
  replay of a 1-lap stint → single `MONITOR_INACTIVE`/`BASELINE` lap; cache key collisions across API
  versions impossible (version segment); idempotent re-ingest of the same session (no duplicate rows).

## 9. Definition of done
All acceptance criteria MET with passing tests (offline — no Redis, no worker, no Postgres, no committed
artefacts; fakes via `dependency_overrides` + `always_eager` + aiosqlite), `/apex-review F14` clean on
the contract checklist, `mypy app` + `ruff` clean, `redis`/`celery`/`fakeredis` pinned in
`pyproject.toml`, INDEX row → DONE. `failsafe-auditor` run on `app/cache/` + `app/tasks/` + `app/replay/`
(RQ5 evidence: cache-down never fails a request; broker-down → typed 503; task fault → `FAILED` never
silent; replay DB-down → 503 no partial write; no `.fit()`; no bare excepts; no stack-trace leak).
Record in the dissertation limitations: (1) `/ingest/session` is 202-async — the terminal counts are
returned by polling `GET /ingest/status` (its `result` field on `COMPLETE`), not inline on the 202
(divergence #31); (2) replay recomputes deviations per call from persisted
laps (read-only) rather than serving a precomputed trace — chosen for determinism + zero write coupling
(divergence #36); (3) worker-down leaves batch jobs `PENDING` (master's "queued until worker recovers")
and only single-lap `/predict` has a synchronous path (§8.2); (4) no RQ5 test exists for worker-down or
replay correctness in the master (§11.4) — F14 covers them with its own unit tests, formal perf timing is
F20; (5) cache invalidation is TTL-based + a driver-profile invalidate-on-write (divergence #37) — no
finer-grained scheme (master is TTL-only).

**Build outcome (2026-07-21).** BUILT + DONE. `app/cache/redis.py` (`RedisCache` + own breaker +
cache-aside), `app/tasks/{ingest,celery}.py` (`run_ingestion` + Celery app + task shell + job stores +
`enqueue_ingestion`), `app/replay/engine.py` (growing-prefix replay), the `app/api/routes/{ingest,replay}.py`
routes (deferred `deferred.py` shrunk to the F05 upload stub), `cache:`/`celery:`/`replay:` config blocks
+ typed `CacheConfig`/`CeleryConfig`/`ReplayConfig`, and **3 new codes** (`QUEUE_UNAVAILABLE`/
`INGEST_TASK_FAILED`/`REPLAY_PARTIAL`). Pins **redis 8.0.1 / celery 5.6.3** (+ `fakeredis` dev) —
billiard 4.2.4 had a cp314 wheel, so the F12-psycopg risk did NOT recur. **~57 F14 tests** (offline:
`fakeredis` + Celery `always_eager` + aiosqlite; no server/worker/artefacts); full suite green, `ruff` +
`mypy app` clean. **`failsafe-auditor` run** on `app/cache` + `app/tasks` + `app/replay` (RQ5 evidence):
confirmed the cache never fails a request, `run_ingestion` is atomic (rollback, no partial write),
broker-down → typed 503, task fault → FAILED+`INGEST_TASK_FAILED`, replay is read-only + never fabricates.
**Findings fixed with regression tests:** (HIGH-1) `RedisJobStore` now bounds every client call with
`asyncio.wait_for` + a socket timeout (can't hang enqueue/status) and **logs** faults instead of
swallowing them; (HIGH-2) `predictions` are now **deduped on re-ingest** (get-or-update by `lap_id` — the
idempotency criterion; the test now asserts the `Prediction` count); (HIGH-3) a **poisoned cache value**
is now logged (was a silent evict); (MED-5) the `/metrics` ingest-job counter is **wired** (`enqueued`
inflow). **Divergence #31 wired for real** (per review): `IngestStatusResponse` gained an optional
`result` field, so a `COMPLETE` poll returns the terminal §8.5 counts — the master's response contract is
fulfilled, not hand-waved.
**Deferred (documented, not silent):** (MED-4) `celery.task_max_retries`/`task_retry_backoff_s` are
declared but the Celery `autoretry` policy is **not wired** — the task fails fast + loud on the first
fault (safe, honest; enforcement is a worker-level policy, F20/ops, like F13's `request_timeout_s`);
worker-side `completed`/`failed` queue metrics need a worker metrics exporter (F20); `IngestSessionRequest.session`
accepts any 1–8-char token (downstream-validated), not a pinned R/Q/FP set.

## 10. Risks & mitigations (from master risk register where relevant)
- **Cache must never be a hard dependency (constitution / §9 / RQ5).** → `RedisCache` owns a breaker,
  never raises into a handler, and every read has a source fallback; `/health` never treats cache as
  critical; the §11.4 "100 predictions, Redis down" test is an acceptance criterion. This is the top
  risk and shapes the cache design.
- **`billiard`/Celery may lack a Python-3.14 (cp314) wheel** (the F12-psycopg risk repeats). →
  Contingency: the **offline suite is broker-independent** (`always_eager`, no worker process needed);
  the real worker can run under a compatible interpreter or the `solo`/threads pool; document the pin +
  rationale in CLAUDE.md as a ratified transitive migration (F04/F06a/F07/F12 precedent). Do **not** let
  a worker-runtime issue block the offline-testable feature.
- **Async result-surfacing tension (§8.5 full body vs 202).** → Resolved decisively in §4.0-b
  (divergence #31): 202 + pending placeholders + poll `/ingest/status`; terminal report in the result
  backend **and** durably in `data_quality_log`/`laps` — never fabricated, never lost.
- **Replay needs persisted laps carrying `monitor_stint`'s columns.** → Build-step: verify the F12
  `Lap`/`lap_features` schema supplies `lap_number`/`tyre_age`/`track_status`/`is_clean`/
  `has_interpolation` + the 28-feature vector; any gap is an **additive** seam fix on the F14 branch
  (precedent: F01 `track_status`, F06a `labels.json`) — reconstruct `is_clean` from `error_code=='OK'`,
  default `has_interpolation` from the `LAP_INTERP` tag. Flagged, not stubbed.
- **DB unavailability → 503, no partial write (§11.4).** → Reuse F12's `session_scope` rollback-on-fault
  + the `DatabaseUnavailableError→503` handler; the batch task fails the whole job (no half-written
  session); tested with a pre-opened breaker.
- **Concurrent requests: races / inconsistent output (§11.4).** → Cache reads are per-key idempotent;
  handles stay frozen (F13); the model-output cache is keyed by a content hash so same input → same
  cached output; reuse F13's 50-concurrent `/predict` consistency test (now cache-fronted). p95 timing
  is F20.
- **Normalisation leakage (master register).** N/A — F14 loads `normalisation.json` read-only through the
  handles and refits nothing. Stated so `/apex-review` ticks it off.
- **New error codes vs reuse discipline.** → Only **3** new codes (`QUEUE_UNAVAILABLE`,
  `INGEST_TASK_FAILED`, `REPLAY_PARTIAL`); everything else reuses pre-seeded codes (the 3 `CACHE_*` were
  seeded for F14; `NOT_FOUND`/`DB_*`/`STALE_DATA`/the F10 states are reused). Flagged as divergence #32.
```
