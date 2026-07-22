# Feature F15 — Failsafe Evaluation Harness (RQ5 — the 7 §11.4 tests)

> One spec, one session. Self-contained: build this from this file + CLAUDE.md only.

**Status:** DONE
**Master spec sections:** §11.4 (Failsafe System Evaluation / RQ5 — **the 7 tests**, the sole scope),
§2.4 (RQ5 itself: "What are the principal failure modes… and how do explicit failsafe mechanisms affect
output reliability?"), §11 intro ("report failure cases alongside success metrics… documented as
findings"), §8.3 (the circuit-breaker policy table the tests assert), §8.5 (`/predict` never-5xx,
`/health` 503-iff-critical, `/metrics` latency), §5.1 (confidence gate — LOW_CONFIDENCE/AMBIGUOUS),
§4.5 (6-NULL tolerance / `LAP_NULL_EXCESS`), §5.2 (`OUT_OF_DISTRIBUTION`), §13 Project Plan **Week 14**
(deliverable: "RQ5 evaluation results table. All 7 tests passing or documented as findings")
**Layer / Week:** Platform (evaluation) / Wk 14
**Depends on:** **F13** (the assembled FastAPI app + `/predict`/`/health`/DB-read/`/fingerprint/unknown`
routes + exception handlers) — **DONE**; **F14** (Redis cache + `RedisCache` fault surface + Celery
queue) — **DONE**. Consumes read-only: **F01** (`ingest_session` + `CircuitBreaker` + `STALE_DATA`),
**F03a** (28-vector assembly + 6-NULL gate), **F06b** (`predict` confidence gate), **F08**
(`project_unknown` OOD), **F12** (`DbCircuitBreaker`/`session_scope`). **All pre-seeded error codes; F15
adds none.**

## 1. Purpose
F15 is APEX's **RQ5 evidence trail made executable**: a harness that assembles the real F13 + F14
system and runs the master's **7 §11.4 failsafe tests** with *real fault injection and measurement* —
Redis disabled, the DB breaker forced open, FastF1 mocked-down, 50 requests fired concurrently — then
emits the **RQ5 evaluation results table** (the Week-14 deliverable). It is the *dynamic* complement to
the static `failsafe-auditor` sub-agent: the auditor reads code to confirm each failure mode is
*handled*; F15 *executes* the assembled system under fault and confirms it *behaves*. F15 is **§11.4
only** — the other three §11 pillars (classifier §11.1, stability §11.2, degradation §11.3) belong to
F20, which invokes F15's harness for §11.4 rather than reimplementing it (divergence #39).

## 2. Upstream / downstream contracts
- **Consumes (all read-only, refits nothing):**
  - **F13** — `create_app()` / `AppState` / `deps.py` providers (`get_apex_state`, `get_predictor`,
    `get_projector`, `get_cache`, `get_db_session`, …) / the exception handlers / the `tests/api`
    fake-handle factories (`fake_prediction`/`fake_projection`/`FakeHandle`/`seed_fingerprint`).
  - **F14** — `RedisCache` (inject a failing client to disable Redis) + its breaker; `enqueue_ingestion`
    (`QueueUnavailableError`).
  - **F01** — `ingest_session(request, config, *, fastf1_fetch, …)` + `CircuitBreaker` / `call_with_breaker`
    / `SourceUnavailableError`; the `STALE_DATA` cache-fallback path (`_fastf1_down`).
  - **F03a** — `assemble_feature_vector` + the 6-NULL / `LAP_NULL_EXCESS` / `IMPUTED` gate.
  - **F06b** `predict` (the `confidence_gate`/entropy gate → `LOW_CONFIDENCE`/`AMBIGUOUS`); **F08**
    `project_unknown` (Mahalanobis OOD → `OUT_OF_DISTRIBUTION`).
  - **F12** — `DbCircuitBreaker` (force `is_open()`), `session_scope` (→ `DatabaseUnavailableError`).
  - `config.yaml` → new `failsafe:` block; **references** `thresholds.null_ceiling` (6),
    `thresholds.confidence_gate` (0.65), `api.latency_slo_ms` (500), `stylespace.ood_sd_threshold` (3.0)
    — never re-declared.
- **Produces:**
  - `app/eval/failsafe.py` — the harness: 7 test functions (one per §11.4 test) each returning a typed
    `FailsafeTestResult`; `run_failsafe_suite(...) → FailsafeReport`; the app-assembler + fault-injection
    helpers + a p95/latency measurement util; the **`artefacts/failsafe_report.json`** results-table
    artefact (content-hashed, timestamp-free — F04/F10 idiom) + a `render_table()` markdown view.
  - `app/eval/__init__.py`.
  - New typed `FailsafeConfig` (`app/contracts/config.py`) + the `failsafe:` YAML block.
  - **No new error codes; no new model artefact** (F15 measures existing behaviour; the report is
    generated evidence, not committed).
- **Consumed by:** **F20** (the full evaluation framework calls `run_failsafe_suite` for §11.4 and folds
  its table into the evaluation chapter, alongside §11.1/§11.2/§11.3 it computes itself), and the
  dissertation's RQ5 section.

## 3. Inputs & outputs (typed)
- **`FailsafeTestResult`** (Pydantic, `app/eval/failsafe.py`):
  `{ test_id: int, name: str, passed: bool, skipped: bool = False, skip_reason: str | None = None,
  measured: dict[str, float | int | str], expected: str, findings: list[str] = [] }`.
  `expected` is the master's verbatim pass criterion; `measured` the observed numbers; `findings` any
  honest caveat (§11 "documented as findings"). A test that *cannot* run offline (needs committed
  artefacts) → `skipped=True` + `skip_reason` (never a false PASS).
- **`FailsafeReport`** (Pydantic): `{ results: list[FailsafeTestResult], total: int, passed: int,
  skipped: int, content_hash: str }`. Timestamp-free; the `content_hash` covers the **reproducible
  outcome** (pass/fail/skip + criterion per test), **not** the run-specific `measured` timing (elapsed/
  p95 vary by nature) — so "same inputs → same outcomes → same hash" (reproducibility #3). `render_table()
  → str` produces the markdown RQ5 results table.
- **Input:** an `AppConfig` (+ optional `artefact_dir` override); the harness assembles its own app and
  test DB. No HTTP input — F15 *drives* the app, it is not a route.

## 4. Method / algorithm

### 4.0 Design notes (the decisions a builder must not re-litigate)
- **(a) F15 = §11.4 only; F20 orchestrates all four pillars (divergence #39).** §11 has four evaluation
  sub-sections; F15 builds *exactly* the §11.4 failsafe harness (7 tests + RQ5 results table, the Week-14
  deliverable). F20 (Wk 20) computes §11.1 (classifier), §11.2 (stability), §11.3 (degradation) itself and
  **calls `run_failsafe_suite` for §11.4** — F20 depends on F15 (INDEX), so F15 is an *upstream deliverable*,
  never a subset to fold in. The spec writes this boundary in so F20 has nothing to duplicate.
- **(b) Dynamic harness, not the static auditor.** The `failsafe-auditor` sub-agent is read-only static
  analysis ("is the mode handled/coded/logged?"). F15 assembles the real app and injects real faults,
  measuring the *behaviour*. Both feed RQ5; F15 is the executable evidence.
- **(c) In-process measurement — validates the mechanism, not production wall-clock (divergence #40).**
  The harness drives the app in-process via `httpx.ASGITransport` with **fake model handles** (no champion
  artefacts are committed — F13's constraint). So measured p95 (Test 7), cache-down degradation (Test 4),
  and breaker-fallback time (Test 2) validate the *failsafe mechanism* — completion, consistency, zero
  failures, no pathological blocking, and that the degradation stays within the master's bound *in this
  harness* — **not** deployment wall-clock with real models + network. True production p95 is **F20**
  (mirrors F13's deferred-p95 note). The harness records both the measured number and this caveat as a
  `finding`.
- **(d) Two tests need committed artefacts for their *rate*; their *mechanism* is tested offline
  (divergence #41).** Test 1 (≥90% LOW_CONFIDENCE on ambiguous input) and Test 6 (real 5-SD detection)
  are model-*behaviour* evaluations: the real rate/detection needs the F06b champion / F08 UMAP artefacts,
  which aren't committed. F15 asserts the **mechanism** offline (a fake predictor returning a sub-gate
  probability surfaces `LOW_CONFIDENCE` on `/predict`; a fake projector with `ood=True` surfaces
  `OUT_OF_DISTRIBUTION` on `/fingerprint/unknown` with coords still computed + no crash), and the
  **real-model rate** is a `@pytest.mark.artefacts` test (skipped offline, run in F20). The results table
  reports each as `PASS (mechanism)` + `SKIP-no-artefact (rate → F20)`, never a fake PASS.
- **(e) Test 2 is pipeline-level (divergence #42).** `STALE_DATA` is an F01 *pipeline* fallback
  (`ingest_session → _fastf1_down`), not surfaced on any HTTP route. So Test 2 drives `ingest_session`
  directly with a `fastf1_fetch` that raises `IngestSourceError` (×3 → breaker opens) + a pre-populated
  Parquet cache, asserting the returned `IngestResult` carries `STALE_DATA` + `Status.DEGRADED`, within
  500 ms, with no uncaught exception. It asserts the pipeline result, not a 200 body.
- **(f) `CACHE_UNAVAILABLE` vs `CACHE_DEGRADED` reconciliation (divergence #43).** §8.3 says the Redis
  breaker logs `CACHE_DEGRADED` on write-drop; §11.4 Test 4 + §8.2 say the log shows `CACHE_UNAVAILABLE`.
  F14 already reconciled this: the outage *onset* logs `CACHE_UNAVAILABLE` (first failures), and
  breaker-open *writes* drop with `CACHE_DEGRADED`. Test 4 asserts `CACHE_UNAVAILABLE` is present in the
  DQ log (it is, at onset) — no code change, just the documented reading.
- **(g) F15 refits nothing, fabricates nothing.** All model handles are fakes or (marked) real artefacts;
  no `.fit()`; the report is generated evidence written to `artefacts/`, never committed. Stated so
  `/apex-review` ticks it off.
- **(h) The harness never fails silently.** A test that errors internally (not the system-under-test
  failing, but the harness itself) → the `FailsafeTestResult` is `passed=False` + a `finding` describing
  it; `run_failsafe_suite` never raises past a single test (one test's fault doesn't abort the suite), and
  a suite with any `passed=False` returns a report with `passed < total` — surfaced, never swallowed.
- **(i) Sequencing note (not a divergence).** The master's plan lists Week-14 failsafe tests *before*
  Week-16's F13/F14 implementation — a plan artefact. APEX's INDEX build order put F13/F14 first (both
  DONE), so F15 runs against the assembled system now, as intended.

### 4.1 The 7 §11.4 tests (fault · assertion · pass criterion · offline?)

| # | Test (§11.4) | Fault injected | F15 asserts | Master pass criterion (verbatim) | Offline? |
|---|---|---|---|---|---|
| 1 | **Confidence gate effectiveness** | ambiguous telemetry (avg of 2 drivers) | `/predict` returns 200 `LOW_CONFIDENCE` for a sub-gate prediction (mechanism); real rate marked | "LOW_CONFIDENCE for 90% of ambiguous inputs rather than a wrong confident prediction" | mechanism ✅ · rate 🔶 F20 |
| 2 | **Circuit breaker activation** | `fastf1_fetch` raises ×3 → F01 breaker opens; Parquet cache present | `ingest_session` → `STALE_DATA` + `Status.DEGRADED`, ≤ `breaker_fallback_ms`, no exception | "No uncaught exception. System returns cached data with STALE_DATA flag within 500ms." | ✅ (pipeline) |
| 3 | **NULL feature tolerance** | vectors with 1/3/6/7 NULLs | 6-NULL → accepted + `IMPUTED`; 7-NULL → `LAP_NULL_EXCESS` reject; `/predict` 7-NULL → 400 `INVALID_INPUT` | "6 NULLs accepted with imputation. 7+ NULLs rejected with LAP_NULL_EXCESS. No silent imputation." | ✅ |
| 4 | **Cache unavailability resilience** | inject a failing `RedisCache` client | 100 `/predict` all 200; avg latency increase ≤ `cache_degradation_ms`; `CACHE_UNAVAILABLE` in DQ log | "Zero failed predictions. Response time increases by < 200ms average. Log shows CACHE_UNAVAILABLE throughout." | ✅ |
| 5 | **Database unavailability** | pre-open `DbCircuitBreaker` | a DB-backed read → **503 `DB_UNAVAILABLE`**; no partial write; no unhandled exception | "API returns 503 with DB_UNAVAILABLE reason. No partial writes. No unhandled exceptions." | ✅ |
| 6 | **UMAP out-of-distribution** | fake projector `ood=True` (5-SD synthetic) | `/fingerprint/unknown` → `OUT_OF_DISTRIBUTION` flag, coords present, no crash (mechanism); real detection marked | "Warning returned. Projection computed but flagged. UI shows dashed marker. No crash." | mechanism ✅ · detection 🔶 F20 |
| 7 | **Concurrent request handling** | 50 concurrent `/predict` | all 50 → 200; identical body for identical input; p95 ≤ `api.latency_slo_ms` | "All 50 requests complete. Results consistent (same input = same output). p95 latency < 500ms." | ✅ (in-process, §4.0-c) |

### 4.2 Harness architecture (`app/eval/failsafe.py`)
- **`_assemble(config, *, cache=None, artefacts=False) → _Harness`** — builds `create_app(config)` + an
  in-memory aiosqlite DB (migrated + alembic-head-stamped, the `tests/api` recipe), injects fake handles
  (or real artefacts when `artefacts=True`), overrides `get_apex_state`, and returns an
  `httpx.AsyncClient(ASGITransport(app, raise_app_exceptions=False))` + the `AppState` + DB factory. This
  is F15's **own** fixture — the F13 `api` fixture leaves `cache`/`celery_app`/`job_store` unset, which the
  harness sets per-fault. `httpx` moves dev → main (the harness is shipped code that drives the ASGI app).
- **7 test functions** `test_1_confidence_gate(...) … test_7_concurrent(...)` — each async, self-contained
  (assemble → inject fault → drive → measure → assert), returning a `FailsafeTestResult`. They **never
  raise** past themselves.
- **`run_failsafe_suite(config, *, artefact_dir=None) → FailsafeReport`** — runs all 7 (wrapping each in
  its own try, so one test's fault never aborts the suite, §4.0-h), writes `artefacts/failsafe_report.json`,
  returns the report. F20 extends Tests 1/6 with the real-artefact *rate* assertions using its committed
  champion/umap artefacts (the mechanism the harness asserts offline is the same code path).
- **Measurement util** `measure_post(client, path, body, n) → (list[float], list[int])` + `p95(samples)`
  (wall-clock via `time.perf_counter`; p95 = `statistics.quantiles(samples, n=100, method="inclusive")[94]`).

### 4.3 The RQ5 results-table artefact (§13 Week-14 deliverable)
`FailsafeReport` → `artefacts/failsafe_report.json`, written with the F04/F10 reproducibility idiom
(`json.dumps(..., sort_keys=True, indent=2, allow_nan=False)`, **embedded `content_hash` = sha256 of the
canonical OUTCOME body, no timestamp**) so two runs on the same inputs share a `content_hash` (the file's
`measured` timing is run-specific — the hash certifies the outcomes, not the milliseconds). `render_table()` emits
the markdown table (Test · Result · Measured · Expected · Findings) the dissertation embeds. The artefact
is generated, **not committed** (like every other `artefacts/*`); F20 regenerates it in the full run.

## 5. config.yaml keys this feature adds
```yaml
# --- F15 RQ5 failsafe evaluation harness (§11.4 — the 7 tests; §13 Week-14 deliverable) -----------
# The knobs the 7 §11.4 tests run with. Shared thresholds are REFERENCED, never re-declared:
# the 6-NULL ceiling is thresholds.null_ceiling, the 0.65 gate is thresholds.confidence_gate, the
# p95 target is api.latency_slo_ms (500), the OOD DETECTION threshold is stylespace.ood_sd_threshold
# (3.0 — the harness INJECTS at ood_injection_sd=5.0, which must exceed it). No new error code.
failsafe:
  n_predictions: 100          # §11.4 Test 4 — consecutive predictions with Redis disabled
  n_concurrent: 50            # §11.4 Test 7 — concurrent /predict requests
  null_boundary: [1, 3, 6, 7] # §11.4 Test 3 — NULL-count boundary points (must span the 6/7 ceiling)
  n_ambiguous_samples: 50     # §11.4 Test 1 — ambiguous inputs to measure the LOW_CONFIDENCE rate
  ambiguous_low_conf_rate: 0.90  # §11.4 Test 1 — min LOW_CONFIDENCE fraction on ambiguous input
  cache_degradation_ms: 200   # §11.4 Test 4 — max avg response-time increase with Redis down
  breaker_fallback_ms: 500    # §11.4 Test 2 — max time to the STALE_DATA cached fallback
  ood_injection_sd: 5.0       # §11.4 Test 6 — SD-outside-training injection magnitude (> detection SD)
  report_filename: failsafe_report.json  # RQ5 results-table artefact under paths.artefacts_dir
```
New typed `FailsafeConfig` in `app/contracts/config.py` (standing rule — feature config lives in
`config.py`, `extra="forbid"`), wired as `AppConfig.failsafe: FailsafeConfig | None = None`. Validators:
`n_predictions`/`n_concurrent`/`n_ambiguous_samples` ≥ 1; `null_boundary` non-empty and **contains both
`null_ceiling` and `null_ceiling+1`** (the accept/reject boundary — cross-block check on `AppConfig`);
`0 < ambiguous_low_conf_rate ≤ 1`; the `_ms` bounds > 0; **`ood_injection_sd > stylespace.ood_sd_threshold`**
(cross-block — you must inject *beyond* the detection threshold); non-empty `report_filename`.

## 6. Error codes & failure modes (the failsafe contract F15 ASSERTS)
F15 adds **no** error codes — it *asserts* the pre-seeded ones the system emits under fault. This table
is the mapping the harness checks (the system-under-test's failure modes; F15's own faults are §4.0-h).

| §11.4 test | Fault | Code/behaviour asserted | Where | Logged? |
|---|---|---|---|---|
| 1 gate | ambiguous input | `LOW_CONFIDENCE` (200 body) | F06b gate via `/predict` | (per F06b) |
| 2 breaker | FastF1 ×3 down | `STALE_DATA` + `Status.DEGRADED`, ≤500ms, no exception | F01 `ingest_session` | yes → `data_quality_log` |
| 3 NULL | 7 NULLs | `LAP_NULL_EXCESS` reject / `IMPUTED` at 6 / `INVALID_INPUT` 400 | F03a + `/predict` | yes |
| 4 cache | Redis down | 100×200, `CACHE_UNAVAILABLE` throughout, ≤200ms avg | F14 cache + `/predict` | yes (onset) |
| 5 DB | breaker open | **503 `DB_UNAVAILABLE`**, no partial write, no unhandled exc | F12 `session_scope` + DB read route | yes |
| 6 OOD | 5-SD input | `OUT_OF_DISTRIBUTION` flag, coords present, no crash | F08 `project_unknown` via `/fingerprint/unknown` | (per F08) |
| 7 concurrent | 50 parallel | all 200, identical bodies, p95 ≤ `latency_slo_ms`, no race | `/predict` over ASGITransport | n/a |
| harness-internal fault | (F15 bug) | `FailsafeTestResult.passed=False` + finding; suite continues | `app/eval/failsafe.py` | (surfaced in report) |

## 7. Acceptance criteria (each item must be mechanically testable by /spec-check)
- [ ] **Test 1** — a sub-gate prediction on ambiguous input surfaces **200 `LOW_CONFIDENCE`** via `/predict`
      (mechanism); the ≥90%-rate assertion is present and `@pytest.mark.artefacts`-gated (skipped offline).
- [ ] **Test 2** — `ingest_session` with a down `fastf1_fetch` (×3) + a Parquet cache returns
      **`STALE_DATA` + `Status.DEGRADED`** within `breaker_fallback_ms`, **no uncaught exception**.
- [ ] **Test 3** — a 6-NULL vector is **accepted + `IMPUTED`**, a 7-NULL vector **rejected with
      `LAP_NULL_EXCESS`** (no silent imputation) via the real `assemble_feature_vector` gate (the
      `/predict` → 400 `INVALID_INPUT` HTTP mapping is F13's, tested there).
- [ ] **Test 4** — with Redis disabled, **100 `/predict` calls all return 200** (zero failures), the avg
      latency increase is **≤ `cache_degradation_ms`** (in-process, §4.0-c), and **`CACHE_UNAVAILABLE`** is
      present in the DQ log.
- [ ] **Test 5** — a DB-backed read with the breaker open returns **503 `DB_UNAVAILABLE`**, performs **no
      partial write**, and raises **no unhandled exception**.
- [ ] **Test 6** — a 5-SD OOD projection surfaces **`OUT_OF_DISTRIBUTION`** with coords still present and
      **no crash** (mechanism); the real-detection assertion is present and artefact-gated.
- [ ] **Test 7** — **50 concurrent `/predict`** all complete 200 with **identical bodies for identical
      input**, and **p95 ≤ `api.latency_slo_ms`** (in-process).
- [ ] `run_failsafe_suite` returns a `FailsafeReport` with all 7 results, writes
      **`artefacts/failsafe_report.json`** (content-hashed, timestamp-free; the outcome `content_hash` is
      stable across two runs — `measured` timing is run-specific), and `render_table()` emits the RQ5
      markdown results table **with the Findings column**.
- [ ] A harness-internal fault yields `passed=False` + a finding and **does not abort the suite** (§4.0-h);
      the harness **never writes to the real `cache/` or `data_quality_log`** (both redirected to a temp dir).
- [ ] **No `.fit()` / no refit / no fabrication** in `app/eval/`; the report is generated, never committed.
- [ ] All knobs read from `config.yaml` `failsafe:` (or the referenced shared thresholds); **no magic
      numbers**; the null boundary spans `null_ceiling`/`+1`; `ood_injection_sd > ood_sd_threshold`.
- [ ] `mypy app` clean, `ruff` clean; `httpx` pinned in `pyproject.toml` **main** deps.

## 8. Test plan
- **`tests/failsafe/test_rq5.py`** — one test per §11.4 test invoking its harness function + asserting
  the criterion AND `FailsafeTestResult.passed`. The offline 5 (2,3,4,5,7) run fully; Tests 1/6 run their
  mechanism offline + carry a companion `@pytest.mark.artefacts` real-rate test (skipped when
  `artefacts/classifier.pkl` / `umap_model.pkl` are absent, mirroring F12's `@pytest.mark.postgres`).
- **Harness fixtures** reuse `tests/api/conftest.py` (`fake_prediction`/`fake_projection`/`FakeHandle`/
  `seed_fingerprint`) + the aiosqlite recipe; add the fault-injection doubles (`_DownClient` for Redis,
  breaker-pre-open for DB, a raising `fastf1_fetch` for F01).
- **Report tests** — `run_failsafe_suite` produces a `FailsafeReport` with `total==7`; the artefact is
  written and the **outcome content-hash is stable on a second run**; `render_table()` contains all 7 rows;
  a run leaves the real `cache/` + `data_quality_log` untouched; the suite survives a test that raises.
- **Property/edge** — a deliberately-broken harness step surfaces `passed=False` (not a raise);
  `null_boundary` missing `7` fails config validation; `ood_injection_sd ≤ ood_sd_threshold` fails
  validation; the suite runs with **no committed artefacts** (Tests 1/6 rate → skipped, not failed).
- **Consolidation, not duplication** — the harness *reuses* the existing fragment tests' mechanisms
  (F13 `test_50_concurrent_predicts_are_consistent`, F14 `test_100_predictions_succeed_with_redis_down`,
  F13 `test_db_backed_read_db_down_is_503`); those unit tests stay, F15 promotes them into the measured,
  reported §11.4 suite.

## 9. Definition of done
All acceptance criteria MET with passing tests (offline — no Redis, no committed artefacts; the two
rate-tests skipped via `@pytest.mark.artefacts`), `/apex-review F15` clean on the contract checklist,
`mypy app` + `ruff` clean, `httpx` promoted to main deps + pinned, INDEX row → DONE. `failsafe-auditor`
run on `app/eval/` (it should find nothing to fix — F15 is the harness the auditor complements; the check
is that the harness itself fails loud, §4.0-h). Record in the dissertation limitations: (1) the harness
measures **in-process** latency with **fake handles** — p95/degradation validate the *mechanism*, and
production wall-clock with real models is **F20** (divergence #40); (2) Tests 1 (90% gate rate) + 6 (real
5-SD detection) are **model-behaviour** evaluations whose rate is **artefact-gated → F20** (only the
mechanism is asserted offline, divergence #41); (3) Test 2 asserts the **pipeline** `STALE_DATA` result,
not an HTTP body (divergence #42); (4) the `CACHE_UNAVAILABLE`/`CACHE_DEGRADED` wording is reconciled per
F14 (divergence #43); (5) the RQ5 results table is emitted as `failsafe_report.json` — the master names
the deliverable but no file/format (divergence #44).

## 10. Risks & mitigations (from master risk register where relevant)
- **In-process p95 is not deployment p95** → §4.0-c: F15 asserts completion + consistency + no
  pathological blocking + the master's *relative* bound in-harness, records the caveat as a `finding`, and
  defers real wall-clock timing to F20. The top honesty risk — mitigated by explicit `findings`, never a
  silent "PASS" implying production numbers.
- **No committed artefacts → Tests 1/6 rate cannot run** → §4.0-d: the *mechanism* is tested offline with
  fakes; the rate is `@pytest.mark.artefacts`-gated (skipped, reported as SKIP-not-FAIL). No false PASS.
- **Harness flakiness (concurrency/timing)** → measure with a fixed `n`, assert the *bound* not an exact
  number; the concurrency test asserts consistency + a generous in-process p95; no wall-clock race.
- **Silent PASS / masking a real failure** → the report distinguishes PASS / FAIL / SKIP with the measured
  number + expected criterion + findings; a harness-internal fault is `passed=False`, never swallowed
  (§4.0-h). This is the RQ5 evidence — it must be honest above all (master §11 "documented as findings").
- **Scope creep into F20** → §4.0-a draws the boundary: F15 is §11.4 only; F20 consumes it. The spec
  states F20 calls `run_failsafe_suite` rather than reimplementing.
- **Normalisation leakage** → N/A: F15 refits nothing, injects/asserts only. Stated so `/apex-review`
  ticks it off.
- **`STALE_DATA` not on an HTTP route** → §4.0-e: Test 2 drives the F01 pipeline directly; documented,
  not stubbed.

**Build outcome (2026-07-22).** BUILT + DONE. `app/eval/failsafe.py` (7 `rq5_*` tests + `run_failsafe_suite`
+ `FailsafeReport`/`render_table` + report writer) + `app/eval/harness.py` (self-contained app-assembly +
fakes + fault-injection + `measure_post`/`p95`), the `failsafe:` config block + typed `FailsafeConfig` (+ 2
cross-block validators), and the `artefacts` pytest marker. **No new error codes** (asserts pre-seeded
ones); `httpx` promoted **dev → main** (the harness drives the ASGI app). All 7 tests PASS offline (fakes +
aiosqlite; Tests 1/6 mechanism + `@pytest.mark.artefacts` rate skipped). **18 tests** (`tests/failsafe/`);
full suite green, `ruff` + `mypy app` clean. **`failsafe-auditor` run on `app/eval/` — findings fixed with
regression tests:** **(HIGH)** Test 3 wrote to the real `logs/data_quality_log.jsonl` and Test 2 to the real
`cache/` — `_harness_config` now redirects **both** the DQ log AND `cache_dir` to a temp dir (a new test
asserts a run leaves both untouched; the already-polluted gitignored files were cleaned); **(HIGH)** a test
raising in its pre-`try` setup could abort the suite — every `rq5_*` moved its setup inside `try` AND
`run_failsafe_suite` wraps each call in its own try (a new test monkeypatches a raising test and asserts the
suite still returns 7 results); **(MED)** `render_table` now includes the mandated **Findings** column;
**(MED)** Test 1's finding relabelled (route-passthrough, not the real gate — that's unit-tested in
`tests/ml`) and Test 5's "no partial write" comment corrected (breaker fast-fails before the write, not a
rollback — rollback is F12-tested); spec's non-existent `include_artefact_tests` param + "byte-identical
file" wording corrected (the `content_hash` certifies the OUTCOME; `measured` timing is per-run).
**Divergences #39–#44.** `/apex-review` + `/spec-check F15` clean. Consumed by F20 (calls
`run_failsafe_suite` for the §11.4 pillar + extends Tests 1/6 with real-artefact rates).
