# Feature F17 — Screen 3 (Ingestion & Prediction) + Screen 4 (Comparison)

> One spec, one session. This document is the single source of truth a session loads
> to build this feature. It must be self-contained: a session should not need the
> 42-page master spec open, only this file plus CLAUDE.md.

**Status:** SPEC WRITTEN
**Master spec sections:** §10.1 (design system, reused), §10.4 (Screen 3 — Ingestion & Prediction), §10.5 (Screen 4 — Comparison View), Week-18 milestone row (§ risk table p.35)
**Layer / Week:** Frontend / Wk 18

---

## 1. Purpose (one paragraph)
F17 adds the two interactive screens the APEX frontend has been missing: **Screen 3 — Ingestion & Prediction** (the live `/predict` flow plus FastF1-session ingestion, manual CSV upload, and unknown-driver projection) and **Screen 4 — Comparison** (two drivers side by side with a signed delta radar, most-distinctive features, a compatibility projection, and a setup delta). It is the first UI that exercises `POST /predict`, `POST /fingerprint/unknown`, the async `/ingest/*` job, and `GET /compare`. Like F16 it is a **pure frontend feature** — no backend code, no new `config.yaml` block, no new backend error code — and it is where the constitution's "no silent failure" rule is proven under the messiest endpoint mismatch in the system: roughly half of the master's Screen-3/4 UI elements have **no backing endpoint or only a partial one**, and every one of them is rendered as an explicit unavailable/proxy state, never fabricated.

---

## 2. Upstream / downstream contracts
- **Consumes (backend, all F13/F14, DONE):**
  - `POST /predict` (F06b + F07) → `PredictResponse`
  - `POST /fingerprint/unknown` (F08) → `UnknownFingerprintResponse`
  - `GET /compare/{id1}/{id2}` (F08) → `CompareResponse`
  - `GET /drivers/{driver_id}/setup/{circuit_id}` (F11) → `SetupResponse`
  - `GET /drivers` (F13) → `DriverSummary[]`
  - `POST /ingest/session` (F14) → `IngestSessionResponse` (202); `GET /ingest/status/{session_id}` (F14) → `IngestStatusResponse`
  - `POST /ingest/upload` (**503 stub → F05**) → `UploadResponse` (schema only; live route raises 503)
- **Consumes (frontend, F16a/F16b, DONE — reused, not rebuilt):** the design system (`components/ds/*`), the API client (`lib/api/*` — `ApiResult`, `fetchApi`, generated types, zod validators, `UI_*` codes), the chart primitives (`FingerprintRadar`, `GaugeNeedle`, `ShapBeeswarm`), `lib/design/*` (animation timings, `confidence_state→colour`, circuits, teams), and the vitest+RTL test patterns.
- **Produces:** two new App-Router routes (`app/ingest/page.tsx`, `app/compare/page.tsx`); new components under `components/ingestion/*` and `components/comparison/*`; two new chart primitives (`components/charts/ShapForcePlot.tsx`, `components/charts/DeltaRadar.tsx`); new API-client helpers + zod validators + type aliases; a refreshed `frontend/openapi.snapshot.json` and an extended pytest drift guard. **No Pydantic result object and no backend artefact.**
- **Consumed by:** F18 (Screen 5 Style Space — the unknown-driver tab embeds it), F19 (Screen 6 — reuses the pipeline/progress + async-poll patterns), F20/F21 (evaluation + write-up screenshots).

---

## 3. Inputs & outputs (typed)

F17 is a frontend feature: its "result objects" are (a) the backend Pydantic schemas it **consumes read-only** (defined in F13, quoted verbatim below) and (b) the TypeScript types + zod validators it **adds** to the client. No new Pydantic model is created.

### 3.1 Backend response schemas consumed (verbatim field lists, `app/api/schemas.py`)

```
PredictRequest            { feature_vector: dict[str, float|None], circuit: str }
PredictResponse           { predicted_driver: str|None, confidence: float,
                            confidence_state: ConfidenceState|None,
                            top_3: [{driver: str, probability: float}],
                            shap_values: dict[str,float]|None,   # feature-id → signed SHAP toward predicted
                            shap_available: bool, model_version: str }

UnknownFingerprintRequest  { laps: [dict[str,float|None]], circuit: str, circuit_type: str|None }
UnknownFingerprintResponse { fingerprint_id: str, fingerprint_vector: [float]|None,   # 5D
                             radar_values: dict[str,float],       # ⚠ route returns {} — div #59
                             style_description: str,               # ⚠ route returns "" — div #59
                             nearest_neighbours: [{driver: str, cosine_similarity: float, distance: float}],
                             risk_flags: [str],
                             data_quality: {n_laps_used: int, n_laps_rejected: int, imputed_features: [str]},
                             confidence_state: ConfidenceState|None,
                             warnings: [ErrorCode] }                # SPARSE_PROJECTION / OUT_OF_DISTRIBUTION

CompareResponse           { driver_1_radar: dict[str,float], driver_2_radar: dict[str,float],
                            delta_shap: dict[str,float],           # per-DOMAIN d1−d2 (not per-feature — div #61)
                            most_distinctive_features: [str],      # domains by |delta|
                            compatibility_delta: float }           # Σ|delta|, circuit-agnostic — div #62

SetupResponse             { driver_id, circuit_id, parameters: [SetupParameterSchema], ... }  # 10 params, F11
IngestSessionRequest      { year: int(1950-2100), event: str(1-128), session: str(default "R", 1-8) }
IngestSessionResponse     { session_id: str, laps_accepted: int, laps_rejected: int,
                            rejection_summary: dict[str,int], quality_report: dict }   # 202; counts are 0 placeholders on enqueue
IngestStatusResponse      { status: "PENDING"|"PROCESSING"|"COMPLETE"|"FAILED",
                            progress: float, error_detail: str|None, result: dict|None }
UploadResponse            { upload_id, laps_accepted, laps_rejected, cleaning_report, warnings }  # 503 stub → F05
```

### 3.2 Frontend types/validators F17 adds (`lib/api/`)
- **`types.ts` aliases** (re-exported from the regenerated `schema.d.ts`): `PredictRequest`, `PredictResponse`, `TopK`, `UnknownFingerprintRequest`, `UnknownFingerprintResponse`, `NeighbourSchema`, `DataQualitySchema`, `IngestSessionRequest`, `IngestSessionResponse`, `IngestStatusResponse`, `UploadResponse`. (`CompareResponse`, `SetupResponse`, `SetupParameterSchema`, `ConfidenceState`, `ErrorCode` already exist.)
- **`endpoints.ts` helpers:** `postPredict(body)`, `postUnknownFingerprint(body)`, `postIngestSession(body)`, `getIngestStatus(sessionId)`, `postUpload(file)`. (`getCompare`, `getSetup`, `getDrivers` already exist. `fetchApi` already accepts an `init` so `method:'POST'` needs no new machinery.)
- **`validators.ts` zod validators** (all `.passthrough()`, guarding only load-bearing fields): `zPredict`, `zUnknownFingerprint`, `zIngestSession`, `zIngestStatus`, `zUpload`.
- **Output contract everywhere:** each helper returns `ApiResult<T>` — `ApiOk<T> {ok:true, data, status, partial, flags}` on 2xx/206, `ApiErr {ok:false, code, detail, status}` otherwise. `fetchApi` **never throws** into a component and never returns `any`; a schema mismatch → `UI_SCHEMA_MISMATCH` (no malformed render).

---

## 4. Method / algorithm

### 4.0 Global rules (inherited from F16, non-negotiable)
- **Four data-fetch states, per panel (no silent failure):** *loading* → `Skeleton` (telemetry-wave, not a spinner); *empty* → `EmptyState`; *error* → `ErrorCard`/`PageError` (code in monospace, Retry iff `isRetryable`); *success* → data. Driven by the `ApiResult` union — the discriminant is `ok`, and empty = `ok:true` + empty payload.
- **Confidence colour is backend-resolved (div #46, continued):** `ConfidenceBar`/`ConfidenceBadge` take `confidence_state` and map via `lib/design/confidence.ts`. The master's literal "green >0.80 / amber 0.65–0.80 / red <0.65" is **not** re-implemented in the client — the gates live server-side (`config.yaml classifier.confidence_gate 0.65`, `confidence_high 0.85`); the UI trusts the resolved `confidence_state`.
- **Reproducibility:** all D3 layout that uses jitter reuses the seeded `LAYOUT_SEED`; timings/staggers/easing come from `lib/design/animation.ts`, never hard-coded per component.
- **No new backend config or code (div #53):** new constants (pipeline-stage timing, force-plot/delta-radar/gauge durations) are **added to `lib/design/animation.ts`**, not `config.yaml`. `styleSpace: 1200` is already declared there for the unknown-driver marker drift.

### 4.1 Screen 3 — Ingestion & Prediction (`app/ingest/page.tsx`)
Master §10.4 verbatim scope: *"Three tabs: FastF1 Session, Manual Upload, Unknown Driver. This screen shows the system's own thinking — the pipeline stages animate as they complete."* Route is a client screen (`'use client'`) — interactive tabs + POSTs; a small Zustand store (`lib/state/ingestion.ts`) holds active tab + FastF1 selection + poll state. Tab shell reuses `ds/Tabs`.

**(a) Shared — `PipelineProgress` component.** Master: a horizontal bar of named stages (Fetching → Validating → Extracting Features → Running Model → Computing SHAP), each lighting green on completion, red + error code inline on failure, monospace timing per stage.
- **Reality / div #55:** the only pipeline the API exposes coarsely is the async ingestion job — `/ingest/status` returns a `status` enum + a single `progress` float + a terminal `quality_report`/`error_detail`, **not** five named per-stage timings. `PipelineProgress` renders the stage rail as a client-side visual of the known pipeline, but *state transitions are driven only by real job signal* (progress + status; `FAILED` → the failing stage turns red and shows `error_detail`). It does **not** fabricate per-stage green/red timings the API never reports.

**(b) Prediction panel — `PredictionPanel` (the real `/predict` flow).**
- **Reality / div #54:** the master's FastF1 flow "pick a session → predicted driver + confidence + SHAP force" is **not end-to-end wireable** — ingestion is async, writes laps to the DB, and returns *no prediction*; and no endpoint yields "the feature vector for session Y, driver Z". So F17 **separates** the two concerns the master conflates: the FastF1 tab drives the real ingestion job (§4.1c), and prediction is a **direct `/predict` panel** taking a `{feature_vector, circuit}` body (the one place the whole app exercises the classifier + SHAP). On success it renders: `predicted_driver`, a reused `ConfidenceBar` (`confidence` + `confidence_state`), the `top_3` list, and a **SHAP force plot** (§4.1f).

**(c) FastF1 Session tab — `FastF1Tab`.**
- Dependent selectors (master: Season → Round → Driver → Session). **Div #57:** no session-catalog endpoint exists (no `/seasons`, `/rounds`, `/sessions`; cf. F16 finding G, no `/circuits`). F17 populates **Season** from the static known set (2022/2023/2024), **Driver** from `getDrivers()`, and renders **Round** and **Session type** as bounded free inputs (round int; session ∈ {FP1,FP2,FP3,Q,R}). The **"laps available" count** has no endpoint → rendered as an explicit `EmptyState`/"—" ("clean-lap count not exposed by the API"), never a guessed number.
- Submit → `postIngestSession({year, event, session})` → **202** with a `session_id` → poll `getIngestStatus(session_id)` on an interval (client-side `useEffect` loop, interval from `animation.ts`) until `COMPLETE`/`FAILED`. `PipelineProgress` reflects `progress`/`status`; on `COMPLETE` the `result` counts + `quality_report` render as a table; on `FAILED` the rail shows the failing stage red + `error_detail`.
- Enqueue when broker down → **503 `QUEUE_UNAVAILABLE`** → `ErrorCard` (retryable), no fake progress.

**(d) Manual Upload tab — `ManualUploadTab`.** Master: drag-and-drop CSV → cleaning report (accepted/rejected counts, per-lap rejection reasons + codes), `UPLOAD_002` amber warnings with dismiss.
- **Reality / div #58:** `POST /ingest/upload` is a **503 stub deferred to F05**; `UPLOAD_001`–`UPLOAD_004` exist in `errors.py` but are wired to no live route. Following the F16 honesty pattern (#48/#49), F17 **builds the drag-drop `DropZone` + cleaning-report table shell**, POSTs the *real* endpoint, and surfaces the honest **503 (`HEALTH_FAIL`) `ErrorCard`: "Manual upload ships in F05."** It never renders a fabricated cleaning report. When F05 lands, the wiring is a one-line data swap (the report table already accepts the `UploadResponse` shape).

**(e) Unknown Driver tab — `UnknownDriverTab`.** Master: upload unlabelled laps → generate fingerprint, radar animates axis-by-axis (SSE), style-space embed (Screen 5), NN top-3, risk flags.
- Submit laps → `postUnknownFingerprint({laps, circuit, circuit_type?})`. **Live/real:** `nearest_neighbours` (top-3 as filled progress bars, `NearestNeighbours`), `risk_flags` (red `RiskFlags` cards, label "outside known F1 driver distribution"), `data_quality` summary, and — **div #60** — the `confidence_state` + `warnings` (`SPARSE_PROJECTION` <30 laps / `OUT_OF_DISTRIBUTION`). This tab is the **honest home of live LOW_CONFIDENCE/OOD/sparse treatment** (F16 finding F: known-driver feeds carry no `confidence_state`; projected drivers do).
- **Div #59:** `radar_values`/`style_description` come back **empty from the route**, there is **no SSE stream** (single POST), and the style-space embed is **Screen 5 (F18, unbuilt)**. So the axis-by-axis radar renders an explicit `EmptyState` ("unknown-driver radar not populated by the API"; the 5D `fingerprint_vector` is available but is not the 6-domain radar), and the embed renders a deferred "Style Space (Screen 5) arrives in F18" state. NN/risk/quality/confidence are shown live.
- Boundaries: `<min_projection_laps` (5) → **400 `INVALID_INPUT`** `ErrorCard`; `>max_request_laps` (5000) → 400; `<30` laps → **200** with a `SPARSE_PROJECTION` warning banner (a state, not an error).

**(f) `ShapForcePlot` (new chart primitive, `components/charts/ShapForcePlot.tsx`).**
- **Div #56:** `/predict` surfaces only `shap_values` (signed per-feature-id dict) + `shap_available`; the full force rows (`base_value`, human-readable name, `feature_value`, toward/away `direction`) are computed internally (`app/ml/explain.py ExplanationResult.force`) but **dropped at the API boundary**. So `ShapForcePlot` renders one signed bar per feature id — red toward / blue away by sign, bars animating from zero, feature value read from the request's own `feature_vector`, hover = feature id + value + contribution — **without a base-value baseline line**. `shap_available=false` → `EmptyState` "Explanation unavailable (`SHAP_UNAVAILABLE`)", prediction still shown. Surfacing `force`/`base_value` is a **named backend seam** (not done in F17, to keep it frontend-only).

### 4.2 Screen 4 — Comparison View (`app/compare/page.tsx`)
Master §10.5: two drivers side by side. Client screen; driver-1/driver-2/circuit held in `lib/state/comparison.ts` (or URL query). Reuses `FingerprintRadar`, `GaugeNeedle`, `ds/*`.
- **Two radars + swap:** `getCompare(id1, id2)` → `driver_1_radar`, `driver_2_radar` → two `FingerprintRadar`s animating on load. **Swap button** swaps the two ids client-side; radars cross-fade; the delta radar flips sign.
- **`DeltaRadar` (new chart primitive):** renders the signed per-domain `delta_shap` — positive (driver-1 higher) in red, negative (driver-2 higher) in blue. **Div #61:** `delta_shap` is a per-**domain** radar-value delta, not the per-feature SHAP delta the master names; F17 labels it per-domain honestly.
- **Most-distinctive features:** `most_distinctive_features` (domains by |delta|) as a horizontal bar chart, bars from zero (top-5 slice; only 6 domains exist — div #61). Hover shows the plain-English meaning of the difference (reuse the F16 NL domain strings).
- **Compatibility projection (circuit selector):** master: "given Car X's setup for this circuit, which driver gets better compatibility?" **Div #62:** there is **no circuit-aware compatibility endpoint** — `/compare` is circuit-agnostic (`compatibility_delta` = Σ|domain delta|). F17 reuses the F16 **compatibility proxy (#47)**: per-driver compatibility for the selected circuit = **mean per-parameter `confidence`** from `getSetup(driver, circuit)`; a `GaugeNeedle` shows the signed delta of the two proxies, **labelled a proxy**. A driver whose setup 404s → honest **"—"** (absence ≠ failure; a 503 → distinct code marker).
- **Setup delta:** `getSetup(d1, circuit)` + `getSetup(d2, circuit)` → a `SetupDeltaTable` diffing the 10 parameters, sorted by |Δ|, bars animating; circuit selector drives re-fetch. **Fully real** (reuses `SetupParameterSchema`).
- **Errors:** unknown driver on `/compare` or `/setup` → **404 `NOT_FOUND`** `ErrorCard`.

### 4.3 Contract plumbing
- **`openapi.snapshot.json` refresh:** regenerate `frontend/openapi.snapshot.json` from `create_app().openapi()` so it carries `PredictResponse`, `UnknownFingerprintResponse`, `IngestSessionResponse`, `IngestStatusResponse`, `UploadResponse` (capturing the existing backend contract — not backend code); then `npm run gen:api` to refresh `lib/api/schema.d.ts`.
- **Drift guard extension:** extend `tests/api/test_openapi_frontend_contract.py` to assert `PredictResponse`, `UnknownFingerprintResponse`, `CompareResponse`, `IngestStatusResponse` still carry the fields F17 reads — a backend rename fails CI here.

---

## 5. config.yaml keys this feature adds
```yaml
# NONE. (Divergence #53 — continues F16 divergence #45.)
# F17 is a frontend feature: it adds no config.yaml block, no backend code, and no new
# backend error code. Its configuration surface lives entirely in the frontend/ tree:
#   - new animation timings  -> lib/design/animation.ts (DURATION/STAGGER additions:
#       ingestPollIntervalMs, pipelineStage, forceBar, deltaRadar, compatGauge)
#   - unknown-driver marker drift reuses the existing DURATION.styleSpace = 1200
#   - confidence_state -> colour reuses lib/design/confidence.ts (no re-derived gate)
#   - API base URL reuses NEXT_PUBLIC_API_BASE_URL (.env.example, fallback :8001)
# The only deferred backend touch is adding the Vercel origin to api.cors_origins at
# deploy time (F19), not part of this feature.
```

---

## 6. Error codes & failure modes (the failsafe contract)

No new code is added. F17 **reuses** the backend `ErrorCode` enum (surfaced via `ErrorResponse.error_code`) and the three frontend-local `UI_*` codes. "Logged?" = `logClientError` via `shouldLog()` (5xx + DB/HEALTH/`UI_*`; an honest 4xx/404 and in-body uncertainty states are **not** logged).

| Failure mode | Detection | Error code | System response | Logged? | User-facing message |
|--------------|-----------|-----------|-----------------|---------|---------------------|
| `/predict` >6 NULL, too many keys, or <5 laps | 400 from API | `INVALID_INPUT` | `ErrorCard`; form stays; no crash | no | server `detail` (e.g. "too many NULL features") |
| `/predict` genuine model fault | 503 | `INFERENCE_ERROR` | `ErrorCard` + Retry | yes | "Prediction failed — retry" |
| classifier/handle not loaded | 503 | `HEALTH_FAIL` | `ErrorCard` + Retry | yes | "Model not ready" |
| SHAP unavailable for a prediction | `shap_available=false` (in body) | `SHAP_UNAVAILABLE` | force plot → `EmptyState`; **prediction preserved** | no | "Explanation unavailable" |
| Low-confidence / ambiguous prediction | `confidence_state` (in body, 200) | `LOW_CONFIDENCE` / `AMBIGUOUS` | `ConfidenceBadge` + dashed treatment; answer shown as uncertain, never forced | no | "⚠ LOW CONFIDENCE" / "⚠ AMBIGUOUS" |
| Unknown driver, <30 laps | `warnings` (in body, 200) | `SPARSE_PROJECTION` | warning banner; result still shown | no | "Sparse projection (<30 laps)" |
| Unknown driver outside distribution | `warnings` / `risk_flags` (200) | `OUT_OF_DISTRIBUTION` | warning banner + red risk cards | no | "Outside known F1 driver distribution" |
| Unknown driver <5 / >5000 laps | 400 | `INVALID_INPUT` | `ErrorCard` | no | server `detail` |
| Ingestion enqueue, broker down | 503 | `QUEUE_UNAVAILABLE` | `ErrorCard` + Retry; no fake progress | yes | "Queue unavailable — retry" |
| Ingestion job failed | `status=FAILED` + `error_detail` | (job's `error_detail` code) | failing pipeline stage turns red; code shown inline | job-reported | the `error_detail` string |
| Manual upload (not yet implemented) | 503 stub | `HEALTH_FAIL` | honest `ErrorCard` "manual upload ships in F05"; never a fabricated report | yes | "Manual upload arrives in F05" |
| Compare/setup unknown driver | 404 | `NOT_FOUND` | `ErrorCard` (compare) / honest "—" (compat proxy) | no | "Driver not found" / "—" |
| Compat-proxy setup outage (not absence) | 503 | `DB_UNAVAILABLE`/`HEALTH_FAIL` | distinct code marker `⚠ <CODE>` (failure ≠ absence) | yes | "⚠ DB_UNAVAILABLE" |
| Network unreachable | fetch reject | `UI_NETWORK` | `ErrorCard` + Retry | yes | "Network error" |
| API too slow | AbortController @10s | `UI_TIMEOUT` | `ErrorCard` + Retry | yes | "Request timed out" |
| Response shape drift | zod `safeParse` fail | `UI_SCHEMA_MISMATCH` | `ErrorCard`; **no malformed render** | yes | "Unexpected response from server" |

---

## 7. Acceptance criteria (each item must be mechanically testable by /spec-check)

**Screen 3 — Prediction panel**
- [ ] Given a `{feature_vector, circuit}`, the panel POSTs `/predict` and renders `predicted_driver`, a `ConfidenceBar` (from `confidence` + `confidence_state`), and the `top_3` list.
- [ ] The confidence colour is taken from `confidence_state` via `lib/design/confidence.ts`; no `0.65`/`0.80`/`0.85` literal appears in Screen-3 components (div #46/#53 — grep-assertable).
- [ ] `ShapForcePlot` renders one signed bar per entry in `shap_values` (red toward / blue away by sign); `shap_available=false` → `EmptyState`, prediction still shown (div #56).
- [ ] `/predict` 400 `INVALID_INPUT` → `ErrorCard` with the code, no crash; 503 `INFERENCE_ERROR`/`HEALTH_FAIL` → retryable `ErrorCard`.
- [ ] `LOW_CONFIDENCE`/`AMBIGUOUS` `confidence_state` → `ConfidenceBadge` + dashed treatment (never a forced/hidden answer).

**Screen 3 — FastF1 Session tab**
- [ ] Submit → `postIngestSession` (202) → polls `getIngestStatus`; `PipelineProgress` reflects `status`/`progress`; `COMPLETE` renders `result` counts + `quality_report`; `FAILED` shows the failing stage red + `error_detail` (div #55).
- [ ] Enqueue 503 `QUEUE_UNAVAILABLE` → `ErrorCard`, no fabricated progress.
- [ ] Season selector = static {2022,2023,2024}; Driver selector populated from `getDrivers()`; "laps available" renders an explicit unavailable state, never a guessed count (div #57).

**Screen 3 — Manual Upload tab**
- [ ] Drag-drop a CSV → POST `/ingest/upload` → 503 → honest "manual upload ships in F05" `ErrorCard`; **no fabricated cleaning report** is ever rendered (div #58).

**Screen 3 — Unknown Driver tab**
- [ ] `postUnknownFingerprint` success renders `nearest_neighbours` (top-3 progress bars), `risk_flags` (red cards), and the `data_quality` summary.
- [ ] `confidence_state` + `warnings` render the LOW_CONFIDENCE/OOD/`SPARSE_PROJECTION` treatment against live signal (div #60).
- [ ] Because `radar_values` is empty, the axis-by-axis radar renders an explicit `EmptyState`; the style-space embed renders a deferred "arrives in F18" state (div #59) — no zeroed/fake radar.
- [ ] `<5` laps → 400 `INVALID_INPUT` `ErrorCard`; `<30` laps → 200 with a `SPARSE_PROJECTION` warning (state, not error) — both boundaries tested.

**Screen 4 — Comparison**
- [ ] `getCompare(id1,id2)` renders two `FingerprintRadar`s + a `DeltaRadar` (positive red / negative blue) from `delta_shap`.
- [ ] `most_distinctive_features` render as horizontal bars, labelled per-domain (div #61).
- [ ] Swap button swaps the drivers and flips the `DeltaRadar` sign (client-side).
- [ ] Setup delta table diffs the 10 params from `getSetup(d1,circuit)` + `getSetup(d2,circuit)`; the circuit selector drives a re-fetch.
- [ ] Compatibility projection gauge shows the delta of two setup-reliability **proxies** (mean per-param `confidence`), labelled a proxy; a driver's setup 404 → honest "—" (div #62/#47).
- [ ] `/compare` 404 `NOT_FOUND` → `ErrorCard`.

**Cross-cutting**
- [ ] Every panel renders all four `ApiResult` states (loading `Skeleton`, empty `EmptyState`, error `ErrorCard`, success) — no silent failure.
- [ ] `UI_NETWORK`/`UI_TIMEOUT`/`UI_SCHEMA_MISMATCH` each reachable and rendered (fetch reject / abort / zod mismatch) for at least one new endpoint.
- [ ] Every error code / failure mode in §6 has a vitest case that triggers it and asserts the rendered state.
- [ ] `openapi.snapshot.json` refreshed to include Predict/Unknown/Ingest/Upload schemas; `tests/api/test_openapi_frontend_contract.py` extended to assert their fields (pytest green).
- [ ] No new `config.yaml` block, no backend code, no new backend error code (grep-assertable diff); new timings live only in `lib/design/animation.ts` (div #53).
- [ ] `tsc --noEmit` (strict) clean, ESLint clean, `next build` compiles `/ingest` and `/compare`, vitest suite green.

---

## 8. Test plan
- **Happy-path fixtures (endpoint mocks, `vi.mock('@/lib/api/endpoints', …)` returning canned `ApiResult` — MSW is not wired in this repo):**
  - `PredictResponse` with HIGH `confidence_state`, populated `shap_values`, `top_3`.
  - `UnknownFingerprintResponse` with NN, risk flags, `data_quality`, empty `radar_values`, `SPARSE_PROJECTION` warning.
  - `CompareResponse` with two radars + signed `delta_shap`; two `SetupResponse`s for the delta table + compat proxy.
  - Ingestion: an enqueue 202 then a `PENDING→PROCESSING→COMPLETE` `getIngestStatus` sequence with a terminal `result`.
- **Failure fixtures (one per §6 row):** 400 `INVALID_INPUT`; 503 `INFERENCE_ERROR`; 503 `HEALTH_FAIL` (classifier + upload stub); `shap_available=false`; `LOW_CONFIDENCE`/`AMBIGUOUS`; `SPARSE_PROJECTION`; `OUT_OF_DISTRIBUTION`; 503 `QUEUE_UNAVAILABLE`; job `FAILED` + `error_detail`; 404 `NOT_FOUND`; 503 vs 404 on the compat proxy (failure ≠ absence); `UI_NETWORK` (fetch reject); `UI_TIMEOUT` (abort); `UI_SCHEMA_MISMATCH` (zod-invalid body).
- **Property/edge:** unknown-driver laps at exactly 5 (pass) and 4 (400) and 29/30 (sparse vs not); empty `shap_values` → EmptyState not empty plot; swap idempotence (swap twice = identity); compat proxy when one driver 404s and the other succeeds (mixed "—" + value); polling stops on terminal status (no infinite loop); a `delta_shap` of all-zero → flat delta radar (not blank).
- **Contract test (pytest):** the extended drift guard over `create_app().openapi()`.
- **e2e (Playwright):** green-field — `frontend/e2e/` does not yet exist and `npx playwright install chromium` has not been run. A minimal `ingest` + `compare` smoke spec is a stretch goal; if the install is unavailable in-session, record it as a documented limitation (as F16b did) rather than faking it.

---

## 9. Definition of done
All §7 acceptance criteria MET with passing vitest tests; `tsc --noEmit` + ESLint clean; `next build` compiles `/ingest` and `/compare`; `tests/api` (incl. the extended drift guard) green; `/apex-review F17` clean on the contract checklist (no silent failure, no re-derived confidence gate, error-code discipline, config-driven timings); INDEX row set to DONE. **Limitations to record for the dissertation:** (1) the master's FastF1 "session → prediction" flow is not end-to-end wireable and is honestly split (div #54); (2) five master UI elements (5-stage pipeline timings, SHAP base-value force plot, unknown-driver radar, "laps available", circuit-aware compatibility) have no backing endpoint and are rendered as explicit unavailable/proxy states with named seams (div #55–#62); (3) Manual Upload is a live-503 shell pending F05 (div #58); (4) Playwright e2e may remain green-field if the browser install is unavailable.

---

## 10. Risks & mitigations (from master risk register where relevant)
- **Week-18 "High — Three.js most complex UI" (master p.35 risk table):** that risk belongs to **Screen 5 / F18**, which F17 explicitly does *not* build; the unknown-driver style-space embed is a deferred state (div #59), so F17 carries none of the WebGL risk. *Mitigation: scope discipline — F17 is Screens 3 & 4 only, per the INDEX split.*
- **Honesty landmine — half the master's Screen-3/4 elements lack a backing endpoint.** The temptation is to fabricate (fake 5 stage timings, a zeroed radar, a guessed lap count, a made-up compatibility score). *Mitigation: every such element is rendered as an explicit unavailable/proxy state with a named backend seam (div #54–#62), exactly as F16 did for #47–#51; `/apex-review` + a dedicated "no fabrication" test per landmine enforce it.*
- **The ingestion/prediction conflation (div #54)** could produce a confusing UI if the two flows are merged. *Mitigation: FastF1 tab = real async job (status/progress/report); a separate `PredictionPanel` = the real `/predict` — each honest about what it does.*
- **Async polling can leak timers / hammer the API.** *Mitigation: poll interval from `animation.ts`; `useEffect` cleanup stops polling on terminal status and on unmount; an edge test asserts polling stops on `COMPLETE`/`FAILED`.*
- **Contract drift** between the frontend types and the live backend (new endpoints not in the committed snapshot). *Mitigation: refresh `openapi.snapshot.json` + extend the pytest drift guard so a backend rename fails CI (§4.3).*
- **Manual Upload depends on F05 (TODO).** *Mitigation: build the shell against the real 503 and surface it honestly; the report table already accepts the `UploadResponse` shape so F05 is a one-line data swap (div #58).*
- **Scope creep** — F17 is already the widest frontend feature (two screens, five landmines). *Mitigation: no backend code (div #53); the two candidate backend seams (force `base_value`, unknown `radar_values`) are named, not built. If the session runs long, the F16→F16a/F16b precedent allows a split (F17a Screen 3 / F17b Screen 4).*

---

### Divergences introduced by this spec (continuing F16's #45–#52)
- **#53** — Frontend-only: no `config.yaml` block, no backend code, no new backend error code; reuses `UI_*` (continues #45).
- **#54** — Screen 3 separates ingestion (async job) from prediction (`/predict` panel); the master's session→prediction flow is not end-to-end wireable.
- **#55** — `PipelineProgress` driven by the real coarse job lifecycle (`status`/`progress`/`error_detail`), not the master's 5 named per-stage timings.
- **#56** — `ShapForcePlot` is a partial: only signed `shap_values` are surfaced; `base_value`/names/direction are dropped at the API boundary (named seam).
- **#57** — No session-catalog endpoint: seasons static, drivers from `/drivers`, round/session bounded inputs, "laps available" an explicit unavailable state.
- **#58** — Manual Upload attempts the real `/ingest/upload` (503 stub, F05) and renders an honest "ships in F05" state; UPLOAD_001–004 defined but unwired.
- **#59** — Unknown-driver `radar_values`/`style_description` empty + no SSE + style-space embed = Screen 5 (F18): radar/embed are explicit unavailable states; NN/risk/quality/confidence live.
- **#60** — Screen 3 is the honest home of live `LOW_CONFIDENCE`/`OUT_OF_DISTRIBUTION`/`SPARSE_PROJECTION` (resolves F16 finding F for projected drivers).
- **#61** — Screen 4 `delta_shap`/`most_distinctive_features` are per-**domain** radar deltas, not per-feature SHAP (labelled honestly).
- **#62** — Screen 4 compatibility projection = setup-reliability proxy (reuse #47); `/compare` is circuit-agnostic, no "which driver better for circuit X" endpoint.
