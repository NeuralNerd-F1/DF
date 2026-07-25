# APEX Feature Index & Build Order

One feature = one spec = one focused session. Build in dependency order: never start a
feature whose upstream prerequisites are not `DONE`. Status values:
`TODO -> SPEC WRITTEN -> IN PROGRESS -> DONE`.

Generate each spec with `/feature-spec <id> <area>`, build it with `/plan-feature <id>`
then implement, gate it with `/spec-check <id>` and `/apex-review <id>`.

## Layer 0 — Foundation contracts (do first, tiny, unblocks everything)
| ID  | Feature | Depends on | Wk | Status |
|-----|---------|-----------|----|--------|
| F00 | Error-code Enum, result objects, `config.yaml`, repo skeleton, Docker Compose | — | 1 | DONE (2026-06-25) |

## Layer 1 — Data pipeline
| ID  | Feature | Depends on | Wk | Status |
|-----|---------|-----------|----|--------|
| F01 | Data ingestion (FastF1/OpenF1/Ergast, caching, circuit breakers) | F00 | 1 | DONE (2026-06-26) |
| F02 | Lap filter chain / validation contract (9 filters, error codes) | F00, F01 | 2 | DONE (2026-06-26) |
| F03a | Scalar-channel features (F01–F18, F22–F24, F26) + vector assembly, `FeatureComputationError`, 6-NULL tolerance | F00, F02 | 3 | DONE (2026-06-26) |
| F03b | X/Y cornering (F19–F21), corner-balance (F25), tyre-age slopes (F27–F28) — plug into F03a's vector | F00, F02, F03a | 3 | DONE (2026-06-26) |
| F04 | Circuit-aware normalisation (train-only, saved artefact) | F03a, F03b | 4 | DONE (2026-06-26) |
| F05 | Manual CSV upload pipeline + edge cases (UPLOAD_001–004) | F02, F03 | 16 | DONE (2026-07-25) |

## Layer 2 — Machine learning
| ID  | Feature | Depends on | Wk | Status |
|-----|---------|-----------|----|--------|
| F06 | Model 1 — XGBoost classifier + confidence gate + calibration + Optuna + MLflow | F04 | 5–6 | DONE (F06a 2026-06-30, F06b 2026-07-01) |
| F07 | SHAP explainability (beeswarm, force, radar, NL templates) | F06 | 7 | DONE (2026-07-01) |
| F08 | Model 2 — UMAP style space + PCA fallback + unknown-driver projection | F04 | 8 | DONE (2026-07-02) |
| F09 | Corner-type decomposition + sub-fingerprints (ablation) | F03a, F03b | 9 | DONE (2026-07-18) |
| F10 | Model 3 — Degradation detector (deviation, rolling mean, alert threshold) | F04, F08 | 11–12 | DONE (2026-07-17) |
| F11 | Car personalisation engine (3 layers: mapping, constraints, confidence) | F07, F08 | 13 | DONE (2026-07-17) |

## Layer 3 — Platform
| ID  | Feature | Depends on | Wk | Status |
|-----|---------|-----------|----|--------|
| F12 | PostgreSQL schema + Alembic migrations | F00 | 15 | DONE (2026-07-19) |
| F13 | FastAPI endpoints + Pydantic contracts + health check | F06–F11, F12 | 15–16 | DONE |
| F14 | Redis cache + Celery queue + replay engine | F13 | 16 | DONE (2026-07-21) |
| F15 | Failsafe evaluation harness (RQ5 — 7 tests) | F13, F14 | 14 | DONE (2026-07-22) |

## Layer 4 — Frontend
| ID  | Feature | Depends on | Wk | Status |
|-----|---------|-----------|----|--------|
| F16a | Design system + Screen 1 Command Centre | F13 | 17 | DONE (2026-07-22) |
| F16b | Screen 2 Driver Profile | F13, F16a | 18 | DONE (2026-07-22) |
| F17 | Screen 3 Ingestion/Prediction + Screen 4 Comparison | F13, F16a | 18 | DONE (2026-07-23) |
| F18 | Screen 5 Style Space (Three.js + D3 2D fallback) | F16a | 18 | DONE (2026-07-24) |
| F19 | Screen 6 Historical Replay Monitor | F14, F16a | 19 | TODO |

## Layer 5 — Evaluation & writing
| ID  | Feature | Depends on | Wk | Status |
|-----|---------|-----------|----|--------|
| F20 | Evaluation framework (classifier, stability, degradation, failsafe) | F06–F15 | 20 | SPEC WRITTEN |
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

## Amendments
- **F05 BUILT — 2026-07-25 (feature/F05-manual-upload): DONE.** `app/pipeline/upload.py`
  (`process_upload` + the eight §3.4 edge-case handlers + `_to_session_frame` + `UploadResult`/
  `CleaningReport`/`LapRejection`) + live `app/api/routes/upload.py` (`POST /ingest/upload` multipart →
  `UploadResponse`, retires the F13 `deferred.py` 503 stub) + `get_medians` provider on `AppState` +
  typed `UploadConfig` + the `upload:` config block. **No new error codes** (all pre-seeded). Runs the
  **real F02 `validate_session` + F03 feature extraction** behind CSV pre-processing; **synchronous +
  stateless** (writes nothing to the DB, mirrors F13 `/predict` #30). The CSV→wide-frame adapter
  synthesizes documented context defaults (green `track_status`, per-lap `lap_time` from Time, no-pit,
  tyre-age-UNKNOWN → F27/F28 NULL); imputation is **injected train-only `FeatureMedians`** (constitution-#2
  leakage scan + ml-auditor PASS); NULLs bounded by {X/Y cornering, F15 distance, F27/F28} so a full-
  telemetry lap → IMPUTED, a bare 5-channel CSV → LAP_NULL_EXCESS. Adds **`python-multipart`** (first
  multipart route) pinned. **Gates:** 51 F05 tests + full suite `PYTEST_EXIT=0`, `mypy app` + `ruff` clean,
  `/spec-check F05` DONE, `/apex-review F05` clean, `/security-review` no findings; failsafe-auditor findings
  folded in (dedup logs `ORPHAN_SAMPLES` not `LAP_PARTIAL`; corrupt-drop/dedup/all-rejected now warn;
  `combination_excluded` surfaced; route caps the multipart read at `max_file_bytes+1`). Regenerated
  `frontend/openapi.snapshot.json` + `schema.d.ts` for the new multipart body (tsc + eslint clean).
  **Limitations (dissertation):** (1) sync + stateless; (2) documented context defaults; (3) UPLOAD_002
  `Speed median > 350 ⇒ mph` copied verbatim though physically inverted (supervisor-flagged); (4) unknown-
  driver projection is F08 via a follow-on `/fingerprint/unknown`; (5) **`FeatureMedians` inject-only — no
  committed artefact/loader yet** (parallel to F13's uncommitted handles) → a live upload returns the
  cleaning report with `clean_vectors=[]` (logged `MODEL_NOT_FOUND` + warned) until a producer lands.
  Depends on **F02 + F03a/F03b + F04 + F13 (all DONE)**; consumed by F13/F14 (route live), F17 (Manual
  Upload tab live), F08 (unknown-driver projection), F20 (upload ingress). Branch off master.
- **F05 spec — 2026-07-25 (feature/F05-manual-upload): SPEC WRITTEN.** `specs/F05-manual-upload.md` — the
  **manual CSV upload pipeline** (master §3.4 edge-case handling + §8.5 `POST /ingest/upload`). Reuses the
  **same F02 filter chain + F03 feature extraction** behind the eight §3.4 pre-processing edge cases and
  fills the one endpoint F13 still stubs as 503 (unblocking F17's live-503 Manual Upload shell). **Adds no
  new error codes** — `UPLOAD_001` (missing columns→400), `UPLOAD_002` (mph-units warning→200),
  `UPLOAD_003` (>5% corrupt→400), `UPLOAD_004` (empty→400) **and** `RESAMPLED`/`IMPUTED`/`LAP_PARTIAL`/
  `SPARSE_PROJECTION`/`DATA_INSUFFICIENT`/`FEATURE_001`/`LAP_NULL_EXCESS`/`INVALID_INPUT` are all
  pre-seeded in `errors.py`; the committed `UploadResponse` schema and `lap_validation.required_channels`
  (= the §3.4 required 5) are reused unchanged. Produces `app/pipeline/upload.py` (`process_upload →
  UploadResult`/`CleaningReport`) + live `app/api/routes/upload.py` (retires `deferred.py`) + typed
  `UploadConfig`. **Key design calls (divergences):** upload is **synchronous + stateless** (returns the
  cleaning report in the response, not a 202-job; writes nothing to the DB — mirrors F13 `/predict` #30;
  `clean_vectors` handed to downstream `/predict`/`/fingerprint/unknown`); the CSV lacks session context so
  `_to_session_frame` synthesizes **documented defaults** (green `track_status`, per-lap `lap_time`
  from Time, no-pit, tyre-age-UNKNOWN → F27/F28 NULL); imputation uses the **injected train-only
  `FeatureMedians`** with a no-`.fit()`/no-`.median()`-on-upload leakage scan (constitution #2); `upload_id`
  = `sha256(bytes)[:16]` (idempotent); DoS guards (`max_file_bytes`/`max_rows` → `INVALID_INPUT`) reject
  before parse; the §3.4 UPLOAD_002 threshold (`Speed median > 350 ⇒ mph`) is **copied verbatim** though
  physically inverted — flagged for the supervisor. Depends on **F02 + F03a/F03b + F04 + F13 (all DONE)**;
  consumed by F13/F14 (route live), F17 (tab live), F08 (unknown-driver projection), F20 (upload ingress).
  Built on `feature/F05-manual-upload` off master. **Next: build (in progress via /ship).**
- **F18 BUILT — 2026-07-24 (worktree-F18-style-space): DONE.** Screen 5 Style Space Map (`app/style-space`) — a
  Three.js 3D scatter of the F08 UMAP style space with a **silent D3 2D fallback**. **Frontend-only** (div #69 — no
  `config.yaml`, no backend code, no new backend error code); adds **`three` 0.169.0 + `@types/three`** (`d3` already
  present). Pure utils `lib/style-space/{hasWebGL,styleSpace}.ts` (`joinStyleWithDrivers` #65, client-side
  `nearestNeighbours` #66, `teammatePairs`, `hasWebGL()` boundary #68); components `components/style-space/`
  (`StyleSpaceScreen` root = join + WebGL branch; `StyleSpace2D` = the jsdom-testable SVG scatter; `StyleSpaceScene` =
  Three.js, **dynamically imported ssr:false** so `three` is code-split out of the SSR/2D path; `DriverTooltip`,
  `DriverSidebar`, `TeammateLinesToggle`, `SeasonFilter` [deferred #64], `UnknownProjectionPanel` reusing F17's
  `postUnknownFingerprint` + `NearestNeighbours`). The scene wraps all WebGL setup in `try/catch → onInitError`, so an
  init failure degrades silently to 2D (master §10.6 "no error"). Reuses `FingerprintRadar` (mini radar) + `teamColour()`
  (sphere/node colour via the #47 `/drivers` join). Honesty landmines rendered honestly: season filter deferred (#64),
  unknown sphere 5D-only + NN-only-on-fallback (#67), team via join (#65), NN client-computed (#66). **Gates green:**
  `tsc` clean, ESLint (`--max-warnings 0`) clean, **126 vitest tests (33 files)**, `next build` compiles `/style-space`
  (three code-split → 125 kB first-load), OpenAPI drift guard **12/12** + `mypy app` clean, `/apex-review` clean.
  **Limitations:** Three.js WebGL render is e2e-only (green-field, no `npx playwright install`) — the D3 2D fallback +
  all interactions/joins/NN/teammate are fully unit-tested; season filter + unknown-2D-position are deferred by design.
  **Divergences #63–#69.** Consumed by F19 + F20/F21. Branch stacks on F17 (PR #2). **Next: F19 (Screen 6 Replay).**
- **F18 spec — 2026-07-24 (worktree-F18-style-space): SPEC WRITTEN.** `specs/F18-style-space-map.md` — **Screen 5, the
  Style Space Map** (master §10.6): a Three.js 3D scatter of the F08 UMAP style space (3 of 5 dims), glowing
  team-coloured spheres, auto-rotating orbit camera, hover tooltip (code/team/season/NN/mini-radar), click sidebar,
  teammate-proximity lines, unknown-driver white sphere, and a **silent D3.js 2D fallback** when WebGL is unavailable.
  Master-flagged **HIGH risk** ("Three.js most complex UI"). **Frontend-only** (div #69 — no `config.yaml`, no backend
  code, no new backend error code); the one dep addition is **`three` + `@types/three`** (not currently installed;
  `d3 7.9.0` already present). Consumes the **already-built-but-unused** F16a seam (`getStyleSpace`/`zStyleSpace`,
  shipped "for F18") + `/drivers` (team colour via the #47 seam + mini-radar) + F17's `postUnknownFingerprint` (the
  white sphere's 5D vector). **Key recon findings driving the divergences:** `/stylespace` returns **one row per driver**
  (coords_2d + coords_5d; `season` always null — the route drops the backend-computed `season_coords_*`), so the
  **season filter is a deferred/disabled state** (#64, named additive seam); **no `team`/neighbours** on the payload →
  team via `/drivers` join (#65), known-driver NN computed client-side from `coords_5d` (#66); the unknown white sphere
  has only a 5D `fingerprint_vector` and degrades to NN-only on `PROJECTION_FALLBACK` (#67, continues #59); the **D3 2D
  fallback is the silent WebGL-unavailable path** and the naturally jsdom-testable surface, with `hasWebGL()` as the
  injectable boundary — Three.js mocked for a smoke test, WebGL render left to Playwright e2e (green-field, as F16b/F17)
  (#68); master's r128 pin reconciled to the current installable `three` ~0.16x (#63). **Divergences #63–#69.** Depends
  on **F16a + F13 + F08 (all DONE)** + reuses F17; branch stacks on the F17 tip (reuses the design system + nav +
  client, conflict-free). **Contingency:** split F18a (data + join + **D3 2D scatter** + interactions + tests = the
  shippable core) / F18b (Three.js 3D + camera + WebGL) if one session runs long. **Next: `/plan-feature F18` then build.**
- **F17 BUILT — 2026-07-23 (worktree-F17-frontend-spec): DONE.** Screen 3 (`app/ingest`) + Screen 4 (`app/compare`)
  built on the F16a/b design system — **pure frontend, no backend app code / no `config.yaml` block / no new backend
  error code** (div #53). New API-client seams (`postPredict`/`postUnknownFingerprint`/`postIngestSession`/
  `getIngestStatus`/`postUpload` + 5 zod validators + 11 type aliases; `errors.ts` messages; `animation.ts` timings;
  a `delta.up`/`delta.down` blue token). Two chart primitives (`ShapForcePlot` signed-bars, `DeltaRadar`). Screen 3 =
  `PredictionPanel` (the one real `/predict` + SHAP flow) + `PipelineProgress` + `FastF1Tab` (async job via
  `useIngestionJob` poll hook) + `ManualUploadTab`/`DropZone` (live-503 F05 shell) + `UnknownDriverTab`
  (NN/risk/quality/live-confidence). Screen 4 = `ComparisonScreen` (two radars + delta radar + distinctive domains) +
  `SetupDeltaTable` + `CompatibilityProjection` (#47 proxy). All ~10 honesty landmines rendered as explicit
  unavailable/proxy states with "no-fabrication" tests. **Gates green:** `tsc` clean, ESLint (`--max-warnings 0`) clean,
  **95 vitest tests (25 files)**, `next build` compiles `/ingest` + `/compare`, extended pytest drift guard + full
  `tests/api` green. `/apex-review F17` clean (no silent failure; confidence colour from `confidence_state`, no re-derived
  gate; config-driven timings; each failure mode tested). **Limitations:** Playwright e2e left green-field (no
  `npx playwright install`, as F16b); two named backend seams not built (force `base_value` #56, unknown `radar_values`
  #59). **Divergences #53–#62.** Consumed by F18 (embeds Screen 5), F19, F20/F21. **Next: F18 (Screen 5 Style Space).**
- **F17 spec — 2026-07-23 (worktree-F17-frontend-spec): SPEC WRITTEN.** `specs/F17-ingestion-prediction-comparison.md`
  — the **widest frontend feature**: master **§10.4 Screen 3 (Ingestion & Prediction)** (three tabs — FastF1 Session,
  Manual Upload, Unknown Driver — plus a direct `/predict` panel + shared pipeline-progress animation) and **§10.5
  Screen 4 (Comparison)** (two radars + signed delta radar, most-distinctive features, compatibility projection, setup
  delta). **Pure frontend, like F16** (divergence #53 = no `config.yaml` block, no backend code, no new backend error
  code; reuses the F16a/F16b design system, API client, `ApiResult` four-state discipline, `FingerprintRadar`/
  `GaugeNeedle`, and vitest patterns). First UI to exercise `POST /predict`, `POST /fingerprint/unknown`, the async
  `/ingest/*` job (202 + poll), and `GET /compare`. **The honesty landmine is the widest yet — ~half the master's
  Screen-3/4 elements have no/partial backing endpoint, each rendered as an explicit unavailable/proxy state, never
  fabricated:** the master's "FastF1 session → prediction" is not end-to-end wireable — ingestion writes to DB and
  returns no prediction — so ingestion and prediction are **honestly split** (div #54); `PipelineProgress` is driven by
  the real coarse job lifecycle, not 5 faked per-stage timings (#55); the SHAP force plot is a partial (`/predict`
  surfaces only signed `shap_values`; `base_value`/names/direction are dropped at the API boundary — #56); no
  session-catalog endpoint → seasons static + drivers from `/drivers` + "laps available" unavailable (#57); **Manual
  Upload is a live-503 shell** (`/ingest/upload` is an F05 stub; UPLOAD_001–004 defined but unwired — #58);
  unknown-driver `radar_values` come back empty + no SSE + style-space embed = Screen 5/F18 (#59), but its
  `confidence_state`/`warnings` make Screen 3 the **honest home of live LOW_CONFIDENCE/OOD/SPARSE_PROJECTION** (#60,
  resolves F16 finding F for projected drivers); Screen 4 `delta_shap`/`most_distinctive_features` are per-**domain**
  radar deltas not per-feature SHAP (#61); and the compatibility projection reuses the **#47 setup-reliability proxy**
  because `/compare` is circuit-agnostic (#62). Adds two chart primitives (`ShapForcePlot`, `DeltaRadar`), new
  `endpoints.ts`/`validators.ts`/`types.ts` for the POST routes, a refreshed `openapi.snapshot.json` + extended pytest
  drift guard. **Divergences #53–#62.** Depends on **F13 + F16a (both DONE)** — every build prerequisite is green;
  consumed by F18 (embeds Screen 5), F19 (reuses async-poll), F20/F21. **Contingency if too big for one session:**
  split F17a (Screen 3) / F17b (Screen 4), per the F16a/b · F06a/b · F13 precedent. **Next: `/plan-feature F17` then build.**
- **F16b BUILT — 2026-07-22 (feature/F16b-driver-profile): merge-ready.** Screen 2 Driver Profile on a
  branch off the merged master. `app/drivers/[code]/page.tsx` (SSR fingerprint + drivers-for-field-mean +
  setup; `PageError` on 404) + `components/driver-profile/` (`ProfileRadar` = the large radar with a hover
  tooltip [domain, score, NL meaning, field-mean vs the 8 drivers, div #51]; `NlDescription`; `SetupPanel`
  = 10 gauges w/ WIDE→range-bar + constraint badges + disclaimer; `MultiSeasonPanel` = seasons/stability/
  pairwise-JS; `ShapBeeswarmPanel`; `CornerTabs`) + charts `GaugeNeedle` + `ShapBeeswarm` (deterministic
  seeded jitter, colour-by-circuit). **Honest gaps rendered as explicit unavailable states, never faked:**
  no per-lap SHAP endpoint → beeswarm empty (div #48); no corner-sub-fingerprint endpoint → corner tabs
  empty (F09 artefact not served, div #49); no per-season fingerprint endpoint → per-season radar switching
  shown honestly (only the multi-season stability/JS-divergence that IS exposed is rendered). Reuses the
  F16a design system, API client, `FingerprintRadar`, and test patterns. **9 new tests (46 total)**; tsc +
  ESLint clean; `next build` compiles (`/drivers/[code]`). Remaining before DONE: F16b failsafe/review gates
  (running) + Playwright e2e (needs `npx playwright install`). Consumed by F17–F19.
- **F16 SPLIT + F16a BUILT — 2026-07-22 (worktree-F16-frontend-spec): F16a merge-ready, F16b SPEC WRITTEN.**
  F16 proved too large for one session (the split the spec §10 documented as a contingency), so it is split
  **F16a** (design system + Screen 1 Command Centre — BUILT) / **F16b** (Screen 2 Driver Profile — the
  remaining §4.4 work). The combined spec `specs/F16-design-system-and-screens.md` stays the source; its
  §12 "Split" header maps which sections are F16a (done) vs F16b (remaining) — the F06 precedent (combined
  spec kept). **F16a delivered** on `worktree-F16-frontend-spec`: a greenfield **`frontend/`** tree
  (Next.js 14.2.35 / TS-strict / D3 7 / Zustand 4 / Tailwind 3), the master §10.1 **design system** (tokens
  in `tailwind.config.js`; `ds/` primitives — Skeleton=telemetry-wave-not-spinner, ErrorCard=red-border+
  monospace-code+Retry, ConfidenceBar=colour-from-state-not-0.65/0.85, ConfidenceBadge, EmptyState, Card,
  Button, Tabs), a **validated API client** (TS types + `ErrorCode` generated from a committed
  `openapi.snapshot.json`; zod runtime validators; never-throwing `fetchApi`→4-state `ApiResult`; `UI_*`
  client codes), the **`FingerprintRadar`** D3 chart, and **Screen 1 Command Centre** (SSR `/drivers`→grid;
  per-driver stability + compatibility proxy [mean setup reliability, div #47; 404→"—"]; compat/stability
  sort; degradation sort disabled+tooltip div #50; LOW_CONFIDENCE card treatment; hover top-3; empty/error
  states). **Backend seam `DriverSummary.team`** (div #47) + a **pytest OpenAPI drift guard** (frontend
  contract). **Gates:** `tsc` + ESLint clean, **31 vitest tests**, `next build` compiles, `tests/api` +
  `tests/contracts` green. **Divergences #45–#52** + two new build-time findings: known-driver feed carries
  **no per-driver `confidence_state`** (LOW_CONFIDENCE treatment built+tested but renders confident live —
  honest, not faked) and **no `/circuits` endpoint** (static reference-data list, held-out circuits omitted).
  **Also fixed + merged to master a real repo bug (`dc8ad14`):** `.gitignore` `cache/` swallowed F14's
  `app/cache/` source so master didn't import from a clean checkout. **F16b next** = Screen 2 (ProfileRadar
  tooltips + field-mean, season switcher+ghost, multi-season overlay+JS-divergence, SHAP beeswarm [div #48],
  corner tabs [div #49], NL description, 10-param setup panel) + Playwright + the review gates.
- **F16 spec — 2026-07-22 (worktree-F16-frontend-spec): SPEC WRITTEN.** `specs/F16-design-system-and-screens.md`
  — the **first frontend feature**: the master §10.1 **design system** in Tailwind (dark-only F1-TV/MoTeC;
  verbatim tokens `#0D0D0D`/`#E8002D`/`#C0392B`/`#F0F0F0`/`#888888`/`#1A1A1A`/`#2B2B2B`, Inter + JetBrains
  Mono, `easeCubicInOut`, skeleton-not-spinner loading, red-left-border error) + **Screen 1 Command Centre**
  (§10.2 — 8-driver grid of animated 6-axis fingerprint radars, 80ms/800ms bloom, circuit selector, hover
  top-3 SHAP, LOW_CONFIDENCE dashed/badge/reduced-opacity, sort, click-through) + **Screen 2 Driver Profile**
  (§10.3 — large radar, season switcher w/ 2s ghost, multi-season overlay + JS divergence, SHAP beeswarm,
  corner tabs, NL description, 10-param setup panel w/ WIDE→range-bar). Greenfield **`frontend/`** tree:
  Next.js 14 App Router / TS-strict / D3 7 / Zustand 4 / Tailwind 3 (Three.js is F17, not here); Vitest + RTL
  + Playwright + MSW offline tests. **The frontend embodiment of the non-negotiables:** four explicit
  data-fetch states (loading/empty/error/success) via a typed `ApiResult` union — **no silent failure**;
  confidence colour driven by the backend-resolved `confidence_state`, **never a re-derived 0.65/0.85** gate
  (div #46); **types + `ErrorCode` generated from `/openapi.json`** (no hardcoded strings) + a pytest drift
  guard; deterministic (seeded) D3 layout (reproducibility). **No new backend error code, no `config.yaml`
  block** (div #45) — the only new "codes" are frontend-local `UI_NETWORK`/`UI_TIMEOUT`/`UI_SCHEMA_MISMATCH`
  for pre-API failures. **The honesty landmine:** four master UI elements have **no** F13 endpoint —
  compatibility score (→ setup-reliability **proxy**, div #47), per-lap SHAP **beeswarm** (div #48), corner
  sub-fingerprint **tabs** (F09 artefact, not served, div #49), degradation-sensitivity **sort** (disabled,
  div #50); each is derived honestly or rendered as an **explicit unavailable state — never fabricated**
  (constitution #1), with a named additive-seam follow-up. **Divergences #45–#52.** Depends on F13 (DONE);
  consumed by F17–F19 (reuse design system + charts + API client) + F20/F21. **Contingency if too big for
  one session:** split F16a (scaffold + design system + Command Centre) / F16b (Driver Profile), per the
  F03a/b · F06a/b · F13-split precedent. **Next: build F16.**
- **F15 BUILT — 2026-07-22 (feature/F15-failsafe-harness): DONE.** `app/eval/failsafe.py` (the 7 `rq5_*`
  §11.4 failsafe tests + `run_failsafe_suite` → content-hashed RQ5 results table + `render_table`) +
  `app/eval/harness.py` (self-contained app-assembly + fakes + fault-injection + `measure_post`/`p95`) — the
  **executable RQ5 harness** (the *dynamic* complement to the static `failsafe-auditor`): assembles the real
  F13+F14 system and runs the master §11.4 tests with real fault injection + measurement (confidence gate,
  FastF1-breaker→STALE_DATA, NULL 6/7 tolerance, Redis-down 100-preds, DB-down 503-no-partial-write, UMAP
  5-SD OOD, 50-concurrent). New `failsafe:` block + typed `FailsafeConfig` (2 cross-block validators);
  **NO new error codes** (asserts pre-seeded ones); `httpx` promoted **dev→main**. **Scope = §11.4 ONLY**
  (div #39); **F20 consumes `run_failsafe_suite`** for that pillar + extends Tests 1/6 with real-artefact
  rates. **Divergences #39–#44** (§11.4-only; in-process measurement validates the *mechanism* not
  production p95 → F20; Tests 1/6 rate artefact-gated; Test 2 is pipeline-level; CACHE_UNAVAILABLE/DEGRADED
  reconciled per F14; RQ5 table = `failsafe_report.json`). **18 tests**; full suite green, `ruff` + `mypy app`
  clean. **`failsafe-auditor` run on `app/eval/`; HIGH findings fixed with regression tests** (harness no
  longer pollutes the real `cache/` or `data_quality_log` — both redirected to temp; the suite survives a
  test that raises in pre-`try` setup; `render_table` gained the Findings column). Consumed by F20 (RQ5
  evidence). **Next: F16 (frontend) or F05 (upload).**
- **F14 BUILT — 2026-07-21 (feature/F14-cache-queue-replay): DONE.** `app/cache/redis.py` (`RedisCache`
  + own §8.3 breaker + cache-aside, never-raising, non-critical), `app/tasks/{ingest,celery}.py`
  (`run_ingestion` = atomic persist→predict→fingerprint batch over a `LapRecord` contract; Celery app +
  task shell + `InProcessJobStore`/`RedisJobStore` + `enqueue_ingestion`), `app/replay/engine.py`
  (read-only growing-prefix replay through F10 `monitor_stint`), `app/api/routes/{ingest,replay}.py` (fill
  F13's deferred 503s; `deferred.py` shrunk to the F05 `/ingest/upload` stub), real `cache_connected` at
  `/health` + cache/queue metrics. New `cache:`/`celery:`/`replay:` blocks + typed
  `CacheConfig`/`CeleryConfig`/`ReplayConfig`; **3 new codes** `QUEUE_UNAVAILABLE`/`INGEST_TASK_FAILED`/
  `REPLAY_PARTIAL` (div #32). **Divergences #31–#38** (202-async result via `/ingest/status.result`;
  3 codes; cache-key scheme; non-driver TTLs; Celery retry constants; replay read-only writes-nothing =
  the batch/ingestion path is the `predictions`/`fingerprints` writer, resolving F13 #30; invalidate-on-write;
  #38 = the FastF1→F04 `feature_provider` is DI'd, live-wired in F20). Pins **redis 8.0.1 / celery 5.6.3**
  (+ `fakeredis` dev) — billiard 4.2.4 had a cp314 wheel, F12-psycopg risk did not recur. **~57 tests**
  (offline: fakeredis + Celery `always_eager` + aiosqlite; a `@pytest.mark.redis`/`@pytest.mark.celery`
  gate the real infra). **`failsafe-auditor` run**; HIGH findings fixed with regression tests
  (`RedisJobStore` bounded + logs, not hangs/silent; `predictions` deduped on re-ingest; poisoned cache
  value logged; queue metric wired). Full suite green, `ruff` + `mypy app` clean. **Deferred (documented):**
  Celery `autoretry` policy (fail-fast for now), worker-side completed/failed metrics, `session`-token
  validation. Consumed by F15/F20 (RQ5 cache/concurrency + eval-DB population) + F19 (Screen 6 replay).
  **Next: F15 (failsafe harness) or F05 (upload).**
- **F13 BUILT — 2026-07-21 (master): DONE.** `app/api/` (`main` · `deps` · `errors` · `schemas` ·
  `metrics` + `routes/{health,predict,drivers,style,metrics,deferred}`) — the FastAPI serving layer
  (`create_app()` factory + best-effort lifespan + typed exception handlers + CORS + Prometheus)
  wiring the DONE never-raising handles (F06b/F07/F08/F10/F11) and the F12 persistence seam into the
  master-§8.5 REST surface, with Pydantic v2 request/response schemas and a **real `/health`** (503
  iff DB down or any handle not ready; cache non-critical). **10 synchronous endpoints fully
  implemented; 4 async/upload/replay routes are honest 503 `HEALTH_FAIL` schema-stubs** (F14/F05).
  **3 new codes `NOT_FOUND`/`FORBIDDEN`/`API_ERROR` (div #29).** New `api:` block + typed `ApiConfig`
  (adds `min_projection_laps`, config-driving the `<5`-laps floor — no magic numbers). Pins
  **FastAPI 0.138.1 / starlette 1.3.1 / uvicorn 0.49.0** (venv→pyproject) + **httpx** (dev).
  **Divergence #30 — `/predict` is stateless** (the spec's optional `predictions`-row write is
  dropped: a live `{feature_vector, circuit}` POST has no persisted `lap_id`/`predicted_driver_id`
  for the F12 `predictions` FKs, and fabricating them would violate §1; the table is populated by the
  F14 batch/replay path). **`failsafe-auditor` run on `app/api/`** (RQ5 evidence): confirmed
  `/predict` never 5xx on model output, no `.fit()`, no bare excepts, stateless routes provably
  DB-independent, rollback-before-raise (no partial write), no stack-trace leak, startup degrade
  always logged. **Findings fixed with regression tests:** DB_UNAVAILABLE + DB_INTEGRITY now log to
  `data_quality_log`; the `<min_projection_laps` gate logs; `INFERENCE_ERROR` surfaces its distinct
  code (not collapsed to `HEALTH_FAIL`); `/metrics` rejects a spoofed `X-Forwarded-For` from a direct
  external peer. **~50 `tests/api` tests** (fake handles via `dependency_overrides` + aiosqlite);
  full suite green, `ruff` + `mypy app` clean. Consumed by F14/F05 (fill deferred routes), F16–F19
  (frontend), F15/F20 (RQ5 harness). **Next: F14 (Redis + Celery + replay).**
- **F13 spec — 2026-07-19 (master):** `specs/F13-api-endpoints.md` written (SPEC WRITTEN). The
  **HTTP serving layer** — a FastAPI app (`create_app()` factory + lifespan + exception handlers +
  CORS) wiring the DONE never-raising handles (F06b `predict` · F07 `explain_prediction`/
  `season_stability` · F08 `project_unknown`/`get_style_space` · F10/F11 health + F11
  `recommend_setup`/`onboard_unknown`) and the F12 persistence seam (`session_scope`/`check_database`/
  repository) into the master-§8.5 REST surface, with Pydantic v2 request/response schemas and a
  **real `/health`** (503 iff DB down or any model handle not ready; **cache is non-critical**).
  **Primary divergence #28 (endpoint split):** F13 fully implements the 10 synchronous endpoints
  (`/health`, `/predict`, `/fingerprint/unknown`, the driver reads, `/stylespace`, `/metrics`); the 4
  async/upload/replay endpoints (`/ingest/session`, `/ingest/status`, `/ingest/upload`, `/replay`) get
  their Pydantic schemas here but return **503 `HEALTH_FAIL`** until F14 (Redis+Celery+replay) / F05
  (upload) wire their engines — honest, never faked. **Three new codes (div #29): `NOT_FOUND` (404),
  `FORBIDDEN` (403 external `/metrics`), `API_ERROR` (500 no-unhandled-exception backstop);** reuses
  `INVALID_INPUT`/`DB_UNAVAILABLE`/`DB_INTEGRITY`/`HEALTH_FAIL`/`MODEL_NOT_FOUND` + every model's
  uncertainty state (→ 200 body, `/predict` **never 5xx**). Key design calls: handles are
  **dependency-injected, loaded best-effort in the lifespan** (no artefacts are committed → a fresh
  checkout 503s at `/health`; **endpoint tests inject fake handles via `dependency_overrides`**);
  DB-criticality is per-endpoint (stateless compute stays 200 when DB down; DB-backed reads → 503);
  DB-backed reads serve the `fingerprints` table (empty→404/[] until an F14/F20 batch populates it).
  New typed `ApiConfig` block; pins FastAPI 0.138.1 / starlette / uvicorn (venv-only → pyproject) +
  httpx (dev). Depends on F06–F11 + F12 (all DONE); consumed by F14/F05 (fill deferred routes),
  F16–F19 (frontend), F15/F20 (RQ5 concurrency + DB-outage). See `specs/F13-api-endpoints.md`
  §4.0/§6/§10. **Next: build.**
- **F12 BUILT — 2026-07-19 (master): DONE.** `app/db/` (`base` · `models` · `results` · `breaker` ·
  `session` · `repository` · `health`) + `alembic.ini` + `migrations/` (async `env.py` +
  `0001_initial_schema.py`) + `.env.example`. The 10 §8.4 tables as SQLAlchemy-2.0 async ORM models
  with portable `.with_variant` types (native JSONB/ENUM/UUID/TEXT[] on Postgres; JSON/VARCHAR-CHECK/
  CHAR on SQLite), the clean-lap CHECK, the §8.3 async circuit breaker (`DB_UNAVAILABLE`), never-raising
  `check_database` (`HEALTH_FAIL`), the `data_quality_log` JSONL→Postgres sink (append-only), and the
  `DatabaseConfig` block (DSN **env-only**). **One new code `DB_INTEGRITY`.** **53 `tests/db` tests**
  (1 `@pytest.mark.postgres` skipped offline); full suite **710 passed / 1 skipped**, `mypy app` +
  `ruff` clean. `failsafe-auditor` run on `app/db/` — 4 findings fixed with regression tests
  (unguarded `session.close()` masking the typed error; breaker missing `sqlalchemy.exc.TimeoutError`
  in its retryable set; no client-side `connect_timeout` on the write path; `render_as_batch` for
  SQLite). **Build-time refinements (divergences #26/#27):** driver is **psycopg v3**
  (`psycopg[binary]` 3.3.4 — asyncpg has no cp314 wheel, ratified like F04/F06a/F07); the async breaker
  **reimplements** F01's §8.3 state machine rather than importing it (F01's `ingest.py` pulls pandas +
  a backwards `app.db→app.pipeline` dep); the baseline migration materialises `Base.metadata` via
  **`create_all`/`drop_all`** (drift-free by construction; `alembic check` clean) rather than
  hand-written DDL — offline `--sql` mode unsupported. Divergences **#24** (`DB_INTEGRITY` code) and
  **#25** (DB ENUM domains = the code enums, drift-guarded). Next consumers: F13 (endpoints + `/health`
  + 503 mapping) / F14 (replay + live log routing) / F15+F20 (RQ5 DB-outage failsafe). See
  `specs/F12-postgres-schema.md`.
- **F12 spec — 2026-07-19 (master):** `specs/F12-postgres-schema.md` written (SPEC WRITTEN). The
  **persistence layer**: the 10 master-§8.4 tables as SQLAlchemy-2.0 async ORM models
  (`app/db/models.py`), the initial Alembic migration (`migrations/versions/0001_initial_schema.py`)
  that materialises them, the async engine/session + **§8.3 Supabase-PostgreSQL circuit breaker**
  (`timeout 5s / 3 retries / 1-2-4s backoff / open after 3-in-30s`, reusing F01's `BreakerConfig` +
  `call_with_breaker` → `DB_UNAVAILABLE`), the never-raising `check_database` health probe, and the
  `data_quality_log` **JSONL→Postgres drop-in sink** the F00 logger was built to accept. Portable
  `.with_variant` column types (JSONB/ARRAY/UUID/ENUM) let the whole unit suite run **offline on
  aiosqlite** (`create_all`) with a separate `@pytest.mark.postgres` migration-round-trip +
  no-autogenerate-drift test against docker-compose Postgres. **One new error code — `DB_INTEGRITY`**
  (constraint / app-invariant violation on write; `null_count == count(nulls)` and the clean-lap
  CHECK are the worked invariants); reuses the pre-seeded `DB_UNAVAILABLE`/`HEALTH_FAIL`/
  `DATA_QUALITY_LOG_UNWRITABLE`/`CONFIG_001`. DSN is **env-only** (`database.dsn_env_var`), never in
  YAML. **Divergences #24** (the `DB_INTEGRITY` code string — master-silent) and **#25** (DB ENUM
  domains follow the CODE enums — `confidence_state` = 5-value `ConfidenceState` incl.
  `LOW_CONFIDENCE`+`OUT_OF_DISTRIBUTION` — not §8.4's `HIGH/MEDIUM/LOW/AMBIGUOUS` paraphrase;
  precedented by F06b/F08). New typed `DatabaseConfig` block. Depends on F00 only; consumed by
  F13 (`/health` + route persistence) / F14 (replay + live log routing) / F15+F20 (RQ5 DB-outage
  failsafe test). **Top risk:** `asyncpg` may lack a Python-3.14 wheel → `psycopg[psycopg]` v3 async
  fallback (ratified-migration pattern); the offline suite is driver-independent. See
  `specs/F12-postgres-schema.md` §4.0/§6/§10. **Next: build.**
- **F09 spec — 2026-07-17 (feature/F09-corner-decomposition):** `specs/F09-corner-decomposition.md`
  written (SPEC WRITTEN). Corner-type decomposition (§4.4): each lap's corners are segmented into
  **slow (<100 m) / medium (100–200 m) / high-speed (>200 m)** by **apex radius `r = 1/max(κ)`**,
  reusing F03b's curvature primitive (`_signed_lateral_g`) + F03a's `_corner_indices` corner windows
  (so a "corner" is identical to F25's). Each corner type yields a 5-feature sub-vector
  (`apex_speed`, `lateral_g`, `throttle`, `brake`, `steering_reversal_rate`), aggregated per
  `(driver, corner_type)` into the Screen-2 **sub-fingerprints** (`corner_subfingerprints.json`,
  content-hashed/timestamp-free). Headline deliverable = the **ablation** (§13/§15 "does corner-type
  info improve the classifier?"): a self-contained F06a-CV comparison of **base (28) vs expanded
  (28 + 15)** matrices → macro-F1 delta + `IMPROVES/NEUTRAL/DEGRADES` finding (NEUTRAL/DEGRADES is a
  legitimate result), logged to `apex-ablation`; does NOT retrain the champion. **No new error code**
  — `CORNER_001` is pre-seeded (`errors.py:59`) for the X/Y-unavailable **skip** (F03b's exact
  behaviour, lap continues); reuses `FEATURE_001`/`DATA_INSUFFICIENT`/`TRAIN_LEAKAGE`/etc.
  **Divergences #18–#22** (per-corner classification; the 5 sub-features; `min_corners_per_type`;
  self-contained ablation; radius units tied to F03b's curvature). Target `app/ml/corners.py`;
  depends on F03a/F03b (geometry) + F06a (CV) + F04 (aggregate scale). See §4.0/§9.
- **F09 build — 2026-07-18 (feature/F09-corner-decomposition): DONE.** `app/ml/corners.py` +
  `tests/ml/test_corners.py` (36 tests; full suite **650 green**, mypy + ruff clean; ml-auditor +
  failsafe-auditor folded). Two build-time refinements to the spec: (a) apex radius is `r = 1/κ[apex]`
  (the speed-minimum sample's central-difference κ, robust to the boundary artefacts that dominate
  `1/max(κ)`), fallback to `1/max(κ[window])`; (b) **divergence #23 — F09 is F04-free**: the 5 corner
  sub-features are new derived quantities F04 never scaled (`normalisation.json` has no params for
  them), the ablation scores via scale-invariant XGBoost, and F19 min-max-scales the radar at render —
  so the `NORMALISATION_FALLBACK` path was dropped, `CornerHandle` carries no `normalisation_params`,
  and the cross-block validator requires only `features`. Thin cells now surface `DEGRADED` +
  `DATA_INSUFFICIENT` on the typed result; the expanded ablation matrix is NaN-native (no imputation).
  **Divergences #18–#23.**
- **F11 spec — 2026-07-17 (feature/F11-personalisation):** `specs/F11-personalisation.md` written
  (SPEC WRITTEN). Model 4 (personalisation, §7) is a **3-layer engine**: Layer 1 learned mapping (one
  XGBoost regressor per setup parameter; input = F08 **5D fingerprint** ‖ drop-first circuit one-hot),
  Layer 2 constraint engine (FIA-bound clip → `CONSTRAINT_CLIPPED`; Monaco max-downforce override →
  `CIRCUIT_OVERRIDE_MONACO`; front-wing>10°+brake<54% → `CONSISTENCY_WARNING` flag-only; confidence
  floor <0.5 → `WIDE_CONFIDENCE` ±15 % range), Layer 3 confidence scoring (per-parameter reliability =
  `1 − CV-nRMSE`, OOD-attenuated, never-NULL). Produces the **10-parameter setup vector** (§7.3), a
  SHAP decomposition per recommendation (§6.1, F07 reuse), and the §7.4 unknown-driver onboarding
  package (style-match top-3 + radar + risk flags + data-quality summary, F08 `project_unknown`
  reuse). **No new error code** — the four §7 codes were pre-seeded in `errors.py:99–102`; reuses
  `INVALID_INPUT`/`MODEL_NOT_FOUND`/`SHAP_UNAVAILABLE`/`FALLBACK_THRESHOLD`/`INFERENCE_ERROR`/etc.
  Champion bundle `artefacts/personalisation.pkl` (10 regressors + reliability + explainers), MLflow
  experiment `apex-personalisation`, F06a leakage-quarantine + reproducibility discipline.
  **Divergences #13–#17** (setup-target approximation heuristic; "best-performing" = fastest-quartile
  clean laps; categorical suspension via ordinal regression; drop-first circuit one-hot;
  regression-confidence = `1 − CV-nRMSE`) + the §14 "only ~5 parameters reliably inferrable" finding
  reported honestly (never silently narrowed). Depends on F08 (5D fingerprints) + F07 (SHAP) + F06a
  (training pattern) + F04 (aggregates). See `specs/F11-personalisation.md` §4.0/§9. **Next: build.**
- **F10 BUILT — 2026-07-17 (feature/F10-degradation-detector):** `app/monitoring/detector.py` implements
  Model 3 in full — `calibrate_stint_variation` (the one train-only fit site: quarantine → tripwire →
  26-dim standardiser → Pass A slope stats / Pass B μ-σ / Pass C pit-window medians → deterministic
  content-hashed artefact → MLflow), `load_detector` (schema/hash/empty-thresholds/scaler-width/
  structural-knob refusals; crossing-knob mismatch → `pit_window_valid=False`), `monitor_stint`
  (never-raises; baseline → 26-dim cosine → 3-lap rolling → live-window F27/F28 slope → strict-`>`
  2-consecutive alert → exact 28-key attribution), `health`, and `lead_time_report` (§11.3). A shared
  pure `_score_stint_core` gives calibration/serving parity by construction. **72 tests** in
  `tests/monitoring/` + 5 `MonitoringConfig` tests; full suite green (**~554**), `mypy app` + `ruff`
  clean. `ml-auditor` + `failsafe-auditor` run on `app/monitoring/`; all high/medium findings fixed +
  test-covered (calibration write-fault → typed `MODEL_TRAINING_FAILED`; NaN-safe baseline `b_vec`;
  `_standardise_lap` floors at `normalise.min_std` not the module constant; never-raises pre-try escape
  on a `None` handle; per-candidate `LAP_001` rows on the `BASELINE_INSUFFICIENT` path). Two
  code-placement refinements folded into the spec (§5/§6/§7 + divergence #12): the slope-inactive DQ row
  reuses **`LAP_PARTIAL`** (keeps F03b's `FEATURE_001` count at zero, §7-11), and a single-lap `WATCH`
  crossing is the **persisted per-lap state**, not a mis-coded DQ row. `/spec-check F10` → DONE.
  See `specs/F10-degradation-detector.md`.
- **F10 spec — 2026-07-03 (apex-F10-degradation-detector):** `specs/F10-degradation-detector.md` written
  (SPEC WRITTEN), hardened by a 4-lens adversarial workflow (master-fidelity re-read of the PDF,
  leakage/reproducibility, failsafe, upstream-seams — 33 findings triaged, 2 blockers + all majors fixed).
  **Model 3 is a statistical deviation detector, not an ML model** (§5.3): per-lap cosine distance of the
  F04-normalised **then train-only-standardised** 26-feature vector (F27/F28 excluded — broadcast columns
  would double-count and leak future laps into replay) vs a first-3-clean-laps baseline (window extends
  to 5), 3-lap rolling mean, τ = μ + 2σ per driver per **circuit type** from
  `artefacts/stint_variation.json` (train-only, `_assert_no_leakage`-guarded, content-hashed,
  deterministic — **no timestamp in the artefact**), 2-consecutive-lap confirmation, F27/F28 live-window
  slopes as the §5.3 weighted component (F03b `compute_degradation_slopes` reused via a `StintLap`
  adapter, never invoked on prefixes it would NULL — no `FEATURE_001` flood). Alert payload: exact
  cosine-decomposition attribution (`1−cosθ = ½‖x̂−b̂‖²`, sums to the score), LOW/MEDIUM/HIGH band,
  pit-window median. **No new error code** — the six §5.3 codes were pre-seeded in `errors.py`; reuses
  `LAP_001` (baseline-exclusion/skip flag), `INVALID_INPUT`, `MODEL_NOT_FOUND`, `TRAIN_LEAKAGE`, etc.
  Review blocker fixed: F04 z-scores only 6/28 features, so a raw cosine would measure unit scale — F10
  fits a **train-only 26-dim standardiser** persisted inside the content-hashed artefact
  (**divergence #7**, the F08-#6 twin); further divergences **#8–#12** (slope formula/weight 0.25,
  confidence bands 2.5σ/3.0σ, pit-window method + stint-end proxy, SHAP-as-exact-decomposition,
  hygiene knobs + the structural-vs-crossing config-mismatch load policy). Depends on F04 + F08 — the
  deviation does **not** run in the 5D space (F08 §2's "5D baseline coordinate" paraphrase corrected:
  F08 supplies only the displayed `fingerprint_snapshot`, sparse-by-design under 30 laps). Ships
  `lead_time_report` (§11.3 in full, incl. per-circuit-type optimal threshold + FN/FP characterisation;
  Week-11 deliverable = read-only run over 2022 val-split stints), formally evaluated in F20.
  See `specs/F10-degradation-detector.md` §4.0/§9.
- **F08 aggregate scaler (Gate-2 fix) — 2026-07-02 (feature/F08-style-space):** a CONFIRMED code-review
  finding was fixed on the branch before merge: F04 z-scores only **6 of the 28** features, so the 56-dim
  driver aggregate mixed ~unit-scale columns with raw physical units spanning ~4 orders of magnitude
  (F16 gear-hold ms ≤10⁴, F12 speed variance ≤8·10³ vs 0–1 fractions), letting the largest-magnitude
  features dominate the euclidean UMAP/PCA fit — the style space would have measured unit scale, not
  style. The master is **silent** on aggregate scale (verified against the PDF: §5.2's "no refit"
  concerns the UMAP model; §4.3 is per-lap only), so F08 adds a **train-only aggregate `StandardScaler`**
  (fit inside the leakage-guarded build after the tripwire; `scale_` floored at `normalise.min_std`;
  persisted in `pca_fallback.pkl`; `scaler_sha256` cross-bundle fingerprint in `umap_model.pkl`; applied
  read-only at every forward pass via the single `_project_aggregate`/`ood_check` choke points;
  `stability_report` uses its own in-memory equivalent). Fail-loud: a pre-scaler/wrong-width bundle →
  `MODEL_NOT_FOUND` (never a silent unscaled load); a fingerprint mismatch → `UMAP_FALLBACK` degrade.
  **Master gap-fill, flagged to the supervisor as divergence #6** (spec §9.7). Same pass also pinned the
  previously-untested load-reconstruction degrade chain, the sparse-gate **signal-count** boundary, fixed
  fallback lap accounting (`n_laps_used`/`laps_rejected` honest on every path), added the
  `requirements.lock` BOM regression scan, and hardened the fit-site scan to a whole-module inverse scan.
  No config/error-code changes; aggregates persist RAW; `_risk_flags` deliberately stays in raw space
  (it z-scores per feature itself); the `PROJECTION_FALLBACK` raw-NN ranking standardises query + known
  rows through the same scaler (unscaled cosine is magnitude-dominated — §9.7 applies there too, a
  `/code-review high` catch). The same review added load-time dimension checks vs config (stale umap
  bundle → `UMAP_FALLBACK`; stale pca bundle → `MODEL_NOT_FOUND`), a stats-less-scaler typed refusal,
  backstop preservation of `sparse`/`used_global` + honest pre-fault accounting `(0, n_input)`, and
  empty (never NaN) neighbours on a non-finite query. Coordinates changed vs the raw-space fit; no
  consumer existed (F10/F11/F13/F16/F18
  unbuilt, no fitted artefact committed) — migration-free. See `specs/F08-style-space.md` §4/§6/§7/§9.7.
- **F08 spec — 2026-07-01 (apex-F08-style-space):** `specs/F08-style-space.md` written (SPEC WRITTEN),
  hardened by a 4-lens adversarial workflow (master-fidelity — 0 findings; leakage/reproducibility;
  failsafe; house-convention). **Model 2**: aggregate each driver's clean **normalised training** laps to
  a **56-dim** distribution (mean+std of the 28 features), fit **UMAP → 5D fingerprint + 2D viz** and a
  **PCA fallback**; unknown drivers project by a **forward pass** (`umap.transform`, no retrain) gated for
  sparsity (`SPARSE_PROJECTION`, <30 laps → reuses `thresholds.min_clean_laps`) and out-of-distribution
  (`OUT_OF_DISTRIBUTION`, Mahalanobis > 3 SD on any PC). **Fail-loud fallback chain:** UMAP corrupt/missing
  → PCA (`UMAP_FALLBACK`); `transform()` throws → raw-56D cosine NN (`PROJECTION_FALLBACK`); no usable
  engine → `MODEL_NOT_FOUND` (503). Depends on **F04** only (consumes `normalisation.json` read-only +
  the shared F06a training matrix; **not** the F06 classifier). Like F07, **no new error code** — the four
  §5.2 codes + the `OUT_OF_DISTRIBUTION` `ConfidenceState` are pre-seeded; reuses `TRAIN_LEAKAGE` /
  `MODEL_TRAINING_FAILED` for the two fit sites (`build_style_space` + the dissertation-only
  `stability_report`, both train-only + `_assert_no_leakage`-guarded). Adds a typed `stylespace:` block +
  its own MLflow URI (no coupling to the optional `classifier` block). Mirrors F06b's load/handle/health +
  never-raises serving. **Feeds F10** (5D fingerprint) **and F11** (nearest-neighbours / style-match /
  risk flags); stability (10-seed Procrustes < 0.05) + teammate-proximity functions are produced here,
  formally evaluated in **F20**. **Stack:** adds `umap-learn` (+`pynndescent`); no numpy movement expected
  (`numba` already caps `numpy<2.5`). Two limitations flagged for the supervisor: (1) the master fixes the
  UMAP matrix at **8 points** (small-sample → PCA-primary fallback if stability > 0.05); (2) UMAP is
  **stochastic** — coordinates reproducible within tolerance via fixed seed but `umap_model.pkl` is **not**
  byte-identical (PCA fallback is). See `specs/F08-style-space.md` §9/§10.
- **F07 DONE — 2026-07-01 (feature/F07-shap-explainability):** `app/ml/explain.py` — TreeExplainer
  (exact) over the F06 champion's **base XGBoost** (`calibrated_classifiers_[0].estimator.estimator`);
  local force + 6-domain radar + top-2 NL, global beeswarm artefact, and multi-season Jensen-Shannon
  stability. Explainability **never blocks a prediction** (SHAP fault → `SHAP_UNAVAILABLE`); reuses
  the pre-seeded `SHAP_UNAVAILABLE`/`SHAP_CONSISTENCY_WARN` codes — **no new code**. Adds a typed
  `shap:` block (`ShapConfig`). **Stack movement:** installs **SHAP 0.52**, whose `numba` dep caps
  `numpy<2.5`, so the pin moved **numpy 2.5.0 → 2.4.6** (cp314 wheel; full F01–F06 suite re-run green
  on 2.4.6 before any F07 code) — ratified like F04/F06a; back-noted in CLAUDE.md + `pyproject`.
  Gates: failsafe-auditor + ml-auditor + `/code-review high` all run, findings fixed with regression
  tests (build-global class-order keyed off `explainer.drivers`; zero-contribution domain never named;
  all-NaN season → `None` not 0.0; explainer/handle driver-order guard). See
  `specs/F07-shap-explainability.md`.
- **F06 split — 2026-06-27 (apex-F06a-classifier-training):** F06 (XGBoost classifier) is split
  into **F06a** (training / Optuna ≥200 trials / `CalibratedClassifierCV` / MLflow champion
  registration / held-out evaluation) and **F06b** (inference / confidence gates / serving the
  `PredictionResult` / startup health). **Seam = the registered champion model:** F06a produces +
  registers it; F06b loads it and never retrains. F06a is also where the real
  `artefacts/normalisation.json` first materialises (F04 shipped machinery only). F06a adds two new
  error codes — `TRAIN_LEAKAGE` (build-breaking leakage in a CV/calibration fold) and
  `MODEL_TRAINING_FAILED` (trial-scope + run-scope). The five runtime gate codes
  (`LOW_CONFIDENCE`/`AMBIGUOUS`/`INVALID_INPUT`/`PERF_WARN`/`MODEL_NOT_FOUND`) stay with F06b. The
  original combined spec remains at `specs/F06-driver-classifier.md`; F06a is
  `specs/F06a-classifier-training.md`. ml-auditor is the F06a gate (CV-fold leakage). See
  `specs/F06a-classifier-training.md`. **F06a DONE + merged 2026-06-30.**
- **F06b spec — 2026-06-30 (apex-F06a-classifier-training):** `specs/F06b-classifier-inference.md`
  written (SPEC WRITTEN), critiqued by a 3-lens adversarial workflow. F06b loads the champion +
  `normalisation.json` read-only (joblib↔joblib seam), applies the gates, returns a typed
  `PredictionResult`; **never refits**. Adds **one** new error code `INFERENCE_ERROR` (unexpected-
  internal backstop, mirrors F02 `VALIDATION_ERROR`); reuses the five §5.1 codes + `HEALTH_FAIL`.
  Adds config `classifier.confidence_high` (HIGH/MEDIUM split — master pins only the 0.65 LOW gate),
  `latency_warn_ms`, and `labels_filename`. **Two open divergences flagged for the supervisor:**
  (1) the champion is **28-feature**, not §5.1's 34-dim (circuit-type + season one-hot dropped;
  circuit context enters via F04 normalisation) — **inherited from F06a** (back-noted in F06a §9);
  (2) `confidence_high` HIGH/MEDIUM boundary is undefined by the master. See
  `specs/F06b-classifier-inference.md` §3/§9.
- **F06a labels.json seam-completion — 2026-06-30 (feature/F06b-classifier-inference):** F06a's
  champion is fit on **integer** labels (`classes_ == [0..n-1]`), so the index→driver-name map was
  unavailable at the seam. F06a's `train_classifier` now also writes `artefacts/labels.json` (the
  sorted driver names) next to `classifier.pkl` (+ `labels_artefact_path` on the result), and F06b's
  `load_model` cross-checks it against the champion's class count, refusing to serve (fail-loud
  `MODEL_NOT_FOUND`) on any mismatch. Additive seam-completion (precedent: F01 `track_status`),
  built on the F06b branch.

- **F03 split — 2026-06-26 (feature/F03a-scalar-features):** the 28-feature extraction is
  split into **F03a** (the 22 scalar-channel features F01–F18, F22–F24, F26 computable
  directly from validated telemetry, plus the shared infrastructure: the 28-key vector
  contract, `FeatureComputationError` + catch-mark-NULL-log behaviour, and the §4.5 6-NULL
  tolerance / imputation / `LAP_NULL_EXCESS` gate) and **F03b** (the X/Y-derived cornering
  features F19–F21, the corner-balance F25, and the tyre-age slopes F27–F28, which plug into
  F03a's vector without changing it). F03a adds a new `IMPUTED` error code. Downstream
  consumers (F04, F09) now depend on both. See `specs/F03a-scalar-features.md`.

- **F01 — 2026-06-26 (feature/F01-track-status):** additive change — raw per-lap
  `track_status` (FastF1 `Laps.TrackStatus`) is now persisted to the Parquet cache and
  added to the F01→F02 column contract; new `TRACK_STATUS_UNAVAILABLE` error code. See
  `specs/F01-data-ingestion.md` §11.
- **F02 — 2026-06-26 (feature/F02-lap-filtering):** DONE. Added validator
  (`app/pipeline/validate.py`), `LapStatus` enum, `ValidateConfig` block, and two F02
  error codes from review hardening: `ORPHAN_SAMPLES` (null-lap samples dropped + logged)
  and `VALIDATION_ERROR` (malformed-frame / unexpected-internal backstop, kept distinct
  from per-lap data codes). `ValidateConfig` enforces `Speed ∈ required_channels`.

- **F04 — 2026-06-26 (feature/F04-circuit-aware-normalisation):** DONE. Authored the missing
  spec (`specs/F04-circuit-aware-normalisation.md`) and shipped the **leakage-chokepoint
  machinery**: `app/pipeline/normalise.py` (one writer `fit_normaliser`, two readers
  `load_normalisation_params` / `apply_normalisation` — readers provably contain no
  `.fit`/`groupby`), the typed `NormaliseConfig` block + `NORMALISED_FEATURE_IDS` tuple
  (F09/F11/F13/F14/F19/F20), and the deterministic content-hashed artefact. Reuses the existing
  `NORMALISATION_FALLBACK` code (no new code). **No real `artefacts/normalisation.json` is
  committed** — there is no training matrix until the F06 era; this session ships the machinery
  only. The per-circuit fit floor reuses `thresholds.min_clean_laps` (30), defined per
  driver-circuit in §3.2 but applied per-circuit here (documented design decision).
