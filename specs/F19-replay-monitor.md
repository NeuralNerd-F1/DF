# Feature F19 — Screen 6: Historical Race Replay Monitor

> One spec, one session. This document is the single source of truth a session loads
> to build this feature. It must be self-contained: a session should not need the
> 42-page master spec open, only this file plus CLAUDE.md.

**Status:** SPEC WRITTEN
**Master spec sections:** §10.1 (design system, reused), §10.7 (Screen 6 — Historical Race Replay Monitor), §5.3 (F10 degradation detector — the data source), Week-19 milestone row (§ risk table p.35 — Low risk)
**Layer / Week:** Frontend / Wk 19

---

## 1. Purpose (one paragraph)
F19 builds **Screen 6 — the Historical Race Replay Monitor**: the user picks a session + driver and the screen replays the race lap-by-lap (client-side playback — Play/Pause/Speed/Scrub), animating a **deviation-score gauge**, a **tyre-degradation timeline**, and a **degradation alert banner** as if monitoring a live race, driven by the F14 `/replay` engine over the F10 detector. There is **no live-data dependency** — the replay is a whole-payload pull, fully controlled and reliable for demos/vivas (master §10.7). Like F16–F18 it is a **frontend feature** — no `config.yaml` block, no backend code, no new backend error code — and it is the most endpoint-mismatched screen yet: `/replay` forwards only a fraction of what the F10 detector computes, so several master elements (the alert banner's confidence/drivers/pit-window, the timeline's τ line, the per-lap radar) are rendered as explicit deferred states with named `ReplayLap` seams, never fabricated.

---

## 2. Upstream / downstream contracts
- **Consumes (backend, all DONE):**
  - `GET /replay/{session_id}/{driver_id}` (F14 → F10) → `ReplayResponse` (whole replay, all laps at once).
  - `GET /drivers` (F13) → `DriverSummary[]` — the driver picker (and, if `DriverSummary.id` is the DB UUID, the `driver_id` source; else a UUID input — div #77).
- **Consumes (frontend, F16a — reused):** the design system (`ds/*`), the API client (`ApiResult`/`fetchApi`), `GaugeNeedle` pattern (reference for the arc gauge), `lib/design/*` (animation timings, `confidenceTone`), and the F17/F18 vitest patterns.
- **Produces:** a new route `app/replay/page.tsx`; components under `components/replay/*`; a `lib/state/replay.ts` Zustand playback store; a `getReplay` endpoint helper + `zReplay` validator + `ReplayResponse`/`ReplayLap` type aliases. **No Pydantic result object, no backend artefact, no snapshot regen** (the generated `schema.d.ts`/`openapi.snapshot.json` already carry `ReplayResponse`/`ReplayLap`).
- **Consumed by:** F20 (evaluation — the replay is the visual RQ4/RQ5 degradation evidence; F20 also **populates the DB** so the replay serves real data), F21/F22 (write-up screenshots).

---

## 3. Inputs & outputs (typed)

F19 consumes existing backend schemas read-only and adds TypeScript + a store. No new backend model.

### 3.1 Backend schema consumed (verbatim, `app/api/schemas.py:207-216`)
```
ReplayLap      { lap_number: int,
                 feature_vector: dict[str, float|None],
                 deviation_score: float|None = None,      # rolling+slope τ-comparand; null on BASELINE/SKIPPED/PENDING/NaN
                 alert_state: str,                         # StintMonitorState: INACTIVE|BASELINE|SKIPPED|PENDING|NOMINAL|WATCH|ALERT|HALTED
                 fingerprint_snapshot: dict|null }         # {coords_5d, coords_2d, confidence_state} — NOT 6-domain radar (div #75)
ReplayResponse { laps: ReplayLap[] }                       # the ONLY field; whole replay at once
```
**Dropped by the route (the seams F19 defers — div #72–#75):** the per-lap `threshold` (τ), and the entire `DegradationAlert` (`alert_confidence`, `top_features`/`feature_attribution`, `laps_since_threshold_crossed`, `optimal_window_remaining_laps`). These exist on the F10 `StintMonitorResult`/`LapDeviation`/`DegradationAlert` (`app/monitoring/detector.py`) but `app/replay/engine.py` forwards only the 5 `ReplayLap` fields.

### 3.2 Frontend additions (F19)
- **`types.ts`:** `export type ReplayResponse = components['schemas']['ReplayResponse']`; `ReplayLap`.
- **`validators.ts`:** `zReplayLap` + `zReplay` (`.passthrough()`, guard `laps[]`, `lap_number`, `deviation_score`, `alert_state`, `fingerprint_snapshot`).
- **`endpoints.ts`:** `getReplay(sessionId, driverId)` → `fetchApi<ReplayResponse>(\`/replay/${enc(s)}/${enc(d)}\`, v.zReplay)`.
- **`lib/state/replay.ts`:** Zustand store `{ playing, speed (1|5|10), lapIndex, setPlaying, setSpeed, setLapIndex, reset }`.
- **`lib/design/animation.ts`:** `DURATION.replayTick` (base ms/lap at 1×); the radar ghost reuses `DURATION.seasonGhost` (2000) — but the per-lap radar is deferred (div #75).
- **Output contract:** every fetch stays on the four-state `ApiResult` discipline.

---

## 4. Method / algorithm

### 4.0 Global rules (inherited)
- Four data-fetch states per source (loading `Skeleton` / empty `EmptyState` / error `PageError`/`ErrorCard` / success). No silent failure.
- No re-derived confidence gate; timings from `animation.ts` (div #70). No `config.yaml` (div #70).

### 4.1 Data load + selection (`ReplayMonitorScreen`, `app/replay/page.tsx`)
- **Session/driver selection (div #77):** `/replay` takes DB **UUIDs**. There is **no session-catalog endpoint** (sessions are listed nowhere) — the session is a **UUID input**; the driver comes from `getDrivers()` if `DriverSummary.id` is the DB UUID, else a UUID input. Honest unavailable catalog (cf. F17 #57).
- Fetch `getReplay(sessionId, driverId)`. **Fresh-checkout reality:** with no artefacts/DB, the endpoint returns **503 `HEALTH_FAIL`** (monitor handle / DB not loaded) — the screen renders `PageError` honestly (the replay serves real data only once F20 populates the DB; the UI shell + all animations are exercised against fixtures in tests). 206 `REPLAY_PARTIAL` → render + a "partial replay" notice (`ApiResult.partial`). 404 `NOT_FOUND` → `EmptyState`/`ErrorCard`.

### 4.2 Playback (client-side — div #71)
- The whole replay is pre-fetched; playback is **client-side** over the `laps[]` (master §10.7 / `engine.py:8-10` — no websocket). A `useEffect` interval advances `lapIndex` at `DURATION.replayTick / speed` while `playing`; Play/Pause toggles it; Speed ∈ {1×, 5×, 10×}; the Scrub bar sets `lapIndex` directly. `useEffect` cleanup clears the timer on pause/unmount/terminal lap.

### 4.3 Deviation gauge (`DeviationGauge` — div #72)
- An arc gauge whose value = the current lap's `deviation_score` and whose **colour comes from `alert_state`** (NOMINAL→green, WATCH→amber, ALERT→red; INACTIVE/BASELINE/SKIPPED/PENDING → grey "—"). **Div #72:** it is NOT a numeric compare against τ, because `threshold` is not exposed per lap (dropped by the route) — the gauge max is the client-derived stint deviation range, labelled as such. Named seam: `ReplayLap.threshold`. `deviation_score == null` → "—" (honest, never 0).

### 4.4 Tyre-degradation timeline (`DeviationTimeline` — div #73)
- A left-building line of `deviation_score` per replayed lap (SVG, deterministic); `alert_state === 'ALERT'` laps highlighted/pulsed red; `null` scores render as gaps (not 0). **Div #73:** the master's dashed **τ threshold line is DEFERRED** (τ not exposed) — rendered as a labelled "threshold line unavailable" note, never a fabricated line. Named seam: `ReplayLap.threshold`.

### 4.5 Alert banner (`AlertBanner` — div #74)
- Fires when the current (or any reached) lap has `alert_state === 'ALERT'`: a full-width banner animates down showing the **alert lap number** and **laps-in-alert** (client-derived by counting the `ALERT` run in the state sequence). **Div #74:** the master's **alert confidence / feature-drivers / recommended-pit-window are DEFERRED** — the route drops `res.alert`; the banner shows an explicit "alert detail (confidence, feature drivers, pit window) needs the replay alert-payload seam" note, never fabricated. Named seam: forward `DegradationAlert` on the alert lap. No `ALERT` in the whole replay → no banner (honest; monitoring nominal).

### 4.6 Per-lap fingerprint radar (div #75 — deferred)
- Master §10.7 wants the fingerprint radar to update every lap with a 2 s ghost. **Div #75:** `fingerprint_snapshot` carries F08 **5D/2D coords + confidence**, not the 6-domain `radar_values` `FingerprintRadar` needs, and the system computes **no per-lap 6-domain radar**. So the per-lap radar is an explicit **deferred state** (optionally showing the per-lap `confidence_state` from the snapshot, which IS real) — never fabricated from 5D coords. Named seam: `ReplayLap.radar_values`.

### 4.7 8-driver mode (div #76 — contingency F19b)
- Master §10.7: replay all 8 drivers in a grid, each with a mini gauge + timeline + alert banner. **Div #76:** there is no batch replay endpoint — this is an **N-parallel `getReplay` fan-out** (one per `/drivers` driver), each cell an independent four-state panel. This is the **F19b split** (contingency) — the single-driver monitor (§4.1–4.6) is F19a, the shippable core.

---

## 5. config.yaml keys this feature adds
```yaml
# NONE. (Divergence #70 — continues #69/#53.)
# Frontend feature: no config.yaml block, no backend code, no new backend error code.
#   - replay-tick / playback constants live in lib/design/animation.ts (DURATION.replayTick)
#   - the getReplay endpoint + zReplay validator + ReplayResponse alias are frontend-only
#   - the generated schema.d.ts / openapi.snapshot.json ALREADY carry ReplayResponse/ReplayLap
#     (F13/F14 defined them) → no snapshot regeneration.
```

---

## 6. Error codes & failure modes (the failsafe contract)
No new code. Reuses the backend `ErrorCode` enum + the three `UI_*` codes.

| Failure mode | Detection | Error code | System response | Logged? | User-facing message |
|--------------|-----------|-----------|-----------------|---------|---------------------|
| Monitor / DB not loaded (fresh checkout) | 503 | `HEALTH_FAIL` | `PageError` + Retry | yes | "Replay not ready (no session data loaded)" |
| DB breaker open at runtime | 503 | `DB_UNAVAILABLE` | `PageError` + Retry | yes | "Database temporarily unavailable" |
| Unknown/malformed session or driver, or no laps | 404 | `NOT_FOUND` | `EmptyState`/`ErrorCard` | no | "No replay for this session + driver" |
| Truncated / non-contiguous laps | 206 | `REPLAY_PARTIAL` | render + "partial replay" notice | no (flag) | "Showing a partial replay" |
| Baseline / skipped lap (no score) | `deviation_score == null` | — | gauge "—", timeline gap | no | *(implicit)* |
| No ALERT across the replay | no `alert_state==='ALERT'` | — | no banner (nominal) | no | *(none)* |
| Alert detail (confidence/drivers/pit) | route drops `res.alert` | — | banner shows lap + laps-in-alert; details **deferred** | no | "Alert detail needs the replay alert-payload seam" |
| τ threshold line | τ not exposed per lap | — | timeline line **deferred** | no | "Threshold line unavailable" |
| Per-lap radar | `fingerprint_snapshot` is 5D coords | — | radar **deferred** (confidence shown) | no | "Per-lap radar not exposed" |
| Session catalog | no endpoint lists sessions | — | UUID input, no catalog | no | "Enter a session id" |
| Network / timeout / shape drift | fetch reject / abort / zod fail | `UI_NETWORK` / `UI_TIMEOUT` / `UI_SCHEMA_MISMATCH` | `PageError`/`ErrorCard` | yes | standard `UI_*` messages |
| 8-driver cell fails | per-driver 404/503 | (surfaced code) | that cell shows its own honest state | per code | per-cell error/empty |

---

## 7. Acceptance criteria (each item must be mechanically testable by /spec-check)
- [ ] `getReplay(s,d)` success renders the monitor; `PlaybackControls` Play advances `lapIndex`, Pause stops it, Speed ∈ {1,5,10} changes the rate, Scrub sets the lap (store-driven, testable).
- [ ] `DeviationGauge` shows the current lap's `deviation_score` and colours by `alert_state` (NOMINAL/WATCH/ALERT → green/amber/red); `deviation_score == null` → "—" (div #72).
- [ ] `DeviationTimeline` plots `deviation_score` per lap, highlights `ALERT` laps, renders nulls as gaps; the τ threshold line is an explicit **deferred** note (div #73).
- [ ] `AlertBanner` appears iff a reached lap is `ALERT`, showing the lap + client-derived laps-in-alert; confidence/feature-drivers/pit-window are **deferred** notes (div #74); no ALERT → no banner.
- [ ] The per-lap fingerprint radar renders an explicit **deferred** state (div #75) — never a radar fabricated from 5D coords.
- [ ] Session/driver selection: driver from `getDrivers()`, session via UUID input (no catalog — div #77).
- [ ] `/replay` 503 → `PageError`; 404 → `EmptyState`/`ErrorCard`; 206 → render + partial notice; `UI_NETWORK`/`UI_TIMEOUT`/`UI_SCHEMA_MISMATCH` each reachable.
- [ ] Playback interval is cleared on pause/unmount/terminal lap (no leak — edge test).
- [ ] `getReplay` + `zReplay` + `ReplayResponse`/`ReplayLap` aliases added; **no snapshot regen** (already present); a StyleSpace-style drift-guard test asserts `ReplayLap` fields.
- [ ] No new `config.yaml` block, no backend code, no new backend error code (grep-assertable); new timings only in `animation.ts` (div #70).
- [ ] Every §6 failure mode has a vitest case; `tsc --noEmit` clean, ESLint clean, `next build` compiles `/replay`, vitest green.
- [ ] *(F19b, contingency)* 8-driver mode = N-parallel `getReplay` fan-out, each cell an independent four-state panel (div #76).

---

## 8. Test plan
- **Happy-path (endpoint mocks):** a `ReplayResponse` fixture with a NOMINAL→WATCH→ALERT `alert_state` progression + `deviation_score` trace (incl. a null baseline lap) → assert gauge value+colour per `lapIndex`, timeline points + ALERT highlight, banner on the ALERT lap.
- **Playback:** store-driven — Play advances `lapIndex` (fake timers), Pause halts, Speed changes cadence, Scrub jumps; cleanup stops the timer.
- **Failure fixtures (one per §6 row):** 503 `HEALTH_FAIL`; 503 `DB_UNAVAILABLE`; 404 `NOT_FOUND`; 206 `REPLAY_PARTIAL` (partial notice); null-score lap (gap + "—"); no-ALERT (no banner); deferred alert-detail / τ-line / per-lap-radar notes present; `UI_NETWORK`/`UI_TIMEOUT`/`UI_SCHEMA_MISMATCH`.
- **Property/edge:** scrub to the last lap → playback stops; an all-null-score replay → timeline all gaps, gauge "—"; a replay with WATCH but never ALERT → amber gauge, no banner.
- **Contract (pytest):** the `ReplayLap`/`ReplayResponse` drift-guard test.
- **e2e (Playwright):** green-field (no `npx playwright install`, as F16b–F18) — documented limitation.

---

## 9. Definition of done
All §7 criteria MET with passing vitest tests; `tsc` + ESLint clean; `next build` compiles `/replay`; `tests/api` (incl. the drift guard) green; `/apex-review F19` clean (no silent failure; config-driven timings; deferred elements honest, never fabricated); INDEX row → DONE. **Limitations for the dissertation:** (1) `/replay` forwards only `deviation_score` + `alert_state` + 5D coords per lap — the alert banner's confidence/drivers/pit-window, the timeline τ line, and the per-lap 6-domain radar are deferred behind named `ReplayLap` seams (div #72–#75); (2) no session catalog → UUID input (div #77); (3) a fresh checkout 503s (no data) until F20 populates the DB — the screen renders that honestly; (4) 8-driver mode is the F19b contingency; (5) Playwright e2e green-field.

---

## 10. Risks & mitigations (from master risk register)
- **Week-19 "Low risk" (master p.35).** F19 carries no WebGL/model risk — the visualisations are SVG (deterministic, jsdom-testable).
- **`/replay` forwards a fraction of the F10 output — the alert banner (Screen 6's centrepiece) is largely deferred.** *Mitigation:* build the real parts (playback, state-coloured gauge, deviation timeline, ALERT-state banner) and render each missing element as an explicit deferred state with a **named `ReplayLap` widening seam** (div #72–#75), exactly as F16–F18 handled their landmines. The recommended follow-up (F19b or a backend PR) forwards `threshold` + the `DegradationAlert` per lap — the data already exists in `StintMonitorResult`, only the DTO drops it — turning the gauge/timeline/banner fully real.
- **Fresh checkout 503s (no data).** *Mitigation:* render the honest `PageError`; exercise all animation/playback logic against `ReplayResponse` fixtures in vitest; the real data path lights up when F20 populates the DB.
- **Playback timer leaks / runaway.** *Mitigation:* interval from `animation.ts`; `useEffect` cleanup stops on pause/unmount/terminal lap; an edge test asserts it.
- **No session catalog.** *Mitigation:* UUID input + driver-from-`/drivers`, honest "no catalog" (div #77); the named seam is a `/sessions` list endpoint.
- **Scope — 8-driver grid.** *Mitigation:* **contingency split F19a (single-driver monitor, shippable) / F19b (8-driver fan-out grid)** per the F16a/b · F17 · F18 precedent.

---

### Divergences introduced by this spec (continuing F18's #63–#69)
- **#70** — Frontend-only: no `config.yaml`, no backend code, no new backend error code (continues #69); adds `getReplay`/`zReplay`/`ReplayResponse` alias only (snapshot already carries the schema).
- **#71** — Whole-payload pull + **client-side playback** (Zustand lap position), no websocket (matches `engine.py:8-10`); Play/Pause/Speed/Scrub operate over the pre-fetched laps.
- **#72** — Deviation gauge coloured by `alert_state` (NOMINAL/WATCH/ALERT), not a numeric τ compare (`threshold` not exposed); gauge max = client stint range. Named seam `ReplayLap.threshold`.
- **#73** — Timeline plots `deviation_score` with ALERT highlighting; the master's dashed **τ threshold line is deferred** (τ not exposed). Named seam `ReplayLap.threshold`.
- **#74** — Alert banner fires on `alert_state==='ALERT'` (lap + client-derived laps-in-alert); **confidence / feature-drivers / pit-window deferred** (route drops `res.alert`). Named seam: forward `DegradationAlert` per alert lap.
- **#75** — Per-lap fingerprint radar **deferred**: `fingerprint_snapshot` is 5D/2D coords + confidence, not the 6-domain radar, and no per-lap 6-domain radar is computed — explicit deferred state, never fabricated. Named seam `ReplayLap.radar_values`.
- **#76** — 8-driver mode = **N-parallel `getReplay` fan-out** (no batch endpoint); the F19b contingency split.
- **#77** — No session-catalog endpoint: session picked by **UUID input** (driver from `/drivers`); honest unavailable catalog (cf. F17 #57). Named seam: a `/sessions` list endpoint.
