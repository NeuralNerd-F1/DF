# Feature F18 — Screen 5: Style Space Map (Three.js + D3 2D fallback)

> One spec, one session. This document is the single source of truth a session loads
> to build this feature. It must be self-contained: a session should not need the
> 42-page master spec open, only this file plus CLAUDE.md.

**Status:** SPEC WRITTEN
**Master spec sections:** §10.1 (design system, reused), §10.6 (Screen 5 — Style Space Map), §9 (Three.js r128 + D3 2D fallback), Week-18 milestone row (§ risk table p.35 — **High risk: "Three.js most complex UI component"**)
**Layer / Week:** Frontend / Wk 18

---

## 1. Purpose (one paragraph)
F18 builds **Screen 5 — the Style Space Map**: a Three.js 3D scatter of all drivers in the F08 UMAP style space (3 of the 5 UMAP dimensions), with glowing team-coloured spheres, an auto-rotating orbit camera, hover tooltips, a click-through fingerprint sidebar, teammate-proximity lines, an unknown-driver projection sphere, and a **silent D3.js 2D fallback** when WebGL is unavailable. It is the master's most visually ambitious screen and the highest-risk UI in the system. Like F16/F17 it is a **frontend feature** — no `config.yaml` block, no backend code, no new backend error code — its one dependency addition being `three`; every master element without a backing endpoint (per-season sphere positions) is rendered as an explicit deferred state, never fabricated.

---

## 2. Upstream / downstream contracts
- **Consumes (backend, all DONE):**
  - `GET /stylespace` (F13 → F08) → `StyleSpaceResponse` (per-driver `coords_2d` + `coords_5d`).
  - `GET /drivers` (F13) → `DriverSummary[]` — the **`team`** (sphere colour, reuses the #47 seam) and **`radar_values`** (tooltip/sidebar mini radar) join, keyed by `code`.
  - `POST /fingerprint/unknown` (F08 via F17's `postUnknownFingerprint`) → `UnknownFingerprintResponse` — the unknown-driver white sphere's 5D `fingerprint_vector` + `nearest_neighbours`.
- **Consumes (frontend, F16a/F17 — reused, not rebuilt):** the design system (`components/ds/*`), the API client (`ApiResult`/`fetchApi`, and the **already-built-but-unused** `getStyleSpace`/`zStyleSpace`/`zStyleDriver`/`StyleSpaceResponse` — F16a shipped these "for F18"), `FingerprintRadar` (mini radar), `lib/design/teams.ts` `teamColour()` (sphere glow), `lib/design/animation.ts` (`DURATION.styleSpace = 1200`, pre-seeded), and the vitest patterns.
- **Produces:** a new route `app/style-space/page.tsx`; components under `components/style-space/*`; a `hasWebGL()` probe util; new `three` + `@types/three` deps; new frontend animation timings. **No new API-client validator** (the style-space, drivers, and unknown-fingerprint validators all already exist). **No Pydantic result object, no backend artefact.**
- **Consumed by:** F17 (resolves F17 div #59 — the deferred style-space embed can now point at this screen), F19 (reuses camera/transition patterns where relevant), F20/F21 (evaluation screenshots + the stability/teammate-proximity visual evidence).

---

## 3. Inputs & outputs (typed)

F18 is a frontend feature: it consumes existing backend schemas read-only and adds TypeScript components + one dependency. **No new zod validator, no new backend model.**

### 3.1 Backend schemas consumed (verbatim, `app/api/schemas.py`)
```
StyleSpaceDriver   { id: str, code: str, coords_2d: float[], coords_5d: float[], season: str|null }
                   # id == code == driver name; season ALWAYS null on the wire (route drops it — div #64)
StyleSpaceResponse { drivers: StyleSpaceDriver[], unknown_drivers: StyleSpaceDriver[] }
                   # ONE row per driver; unknown_drivers ALWAYS [] (hardcoded by the route)
DriverSummary      { id, code, team: str|null, radar_values: dict[str,float], ... }   # the join source
UnknownFingerprintResponse { fingerprint_vector: float[]|null (5D), nearest_neighbours: [...], warnings: [...], ... }
```

### 3.2 Frontend additions (F18)
- **Deps:** `three` + `@types/three` in `frontend/package.json` (div #63). `d3 7.9.0` already present (the 2D fallback).
- **Components (`components/style-space/`):** `StyleSpaceScreen` (route root; fetch + join + WebGL branch), `StyleSpaceScene` (Three.js 3D), `StyleSpace2D` (D3 SVG fallback), `DriverTooltip`, `DriverSidebar`, `TeammateLinesToggle`, `SeasonFilter` (deferred state), `UnknownSphereInput` (optional projection).
- **Utils (`lib/style-space/` or `lib/design/`):** `hasWebGL()` (probe), `nearestNeighbours(code, drivers)` (client-side from `coords_5d`), `teammatePairs(drivers, teamByCode)` (intra-team cosine distance), `joinStyleWithDrivers(style, drivers)`.
- **State:** page-local React state (selected driver, teammate-lines toggle, season filter, unknown projection) — no new Zustand store required (F17 precedent).
- **Output contract:** every fetch stays on the four-state `ApiResult` discipline; the client never throws.

---

## 4. Method / algorithm

### 4.0 Global rules (inherited, non-negotiable)
- **Four data-fetch states** per data source (loading `Skeleton` / empty `EmptyState` / error `PageError`/`ErrorCard` / success) — no silent failure. **Exception, master-mandated:** the *WebGL-unavailable* fallback is deliberately **silent** (a 2D view, never an error — master §10.6 "User sees no error").
- **No re-derived confidence gate / no config-driven thresholds re-implemented in the client** (div #46). Timings from `lib/design/animation.ts`.
- **Reproducibility:** any stochastic layout jitter seeds from `LAYOUT_SEED`; the UMAP coordinates themselves come from the backend (deterministic per artefact).
- **No new backend config/code/error code** (div #69 / continues #53). New constants live in `animation.ts`; the only new dep is `three`.

### 4.1 Data load + join (`StyleSpaceScreen`, `app/style-space/page.tsx`)
- SSR (or client) fetch **both** `getStyleSpace()` and `getDrivers()`.
  - `/stylespace` 503 (`HEALTH_FAIL`, not loaded) → `PageError`. `/stylespace` 200 empty (`drivers: []`) → `EmptyState` "No style space computed yet." `UI_*` → `PageError`.
  - `/drivers` is the **enrichment** join (team colour + mini-radar): if it fails, spheres still render from `/stylespace` but degrade to **neutral colour + no mini radar** (honest — the scatter is the primary data; the join is enrichment). A `/drivers` outage (5xx) is logged by `fetchApi`.
- **Join by `code`** → each sphere carries: `coords_5d` (3D position = first 3 dims), `coords_2d` (2D fallback position), `team` (→ `teamColour()`, div #65), `radar_values` (mini radar).
- Compute **nearest-neighbours per driver client-side** from `coords_5d` (top-3 by cosine distance — div #66) and **teammate pairs** (same `team`, intra-team cosine distance → line thickness).

### 4.2 WebGL branch + silent fallback (div #68 — the core risk mitigation)
- `hasWebGL()` probes a throwaway `<canvas>.getContext('webgl2') ?? getContext('webgl')`. In jsdom this returns `null` → **false** → the D3 2D path renders (this is what vitest exercises).
- `hasWebGL()` true → `StyleSpaceScene` (Three.js). Any throw during Three.js init (or a runtime `webglcontextlost`) is caught → fall back to `StyleSpace2D`. **No error UI on fallback** (master §10.6). The Three.js module is **dynamically imported** (`next/dynamic`, `ssr:false`) so the `three` bundle never ships to the fallback/SSR path.

### 4.3 `StyleSpaceScene` (Three.js 3D — master §10.6)
- `WebGLRenderer` + `PerspectiveCamera` + `Scene`; **8 driver spheres** (`SphereGeometry` + emissive `MeshStandard`/`MeshBasic` glow in `teamColour(team)`), positioned at `coords_5d[0..2]` (3 of 5 dims).
- **Auto-rotating camera at 0.2°/s**; `OrbitControls` for click-drag (user override). Rotation speed + easing from `animation.ts`.
- **Hover** (raycaster): sphere pulses; `DriverTooltip` shows `code`, `team`, `season` (→ "—", div #64), **nearest neighbours** (client-computed), and a **mini `FingerprintRadar`** (`radar_values`).
- **Click:** camera tweens to face the sphere (`DURATION.styleSpace` = 1200 cubic); `DriverSidebar` opens with the fingerprint summary (radar + code + team; optional `nl_description` via `getFingerprint(code)`).
- **Unknown-driver white sphere:** given an `UnknownFingerprintResponse`, a white sphere animates from origin to `fingerprint_vector[0..2]` over **1.5 s** (div #67); label "Unknown Driver". If `fingerprint_vector` is `null` (`PROJECTION_FALLBACK`), it renders **unplaced** with a nearest-neighbours-only panel — never a fabricated origin position (continues F17 #59).
- **Teammate proximity lines toggle:** `LineSegments` between same-team spheres; thickness ∝ intra-team cosine similarity (thicker = more similar).
- **Season filter (deferred — div #64):** rendered as an explicit disabled control with a note ("per-season positions need a `/stylespace` season-coords seam"); it does **not** fabricate per-season motion. The backend already computes `season_coords_2d/5d` in `StyleSpaceHandle` but the route drops them — the named additive seam is `StyleSpaceDriver.season_coords_*`.

### 4.4 `StyleSpace2D` (D3 SVG fallback — the testable path)
- SVG scatter of `coords_2d`; one `<circle>` per driver, filled `teamColour(team)`, `data-testid`/`data-driver` for tests. **Same interactions** (hover tooltip, click sidebar, teammate lines, unknown sphere) rendered in 2D. Deterministic (no animation in tests via an `animate={false}` prop), exactly like `FingerprintRadar`/`DeltaRadar`. The unknown white sphere in 2D is placed via a client 2D reduction of `fingerprint_vector` (or, absent a server `coords_2d` for it, shown in an NN-only panel — div #67).

### 4.5 Contract plumbing
- **No `openapi.snapshot.json` change** unless the optional `season_coords_*` seam is built (out of scope for F18; deferred). The existing `zStyleSpace` already tolerates the empty/optional arrays.
- Extend `tests/api/test_openapi_frontend_contract.py` with a `StyleSpaceDriver`/`StyleSpaceResponse` field-membership test (guards `coords_2d`/`coords_5d`/`code` for F18) — additive, same pattern as F17.

---

## 5. config.yaml keys this feature adds
```yaml
# NONE. (Divergence #69 — continues #45/#53.)
# F18 is a frontend feature: no config.yaml block, no backend code, no new backend error code.
#   - new animation timings -> lib/design/animation.ts (cameraRotateDegPerSec, unknownDrift=1500,
#       spherePulse; season transition reuses DURATION.styleSpace = 1200, already declared)
#   - the ONE new dependency is `three` (+ @types/three) in frontend/package.json (like d3)
#   - team colour reuses lib/design/teams.ts; API base URL reuses NEXT_PUBLIC_API_BASE_URL
```

---

## 6. Error codes & failure modes (the failsafe contract)
No new code. Reuses the backend `ErrorCode` enum + the three `UI_*` codes. "Logged?" via `shouldLog()` (5xx + DB/HEALTH/`UI_*`).

| Failure mode | Detection | Error code | System response | Logged? | User-facing message |
|--------------|-----------|-----------|-----------------|---------|---------------------|
| Style space not loaded | 503 | `HEALTH_FAIL` | `PageError` + Retry | yes | "Style space not ready" |
| Empty style space | 200, `drivers: []` | — | `EmptyState` | no | "No style space computed yet" |
| `/drivers` join fails (enrichment) | 5xx / `UI_*` | (surfaced code) | spheres render **neutral, no mini radar** (honest degrade); scatter still shown | yes (5xx) | subtle "team/radar unavailable" note |
| **WebGL unavailable** | `hasWebGL()` false | — | **silent D3 2D fallback, NO error** (master §10.6) | no | *(none — 2D view)* |
| Three.js init throws / context lost | try/catch, `webglcontextlost` | — | fall back to D3 2D (silent) | yes (dev log) | *(none)* |
| Unknown-driver <5 / >5000 laps | 400 | `INVALID_INPUT` | `ErrorCard` on the projection input | no | server `detail` |
| Unknown-driver sparse / OOD | `warnings` (200) | `SPARSE_PROJECTION` / `OUT_OF_DISTRIBUTION` | white sphere + warning banner (reuse F17) | no | "Sparse (<30 laps)" / "Outside known distribution" |
| Unknown-driver projection fallback | `fingerprint_vector == null` | (in-body) | **unplaced** white sphere + NN-only panel; no fabricated origin | no | "Position unavailable — nearest matches shown" |
| Season filter (no per-season data) | route drops `season_coords` | — | explicit **deferred/disabled** control | no | "Per-season positions not yet exposed" |
| Network unreachable / timeout / shape drift | fetch reject / abort / zod fail | `UI_NETWORK` / `UI_TIMEOUT` / `UI_SCHEMA_MISMATCH` | `PageError`/`ErrorCard` | yes | standard `UI_*` messages |

---

## 7. Acceptance criteria (each item must be mechanically testable by /spec-check)
- [ ] `getStyleSpace()` success → a scatter renders **one node per driver** (D3 2D in jsdom); node count == `drivers.length`.
- [ ] **WebGL unavailable (jsdom, `hasWebGL()` false) → the D3 2D fallback renders with NO `ErrorCard`/`PageError`** (silent fallback, div #68).
- [ ] `hasWebGL()` true (mocked) → the Three.js branch (`StyleSpaceScene`, dynamically imported) is selected and mounts without throwing (three mocked).
- [ ] Sphere/node colour comes from `teamColour(team)` via the `/stylespace`×`/drivers` join on `code`; a driver missing from `/drivers` (or `team` null) → neutral colour, never a fabricated team (div #65).
- [ ] Hover → `DriverTooltip` shows `code`, `team`, `season` ("—", div #64), client-computed nearest-neighbours (div #66), and a mini `FingerprintRadar` from `radar_values`.
- [ ] Click → `DriverSidebar` opens with the driver's fingerprint summary (radar + code + team).
- [ ] `nearestNeighbours()` returns the top-3 by `coords_5d` distance (unit-tested on a fixed fixture); teammate pairs = same-`team` drivers with thickness ∝ intra-team distance (unit-tested).
- [ ] Teammate-lines toggle shows/hides the connecting lines.
- [ ] **Season filter renders an explicit deferred/disabled state** (div #64) — no fabricated per-season motion.
- [ ] Unknown-driver white sphere placed from `fingerprint_vector[0..2]`; `fingerprint_vector == null` → unplaced + NN-only, no fabricated origin (div #67); `<5` laps → `INVALID_INPUT` `ErrorCard`; sparse/OOD warnings surfaced.
- [ ] `/stylespace` 503 → `PageError`; empty → `EmptyState`; `UI_NETWORK`/`UI_TIMEOUT`/`UI_SCHEMA_MISMATCH` each reachable.
- [ ] `three` + `@types/three` added to `package.json`; the D3 fallback uses the existing `d3`.
- [ ] No new `config.yaml` block, no backend code, no new backend error code (grep-assertable); new timings only in `lib/design/animation.ts` (div #69).
- [ ] Every §6 failure mode has a vitest case that triggers it and asserts the rendered state.
- [ ] `tsc --noEmit` (strict) clean, ESLint clean, `next build` compiles `/style-space`, vitest suite green; the extended OpenAPI drift guard (StyleSpace fields) is green.

---

## 8. Test plan
- **Happy-path (endpoint mocks):** `StyleSpaceResponse` with N drivers (coords_2d/coords_5d) + a `DriverSummary[]` with teams/radar → assert D3 2D nodes, colours, tooltip, sidebar.
- **Fallback / branch:** `hasWebGL()` mocked false → D3 2D, no error; mocked true + `vi.mock('three')` → Scene mounts (smoke). A Three.js init-throw → falls back to 2D.
- **Pure-function units:** `nearestNeighbours` (fixed coords → known top-3), `teammatePairs` (same-team detection + distance ordering), `joinStyleWithDrivers` (join by code, missing-driver degrade).
- **Failure fixtures (one per §6 row):** 503 `HEALTH_FAIL`; empty 200; `/drivers` 5xx degrade; unknown `INVALID_INPUT` (400); `SPARSE_PROJECTION`/`OUT_OF_DISTRIBUTION`; `fingerprint_vector==null` fallback; `UI_NETWORK`/`UI_TIMEOUT`/`UI_SCHEMA_MISMATCH`.
- **Property/edge:** empty style space → EmptyState not a blank canvas; a driver in `/stylespace` but absent from `/drivers` → neutral node (no crash); teammate lines with a lone driver on a team → no line; season filter disabled regardless of data.
- **Contract (pytest):** the StyleSpace drift-guard test.
- **e2e (Playwright — WebGL):** the real Three.js render is **green-field** (jsdom has no WebGL; `npx playwright install` not run, as F16b/F17). A `/style-space` WebGL smoke is a stretch goal; if the browser install is unavailable, record it as a documented limitation rather than faking WebGL coverage.

---

## 9. Definition of done
All §7 criteria MET with passing vitest tests; `tsc --noEmit` + ESLint clean; `next build` compiles `/style-space`; `tests/api` (incl. the StyleSpace drift guard) green; `/apex-review F18` clean (no silent failure — except the master-mandated silent WebGL fallback; config-driven timings; no re-derived gate); INDEX row → DONE. **Limitations for the dissertation:** (1) per-season sphere positions are computed backend-side but dropped by `/stylespace`; the season filter is a deferred state pending the `season_coords_*` seam (div #64); (2) the Three.js WebGL render is exercised by e2e only (green-field if no browser install), while the D3 2D fallback + all interactions/joins are unit-tested; (3) the unknown white sphere has only a 5D coord and degrades to NN-only on the projection fallback (div #67).

---

## 10. Risks & mitigations (from master risk register)
- **Week-18 "High — Three.js most complex UI component" (master p.35).** *Mitigation:* the **D3 2D fallback is a first-class, fully-unit-tested path** (SVG, jsdom-native), and `hasWebGL()` makes the WebGL/2D choice an injectable boundary — so the screen is correct and testable even where WebGL is absent; Three.js is dynamically imported so a failure never breaks the page (silent 2D). This is the master's own design (§10.6) turned into the risk mitigation.
- **jsdom has no WebGL; no canvas mock exists.** *Mitigation:* unit-test the D3 path + branch selection + pure joins/NN/teammate; `vi.mock('three')` for a mount smoke; WebGL render → Playwright e2e (green-field, documented).
- **`three` bundle size / SSR.** *Mitigation:* `next/dynamic({ ssr:false })` dynamic import; the fallback and SSR never load `three`.
- **Master pins Three.js r128 (Dec 2020), not installable as a modern module.** *Mitigation:* pin the current stable `three` (~0.16x) + matching `@types/three`, documented as div #63 (the r128 reference is honoured in spirit — a WebGL scatter with a D3 fallback — not by an unshippable version).
- **Season filter has no backing data (route drops it).** *Mitigation:* deferred/disabled state + named additive seam `StyleSpaceDriver.season_coords_*` (the data already exists in `StyleSpaceHandle`); never fabricated (div #64).
- **Scope — F18 is the widest+riskiest screen.** *Mitigation:* frontend-only (no backend code); **contingency split** F18a (data + join + **D3 2D scatter** + all interactions + tests — the shippable core) / F18b (Three.js 3D scene + camera + WebGL), per the F16a/b · F17 precedent, if one session runs long.

---

### Divergences introduced by this spec (continuing F17's #53–#62)
- **#63** — Adds `three` + `@types/three`; master's r128 pin reconciled to the current installable `three` (~0.16x) — a WebGL scatter + D3 fallback, not an unshippable version.
- **#64** — `/stylespace` returns one row per driver with `season` always null (the route drops the backend-computed `season_coords_*`); the **season filter is a deferred/disabled state**, with `StyleSpaceDriver.season_coords_*` named as the additive seam.
- **#65** — Sphere team colour via a client-side `/stylespace`×`/drivers` join on `code` (reuses the #47 `DriverSummary.team` seam); `StyleSpaceDriver` carries no `team`.
- **#66** — Known-driver nearest-neighbours (hover tooltip) computed **client-side** from `coords_5d`; the server exposes neighbours only on the unknown-driver path.
- **#67** — Unknown white sphere placed from the 5D `fingerprint_vector` (first 3 dims); no server `coords_2d` for it, and `fingerprint_vector == null` on `PROJECTION_FALLBACK` → unplaced + NN-only, never a fabricated origin (continues #59).
- **#68** — The D3.js 2D fallback is the **silent** WebGL-unavailable path (master §10.6 "no error"); `hasWebGL()` is the injectable boundary — jsdom/vitest exercises the D3 path, `three` is mocked for a smoke test, and the WebGL render is Playwright-e2e (green-field).
- **#69** — Frontend-only: no `config.yaml` block, no backend code, no new backend error code (continues #53); the only additions are the `three` dep + frontend animation timings.
