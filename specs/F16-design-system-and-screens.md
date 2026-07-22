# Feature F16 — Design System + Screen 1 (Command Centre) + Screen 2 (Driver Profile)

> One spec, one session. Self-contained: build this from this file + CLAUDE.md only.

**Status:** SPEC WRITTEN — split into **F16a** (design system + Screen 1 Command Centre; BUILT + merge-ready)
and **F16b** (Screen 2 Driver Profile; next session). See §12 for the F16a/F16b map.
**Master spec sections:** §9 (Technology Stack — frontend pins, pp.24–26), §10 preamble (Animated UI
ethos, p.26), §10.1 (Global Design System — tokens, loading/error states, p.26), §10.2 (Screen 1 Command
Centre, pp.26–27), §10.3 (Screen 2 Driver Profile, p.27), §6 (SHAP radar / beeswarm / 6 domains / NL
generation, p.16), §5 + §5.2 (confidence-gate + Model-1/Model-2 uncertainty states & user messages,
pp.11–14), §8.3 (API circuit breakers → 503, p.20), §8.5 (API endpoint contracts + status codes, pp.22–24),
§13 (roadmap Week 17 deliverable, pp.34/37), §12 (risk register — WebGL fallback row, p.38).
**Layer / Week:** Frontend / Wk 17
**Depends on:** **F13** (the FastAPI serving layer — every read endpoint Screens 1 & 2 consume, the
`ErrorResponse` envelope, the `ConfidenceState` enum, the `/openapi.json` schema the TS types are generated
from, the CORS allow-list which already whitelists `http://localhost:3000`) — **DONE**. Indirectly renders
the outputs of F06b (`/predict` — not on Screens 1/2), F07 (`shap_radar` / `nl_description`), F08 (5D
fingerprint / `/stylespace` — Screen 5, not F16), F10 (degradation — no Screen 1/2 feed), F11 (`/setup`
10-parameter recommendation), F09 (corner sub-fingerprints — see the binding gap in §4.2). **F16 talks to
all of these ONLY over HTTP; it imports no Python, loads no artefact, calls no `.fit()`.**

## 1. Purpose
F16 stands up APEX's **frontend**: a Next.js 14 / TypeScript-strict app that implements the master §10.1
**design system** in Tailwind (dark-only, F1-TV/MoTeC visual language) and the first two of the six screens
— **Screen 1 Command Centre** (an 8-driver grid of live animated fingerprint radars, sortable, circuit-aware)
and **Screen 2 Driver Profile** (one driver's full animated radar + SHAP beeswarm + multi-season overlay +
10-parameter setup panel). It is the first embodiment of non-negotiable #1 in the browser: **the UI never
fails silently** — every request has an explicit skeleton/empty/error state, every backend `error_code` is
shown verbatim in monospace, every model uncertainty state (`LOW_CONFIDENCE`, `AMBIGUOUS`,
`OUT_OF_DISTRIBUTION`, `WIDE_CONFIDENCE`, `SHAP_UNAVAILABLE`) is rendered as an honest visual state, and
nothing is ever fabricated to fill a gap.

## 2. Upstream / downstream contracts
- **Consumes (all read-only over HTTP; F16 imports no backend module):**
  - **F13 API** — the endpoints in §4.2; the uniform `ErrorResponse { error_code, detail, flags[] }`
    envelope (`app/api/schemas.py`); `ConfidenceState` = `HIGH | MEDIUM | LOW_CONFIDENCE | AMBIGUOUS |
    OUT_OF_DISTRIBUTION` (`app/contracts/enums.py`); the `/openapi.json` document (served while
    `api.docs_enabled: true`) from which **all** TS request/response types **and** the `ErrorCode` union are
    **generated** (never hand-written) — this is how "error codes are an Enum, strings never hardcoded"
    (constitution) survives into TypeScript.
  - The 6 radar domains (structural, not config): **throttle, braking, speed, gear, engine, cornering**
    (master §6; `app/ml/explain.py DOMAIN_FEATURES`). The frontend receives these as `radar_values` /
    `shap_radar` dict keys and must not re-derive them.
  - `NEXT_PUBLIC_API_BASE_URL` (env-only) — the API origin. Never hardcoded.
- **Produces:** a new **`frontend/`** tree (greenfield — the repo has no frontend today):
  - **Tooling:** `frontend/package.json`, `tsconfig.json` (`"strict": true`, `noImplicitAny`),
    `next.config.js`, `tailwind.config.js` (the §10.1 tokens — the design-system single source of truth),
    `postcss.config.js`, `.env.example` (`NEXT_PUBLIC_API_BASE_URL`), an OpenAPI→TS codegen step.
  - **App Router:** `app/layout.tsx` (root, dark theme, Inter + JetBrains Mono), `app/page.tsx` (Screen 1),
    `app/drivers/[code]/page.tsx` (Screen 2), `app/globals.css`, an `error.tsx` + `not-found.tsx` boundary.
  - **Design-system primitives** (`components/ds/`): `Card`, `Skeleton` (telemetry-wave SVG), `ErrorCard`
    (red left border + monospace code + retry), `ConfidenceBadge`, `ConfidenceBar`, `Button`, `Tabs`.
  - **D3 chart components** (`components/charts/`): `FingerprintRadar` (6-axis, animated), `ShapBeeswarm`,
    `GaugeNeedle` (+ range-bar variant), `HorizontalBars` (hover SHAP top-3). D3.js 7.x, deterministic.
  - **Screen components:** `components/command-centre/` (`DriverGrid`, `DriverCard`, `CircuitSelector`,
    `SortControls`) and `components/driver-profile/` (`ProfileRadar`, `SeasonSwitcher`, `OverlayMode`,
    `ShapBeeswarmPanel`, `CornerTabs`, `NlDescription`, `SetupPanel`).
  - **Lib:** `lib/api/` (typed fetch client + generated `types.ts` + response runtime-validation + error
    mapping), `lib/state/` (Zustand stores: driver selection, circuit filter, sort key), `lib/design/`
    (`tokens.ts`, `animation.ts` — durations/staggers/easing, `confidence.ts` — state→colour map).
  - **Tests:** `frontend/**/__tests__` (Vitest + React Testing Library) + `frontend/e2e/` (Playwright over
    MSW-mocked API). Optionally `tests/api/test_openapi_frontend_contract.py` (pytest drift guard, §8).
  - **No new backend error code, no new `config.yaml` block, no model artefact.** (See divergence #45.)
- **Consumed by:** **F17** (Screens 3–5 reuse the design system, chart primitives, API client, and Zustand
  stores), **F18/F19** (Screen 6 replay reuses the same), **F20/F21** (screenshots / evaluation write-up).

## 3. Inputs & outputs (typed)
F16 has no Python DTO surface. Its "types" are the **generated** TS mirrors of the F13 schemas plus a small
frontend-only view-model layer. The load-bearing shapes:

- **`ErrorCode`** *(generated from OpenAPI)* — the exact backend union (`app/contracts/errors.py`). Rendered
  verbatim in the error state. The frontend **adds none**.
- **`UiErrorCode`** *(new, frontend-only — `UI_NETWORK | UI_TIMEOUT | UI_SCHEMA_MISMATCH`)* — for failures
  that occur **before/around** the API boundary and therefore have no server `error_code` (network
  unreachable, request timeout, a response that fails runtime schema validation). Namespaced `UI_*` so it
  can never be confused with a backend code; rendered in the **same** monospace error-state pattern
  (divergence #45). This is the only "code" F16 introduces and it lives in the frontend, not the Python enum.
- **`ApiResult<T>`** *(new, internal)* — the fetch-client return: a discriminated union
  `{ ok: true, data: T, flags: ErrorCode[], status: 200|206 }` | `{ ok: false, code: ErrorCode | UiErrorCode,
  detail: string, status: number }`. The client **never throws into a component** and **never returns raw
  `any`** — a component always receives a typed result and renders the matching state (§4.0-a).
- **Generated response types consumed:** `HealthResponse`, `DriverSummary` (`id`, `code`, `radar_values`),
  `FingerprintResponse` (`vector_5d`, `shap_radar`, `nl_description`, `stability`), `StabilityResponse`
  (`stability_score`, `pairwise_js`, `seasons_available`, `partial`), `SetupResponse` +
  `SetupParameterSchema` (`value`, `value_range`, `rating` ∈ HIGH/MEDIUM/LOW/WIDE, `confidence`, `fia_min`,
  `fia_max`, `constraint_flags`, `shap_top_features`, `shap_available`), `CompareResponse`,
  `StyleSpaceResponse`. (Screens 1/2 do not use `/predict` — that is Screen 3, F17.)

## 4. Method / algorithm

### 4.0 Design notes (decisions a builder must not re-litigate)
- **(a) No silent failure in the browser (constitution #1 → the RQ5 frontend evidence).** Every data-fetch
  boundary resolves to exactly one of four explicit UI states: **loading** (animated skeleton with the
  telemetry-wave SVG — *never a spinner*, §10.1), **empty** (a 200 with an empty array → a purpose-built
  empty state, *not* an error — `/drivers` and `/stylespace` are "200 always, `[]` if no data", §8.5),
  **error** (any non-2xx or client-side failure → the §10.1 pattern: red left border on the affected card,
  the `error_code` in monospace, a Retry button where the call is retryable), and **success**. There is no
  fifth "blank/undefined" state: the typed `ApiResult` union makes an unhandled branch a compile error under
  `strict`. Errors are surfaced **per affected card/panel**, so one failed request never blanks the screen.
- **(b) The frontend NEVER re-derives the confidence gate (constitution #4).** The backend resolves
  `confidence_state` server-side using `thresholds.confidence_gate: 0.65` and `classifier.confidence_high:
  0.85`; those constants are **not** echoed in any response. The UI colours the confidence bar from the
  **categorical** `confidence_state` — `HIGH → green`, `MEDIUM → amber`, `LOW_CONFIDENCE → red`, `AMBIGUOUS →
  red` — and sizes the fill from the **numeric** `confidence`. It does **not** re-threshold the number at
  0.80/0.65. This deliberately supersedes the master's numeric colour thresholds (§10.4: green >0.80 / amber
  0.65–0.80 / red <0.65), which would make the UI disagree with the backend's own label in the 0.80–0.85
  band (the master wrote 0.80 before F06b pinned `confidence_high` at 0.85). **The state the API assigned is
  authoritative; the UI reflects it, never re-computes it.** (Divergence #46.) The `state→colour` map lives
  once in `lib/design/confidence.ts` — no `0.65`/`0.85` literal appears in any component.
- **(c) Types + error codes are GENERATED from the API's OpenAPI schema (constitution — no hardcoded
  strings).** A build step (`openapi-typescript` against `${NEXT_PUBLIC_API_BASE_URL}/openapi.json`, or a
  committed snapshot) emits `lib/api/types.ts` containing every request/response model **and** the
  `ErrorCode` union. Components import these — they never hand-type an endpoint shape or an error-code
  string. A pytest drift guard (§8) asserts the committed OpenAPI snapshot still carries the fields the
  frontend reads, so a backend rename fails CI, not the browser.
- **(d) SSR for initial data, CSR for animation (§9 Next.js row).** Server Components fetch the first paint
  (the `/drivers` grid, a driver's `/fingerprint`) so the page is populated on load; Client Components own
  every animation and interaction (D3 transitions, hover, tabs, Zustand state). The API base URL is read
  from `NEXT_PUBLIC_API_BASE_URL` on both sides; nothing is hardcoded.
- **(e) No magic numbers (constitution).** Design tokens (the §10.1 hex values, fonts) live **only** in
  `tailwind.config.js`; animation durations/staggers/easing (`80ms`, `100ms`, `600ms`, `800ms`, `1200ms`,
  `2000ms`, `easeCubicInOut`) live **only** in `lib/design/animation.ts`; chart geometry constants live with
  their chart. No component contains a raw timing or colour literal, and there are **no inline styles**
  (§9 Styling; enforced by an ESLint rule). Dark theme is the only theme.
- **(f) Reproducibility (constitution #3) applies to stochastic rendering.** The SHAP beeswarm's dot jitter
  / any D3 force layout uses a **fixed seed** (`d3.randomLcg(seed)`), so a given dataset renders
  byte-identically every time — snapshots are stable and the view is deterministic. No `Math.random()` in
  layout.
- **(g) F16 is frontend-only and refits nothing.** No `.fit()`, no Python import, no artefact load anywhere
  in `frontend/`. Stated so `/apex-review` ticks it off. (Any *additive backend seam* taken to unblock a
  binding gap in §4.2 is the sole, explicitly-flagged exception and is read-only.)

### 4.1 Design system (master §10.1 — copied verbatim into `tailwind.config.js`)
| Token | Value | Use |
|---|---|---|
| `background` | **`#0D0D0D`** | near-black page background, all screens |
| `accent` (primary) | **`#E8002D`** | F1 red — headings, active states, alert indicators |
| `accent-deep` (secondary) | **`#C0392B`** | deep red — SHAP force values, warnings |
| `text-primary` | **`#F0F0F0`** | near-white body text |
| `text-secondary` | **`#888888`** | mid-grey labels / metadata |
| `card` | **`#1A1A1A`** | card surface (lighter than background) |
| `border` | **`#2B2B2B`** | subtle separation |
| font (UI) | **Inter** | all UI text |
| font (mono) | **JetBrains Mono** | telemetry values, lap numbers, feature values, **error codes** |

Motion (→ `lib/design/animation.ts`): easing **D3 `easeCubicInOut`** for all chart transitions; **600ms**
default chart transition; **1200ms** style-space (F17, declared here for reuse); Screen-1 load bloom **80ms**
per-axis stagger / **800ms** total; circuit change **600ms**; Screen-2 radar load **100ms** per-axis
stagger; season-ghost hold **2000ms** before fade. **Loading** = skeleton + telemetry-wave SVG, **no
spinners**. **Error** = red left border + monospace `error_code` + Retry. *Note:* the master gives **no**
numeric spacing scale and **no** team-colour palette (only "team colour bar" / `drivers.team`); F16 defines a
standard Tailwind spacing scale and derives the team-colour bar from a small committed `team → hex` map keyed
by the driver's team (documented in §10; a gap the master leaves open).

### 4.2 Master UI element → backing endpoint (the binding contract, incl. the honesty gaps)
Every Screen-1/2 element the master lists, mapped to the F13 endpoint that feeds it. Where **no** endpoint
exists, the resolution is **honest** — derive from existing data, or render an explicit unavailable state —
**never fabricate** (constitution #1).

| Screen · element | Feed (F13) | Binding status |
|---|---|---|
| S1 · 8-driver grid + per-card radar | `GET /drivers` → `radar_values` | **direct.** 200-always; `[]` → empty grid |
| S1 · fingerprint stability indicator | `GET /drivers/{code}/stability` → `stability_score` | **direct.** 206 if <3 seasons → "partial" |
| S1 · hover top-3 distinctive SHAP | `GET /drivers/{code}/fingerprint` → `shap_radar` | **direct** (top-3 by |value|) |
| S1 · circuit compatibility score | *(no endpoint)* | **proxy (div #47):** mean per-param `confidence` from `GET /drivers/{code}/setup/{circuit}`, labelled a proxy; 404 → honest "—", never a number |
| S1 · sort by compatibility / stability | derived from the two above | **direct** |
| S1 · sort by degradation sensitivity | *(no per-driver scalar)* | **disabled + tooltip (div #50)** until a backend scalar exists; no fabricated order |
| S1 · LOW_CONFIDENCE card encoding | `confidence_state` on the fingerprint feed | **direct** (§4.0-b) |
| S2 · large 6-axis radar + labels/scores | `GET /drivers/{code}/fingerprint` → `shap_radar` | **direct** |
| S2 · tooltip "comparison to field mean" | computed client-side from `GET /drivers` | **derived (div #51):** per-domain mean over the 8 drivers |
| S2 · NL description | `fingerprint.nl_description` | **direct**; updates on season/corner change; fades out/in |
| S2 · season switcher 2022/23/24 + ghost | `fingerprint` per season + `stability` | **direct** (per-season fingerprint) |
| S2 · multi-season overlay + JS divergence | `stability.pairwise_js` | **direct** |
| S2 · setup panel (10 gauges) + WIDE range-bar | `GET /drivers/{code}/setup/{circuit}` → `parameters[]` | **direct.** `rating==WIDE` / `value_range` → range bar |
| S2 · SHAP beeswarm (per-lap dots) | *(no endpoint — F13 exposes only aggregated `shap_radar`)* | **honest empty state (div #48)** built + animated to a documented per-lap-SHAP contract; renders "per-lap explanations unavailable" until an additive seam serves it |
| S2 · corner tabs Slow/Medium/Fast | *(no endpoint — F09 `corner_subfingerprints.json` is an artefact, not served)* | **honest empty state (div #49)** same treatment as beeswarm |

For #48/#49 the component is fully built and animated against a precise contract (so wiring it later is a
one-line data swap), but with **no data it shows the explicit unavailable state** — exactly F13's "honest
503 stub, never faked" precedent. The unblocking backend seam (a small read-only `GET
/drivers/{code}/laps-shap` and `GET /drivers/{code}/corners`) is **out of F16 scope**, flagged for the
supervisor; if approved it becomes an additive seam on the F16 branch (precedent: F01 `track_status`, F06a
`labels.json`, F14's seam note).

### 4.3 Screen 1 — Command Centre (master §10.2)
Landing screen. `app/page.tsx` (Server Component) SSR-fetches `GET /drivers`; the `DriverGrid` (Client)
hydrates 8 `DriverCard`s. Each card: driver **code**, **team-colour bar** (§4.1 map), animated **6-axis
radar**, **circuit-compatibility proxy** (§4.2 #47), **stability indicator** (§4.2). Behaviours:
- **Load bloom:** all 8 radars animate together; each axis draws outward in sequence, **80ms** stagger,
  **800ms** total, `easeCubicInOut`.
- **Circuit selector** (top nav, Zustand `circuitFilter`): change → all 8 radars transition to the new
  circuit's fingerprint over **600ms**, and the compatibility proxies refetch per circuit.
- **Hover:** card expands **inline** (no navigation) showing the **top-3 distinctive SHAP features** as
  horizontal bars animating from zero.
- **`LOW_CONFIDENCE`:** dashed border + warning badge; radar still shown at **reduced opacity**; message
  "Insufficient telemetry signal for confident identification. Top candidates shown." (§5 Model-1 table).
- **Click:** card-expand transition (scale + fade to fill) → route to `app/drivers/[code]` (Screen 2).
- **Sort controls:** compatibility / stability (working) + degradation-sensitivity (disabled, §4.2 #50);
  cards reorder with a layout animation.
- **Empty:** `/drivers` → `[]` renders the empty-grid state (not an error). Any card whose feed errors shows
  the §10.1 error card in place, leaving the others live.

### 4.4 Screen 2 — Driver Profile (master §10.3)
`app/drivers/[code]/page.tsx` SSR-fetches `GET /drivers/{code}/fingerprint`; split layout — **left** =
`ProfileRadar` + metadata, **right** = `ShapBeeswarmPanel` + season timeline. Behaviours:
- **Large 6-axis radar:** each axis labelled with domain + current score; **hover tooltip** = domain,
  score, NL meaning, **comparison to field mean** (§4.2 #51); load animates axis-by-axis, **100ms** stagger.
- **Season switcher** (2022/2023/2024): switch animates the radar; a **ghost** of the previous season stays
  at reduced opacity for **2000ms** then fades. 206 (missing season) → that tab shows a "partial" note.
- **SHAP beeswarm:** dots animate from the centre line, each a lap, **colour-coded by circuit**; hover =
  lap number, session, predicted confidence. Deterministic jitter (§4.0-f). No per-lap feed yet → explicit
  unavailable state (§4.2 #48). `shap_available==false` on the source → the panel degrades gracefully
  ("explanation unavailable"), radar/NL unaffected (`SHAP_UNAVAILABLE`, §6.2).
- **Multi-season overlay mode:** toggle overlays all three seasons on one radar (solid / dashed / dotted);
  the **JS divergence** score is shown between each pair (`stability.pairwise_js`).
- **Corner tabs Slow / Medium / Fast:** re-animate the radar for that corner-class sub-fingerprint — honest
  unavailable state until served (§4.2 #49).
- **NL description:** from the top-2 SHAP domains; updates on season/corner change; text fades out then in.
- **Setup panel (below radar):** 10 parameters, each an **animated gauge needle** settling to the
  recommended value; a **`WIDE_CONFIDENCE`** parameter (`rating==WIDE` / `value_range` present) renders a
  **range bar** instead of a point needle; `constraint_flags` (`CONSTRAINT_CLIPPED`,
  `CIRCUIT_OVERRIDE_MONACO`, `CONSISTENCY_WARNING`) shown as badges; 404 → "no setup data for this circuit".

### 4.5 API client & error mapping (`lib/api/`)
One typed `fetchApi<T>(path): Promise<ApiResult<T>>`: reads `NEXT_PUBLIC_API_BASE_URL`; on network failure
→ `{ok:false, code: UI_NETWORK}`; on timeout → `UI_TIMEOUT`; on non-2xx parses `ErrorResponse` →
`{ok:false, code: error_code, detail}` (422 uses Pydantic's `{detail:[…]}` shape — mapped with the field
detail); on 2xx **validates the body against the generated schema** and, on mismatch, returns
`UI_SCHEMA_MISMATCH` (never renders unvalidated data); 206 is a success carrying a `partial` flag. `/metrics`
is **never** called from the browser (403 `FORBIDDEN`, §6). The client **never throws** and **never leaks
`any`**.

## 5. config.yaml keys this feature adds
**None.** F16 is a frontend feature and adds **no `config.yaml` block and no backend code** (divergence #45).
Its configuration surface is entirely in the `frontend/` tree:
```text
tailwind.config.js   → the §10.1 design tokens (colours, fonts) — the design-system single source of truth
lib/design/animation.ts → all durations/staggers/easing (80/100/600/800/1200/2000 ms, easeCubicInOut)
lib/design/confidence.ts → the confidence_state → colour map (no 0.65/0.85 literal anywhere; div #46)
frontend/.env.example → NEXT_PUBLIC_API_BASE_URL (the API origin; env-only, never committed with a value)
```
The one **backend** touch is **deferred to deploy, not built here**: the production Vercel origin must be
added to `api.cors_origins` in `config.yaml` (currently `["http://localhost:3000"]`) before public exposure
— a one-line ops change, noted so it is not forgotten.

## 6. Error codes & failure modes (the failsafe contract)
F16 adds **no backend error code**; it **renders** the codes the API already returns and adds only the
frontend-local `UI_*` set for pre-API failures. Every row below is a UI state, mechanically testable.

| Failure mode | Detection | Error code | System response (UI) | Logged? | User-facing message |
|--------------|-----------|-----------|----------------------|---------|---------------------|
| Any endpoint returns 5xx / 500 backstop | non-2xx status | `API_ERROR` / `HEALTH_FAIL` *(reuse, F13)* | error card on the affected panel (red left border + monospace code + **Retry**); rest of screen live | client log | `detail` verbatim + "Retry" |
| DB down on a DB-backed read | 503 | `DB_UNAVAILABLE` *(reuse)* | error card + Retry; `/health` banner if global | client log | "Database temporarily unavailable." |
| Driver not found | 404 | `NOT_FOUND` *(reuse)* | Screen 2 not-found state; card removed from grid | no | "Driver not found." |
| Partial data (missing seasons / <3) | 206 | *(status, no code)* | render available data + a "partial data" marker; season tab noted | no | "Partial data — some seasons missing." |
| Empty dataset | 200 + `[]` | *(none — success)* | purpose-built **empty grid / empty profile** state (not an error) | no | "No driver data yet." |
| Low-confidence fingerprint | `confidence_state==LOW_CONFIDENCE` | `LOW_CONFIDENCE` *(reuse)* | dashed border + warning badge + reduced-opacity radar | no | "Insufficient telemetry signal for confident identification. Top candidates shown." |
| Ambiguous prediction | `confidence_state==AMBIGUOUS` | `AMBIGUOUS` *(reuse)* | red confidence colour + ambiguity badge | no | "Input data does not match F1 telemetry patterns." |
| Wide-confidence setup parameter | `rating==WIDE` / `value_range` set | `WIDE_CONFIDENCE` *(reuse)* | **range bar** instead of point needle | no | "Estimated range (limited data)." |
| SHAP missing for a panel | `shap_available==false` | `SHAP_UNAVAILABLE` *(reuse)* | beeswarm/NL panel degrades to "explanation unavailable"; radar/prediction unaffected | no | "Explanation temporarily unavailable." |
| Stale cached data | `STALE_DATA` in `flags[]` | `STALE_DATA` *(reuse)* | subtle "stale" indicator; data still shown | no | "Showing cached data." |
| Cache (Redis) down | (transparent) | `CACHE_UNAVAILABLE`/`CACHE_DEGRADED` *(reuse)* | **no UI error** — possibly slower only | no | (none — invisible per §8.2) |
| Compatibility proxy unavailable | `/setup` 404 | `NOT_FOUND` *(reuse)* | card shows honest **"—"**, never a fabricated score (div #47) | no | (dash) |
| Per-lap SHAP / corner data not served | endpoint absent | *(UI empty state; div #48/#49)* | beeswarm / corner tab shows explicit "unavailable" state, never fabricated dots | no | "Per-lap explanations not yet available." |
| Network unreachable | fetch rejects | `UI_NETWORK` *(new, frontend-only)* | error card + Retry | client log | "Cannot reach APEX API." |
| Request timeout | abort after timeout | `UI_TIMEOUT` *(new, frontend-only)* | error card + Retry | client log | "Request timed out." |
| Response fails schema validation | runtime validation of a 2xx body | `UI_SCHEMA_MISMATCH` *(new, frontend-only)* | error card; **no render of malformed data** | client log | "Unexpected data format." |

## 7. Acceptance criteria (each item must be mechanically testable by /spec-check)
- [ ] **Design system:** every §10.1 token (`#0D0D0D`, `#E8002D`, `#C0392B`, `#F0F0F0`, `#888888`,
      `#1A1A1A`, `#2B2B2B`, Inter, JetBrains Mono) is present in `tailwind.config.js`; **no inline styles**
      anywhere (ESLint `react/forbid-dom-props`-style rule green); dark theme is the only theme.
- [ ] **Loading = skeleton (telemetry-wave SVG), never a spinner** (no spinner component exists); **error =
      red left border + monospace `error_code` + Retry** on the affected card; **empty = purpose-built empty
      state**, distinct from error.
- [ ] **Confidence colour comes from `confidence_state`** (`HIGH→green / MEDIUM→amber /
      LOW_CONFIDENCE+AMBIGUOUS→red`), fill from numeric `confidence`; **no `0.65`/`0.85` literal in any
      component** (grep-clean); the map lives once in `lib/design/confidence.ts` (div #46).
- [ ] **`LOW_CONFIDENCE`** on a Screen-1 card renders dashed border + warning badge + reduced-opacity radar.
- [ ] **Screen 1:** 8-card grid from `GET /drivers`; radars bloom on load with **80ms** axis stagger /
      **800ms** total / `easeCubicInOut`; circuit selector transitions all radars over **600ms**; hover
      expands a card inline with top-3 SHAP horizontal bars; click routes to the Driver Profile; sort by
      compatibility & stability reorders with a layout animation; **empty `/drivers` → empty grid**;
      degradation-sensitivity sort is present-but-disabled with a tooltip (div #50).
- [ ] **Screen 2:** split layout; 6-axis radar with per-axis label+score, hover tooltip (domain, score, NL
      meaning, field-mean comparison), **100ms** load stagger; season switcher 2022/23/24 with a **2000ms**
      ghost; overlay mode (solid/dashed/dotted) + `pairwise_js`; corner tabs Slow/Medium/Fast; NL fades
      out/in on change; setup panel of 10 gauge needles with **`WIDE_CONFIDENCE` → range bar**; beeswarm
      dots colour-coded by circuit with hover (lap/session/confidence) and **deterministic** jitter.
- [ ] **API client:** base URL from `NEXT_PUBLIC_API_BASE_URL`; every non-2xx → a typed `ApiResult` error
      (nothing swallowed, nothing thrown into a component); 404 / 206 / empty each handled; a 2xx body
      failing schema validation → `UI_SCHEMA_MISMATCH` (no malformed render); the browser never calls
      `/metrics`.
- [ ] **Types + `ErrorCode` are generated from `/openapi.json`** — no hand-written endpoint shape or
      error-code string; the pytest drift guard (§8) passes.
- [ ] **Honesty gaps are honest:** the compatibility proxy shows "—" on 404 (never a number); the beeswarm
      and corner tabs show explicit unavailable states (never fabricated data) (div #47/#48/#49).
- [ ] **F16 is frontend-only:** no `.fit()`, no Python import, no artefact load in `frontend/`.
- [ ] **Every failure mode in §6 has a component/integration test** (happy + error/empty/loading/partial),
      driven by MSW-mocked responses.
- [ ] **No magic numbers / no inline styles:** all timings in `lib/design/animation.ts`, all tokens in
      `tailwind.config.js`; `tsc --noEmit` clean under `strict`; ESLint clean.

## 8. Test plan
- **Tooling:** **Vitest + React Testing Library** for component/unit; **Playwright** for e2e; **MSW** (Mock
  Service Worker) to serve canned API responses — the frontend analogue of F13's `dependency_overrides`
  fakes. No live API, no committed artefacts required; the suite runs fully offline.
- **Happy path:** MSW fixtures for `/drivers` (8), `/drivers/{code}/fingerprint`, `/stability`, `/setup`,
  `/stylespace`. Assert: grid renders 8 cards + radars; Screen 2 renders split layout, radar axes, season
  tabs, setup gauges, NL text.
- **Failure fixtures (one per §6 row):** 500 `API_ERROR` → error card + code + Retry; 503 `DB_UNAVAILABLE` →
  error card; 404 `NOT_FOUND` → not-found; 206 → partial marker; `[]` → empty grid; `LOW_CONFIDENCE`
  fingerprint → dashed/badge/reduced-opacity; `AMBIGUOUS` → red + badge; `WIDE_CONFIDENCE` setup param →
  range bar; `shap_available:false` → degraded beeswarm/NL; `STALE_DATA` flag → stale indicator; network
  error (MSW `res.networkError`) → `UI_NETWORK` + Retry; malformed body → `UI_SCHEMA_MISMATCH`, no render.
- **Animation/determinism:** assert components reference the `lib/design/animation.ts` constants (grep +
  unit test the values: 80/100/600/800/1200/2000 & `easeCubicInOut`); assert beeswarm layout is
  deterministic (same fixture → identical dot coordinates across two renders — the reproducibility check).
- **State/interaction (Playwright):** load bloom present; circuit change transitions all radars; hover
  expands a card with top-3 bars; click navigates to `/drivers/[code]`; season switch shows the ghost;
  overlay toggle shows three line styles + `pairwise_js`; sort reorders; disabled degradation sort shows a
  tooltip and does nothing.
- **Contract drift guard (`tests/api/test_openapi_frontend_contract.py`, pytest):** load the committed
  OpenAPI snapshot and assert it still contains the fields the frontend generates from (`DriverSummary.
  radar_values`, `FingerprintResponse.shap_radar`/`nl_description`, `SetupParameterSchema.rating`/
  `value_range`, `ConfidenceState` members, the `ErrorResponse` envelope). A backend rename fails this test.
- **Edge:** a driver with a single season (radar renders, overlay disabled honestly); a setup with all-WIDE
  params (all range bars); a fingerprint with a zero-magnitude domain (radar axis renders 0, tooltip
  "balanced"); field-mean tooltip on a 1-driver grid (mean == the driver's own value, no divide-by-zero).

## 9. Definition of done
All §7 criteria MET with passing tests (offline — MSW + Vitest + Playwright; no live API, no artefacts),
`/apex-review F16` clean on the applicable contracts (no silent failure in the UI; no re-derived confidence
gate; no hardcoded error strings/tokens/timings; frontend-only, no `.fit()`), `tsc --noEmit` + ESLint clean,
the frontend deps pinned in `frontend/package.json` (Next.js 14.x, React 18, TypeScript 5.x, D3 7.x, Zustand
4.x, Tailwind 3.x), INDEX row → DONE. Run `failsafe-auditor` (RQ5 frontend evidence) on the API-client +
error-boundary layer: confirm every fetch resolves to a typed state, no `catch {}` that swallows, no
`console.error`-and-continue that hides a failure from the user, and no fabricated data on any binding gap.
Record in the dissertation limitations: (1) the circuit-**compatibility** score is a **proxy** for a metric
the backend does not yet compute (div #47); (2) the Screen-2 **SHAP beeswarm** and **corner-type tabs** are
built and animated but render an **explicit unavailable state** until an additive backend seam serves
per-lap SHAP / corner sub-fingerprints (div #48/#49); (3) **degradation-sensitivity** sorting is disabled
until a per-driver scalar exists (div #50); (4) the master specifies **no** accessibility / responsiveness /
FPS acceptance criteria — any added are beyond-master and noted as such; (5) team-colour palette and the
numeric spacing scale are F16-defined (the master leaves both open).

## 10. Risks & mitigations (from master risk register + the two recon reports)
- **F16 is large — scaffold + design system + two complex animated screens in one session (§13 Week 17,
  Risk: Medium).** → Build strictly in the §11 order (scaffold → design system → API client/types → Screen 1
  → Screen 2); the design system + chart primitives + API client are reusable, so the marginal cost of
  Screen 2 is low. **Documented contingency (user chose a single spec):** if a session cannot hold it,
  split into **F16a** (scaffold + design system + Command Centre) and **F16b** (Driver Profile), each its own
  INDEX row — the F03a/b, F06a/b, F13-split precedent + the build guide's "too big for one session" rule.
- **Master UI elements with no backing endpoint (compatibility, per-lap beeswarm, corner tabs, degradation
  sort).** → The #1 honesty risk. Mitigation is structural (§4.2): derive honestly where possible
  (compatibility proxy, field-mean), and render **explicit unavailable states** otherwise — **never
  fabricate** (constitution #1). Each is a flagged divergence with a named additive-seam follow-up. Do
  **not** let "make the screen look complete" invent a number.
- **Confidence-gate duplication (constitution #4).** → The UI colours off the backend `confidence_state`,
  not a client-side 0.65/0.85 (div #46); the map lives once; a grep-for-literals test guards it. The
  backend stays the single source of truth for the gate.
- **Contract drift between the Python API and the TS types.** → Types + `ErrorCode` are **generated** from
  `/openapi.json`, and a pytest drift guard fails CI on a backend rename (§4.0-c, §8). No hand-maintained
  mirror to rot.
- **Silent failure creeping into the UI (the enemy the whole spec is built against).** → The typed
  `ApiResult` union makes an unhandled branch a compile error; the four-state discipline (§4.0-a) is an
  acceptance criterion; `failsafe-auditor` runs on the client layer for RQ5 evidence.
- **Non-determinism in D3 layout breaking reproducibility (constitution #3).** → Beeswarm jitter uses a
  fixed-seed LCG; no `Math.random()` in layout; a snapshot-equality test guards it.
- **Frontend toolchain sprawl / version skew with the master pins.** → Pin Next.js 14.x, React 18, TS 5.x,
  D3 7.x, Zustand 4.x, Tailwind 3.x exactly (master §9); Three.js is **not** installed here (it is Screen 5,
  F17). Vercel hosting config is deploy-time, not a build dependency.
- **No auth on the API (F13 deployment precondition).** → Out of F16 scope; the frontend talks to an
  unauthenticated dev API. Noted so a public deploy does not ship without the ingress/auth layer F13 flags.

## 11. Build order (the exact sequence a session follows)
1. **Scaffold** `frontend/` — Next.js 14 App Router, TS strict, Tailwind 3, ESLint (no-inline-style rule),
   Vitest + RTL + Playwright + MSW, the OpenAPI→TS codegen script, `.env.example`. *(→ §7 tooling criteria)*
2. **Design system** — `tailwind.config.js` §10.1 tokens; `lib/design/{animation,confidence}.ts`;
   `components/ds/*` (`Card`, `Skeleton`, `ErrorCard`, `ConfidenceBadge`, `ConfidenceBar`, `Button`, `Tabs`);
   `app/layout.tsx` + fonts + `globals.css`. *(→ §7 design-system / loading / error / confidence criteria)*
3. **API client + types** — generate `lib/api/types.ts` from `/openapi.json`; `fetchApi<T>` + `ApiResult`
   + runtime validation + `UI_*` mapping; the pytest drift guard. *(→ §7 API-client / generated-types)*
4. **Chart primitives** — `FingerprintRadar`, `HorizontalBars`, `GaugeNeedle`(+range), `ShapBeeswarm`
   (deterministic). *(→ §7 Screen-1/2 chart criteria; §4.0-f)*
5. **Screen 1 Command Centre** — `DriverGrid`/`DriverCard`/`CircuitSelector`/`SortControls`, Zustand
   `circuitFilter`/`sortKey`, load bloom, hover, LOW_CONFIDENCE, empty grid, click-through. *(→ §4.3, §7)*
6. **Screen 2 Driver Profile** — `ProfileRadar`/`SeasonSwitcher`/`OverlayMode`/`ShapBeeswarmPanel`/
   `CornerTabs`/`NlDescription`/`SetupPanel`, honest empty states for the beeswarm/corner gaps. *(→ §4.4, §7)*
7. **Failure-mode tests** — one per §6 row via MSW; determinism + drift-guard; `/apex-review F16` +
   `failsafe-auditor`; `tsc`/ESLint clean; INDEX → DONE. *(→ §7 test criteria, §9)*

## 12. Split into F16a / F16b (ratified 2026-07-22 — the §10 "too big for one session" contingency)
F16 was split at build time. This combined spec stays the source (F06 precedent); the map below says
which parts each sub-feature owns. **`/spec-check F16a` covers only the F16a column.**

**F16a — design system + Screen 1 Command Centre (BUILT, merge-ready on `worktree-F16-frontend-spec`):**
scaffold (§2 tooling), §4.0 design notes, §4.1 design system, §4.2 rows for Screen 1 (grid/stability/
hover-SHAP/compatibility-proxy #47/degradation-sort #50/LOW_CONFIDENCE), §4.3 Screen 1, §4.5 API client,
§5 config surface, §6 rows reachable from Screen 1 + the client (API_ERROR/DB_UNAVAILABLE/NOT_FOUND/206/
empty/LOW_CONFIDENCE/compat-404/UI_NETWORK/UI_TIMEOUT/UI_SCHEMA_MISMATCH), §7 criteria 1–5 + 7–11 (the
Screen-1 half of 5), §8 client + ds + Screen-1 tests + the drift guard, §11 steps 1–5. Charts built:
`FingerprintRadar`. **Delivered green:** tsc + ESLint clean, 31 vitest tests, `next build` compiles,
`tests/api`+`tests/contracts` green; backend seam `DriverSummary.team` (#47) + pytest drift guard.

**F16b — Screen 2 Driver Profile (BUILT, merge-ready on `feature/F16b-driver-profile`):** §4.2 rows for
Screen 2 (field-mean #51, per-lap SHAP beeswarm #48, corner tabs #49), §4.4 Screen 2, §7 criterion 6, §8
Screen-2 tests, §11 step 6. Charts built: **`ShapBeeswarm`** (deterministic seeded jitter §4.0-f),
**`GaugeNeedle`** (+range-bar). Components: `app/drivers/[code]/page.tsx`, `ProfileRadar` (tooltip +
field-mean), `NlDescription`, `SetupPanel`, `MultiSeasonPanel`, `ShapBeeswarmPanel`, `CornerTabs`. Reuses
everything F16a built. **9 tests; tsc + ESLint clean; `next build` compiles.** **Additional build-time
finding (I):** the API exposes **no per-season fingerprint endpoint**, so per-season radar switching/ghost/
overlay of shapes can't be data-backed — F16b renders the multi-season data that IS exposed (seasons,
stability, pairwise-JS) and surfaces the per-season shape overlay honestly as unavailable, never faked.
**Remaining:** F16b failsafe/review gates + **Playwright e2e** (needs `npx playwright install`).

**Build-time findings folded in (beyond div #45–#52):**
- **(F)** the known-driver `/drivers` + `/fingerprint` feeds expose **no per-driver `confidence_state`** —
  so the Screen-1 LOW_CONFIDENCE treatment is built + tested (via prop) but renders **confident** on live
  known-driver cards (the signal exists only for projected/unknown drivers, Screen 3 / F17). Honest, never
  faked. A future additive seam could expose it; not fabricated here.
- **(G)** there is **no `/circuits` endpoint** — the CircuitSelector uses a **static reference-data list**
  (`lib/design/circuits.ts`), held-out Belgium/Italy omitted (quarantine). Reference data, not inference.
- **(H)** `DriverSummary.team` did not exist — resolved by the **additive backend seam** (#47), merged into
  the branch and covered by the drift guard.
