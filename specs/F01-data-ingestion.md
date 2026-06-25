# Feature F01 — Data Ingestion (FastF1 / OpenF1 / Ergast)

**Status:** SPEC WRITTEN
**Master spec sections:** §3.1, §8.2 (Ingestion module), §8.3 (Circuit Breakers), Risk Register row 2
**Layer / Week:** Data pipeline / Wk 1

## 1. Purpose
Pull race telemetry and context data for the 8 drivers × 10 circuits × 3 seasons and
make it available as a local cache the rest of the pipeline reads. Every external call
is wrapped so a flaky API never hangs the system and never silently substitutes data:
on failure it returns a typed result with an error code and falls back to a defined
source or cached Parquet.

## 2. Upstream / downstream contracts
- **Consumes:** FastF1 (primary telemetry), OpenF1 REST (compound/pit/weather), Ergast
  via FastF1 (lap times, stint numbers, tyre age).
- **Produces:** `IngestResult` + local Parquet cache under `cache/<season>/<circuit>/<driver>.parquet`.
- **Consumed by:** F02 (validation) reads the cached sessions.

## 3. Inputs & outputs (typed)
- Input: `SessionRequest { season:int, circuit:str, driver:str, session_type="R" }`.
- Output:
  `IngestResult { status: OK|DATA_SOURCE_UNAVAILABLE, source_used, rows:int, compound_source: OPENF1|ERGAST|UNKNOWN, tyre_age_source: ERGAST|DERIVED, cache_path, flags:list, error_code }`.

## 4. Method
1. Resolve request -> check local Parquet cache first; use it if present and < 7 days
   old (§8.2). Else fetch from FastF1 (~50Hz: Speed, Throttle, Brake, RPM, Gear, DRS,
   X/Y).
2. Enrich with OpenF1 (~3.7Hz): tyre compound, pit in/out flags, weather, session status.
3. Cross-reference Ergast (per-lap): lap times, stint numbers, tyre age.
4. Persist raw telemetry to Parquet (the cache used when sources are unavailable).
All HTTP I/O goes through one `call_with_breaker(endpoint, ...)` helper implementing
the per-source retry/timeout/breaker policy in §6.

## 5. config.yaml keys this feature adds
```yaml
ingest:
  cache_dir: cache
  cache_max_age_days: 7            # §8.2 — use cache if fresher than this
  breakers:
    fastf1:  { timeout_s: 30, retries: 3, backoff_s: [5, 15, 30], open_after: 3, window_s: 60 }
    openf1:  { timeout_s: 10, retries: 3, backoff_s: [2, 5, 10],  open_after: 3, window_s: 60 }
    ergast:  { timeout_s: 10, retries: 2, backoff_s: [3, 8],      open_after: 2, window_s: 60 }
```

## 6. Error codes & failure modes
| Failure mode | Detection | Error code | System response | Logged? | User message |
|---|---|---|---|---|---|
| FastF1 session unavailable | fetch raises / empty | `DATA_001` | Skip session, do NOT substitute, alert operator | yes (data_quality_log) | "Session unavailable; skipped." |
| FastF1 down, cache exists | breaker open | `STALE_DATA` (flag) | Return cached Parquet | yes | served with stale flag |
| FastF1 down, no cache | breaker open + cache miss | `DATA_SOURCE_UNAVAILABLE` | Typed failure, no data invented | yes | "Telemetry source unavailable." |
| OpenF1 timeout >10s | breaker | — | Fall back to Ergast for compound | yes | none |
| OpenF1 + Ergast both fail | both breakers open | `COMPOUND_UNKNOWN` | Mark compound UNKNOWN; exclude from compound-stratified analysis ONLY; core fingerprint unaffected | yes | "Compound unknown for N laps." |
| Ergast unavailable | breaker | `DERIVED_AGE` (tag) | Derive tyre age from FastF1 pit flags; tag affected laps | yes | none |
| Any breaker trips | 3 (or 2) failures in window | — | Open circuit; subsequent calls return immediately with fallback within 500ms | yes | none |

## 7. Acceptance criteria
- [ ] Cache is consulted before any network call; cache < 7 days old is used without
      fetching.
- [ ] `DATA_001` is returned (not an exception) when a FastF1 session is unavailable,
      and the session is skipped, never substituted.
- [ ] OpenF1 timeout > 10s triggers the Ergast fallback path.
- [ ] Both-fail path yields `COMPOUND_UNKNOWN` and leaves core telemetry intact.
- [ ] Ergast-down path derives tyre age from pit flags and tags laps `DERIVED_AGE`.
- [ ] Each breaker opens after exactly its configured failure count within the window
      and returns a fallback within 500ms (no hang).
- [ ] Every failure path writes one row to `data_quality_log`.
- [ ] All timeouts/retries/backoffs read from `config.yaml`.

## 8. Test plan
- Happy path: mocked FastF1+OpenF1+Ergast returning valid frames -> `status=OK`,
  compound from OpenF1, age from Ergast.
- Failure fixtures (mock each source): FastF1 unavailable; FastF1 down + cache present;
  FastF1 down + no cache; OpenF1 timeout; OpenF1+Ergast down; Ergast down.
- Breaker test: inject 3 consecutive FastF1 failures within 60s, assert circuit opens
  and 4th call returns fallback in < 500ms with no exception.
- Cache freshness: 6-day-old vs 8-day-old cache -> used vs refetched.

## 9. Definition of done
All criteria MET with passing tests; `/apex-review F01` and `/spec-check F01` clean;
no bare excepts anywhere in `app/pipeline/ingest.py`. Limitation to record: live race
ingestion is out of scope (historical/cached only).

## 10. Risks & mitigations
- FastF1 gaps for specific driver-circuit-season combos (Risk Register row 2) ->
  combinations below 30 clean laps are dropped downstream in F02; ingestion only
  reports availability, it does not pad.
