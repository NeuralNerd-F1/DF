# Feature F05 — Manual CSV Upload Pipeline (§3.4 edge-case handling)

> One spec, one session. Self-contained: build this from this file + CLAUDE.md only.

**Status:** SPEC WRITTEN
**Master spec sections:** §3.4 (Manual Upload Pipeline — Edge Case Handling, **the sole primary
scope**: the 8-row edge-case table + UPLOAD_001–004), §3.3 (the 9-filter Lap Quality Contract the
upload re-runs — `required_channels`, 20 Hz floor, 115 % laptime, LAP_00x), §4.5 (28-key vector +
6-NULL tolerance + train-only imputation medians), §5.2 (`SPARSE_PROJECTION` <30 laps / unknown-driver
projection), §8.5 (`POST /ingest/upload` contract row: `{ upload_id, laps_accepted, laps_rejected,
cleaning_report, warnings }`, **400 on UPLOAD_001 through UPLOAD_004, 200 with warnings for non-fatal
issues**), §10.4 Screen 3 Manual Upload Tab (the frontend consumer — cleaning-report table, amber
UPLOAD_002 warning), §8.2 (all pipeline errors surface `DATA_/LAP_/UPLOAD_` codes)
**Layer / Week:** Data pipeline / Wk 16

## 1. Purpose
F05 is the **manual CSV upload path** into the APEX pipeline: a user drops a telemetry CSV and the
system runs *the same F02 filter chain + F03 feature extraction* behind eight documented pre-processing
edge cases (wrong columns, mph units, mixed sampling, corrupt rows, single-lap, unknown-driver,
duplicate laps, empty file), returning a **typed cleaning report** — never a fabricated result, never a
silent drop. It fills the one endpoint F13 still stubs as 503 (`POST /ingest/upload`) with the real
logic and unblocks the F17 Manual Upload tab (currently a live-503 shell). It is a *pre-processing +
orchestration* feature — it reuses F02/F03 wholesale and adds **no new error codes** (UPLOAD_001–004 +
`RESAMPLED` are pre-seeded).

## 2. Upstream / downstream contracts
- **Consumes (all read-only; F05 refits nothing, fabricates nothing):**
  - **F02** — `validate_session(telemetry, *, season, circuit, driver, config, log_path) →
    ValidationResult` (the 9-filter chain: green-flag/lap-type/completeness/interp/sampling-floor/
    speed-anomaly/115 %-laptime/tyre-age/30-lap-min) + `LapValidationResult` + the per-lap
    `LAP_001…LAP_006`/`LAP_INTERP`/`LAP_PARTIAL`/`COMBO_001` codes. `config.lap_validation.required_channels`
    = `[Speed, Throttle, Brake, RPM, Gear]` **is** the §3.4 UPLOAD_001 required-column set (referenced,
    not re-declared).
  - **F03a/F03b** — `compute_scalar_features` (F01–F18, F22–F24, F26) + the X/Y geometry (F19–F21, F25)
    + tyre-age slopes (F27–F28), then `assemble_feature_vector(values, *, lap_number, driver,
    circuit_type, medians, config, forced_null, flags, corner_type, …) → FeatureResult` (the 28-key
    vector + §4.5 6-NULL gate: `IMPUTED` at ≤6, `LAP_NULL_EXCESS` reject at >6, `FEATURE_001` for a NULL
    with no train median), `FEATURE_IDS`, `CORNER_001` (X/Y-absent skip → F19–F21/F25 NULL).
  - **`FeatureMedians`** (`app/pipeline/features.py`) — the **train-only** driver×circuit-type imputation
    table (`global_medians` + `driver_circuit_medians`), **injected** (constitution #2): F05 loads it
    read-only from the committed artefact seam and **never derives a median from the uploaded file**.
  - **F04** — `artefacts/normalisation.json` is **not** applied inside F05 (upload returns *raw* assembled
    vectors; per-circuit normalisation happens downstream at `/predict` / `/fingerprint/unknown`). Only
    the train-only median table is consumed here.
  - **F13** — `create_app()` / the exception handlers / `service_unavailable` / the committed
    `UploadResponse` schema (`app/api/schemas.py`) / DoS + CORS middleware / the error→HTTP mapping
    (`INVALID_INPUT`→400, `NOT_FOUND`→404).
  - `config.yaml` → new `upload:` block; **references** `lap_validation.required_channels`,
    `thresholds.sample_rate_floor_hz` (20), `thresholds.min_clean_laps` (30), `thresholds.null_ceiling`
    (6) — never re-declared.
- **Produces:**
  - `app/pipeline/upload.py` — `process_upload(raw: bytes, *, filename, driver_label, circuit, config,
    medians, log_path) → UploadResult`; the §3.4 pre-processing helpers (`_check_required_columns`,
    `_unit_warning`, `_drop_corrupt_rows`, `_dedupe_laps`, `_resample_if_inconsistent`,
    `_to_session_frame`); the typed `UploadResult` + `CleaningReport` + `LapRejection`.
  - `app/api/routes/upload.py` — the **live** `POST /ingest/upload` (multipart file) replacing F13's 503
    stub; `app/api/routes/deferred.py` retires (mirrors how F14 shrank it).
  - New typed `UploadConfig` (`app/contracts/config.py`) + the `upload:` YAML block.
  - **No new error codes; no new model artefact** (F05 orchestrates existing stages).
- **Consumed by:** **F13/F14** (route now live), **F17** (Manual Upload tab goes live), **F08**
  (`project_unknown` — the unknown-driver clean vectors feed `/fingerprint/unknown` downstream), **F20**
  (upload is a robustness/held-out ingress for the evaluation harness).

## 3. Inputs & outputs (typed)
- **Input (pipeline):** `raw: bytes` (the CSV), `filename: str`, `driver_label: str | None` (None ⇒
  unknown-driver mode), `circuit: str | None`, `config: AppConfig`, `medians: FeatureMedians`
  (train-only, injected), `log_path`. **Input (route):** `multipart/form-data` — `file` (required) +
  optional `driver_label`, `circuit`, `unknown_driver: bool`.
- **`LapRejection`** (Pydantic): `{ lap_number: int, error_code: ErrorCode, reason: str }`.
- **`CleaningReport`** (Pydantic): `{ total_rows: int, corrupt_rows_dropped: int,
  corrupt_row_fraction: float, duplicate_laps_removed: int, resampled: bool,
  resample_target_hz: int | None, missing_columns: list[str], laps_total: int, laps_clean: int,
  laps_rejected: int, single_lap: bool, unknown_driver: bool, per_lap_rejections: list[LapRejection],
  features_null_by_id: dict[str, int] }`.
- **`UploadResult`** (Pydantic `BaseResult`): `{ status: Status, error_code: ErrorCode | None,
  upload_id: str, laps_accepted: int, laps_rejected: int, cleaning_report: CleaningReport,
  warnings: list[str], clean_vectors: list[FeatureResult], unknown_driver: bool, logged: bool }`.
  `status=FAILED` + `error_code ∈ {UPLOAD_001, UPLOAD_003, UPLOAD_004, INVALID_INPUT}` on a fatal fault
  (empty `clean_vectors`); `status=OK` otherwise, **even with warnings** (UPLOAD_002/RESAMPLED/single-lap/
  SPARSE_PROJECTION are non-fatal).
- **API mapping** → `UploadResponse` (already committed): `upload_id`, `laps_accepted`, `laps_rejected`,
  `cleaning_report = report.model_dump()`, `warnings`. Fatal `error_code` → **HTTP 400** with the code
  (F13 handler); non-fatal → **200** with `warnings`. `upload_id = "up_" + sha256(raw)[:16]`
  (content-hashed ⇒ idempotent + testable; not a training artefact, so no reproducibility constraint).

## 4. Method / algorithm

### 4.0 Design notes (decisions a builder must not re-litigate)
- **(a) F05 = §3.4 only; it reuses F02/F03, never reimplements them.** The master says *"the same filter
  chain runs with additional pre-processing steps."* F05 adds the eight pre-processing steps and calls
  `validate_session` + `assemble_feature_vector` unchanged. No filter logic is duplicated.
- **(b) Synchronous, stateless (divergence — mirrors F13 `/predict` #30).** The §8.5 upload row returns
  the cleaning report **in the response** (200/400), not a `202`+poll job — the `202`/`/ingest/status`
  async pattern is the FastF1-session path (F14). A single CSV is bounded, so F05 processes it in a
  threadpool (`run_in_threadpool`, pandas is CPU-bound) and returns. **F05 writes nothing to the DB** (no
  `lap_id`/`session_id` FKs to fabricate); the resulting `clean_vectors` are handed to the caller, which
  calls the stateless `/predict` or `/fingerprint/unknown` next. `upload_id` is a content-hash handle for
  client reference/idempotency, **not** a job id.
- **(c) Uploaded CSV lacks session context → documented defaults, never silent (divergence).** The CSV
  carries `Speed/Throttle/Brake/RPM/Gear` (+ lap-number + time), but **not** `TrackStatus`, pit in/out
  events, or tyre age. `_to_session_frame` supplies **explicit** defaults so the F02 chain runs honestly:
  TrackStatus = the green token (`lap_validation.green_track_status_code`, uploaded data = racing), no
  pit events (outlap/inlap filter passes), **tyre age = UNKNOWN → F27/F28 forced NULL via `LAP_PARTIAL`**.
  These defaults are stated in the `CleaningReport` (a `finding`, §11 "documented as findings"), not
  assumed away. Consequence: an upload with **no X/Y** columns NULLs F19–F21+F25 (`CORNER_001`) *and*
  F27/F28 (single-lap) = 6 NULLs — exactly the `null_ceiling`; still accepted with `IMPUTED`.
- **(d) Train-only imputation, no leakage (constitution #2).** Step 8 imputes ≤6-NULL vectors from the
  **injected** `FeatureMedians` (train-only). `app/pipeline/upload.py` contains **no `.fit()`, no
  `.median()`/`.mean()` over the uploaded frame for imputation** — a whole-module scan is a review gate.
  A NULL with no train median → `FEATURE_001` lap-fail (honest), never a fabricated fill.
- **(e) `upload.required_columns` mirrors `lap_validation.required_channels`.** §3.3 and §3.4 name the
  identical set `[Speed, Throttle, Brake, RPM, Gear]`. F05 **references** the F02 list for the UPLOAD_001
  check (single source of truth) rather than re-declaring it.
- **(f) 50 Hz resample target ≠ 20 Hz floor.** §3.4 "resample to 50 Hz via linear interpolation" is the
  *target rate* for **inconsistent** uploads (`RESAMPLED` flag); §3.3's 20 Hz is the *floor* below which a
  lap is excluded. Distinct quantities: `upload.resample_target_hz` (50) vs
  `thresholds.sample_rate_floor_hz` (20). A cross-block validator asserts `resample_target_hz ≥
  sample_rate_floor_hz`.
- **(g) Never raises for a data fault (constitution #1).** `process_upload` returns a typed `UploadResult`
  on every path; a malformed/oversize/non-CSV file → `INVALID_INPUT` typed result (route → 400), a
  parser exception is caught and coded, and every drop/dedup/resample/rejection writes a
  `data_quality_log` row.

### 4.1 The 8 §3.4 edge cases (verbatim detection · response)
| Edge case | Detection (master §3.4) | System response (master §3.4) | Code/flag | Fatal? |
|---|---|---|---|---|
| **Wrong column names** | "Check for exact column name match: Speed, Throttle, Brake, RPM, Gear required." | "Return error **UPLOAD_001** with list of missing columns. Do not proceed." | `UPLOAD_001` | ✅ 400 |
| **Wrong units (speed in mph)** | "If Speed median > 350, likely mph not km/h. Flag for user confirmation." | "Return warning **UPLOAD_002**: 'Speed values appear to be in mph. Convert before uploading?'" | `UPLOAD_002` | ✖ 200+warn |
| **Mixed sampling rates** | "Compute inter-sample time delta. If std > 20 % of mean: inconsistent sampling." | "Resample to 50 Hz via linear interpolation. Tag all laps with **RESAMPLED** flag." | `RESAMPLED` | ✖ 200 |
| **Corrupt rows (non-numeric)** | "Any non-numeric value in numeric columns detected on read." | "Drop row. If >5 % of rows corrupt: return error **UPLOAD_003**. If <5 %: log and continue." | `UPLOAD_003` | ✅ 400 iff >5 % |
| **Single-lap upload** | "Row count insufficient for tyre age slope features (F27, F28)." | "Accept upload. Compute F01–F26 only. Return partial fingerprint with explicit note: 'Tyre degradation features require multiple laps.'" | `LAP_PARTIAL` | ✖ 200+note |
| **Unknown driver — no label** | "User selects Unknown Driver mode explicitly." | "Run full pipeline without driver label. Project into style space. Return fingerprint with uncertainty bounds and nearest-neighbour matches." | (→ F08 `SPARSE_PROJECTION`/OOD) | ✖ 200 |
| **Duplicate lap numbers** | "Lap number column contains repeated values." | "De-duplicate by keeping first occurrence. Log duplicate count." | (dedup, logged) | ✖ 200 |
| **Empty file** | "Zero data rows after header." | "Return error **UPLOAD_004** immediately. Do not proceed." | `UPLOAD_004` | ✅ 400 |

### 4.2 `process_upload` step order
1. **Guard + parse.** `len(raw) > upload.max_file_bytes` → `UploadResult(FAILED, INVALID_INPUT)` (route
   400, "file exceeds N bytes", no silent truncation). `pd.read_csv` (utf-8); a parser error is caught →
   `INVALID_INPUT`. `> upload.max_rows` → `INVALID_INPUT`. `upload_id = "up_" + sha256(raw)[:16]`.
2. **UPLOAD_004 empty.** Zero data rows after header → `UploadResult(FAILED, UPLOAD_004)` immediately, do
   not proceed.
3. **UPLOAD_001 columns.** `required_columns ⊄ df.columns` (exact, case-sensitive) →
   `UploadResult(FAILED, UPLOAD_001, cleaning_report.missing_columns=[…])`, do not proceed.
4. **UPLOAD_002 units.** `df["Speed"].median() > upload.mph_speed_median_kmh` → append warning
   `"UPLOAD_002: Speed values appear to be in mph. Convert before uploading?"` (non-fatal, continue).
5. **UPLOAD_003 corrupt rows.** Coerce numeric columns; a row with any non-numeric in a numeric column is
   corrupt. `frac = corrupt / total`. `frac > upload.max_corrupt_row_fraction` → `UploadResult(FAILED,
   UPLOAD_003)`. Else drop the corrupt rows, log the count, continue.
6. **Dedupe laps.** Duplicate values in `upload.lap_number_column` → keep first occurrence, record
   `duplicate_laps_removed`, log.
7. **Sampling consistency → RESAMPLED.** Per lap, Δt from `upload.time_column`; if `std(Δt) >
   upload.sampling_delta_std_fraction × mean(Δt)` → resample the lap to `upload.resample_target_hz` by
   linear interpolation, set `resampled=True`, tag `RESAMPLED` (report + DQ log).
8. **Adapt → F01 wide frame** (`_to_session_frame`, §4.0-c defaults) → run **F02 `validate_session`**
   (`season`/`circuit`/`driver` = provided or placeholder; single lap ⇒ `single_lap=True`) →
   `ValidationResult`. Then per clean lap run **F03** and `assemble_feature_vector(medians=<train-only>,
   forced_null=["F27","F28"] if single_lap, …)` → `FeatureResult` (IMPUTED ≤6 / `LAP_NULL_EXCESS` >6).
   Collect `clean_vectors`; rejected laps → `per_lap_rejections` with their `LAP_00x`/`FEATURE_001` code.
9. **Warnings.** `driver_label is None` or `unknown_driver` ⇒ `unknown_driver=True`; `laps_clean <
   thresholds.min_clean_laps` ⇒ warning `SPARSE_PROJECTION` (unknown-driver projection is downstream F08);
   `single_lap` ⇒ note "Tyre degradation features require multiple laps."
10. **Assemble** `UploadResult` (counts + `CleaningReport` + `warnings`), `status=OK`, `logged=True`.

## 5. config.yaml keys this feature adds
```yaml
# --- F05 manual CSV upload pipeline (§3.4 edge-case handling; §8.5 POST /ingest/upload) -----------
# The UPLOAD_001 required-column set is lap_validation.required_channels ([Speed,Throttle,Brake,RPM,
# Gear]) — §3.3 and §3.4 name the same 5; it is REFERENCED there, never re-declared. The 20 Hz
# post-interpolation FLOOR is thresholds.sample_rate_floor_hz (§3.3); the 50 Hz below is the RESAMPLE
# TARGET for inconsistent uploads (§3.4) — a different quantity. No new error code (UPLOAD_001–004 +
# RESAMPLED are pre-seeded in errors.py).
upload:
  mph_speed_median_kmh: 350          # §3.4 UPLOAD_002 — Speed median > this (km/h) -> 'appears to be mph' warning
  max_corrupt_row_fraction: 0.05     # §3.4 UPLOAD_003 — > this fraction corrupt -> error; at/below -> drop+log+continue
  resample_target_hz: 50             # §3.4 RESAMPLED — linear-interpolation resample target for inconsistent sampling
  sampling_delta_std_fraction: 0.20  # §3.4 — inter-sample dt std > this x mean -> inconsistent sampling -> resample
  max_file_bytes: 10485760           # DoS guard (10 MiB) — reject before parse (INVALID_INPUT), no silent truncation
  max_rows: 500000                   # DoS guard — reject before parse (INVALID_INPUT)
  lap_number_column: LapNumber       # §3.4 duplicate-lap dedup + per-lap grouping key
  time_column: Time                  # inter-sample dt source for the sampling-consistency check
```
New typed `UploadConfig` in `app/contracts/config.py` (standing rule — feature config lives in
`config.py`, `extra="forbid"`), wired as `AppConfig.upload: UploadConfig | None = None`. Validators:
`mph_speed_median_kmh`/`max_file_bytes`/`max_rows`/`resample_target_hz` > 0; `0 < max_corrupt_row_fraction
< 1`; `sampling_delta_std_fraction > 0`; non-empty `lap_number_column`/`time_column`; **cross-block**
`resample_target_hz ≥ thresholds.sample_rate_floor_hz` (target at/above the floor).

## 6. Error codes & failure modes (the failsafe contract)
F05 **adds no error codes** — it reuses the pre-seeded set. All propagate as typed results (never a
raise); every fatal + every drop/dedup/resample/rejection writes a `data_quality_log` row.

| Failure mode | Detection | Error code | System response | Logged? | User-facing message |
|--------------|-----------|-----------|-----------------|---------|---------------------|
| Missing required column | `required_channels ⊄ columns` (exact match) | `UPLOAD_001` | `UploadResult(FAILED)`; route **400**; `missing_columns` listed; do not proceed | yes | "Missing required columns: {list}. Upload not processed." |
| Speed likely mph | `Speed.median() > mph_speed_median_kmh` | `UPLOAD_002` | warning appended; **200**; processing continues | yes | "Speed values appear to be in mph. Convert before uploading?" |
| Inconsistent sampling | `std(Δt) > 0.20·mean(Δt)` per lap | `RESAMPLED` (flag) | resample to 50 Hz via linear interp; tag laps | yes | "Laps resampled to 50 Hz (inconsistent source sampling)." |
| >5 % corrupt rows | non-numeric in numeric cols; `frac > 0.05` | `UPLOAD_003` | `UploadResult(FAILED)`; route **400**; do not proceed | yes | "More than 5 % of rows are corrupt ({pct} %). Upload rejected." |
| ≤5 % corrupt rows | `frac ≤ 0.05` | (none — logged) | drop corrupt rows; **200**; continue | yes | "{n} corrupt rows dropped ({pct} %)." |
| Single-lap upload | `laps_total == 1` | `LAP_PARTIAL` | accept; F27/F28 NULL; partial note | yes | "Tyre degradation features require multiple laps." |
| Unknown driver / no label | `driver_label is None` ∨ `unknown_driver` | (→ F08) | full pipeline, no label; downstream projection | — | "Processed as unknown driver — projecting into style space." |
| <30 clean laps | `laps_clean < min_clean_laps` | `SPARSE_PROJECTION` (warn) | **200**; projection flagged sparse (F08) | yes | "Fewer than 30 clean laps — projection is sparse." |
| Duplicate lap numbers | repeated `lap_number_column` values | (none — logged) | de-dup, keep first; **200** | yes | "{n} duplicate laps removed (kept first occurrence)." |
| Empty file | zero data rows after header | `UPLOAD_004` | `UploadResult(FAILED)`; route **400**; immediate | yes | "File contains no data rows. Upload rejected." |
| Oversize / non-CSV / unparseable | `> max_file_bytes` / `> max_rows` / parser error | `INVALID_INPUT` | `UploadResult(FAILED)`; route **400**; no partial read | yes | "File too large or not a valid CSV." |
| ≤6 NULL after imputation | `null_count ≤ 6`, train median exists | `IMPUTED` (flag) | impute from train-only medians; lap accepted | yes | (per F03a §4.5) |
| >6 NULL / no train median | `null_count > 6` / no median | `LAP_NULL_EXCESS` / `FEATURE_001` | lap rejected → `per_lap_rejections`; **200** for the upload | yes | "Lap {n} rejected: {code}." |
| Combination <30 laps | F02 30-lap floor | `DATA_INSUFFICIENT`/`COMBO_001` | per F02 (surfaced in report) | yes | (per F02) |

## 7. Acceptance criteria (each item must be mechanically testable by /spec-check)
- [ ] A CSV missing any of `[Speed, Throttle, Brake, RPM, Gear]` → `UploadResult(status=FAILED,
      error_code=UPLOAD_001)` with `cleaning_report.missing_columns` populated; the route returns **400**;
      **no laps processed** ("do not proceed").
- [ ] A CSV with `Speed` median `> upload.mph_speed_median_kmh` → `status=OK`, a **UPLOAD_002** warning in
      `warnings`, and processing **continues** (route 200).
- [ ] A CSV whose per-lap inter-sample Δt std `> 0.20 × mean` → laps **resampled to
      `upload.resample_target_hz` (50)**, `cleaning_report.resampled=True`, `RESAMPLED` logged; route 200.
- [ ] A CSV with `> upload.max_corrupt_row_fraction` (5 %) corrupt rows → `UploadResult(FAILED,
      UPLOAD_003)`, route **400**; a CSV with `≤5 %` corrupt → rows **dropped + logged**,
      `corrupt_rows_dropped` counted, `status=OK`.
- [ ] A single-lap CSV → `cleaning_report.single_lap=True`, **F27/F28 NULL** on the accepted vector
      (`LAP_PARTIAL`), and the single-lap note in `warnings`; the lap is **accepted**.
- [ ] An upload with `driver_label=None` (or `unknown_driver=True`) → `unknown_driver=True`; `<30` clean
      laps additionally yields a **SPARSE_PROJECTION** warning.
- [ ] A CSV with repeated `lap_number_column` values → duplicates removed keeping the first,
      `duplicate_laps_removed` counted + logged; `status=OK`.
- [ ] An empty CSV (header only, zero data rows) → `UploadResult(FAILED, UPLOAD_004)` **immediately**,
      route **400**, before any other check.
- [ ] An oversize (`> max_file_bytes`) / non-CSV / unparseable file → `INVALID_INPUT` typed result, route
      **400**, **no partial read**.
- [ ] Accepted laps run the **real** `validate_session` + `assemble_feature_vector`; a ≤6-NULL vector is
      **IMPUTED from injected train-only `FeatureMedians`**, a >6-NULL vector is `LAP_NULL_EXCESS`-rejected
      into `per_lap_rejections`.
- [ ] `POST /ingest/upload` (multipart) returns the committed `UploadResponse` shape (`upload_id,
      laps_accepted, laps_rejected, cleaning_report, warnings`); F13's 503 stub is **removed**.
- [ ] **No `.fit()` and no median/mean over the uploaded frame for imputation** anywhere in
      `app/pipeline/upload.py` (train-only-medians leakage guard, constitution #2).
- [ ] Every error code above has a unit test that triggers it and asserts the typed result **and** a
      `data_quality_log` row; `process_upload` **never raises** for a data fault.
- [ ] All thresholds read from `config.yaml` `upload:` (or the referenced shared thresholds); **no magic
      numbers**; `upload_id` is the content hash; `mypy app` + `ruff` clean.

## 8. Test plan
- **Happy-path fixtures** (`tests/pipeline/test_upload.py`): a well-formed multi-lap CSV (all 5 required
  columns + `LapNumber` + `Time`, consistent 50 Hz, km/h speeds) → `status=OK`, `laps_accepted > 0`,
  clean `cleaning_report`, empty `warnings`. A synthetic `FeatureMedians` is injected (the failsafe-harness
  recipe: `FeatureMedians(global_medians=dict.fromkeys(FEATURE_IDS, 0.5))`) so imputation runs offline.
- **Failure fixtures — one per row of §4.1 / §6:** missing-column CSV → UPLOAD_001; high-median-speed CSV
  → UPLOAD_002 warning; jittered-Δt CSV → RESAMPLED; 8 %-corrupt CSV → UPLOAD_003 / 3 %-corrupt →
  drop+continue; single-lap CSV → LAP_PARTIAL + note; no-label upload → unknown_driver; `<30`-lap upload →
  SPARSE_PROJECTION; duplicate-`LapNumber` CSV → dedup; header-only CSV → UPLOAD_004; 11 MiB / binary blob
  → INVALID_INPUT.
- **Route tests** (`tests/api/test_upload_route.py`): multipart POST → 200 `UploadResponse` on the happy
  path; **400 + the exact code** for UPLOAD_001/003/004 + INVALID_INPUT; 200 + `warnings` for UPLOAD_002;
  the deferred stub is gone (a `GET`/import check). Reuse F13's `create_app` + `dependency_overrides` to
  inject a fake `FeatureMedians`.
- **Property/edge:** exactly-6-NULL vector (no X/Y + single-lap) accepted with IMPUTED (boundary);
  exactly-7-NULL rejected LAP_NULL_EXCESS; corrupt fraction exactly `0.05` (≤ ⇒ continue — boundary);
  `Speed.median()` exactly `350` (`>` strict ⇒ no warning); a NULL with no train median → FEATURE_001 (no
  fabricated fill); `resample_target_hz == sample_rate_floor_hz` passes config validation, `<` fails;
  `content_hash` stable across two uploads of identical bytes.
- **Leakage scan:** an AST/string scan of `app/pipeline/upload.py` asserts no `.fit(`, `.median(`,
  `.mean(` on the uploaded frame (the ml-auditor / apex-review gate).

## 9. Definition of done
All acceptance criteria MET with passing tests (offline — injected `FeatureMedians`, no committed
artefacts, no DB), `/apex-review F05` clean on the contract checklist (no silent failure; train-only
imputation; config-driven thresholds; each code tested), `/spec-check F05` DONE, `mypy app` + `ruff`
clean, the F13 OpenAPI drift-guard still green (the committed `UploadResponse` is unchanged), INDEX row →
DONE. Run `failsafe-auditor` + `ml-auditor` on `app/pipeline/upload.py` + `app/api/routes/upload.py`.
Record in the dissertation limitations: (1) upload is **synchronous + stateless** — it returns the
cleaning report and hands `clean_vectors` to the downstream `/predict`/`/fingerprint/unknown`, writing
nothing to the DB (divergence, mirrors F13 `/predict` #30); (2) the CSV lacks session context, so
TrackStatus/pit/tyre-age use **documented defaults** (all-green, no-pit, tyre-age-UNKNOWN → F27/F28 NULL,
§4.0-c); (3) the §3.4 UPLOAD_002 detection (`Speed median > 350 km/h ⇒ mph`) is **copied verbatim** as a
contract though it reads physically inverted (mph values are *smaller* than km/h) — flagged for supervisor
review; (4) the unknown-driver projection + nearest-neighbour matches happen in **F08** via a follow-on
`/fingerprint/unknown` call, not inside F05.

## 10. Risks & mitigations (from master risk register where relevant)
- **Imputation leakage from the uploaded file** (the top risk) → §4.0-d: medians are **injected**
  train-only; `upload.py` has no `.fit()`/`.median()` on the upload; a module scan is a review gate. This
  is the constitution-#2 chokepoint for the upload path.
- **CSV lacks the context the 9-filter chain expects** → §4.0-c: `_to_session_frame` supplies *documented*
  defaults surfaced in the `CleaningReport`; nothing is silently assumed — an all-green default is stated,
  not hidden.
- **Malicious / huge / non-CSV upload (DoS)** → `max_file_bytes` + `max_rows` guards reject **before**
  parse (`INVALID_INPUT`), reusing F13's DoS-cap posture; no unbounded read, no silent truncation.
- **UPLOAD_002 threshold reads inverted** → copied verbatim (spec discipline: contracts are copied, not
  corrected), flagged as a divergence for the supervisor rather than silently "fixed".
- **Scope creep into F08/F13** → §4.0-b/§2 draw the boundary: F05 cleans + extracts + returns the report
  and fills the one route; projection/prediction stay in the already-built downstream endpoints.
- **Reproducibility** → `upload_id` is a content hash; the cleaning report is deterministic given the same
  bytes + config + injected medians (no timestamp, no RNG).
```

