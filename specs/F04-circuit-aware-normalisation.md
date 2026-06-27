# Feature F04 — Circuit-Aware Normalisation (train-only, saved artefact)

> One spec, one session. Self-contained: build this from this file + CLAUDE.md only.

**Status:** DONE
**Master spec sections:** §4.3 (Normalisation / circuit-aware scaling), §8.2 (`normalisation.json`
artefact + `NORMALISATION_FALLBACK`), §3.2 (held-out circuits, per-combination minimum),
Project Plan Wk 4, Risk Register (normalisation leakage)
**Layer / Week:** Data pipeline / Wk 4
**Depends on:** F03a, F03b

## 1. Purpose
F04 is APEX's **leakage chokepoint**. The 28-feature lap vector contains six features whose raw
magnitude is dominated by the circuit rather than the driver — the speed-dependent **F09**
(brake_entry_speed_mean), **F11** (speed_min_mean), **F13** (speed_p10), **F14** (speed_p90) and
the lateral-G **F19** (lateral_g_mean), **F20** (lateral_g_peak_mean). F04 **Z-score normalises
these six per circuit** so a driver's relative behaviour, not the track's top speed, reaches the
models. Scaling parameters (mean/std per circuit) are fit on the **training split only**, saved
once to `artefacts/normalisation.json`, and re-loaded — never recomputed — for val/test/upload.
The held-out circuits (Belgium, Italy) must never reach a `.fit()`: if they do, every downstream
accuracy number (F06, F08, F10, F11) is silently inflated and invalid. **The no-leakage contract
is the primary definition of done here, above all else** (non-negotiable #2).

## 2. Upstream / downstream contracts
- **Consumes:**
  - The assembled feature matrix from F03a/F03b — one row per accepted lap, the 28 canonical
    feature keys (`F01..F28`) plus, for fit, a `circuit` column and a `split` column
    (`train`/`val`/`test`). At apply time a single 28-key `feature_vector` dict + its `circuit`.
  - `config.yaml` → new `normalise:` block (the six feature IDs, `grouping_key`, `ddof`,
    `min_std`, `artefact_filename`) + `held_out_circuits` (referenced) + `thresholds.min_clean_laps`
    (30, referenced — the per-circuit fit floor; **not** re-declared).
- **Produces:**
  - `artefacts/normalisation.json` — the **only** persisted artefact: per-circuit and global
    `{mean, std}` for the six features, plus the train/excluded circuit lists and a
    `content_hash`. Written by `fit_normaliser` and nothing else.
  - `NormalisationFitResult` (fit summary) and `NormalisationResult` (per-vector apply output).
- **Consumed by:** F06 (classifier), F08 (style space), F10 (degradation), F11 (personalisation)
  — every model loads this same artefact and applies it; none of them re-fits.

## 3. Inputs & outputs (typed)
- **Fit** (train-time, the only writer):
  `fit_normaliser(laps: DataFrame, *, config: AppConfig, artefact_path: Path | None = None,
  log_path: Path | None = None) -> NormalisationFitResult`, where `laps` is the wide feature
  matrix (`circuit`, `split`, `F01..F28`). Default artefact path =
  `paths.artefacts_dir / normalise.artefact_filename`.
- **Load** (read-only):
  `load_normalisation_params(artefact_path: Path) -> NormalisationParams` — missing / invalid
  JSON / schema mismatch / content-hash mismatch → **raises** `NormalisationError`.
- **Apply** (read-only, pure transform — never re-fits, never raises):
  `apply_normalisation(feature_vector: dict[str, float | None], *, circuit: str,
  params: NormalisationParams, config: AppConfig, log_path: Path | None = None) ->
  NormalisationResult` (`config` supplies the default `data_quality_log` path, consistent with
  every other pipeline entry point).
- **Outputs:**
  - `NormalisationParams(BaseModel) { schema_version:str, normalised_features:list[str],
    grouping_key:str, ddof:int, train_circuits:list[str], excluded_circuits:list[str],
    global_:dict[str,FeatureStats], per_circuit:dict[str,dict[str,FeatureStats]],
    content_hash:str }` where `FeatureStats { mean:float, std:float }`. This is the on-disk shape.
  - `NormalisationFitResult(BaseResult) { n_train_laps:int, n_circuits:int,
    normalised_features:list[str], excluded_circuits:list[str], skipped_circuits:list[str],
    artefact_path:str, content_hash:str, status, error_code, flags, logged }`.
  - `NormalisationResult(BaseResult) { circuit:str, normalised_vector:dict[str,float|None]  # 28
    keys, normalised_features:list[str], used_global:bool, fallback_features:list[str], status,
    error_code, flags, logged }`.
- **Exception:** `NormalisationError(Exception)` carrying `ErrorCode.NORMALISATION_FALLBACK`,
  raised **only** on the fail-loud paths (leakage detected at fit; missing/corrupt artefact at
  load). `apply_normalisation` never raises — every apply-time fault is a coded
  `NormalisationResult` + a logged `data_quality_log` row.

## 4. Method / algorithm
All ops vectorised (pandas `groupby`/`agg`, numpy) — **no per-row Python loops**. The module is
split into exactly **one writer and two readers**, the architectural guarantee against leakage:

**`fit_normaliser` — the ONLY function that writes the artefact or calls `groupby`/derives any
mean/std:**
1. **Exclude held-out and non-train rows FIRST, before any reduction:**
   `train = laps[(laps["split"] == "train") & (~laps["circuit"].isin(config.held_out_circuits))]`.
2. **Fail-loud leakage assert:** `_assert_no_leakage(train, config)` raises `NormalisationError`
   if any held-out circuit survived the filter. **No artefact is written on a poisoned run.**
3. **Per-circuit params:** `train.groupby(grouping_key)[features].agg(["mean", "std"])` with the
   configured `ddof` (population std, `ddof=0`). A circuit with `< thresholds.min_clean_laps` rows
   is recorded in `skipped_circuits` and given **no** per-circuit entry — it falls back to global
   params at apply time.
4. **Global params:** mean/std of the six features over all train rows (the fallback for unknown
   or skipped circuits).
5. **Zero-variance handling (no division by zero):** for any computed `std < min_std`, substitute
   the **global** std for that feature; if the global std is itself `< min_std`, centre only
   (`std = 1.0`). Each substitution logs one `NORMALISATION_FALLBACK` row, detail `"ZERO_VARIANCE"`.
6. **Deterministic write:** build `NormalisationParams`, coerce every float through `.item()` →
   Python float, compute `content_hash = sha256` over the canonical body (the params payload with
   the hash field omitted), embed it, and write with `json.dumps(payload, sort_keys=True,
   indent=2)` — **no timestamp**. Two runs on the same inputs produce a **byte-identical** file.

**`load_normalisation_params` — read-only:** read the file; a missing file, invalid JSON, a
schema mismatch, or a `content_hash` that does not match the body → raise `NormalisationError`
(never a silent empty-params no-op). Source contains no `.fit`/`groupby`.

**`apply_normalisation` — read-only, pure transform, never raises, never fits:** for each of the
six features at `circuit`, `z = (x - mean) / std` using the per-circuit params, falling back to
global params when the circuit is unknown or was skipped (`used_global=True`,
`NORMALISATION_FALLBACK`, one DQ row, `status=DEGRADED`). A NULL among the six is left as-is, added
to `fallback_features`, logged `NORMALISATION_FALLBACK`. The other 22 keys pass through untouched.
The full 28-key vector is always returned. Source contains no `.fit`/`groupby`.

**Design decision — per-circuit fit floor:** `thresholds.min_clean_laps` (30) is documented "per
**driver-circuit** combination" (§3.2). F04 groups by **circuit** only (all drivers pooled), so it
reuses the same constant as the **per-circuit** fit floor rather than introducing a new magic
number. Documented here; config-driven; boundary-tested (30 fitted vs 29 skipped).

## 5. config.yaml keys this feature adds
A new `normalise:` block + a typed `NormaliseConfig` sub-model in `app/contracts/config.py` wired
onto `AppConfig` (standing rule: typed sub-model, `extra="forbid"`, no untyped passthrough). A new
module-level `NORMALISED_FEATURE_IDS` tuple sits beside the existing ID tuples; the validator
asserts `features` equals it exactly and that every ID is a member of `RANGE_CHECKED_IDS`.
`thresholds.min_clean_laps` (30) is **referenced**, not redefined. No new error code (uses the
existing `NORMALISATION_FALLBACK`).
```yaml
# --- F04 circuit-aware normalisation (train-only artefact, leakage chokepoint) ---
# The per-circuit minimum (min_clean_laps=30) lives under thresholds: and is referenced,
# never re-declared. Held-out circuits (held_out_circuits:) never reach a .fit().
normalise:
  features: [F09, F11, F13, F14, F19, F20]  # the 6 circuit-dependent features Z-scored per circuit
  grouping_key: circuit                      # fit mean/std per circuit
  artefact_filename: normalisation.json      # written under paths.artefacts_dir
  ddof: 0                                     # population std (numpy default)
  min_std: 1.0e-8                             # std below this → zero-variance fallback (no div-by-zero)
```

## 6. Error codes & failure modes (the failsafe contract)
| Failure mode | Detection | Error code | System response | Logged? | User-facing message |
|--------------|-----------|-----------|-----------------|---------|---------------------|
| Held-out circuit reaches the fit | held-out circuit survives the train filter | `NORMALISATION_FALLBACK` | `_assert_no_leakage` raises `NormalisationError`; **no artefact written** | n/a (build-breaking; surfaced to caller) | "Normalisation aborted: held-out circuit in training input." |
| Artefact missing / corrupt / schema- or hash-mismatch (load) | file absent / invalid JSON / bad schema / hash ≠ body | `NORMALISATION_FALLBACK` | `load_normalisation_params` raises `NormalisationError` (never a silent skip) | n/a (raised to caller) | "Normalisation params unavailable; cannot proceed un-normalised." |
| Unknown / skipped circuit at apply | `circuit` ∉ `per_circuit` | `NORMALISATION_FALLBACK` | use global params, `used_global=True`, `status=DEGRADED` | yes (circuit, "unknown circuit → global params", `fallback_used=True`) | "Circuit X not in training set; global normalisation applied." |
| Zero-variance feature at fit | `std < min_std` | `NORMALISATION_FALLBACK` | global std, or centre only (`std=1.0`); value never NaN/Inf | yes (circuit, feature, "ZERO_VARIANCE", `fallback_used=True`) | "Feature Fxx constant at circuit X; centred without scaling." |
| NULL input feature at apply | one of the six is `None` | `NORMALISATION_FALLBACK` | leave NULL, add to `fallback_features`, `status=DEGRADED` | yes (circuit, feature, "NULL input → passthrough", `fallback_used=True`) | "Feature Fxx missing; left un-normalised." |
| Circuit below per-circuit floor at fit | `< thresholds.min_clean_laps` rows | `NORMALISATION_FALLBACK` | record in `skipped_circuits`; no per-circuit entry (global at apply); `fallback_used=False` (the fallback is taken later, at apply) | yes (circuit, "SUB_MIN_CIRCUIT; skipped") | "Circuit X had <30 laps; using global normalisation." |

`NORMALISATION_FALLBACK` already exists in `ErrorCode` (§8.2); **no new code is added**.

The **User-facing message** column is *illustrative* end-user copy the API layer (F13) would
render — the contract callers branch on is the `error_code`, never the prose. The
`NormalisationError.message` and `data_quality_log` `error_detail` strings the implementation
emits are richer (developer-facing) than this copy and are intentionally not asserted verbatim by
tests. For the two fit-time conditions (zero-variance, sub-min circuit) the code is carried both on
the per-circuit `data_quality_log` row **and** on the returned `NormalisationFitResult.error_code`
(with `status=DEGRADED`) — a degraded-but-successful fit, not a hard failure.

## 7. Acceptance criteria (each item mechanically testable by /spec-check)
- [ ] **Leakage guard (primary):** `_assert_no_leakage` **raises** `NormalisationError` when a
      held-out circuit (Belgium/Italy) is present in the post-filter train set.
- [ ] **Leakage has zero effect:** adding pathological Belgium/Italy rows to the fit input yields a
      **byte-identical** `normalisation.json` vs. fitting without them (proves held-out rows never
      reach aggregation, because they are excluded before any reduction).
- [ ] **Exclusion is reported:** `NormalisationFitResult.excluded_circuits == config.held_out_circuits`.
- [ ] **Determinism:** `fit_normaliser` run twice on the same input writes **byte-identical** files
      and equal `content_hash` (sorted keys, Python floats, no timestamp).
- [ ] **No refit downstream:** after fit, the artefact's sha256 + mtime are unchanged by any number
      of `apply_normalisation` calls over val/test rows; a **static source scan** asserts the
      `load`/`apply` code contains no `.fit(` or `groupby`.
- [ ] **One writer:** only `fit_normaliser` writes the artefact / contains `groupby` / derives any
      mean/std (static source scan).
- [ ] **Round-trip:** the saved per-circuit params reproduce the in-fit `(x - mean) / std` for a
      known circuit within tolerance.
- [ ] **Unknown-circuit fallback:** `apply_normalisation` on a circuit absent from training →
      `error_code=NORMALISATION_FALLBACK`, `used_global=True`, `status=DEGRADED`, exactly one
      `data_quality_log` row, and the full 28-key vector is returned.
- [ ] **Sub-min circuit:** a circuit with `< min_clean_laps` rows appears in `skipped_circuits` and
      is normalised with global params at apply.
- [ ] **Degraded-fit signalling:** a fit with a skipped circuit or a zero-variance fallback returns
      `status=DEGRADED` + `error_code=NORMALISATION_FALLBACK`; a fully clean fit returns
      `status=OK` + `error_code=None`.
- [ ] **Fail-loud config guard:** `fit_normaliser` and `_assert_no_leakage` raise
      `NormalisationError` (never a bare `AttributeError`) when `config.normalise` is absent.
- [ ] **Zero-variance:** a constant feature at a circuit produces **no NaN/Inf** and exactly one
      logged `ZERO_VARIANCE` row.
- [ ] **NULL input:** a NULL among the six at apply is left as-is, listed in `fallback_features`,
      `error_code=NORMALISATION_FALLBACK`, logged.
- [ ] **Fail-loud load:** a missing or corrupt artefact makes `load_normalisation_params` **raise**
      `NormalisationError` (never returns empty params / silently skips normalisation).
- [ ] **Scope:** only the six features change through `apply_normalisation`; the other 22 keys are
      bit-identical to the input.
- [ ] **Boundaries (pinned both sides):** exactly `min_clean_laps` (30) rows → fitted; 29 → skipped.
      `std` exactly at `min_std` → kept; just below → zero-variance fallback.
- [ ] **Config-driven:** every constant (the six IDs, `grouping_key`, `ddof`, `min_std`,
      `artefact_filename`, the 30-lap floor) is read from `config.yaml`; a source scan asserts
      `normalise.py` has no numeric/string literal for any of them.
- [ ] `NORMALISATION_FALLBACK` is exercised by a unit test on every path that emits it.

## 8. Test plan
- **Fixtures (`tests/pipeline/conftest.py`):** a `normalise_config` fixture (real repo config with
  `paths.data_quality_log` + `paths.artefacts_dir` under `tmp_path`, mirroring `validate_config`),
  and a `feature_frame(...)` builder producing the wide fit-input frame (`circuit`, `split`,
  `F01..F28`) with knobs for circuit count, laps-per-circuit, held-out rows, a constant
  (zero-variance) feature, and hand-computable F09/F11/F13/F14/F19/F20 distributions so
  `(x - mean) / std` is checkable by hand. Reuse `read_dq` for DQ-row assertions.
- **Happy path:** fit a 2–3 circuit train frame; assert per-circuit means/stds equal the
  hand-computed values; round-trip a vector and assert the Z-score; assert the other 22 keys pass
  through unchanged.
- **Failure / boundary fixtures (one per row in §6 + each boundary in §7):** held-out-in-train
  (raises), unknown circuit (global fallback + DQ row), zero-variance feature (centred + DQ row),
  NULL input (passthrough + DQ row), sub-min circuit (skipped), missing/corrupt artefact (load
  raises), determinism (byte-equal twice), leakage-zero-effect (byte-equal with/without held-out),
  30-vs-29 lap floor, `std` at vs below `min_std`, and the two static source-scan tests
  (one-writer / readers-have-no-`.fit`).

## 9. Definition of done
All acceptance criteria MET with passing tests; `/apex-review F04` clean (no silent failure, no
normalisation leakage, config-driven thresholds, error-code discipline); `/spec-check F04` green;
`mypy app` clean; no bare `except` in `app/pipeline/normalise.py`. INDEX row `F04` set to DONE.
**Limitations for the dissertation:** (1) this session ships the **machinery only** — no real
`artefacts/normalisation.json` is committed, because no training matrix exists until the F06 era;
the artefact is produced when F06 first fits on real laps. (2) The per-circuit fit floor reuses
`thresholds.min_clean_laps` (30), defined per driver-circuit in §3.2 but applied per-circuit here —
a deliberate, documented granularity reuse, not a new constant. (3) If platform float-repr drift
ever breaks byte-determinism, values are rounded to a fixed precision before serialising
(documented fallback).

## 10. Risks & mitigations (from master risk register where relevant)
- **Normalisation leakage (master §4.3, non-negotiable #2 — catastrophic, invisible until viva):**
  held-out + non-train rows are excluded **before** any aggregation; `_assert_no_leakage` is
  fail-loud (raises, no artefact written); a dedicated guard test proves held-out rows have
  **zero** effect on the artefact bytes. This is the headline `/apex-review` check.
- **Non-reproducible artefact:** sorted keys + `.item()` Python floats + no timestamp pin
  byte-equality; a determinism test asserts it. Float-repr drift fallback documented in §9.
- **Silent skip of normalisation:** a missing/corrupt artefact **raises** (never a no-op); unknown
  circuit / NULL input log + fall back; no path reaches a model un-normalised without a coded
  `data_quality_log` row — the §8.2 / non-negotiable #1 contract feeding the RQ5 evidence.
- **Division by zero (zero-variance feature):** `std < min_std` → global std or centre only
  (`std=1.0`), never NaN/Inf; logged `ZERO_VARIANCE`.
- **Wrong ledger granularity:** F03a's `_DQLedger` hardcodes `entity_type="lap"`; F04 ships its own
  parameterised ledger so fit-time rows are correctly `entity_type="circuit"`.
