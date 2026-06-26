# Feature F02 — Lap Filter Chain / Validation Contract

**Status:** DONE (2026-06-26)
**Master spec sections:** §3.2 (Data Coverage), §3.3 (Lap Filtering Rules — Data Quality Contract), §8.2 (Validator module), Project Plan Wk 2
**Layer / Week:** Data pipeline / Wk 2

## 1. Purpose
Apply the 9-filter data-quality contract to every ingested lap so that only clean,
green-flag, well-sampled, plausibly-timed laps enter the feature matrix. Each lap is
admitted, tagged, or rejected with a specific error code, and combinations with too
few clean laps are excluded — no lap is ever silently dropped. The validator is the
gatekeeper between raw cached telemetry (F01) and feature extraction (F03/F05).

## 2. Upstream / downstream contracts
- **Consumes:** F01 — cached session telemetry (`cache/<season>/<circuit>/<driver>.parquet`)
  plus the per-lap context F01 attaches (session/track status, pit in/out flags,
  lap time, tyre-age source). Reads thresholds from `config.yaml`.
- **Produces:** `ValidationResult` per driver-circuit combination (containing one
  `LapValidationResult` per lap), and one `data_quality_log` row per rejected/tagged
  lap and per excluded combination. No new artefact files.
- **Consumed by:** F03 (feature extraction reads the accepted clean laps + per-lap
  NULL-feature directives), F05 (manual CSV upload runs the same filter chain).

## 3. Inputs & outputs (typed)
- Input (implemented): `validate_session(telemetry: DataFrame, *, season, circuit, driver,
  config, log_path=None)` — F01's **wide** cached frame (one row per sample; per-lap
  context `track_status`, `pit_in`, `pit_out`, `lap_time`, `tyre_age` left-joined on
  `lap`). An internal adapter groups it into per-lap views; there is no separate
  `SessionLaps`/`LapTelemetry` object. (Conceptually each lap carries telemetry channels
  + track status + pit flags + lap time + tyre-age presence.)
- Output (per lap):
  `LapValidationResult { lap_number:int, status: ACCEPTED|REJECTED,
  error_code: ErrorCode (OK when clean), flags: list[ErrorCode] (LAP_INTERP, LAP_PARTIAL),
  null_features: list[str] (e.g. ["F27","F28"]), detail: dict (channel, dropout_pct,
  sampling_rate_hz, lap_time_s, session_median_s, stationary_fraction — whichever applies) }`.
- Output (per combination):
  `ValidationResult { season, circuit, driver, clean_laps: list[LapValidationResult],
  rejected_laps: list[LapValidationResult], clean_lap_count:int,
  combination_excluded:bool, status: Status (OK normally, FAILED when excluded),
  error_code: None|COMBO_001 }`. The `DATA_INSUFFICIENT` name is the *logged* code on the
  combination's data-quality row (§3.2), not a `Status` value — `Status` is the
  foundation enum `{OK, DEGRADED, FAILED}`, so an excluded combination is `FAILED` and
  consumers discriminate "under-covered" via `error_code == COMBO_001`.

The validator **never raises** — every error is caught, coded, and returned inside the
result object (§8.2 Validator failsafe behaviour).

## 4. Method / algorithm
Filters run **in order**; the **first failure rejects** the lap with its code (§3.3).
Tag filters (LAP_INTERP, LAP_PARTIAL) do not reject — they annotate and continue.

Per lap:
1. **Session flag (LAP_001)** — lap must occur entirely under green flag. Reject if any
   sample is SC / VSC / red-flag (`track_status`).
2. **Lap type (LAP_002)** — reject if the lap contains a pit-in or pit-out event
   (`is_pit_in or is_pit_out`); i.e. outlaps and inlaps.
3. **Telemetry completeness — major (LAP_003)** — for each required channel
   `[Speed, Throttle, Brake, RPM, Gear]`, reject if any run of *consecutive* NaN exceeds
   `validate.max_consecutive_nan_pct` (10%) of the lap's sample count. Log channel name
   and dropout percentage.
4. **Telemetry completeness — minor (LAP_INTERP)** — fill gaps of `1..interp_gap_max_samples`
   (1–5) consecutive NaN in any channel by linear interpolation. Accept the lap; add the
   `LAP_INTERP` flag; log gap size. (Runs after step 3 so only sub-threshold gaps remain.)
5. **Sampling rate (LAP_004)** — compute effective sampling rate *after* interpolation;
   reject if `< thresholds.sample_rate_floor_hz` (20 Hz). Log computed rate.
6. **Speed anomaly (LAP_005)** — reject if the stationary fraction (samples with
   Speed ≤ `validate.min_moving_speed_kmh`, i.e. 0 km/h) is
   `>= validate.max_stationary_fraction` (0.01) — equivalently, fewer than 99% of samples
   are moving (master: "> 99% of samples" must be > 0). Exactly 1% stationary → reject.
   Log anomaly location.
7. **Lap time plausibility (LAP_006)** — reject if
   `lap_time_s > (thresholds.laptime_bound_pct / 100) × session_median` — strictly above
   115% of the **driver's session median**; exactly 115% is "within 115%" and passes. Log
   lap time and session median. The median is taken over the driver's **racing laps**
   (green-flag and non-pit) only — in/out and SC/VSC laps are slow and would inflate the
   median, making the bound too lenient.
8. **Tyre age (LAP_PARTIAL)** — if `tyre_age_known` is false, accept the lap but mark
   features `validate.degradation_features` (`F27, F28`) NULL for it; add `LAP_PARTIAL`
   flag; log. Does **not** reject.
9. **Minimum lap count post-filter (COMBO_001)** — after all laps are filtered, if
   `clean_lap_count < thresholds.min_clean_laps` (30), set `combination_excluded=True`,
   `error_code=COMBO_001`, `status=DATA_INSUFFICIENT`, and log the excluded combination
   to the data-quality report (§3.2 — combination excluded from training, logged as
   `DATA_INSUFFICIENT`).

All operations vectorised over numpy/pandas — no per-row Python loops (constitution tech
stack). A clean lap that passes 1–8 with no tags has `status=ACCEPTED`, `error_code=OK`,
empty `flags`/`null_features`.

## 5. config.yaml keys this feature adds
Shared thresholds already in `config.yaml` are **referenced, not redefined**:
`thresholds.sample_rate_floor_hz` (20), `thresholds.laptime_bound_pct` (115),
`thresholds.min_clean_laps` (30). F02 adds its own block for the values unique to it:
```yaml
# --- F02 lap filter chain / validation contract (§3.2, §3.3) ----------------
validate:
  required_channels: [Speed, Throttle, Brake, RPM, Gear]  # §3.3 completeness check set (MUST include Speed — LAP_005)
  green_track_status_code: "1"       # §3.3 LAP_001 — FastF1 green token; lap green iff every token == this
  max_consecutive_nan_pct: 10        # §3.3 LAP_003 — >N% consecutive NaN in a channel -> reject
  interp_gap_max_samples: 5          # §3.3 LAP_INTERP — gaps of 1..N consecutive NaN linear-interpolated
  min_moving_speed_kmh: 0            # §3.3 LAP_005 — speed must exceed this to count as moving
  max_stationary_fraction: 0.01      # §3.3 LAP_005 — stationary samples must be <1% (>99% moving)
  degradation_features: [F27, F28]   # §3.3 LAP_PARTIAL — NULL'd when tyre age UNKNOWN
```

## 6. Error codes & failure modes (the failsafe contract)
| Failure mode | Detection | Error code | System response | Logged? | User-facing message |
|--------------|-----------|-----------|-----------------|---------|---------------------|
| Lap under SC/VSC/red flag | any sample `track_status` not green | `LAP_001` | Reject lap | yes (lap number + reason) | "Lap N excluded: not green-flag." |
| Outlap / inlap | `is_pit_in` or `is_pit_out` true | `LAP_002` | Reject lap | yes (lap number + reason) | "Lap N excluded: in/out lap." |
| Telemetry dropout | consecutive NaN run > 10% of samples in a required channel | `LAP_003` | Reject lap | yes (channel + dropout %) | "Lap N excluded: <channel> dropout X%." |
| Small telemetry gap | 1–5 consecutive NaN in a channel | `LAP_INTERP` (flag) | Accept; linear-interpolate; tag INTERPOLATED | yes (gap size) | served with interpolated flag |
| Low sampling rate | effective rate `< 20 Hz` after interpolation (exactly 20 Hz passes) | `LAP_004` | Reject lap | yes (computed rate) | "Lap N excluded: sampling X Hz < 20 Hz." |
| Speed/stationary anomaly | stationary fraction (speed ≤ 0) `>= 1%` | `LAP_005` | Reject lap | yes (anomaly location) | "Lap N excluded: stationary-sample anomaly." |
| Implausible lap time | lap time `> 115%` of driver session median (exactly 115% passes) | `LAP_006` | Reject lap | yes (lap time + median) | "Lap N excluded: time outside 115% of median." |
| Tyre age UNKNOWN | `tyre_age_known` false | `LAP_PARTIAL` (flag) | Accept; mark F27,F28 NULL for lap | yes | "Lap N: tyre-age features unavailable." |
| < 30 clean laps post-filter | `clean_lap_count < 30` | `COMBO_001` / `DATA_INSUFFICIENT` | Exclude combination from training | yes (excluded combo in DQ report) | "Driver-circuit excluded: only N clean laps." |
| Samples with no lap key | `telemetry.lap` is null for some rows | `ORPHAN_SAMPLES` | Drop those samples (groupby would anyway), session continues | yes (session row, count) | none |
| Malformed frame / unexpected internal error | no `lap` column, or any unforeseen exception in a lap | `VALIDATION_ERROR` | Typed FAILED (frame) or reject lap (per-lap); NOT laundered into a data code | yes | "Validator could not process input." |

`COMBO_001` is the filter-chain action code; `DATA_INSUFFICIENT` is the combination-level
status logged in the data-quality report when the combination is excluded (§3.2).

## 7. Acceptance criteria (each item mechanically testable by /spec-check)
- [x] Filters apply in the documented order; the first failing filter sets the lap's
      `error_code` and later filters are not evaluated. (A lap engineered to fail both
      LAP_001 and LAP_006 returns `error_code=LAP_001`.)
- [x] A non-green (SC/VSC/red) lap returns `status=REJECTED, error_code=LAP_001`.
- [x] A lap with `is_pit_in` or `is_pit_out` true returns `LAP_002`.
- [x] A required channel whose longest consecutive-NaN run is `> 10%` of the lap's samples
      returns `LAP_003` with `detail.channel` and `detail.dropout_pct` set; a run of
      exactly 10% does not trigger LAP_003.
- [x] A NaN gap of exactly 5 samples is linearly interpolated, the lap is `ACCEPTED`, and
      `LAP_INTERP` is in `flags`; a gap of 6 samples is not interpolated.
- [x] A lap whose post-interpolation effective rate is `< 20 Hz` returns `LAP_004` with
      `detail.sampling_rate_hz` set; a lap at exactly 20 Hz is accepted.
- [x] A lap whose stationary fraction (speed ≤ 0) is `>= 1%` returns `LAP_005`; a lap at
      0.5% stationary is accepted.
- [x] A lap with time `> 115%` of the driver's session median returns `LAP_006` with
      `detail.lap_time_s` and `detail.session_median_s` set; a lap at exactly 115% is
      accepted.
- [x] A lap with `tyre_age_known=False` is `ACCEPTED`, carries `LAP_PARTIAL` in `flags`,
      and lists exactly `["F27","F28"]` in `null_features`.
- [x] A combination with `< 30` clean laps yields `combination_excluded=True`,
      `error_code=COMBO_001`, `status=Status.FAILED`, and a `DATA_INSUFFICIENT` data-quality
      row; a combination with exactly 30 is not excluded (29 vs 30 boundary).
- [x] The validator never raises: an empty or all-NaN lap is returned as a coded
      `REJECTED` result, not an exception.
- [x] Every error code above (LAP_001–006, LAP_INTERP, LAP_PARTIAL, COMBO_001) has a unit
      test that triggers it and asserts `error_code`/`flags`.
- [x] Each rejection, tag, and combination exclusion writes exactly one `data_quality_log`
      row (row count asserted).
- [x] Every threshold (10%, 5-sample gap, 20 Hz, 1%, 115%, 30 laps) is read from
      `config.yaml`; `app/pipeline/validate.py` contains no numeric literal for any of them.

## 8. Test plan
- **Happy-path fixtures:** a synthetic green-flag lap, full channels, ≥20 Hz, no
  stationary samples, lap time near median, tyre age known → `ACCEPTED`, `error_code=OK`,
  empty flags. A combination of 30+ such laps → not excluded.
- **Failure fixtures (one per code):** SC-status lap (LAP_001); pit-out lap (LAP_002);
  channel with 15% consecutive-NaN run (LAP_003); 3-sample NaN gap (LAP_INTERP, accepted);
  decimated lap at ~10 Hz (LAP_004); lap with 2% zero-speed samples (LAP_005); lap at 130%
  of median (LAP_006); lap with `tyre_age_known=False` (LAP_PARTIAL); a 12-clean-lap
  combination (COMBO_001 / DATA_INSUFFICIENT).
- **Property / edge (pinned expected value each side):**
  - Lap time: 115.0% of median → **accepted**; 115.1% → **LAP_006**.
  - Consecutive-NaN run: exactly 10% of samples → **not LAP_003**; 10.5% → **LAP_003**.
  - Interp gap: exactly 5 NaN → **LAP_INTERP, accepted**; 6 NaN → **not interpolated**
    (remains NaN, and is only LAP_003 if that run also exceeds 10% of the lap — tested on a
    short lap where 6 samples > 10%, and a long lap where it does not).
  - Sampling rate: exactly 20 Hz → **accepted**; 19.5 Hz → **LAP_004**.
  - Stationary fraction: 0.5% → **accepted**; 1.0% → **LAP_005**.
  - Clean count: exactly 30 → **not excluded**; 29 → **COMBO_001 / DATA_INSUFFICIENT**.
  - All-NaN / empty lap → caught, coded `REJECTED`, no raise.
  - Lap failing two filters at once → asserts the **earlier** filter's code wins.

## 9. Definition of done
All acceptance criteria MET with passing tests; `/apex-review F02` clean on the contract
checklist; `/spec-check F02` green; `mypy app` clean; no bare `except` anywhere in
`app/pipeline/validate.py`. INDEX row set to DONE. Limitation to record in the
dissertation: the validator trusts F01's session/track-status and pit flags; mislabelled
upstream status is out of scope and would pass undetected.

## 10. Risks & mitigations (from master risk register where relevant)
- **FastF1 gaps for specific combinations (Risk Register row 2 / Wk-1 risk):** validator
  reports and excludes (COMBO_001) rather than padding — the coverage report lists every
  below-threshold combination; no synthetic laps are invented.
- **Patchy X/Y or channel data:** the major-completeness filter (LAP_003) rejects laps
  that would otherwise produce garbage features downstream; small gaps are interpolated
  (LAP_INTERP) so usable laps are not over-rejected.
