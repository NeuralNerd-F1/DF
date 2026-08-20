"""Stage B of real-data training: the FastF1 parquet corpus -> APEX's 28-feature training matrix.

Reads what ``scripts/fetch_fastf1.py`` (Stage A) wrote and drives the REAL pipeline surfaces --
:func:`validate_session` (F02), :func:`compute_scalar_features` / :func:`compute_geometry_features`
/ :func:`compute_degradation_slopes` (F03a/F03b) and :func:`assemble_feature_vector` -- to produce
``driver, circuit, season, split, F01..F28``, the matrix :func:`train_classifier` and
:func:`build_style_space` consume.

This script ORCHESTRATES; it computes no feature maths of its own.

Three things it must get right, all of them documented decisions rather than defaults:

**1. Resampling (§3.4).** FastF1's public telemetry is ~7.5 Hz with irregular gaps,
below ``thresholds.sample_rate_floor_hz`` (20) -- every raw lap would be rejected ``LAP_004``.
The corpus measures inter-sample Δt std/mean ~0.61, so it formally meets APEX's own "inconsistent
sampling" test and §3.4 prescribes a linear-interpolation resample to
``ingest.resample_target_hz``. That is applied here, per lap, before validation.
NOTE FOR THE WRITE-UP: upsampling clears the floor but adds no information --
the true information rate stays ~7.5 Hz, and any feature reading high-frequency content
(modulation rate, jerk) is interpolation-smoothed. Recorded in the matrix metadata.

**2. Train-only medians (non-negotiable #2).** ``assemble_feature_vector`` imputes NULLs from a
:class:`FeatureMedians` table that must be computed on TRAINING laps only. The build runs in three
stages: :func:`extract_session` resamples/validates and computes F03a/F03b ONCE per lap;
:func:`derive_medians` builds the median table from the train subset of those records alone
(held-out circuits are filtered out and never contribute); :func:`assemble_rows` then applies the
§4.5 NULL gate and imputation to every record, held-out included, since those are the test split.
Features are deliberately computed once, not once per pass -- an earlier two-pass version recomputed
them purely to assemble the vectors and stretched a four-season build to ~7 hours.

**3. Circuit type.** Nothing in the repo maps a circuit to one of ``config.circuit_types`` -- it is
always caller-supplied. This module owns that mapping explicitly (``_CIRCUIT_TYPE``); an unmapped
circuit is reported, not silently defaulted.

Run (main venv -- NOT the FastF1 one):

    .venv/Scripts/python.exe scripts/build_training_matrix.py --seasons 2022 2023 2024 2025
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.contracts.config import AppConfig, load_config  # noqa: E402
from app.contracts.errors import ErrorCode  # noqa: E402
from app.contracts.logging import DataQualityLogError, log_data_quality  # noqa: E402
from app.contracts.results import Status  # noqa: E402
from app.pipeline.features import (  # noqa: E402
    FEATURE_IDS,
    FeatureMedians,
    StintLap,
    assemble_feature_vector,
    compute_degradation_slopes,
    compute_geometry_features,
    compute_scalar_features,
)
from app.pipeline.validate import validate_session  # noqa: E402

# Circuit -> config.circuit_types member. Grouped on the dominant demand of the layout:
#   street      — public-road circuits: low grip, walls, low average speed
#   high_speed  — long straights / high full-throttle fraction
#   technical   — high corner density, heavy braking, low speed
#   mixed       — balanced layouts that sit in neither extreme
# An unmapped circuit is REPORTED (never silently bucketed) — the type drives F04 grouping,
# the F10 τ lookup and F11's setup mapping, so a wrong guess is worse than a loud gap.
_CIRCUIT_TYPE: dict[str, str] = {
    "Bahrain": "high_speed",
    "Saudi Arabia": "street",
    "Australia": "mixed",
    "Azerbaijan": "street",
    "Miami": "mixed",
    "Emilia-Romagna": "technical",
    "Monaco": "street",
    "Spain": "technical",
    "Canada": "mixed",
    "Austria": "high_speed",
    "Great Britain": "high_speed",
    "Hungary": "technical",
    "Belgium": "high_speed",      # held out
    "Netherlands": "technical",
    "Italy": "high_speed",        # held out
    "Singapore": "street",
    "Japan": "technical",
    "Qatar": "high_speed",
    "USA": "mixed",
    "Mexico": "mixed",
    "Brazil": "mixed",
    "Las Vegas": "street",
    "Abu Dhabi": "mixed",
    "France": "high_speed",
    "Portugal": "technical",
    "China": "mixed",
    "Turkey": "mixed",
}

# Resampling treats channels by KIND, which is load-bearing. Linearly interpolating a CATEGORICAL
# channel invents transitions: linear-interp Gear changes value at essentially every sample, so
# F15 (gear changes per km) explodes past its [0,15] range and NULLs out — which then fails every
# lap, because a NULL with no train median cannot be imputed. Gear and DRS are category codes and
# so carry the previous observed value (zero-order hold).
#
# Brake sits with the CONTINUOUS channels despite arriving as a bool, and this is a documented
# limitation rather than a clean call. FastF1 publishes brake as 1-bit on/off; APEX's F06/F08
# (brake_peak_mean, brake_release_rate in bar/50 ms) were specified against a brake-PRESSURE
# trace, which the public feed does not carry. Held as a step, the signal is a square wave whose
# "release" is a single -100 edge — F08 saturates past its [0,100] range and NULLs for every lap.
# Interpolating treats the bool as a coarse quantisation of the underlying pressure, which is the
# least-wrong available reading. FOR THE WRITE-UP: on real data F06/F08 are derived from a boolean
# proxy and are NOT brake pressure; they should not be interpreted as such.
_CONTINUOUS = ["Speed", "Throttle", "RPM", "X", "Y", "Brake"]
_DISCRETE = ["Gear", "DRS"]

# data_quality_log detail tags for the faults this ORCHESTRATOR owns (the pipeline surfaces log
# their own). Mirrors the REASON_* convention in features.py / normalise.py.
REASON_SESSION_UNREADABLE = "SESSION_UNREADABLE"  # a driver-session parquet would not read
REASON_SEASON_ABSENT = "SEASON_ABSENT"            # a requested season has no corpus on disk
REASON_CONTEXT_ABSENT = "CONTEXT_COLUMN_ABSENT"   # parquet lacks a column F10/F11 need
REASON_CIRCUIT_TYPE_ABSENT = "CIRCUIT_TYPE_ABSENT"  # circuit outside _CIRCUIT_TYPE
REASON_NO_TRAIN_MEDIAN = "NO_TRAIN_MEDIAN"        # feature un-imputable for the whole run

# The per-lap context columns Stage A's contract guarantees and F10/F11 consume.
_CONTEXT_COLS = ("stint", "tyre_age", "compound", "track_status", "lap_time")


def record_dq(
    log_path: Path, *, code: ErrorCode, entity_type: str, entity_id: str, detail: str,
    fallback_used: bool = False,
) -> None:
    """Append one ``data_quality_log`` row for a fault this script owns.

    A build must not die because the audit sink is unwritable, but the sink failing must never be
    silent either (non-negotiable #1 applies to the logger itself) -- so the ``DataQualityLogError``
    is surfaced on stderr rather than swallowed.
    """
    try:
        log_data_quality(
            error_code=code, entity_type=entity_type, entity_id=entity_id,
            error_detail=detail, fallback_used=fallback_used, log_path=log_path,
        )
    except DataQualityLogError as exc:
        print(f"  !! data_quality_log UNWRITABLE ({exc}); lost row: "
              f"{code.value} {entity_type}/{entity_id}: {detail}", file=sys.stderr)


def resample_lap(lap: pd.DataFrame, hz: int) -> pd.DataFrame:
    """Linear-interpolate one lap's channels onto a uniform ``hz`` grid (§3.4).

    Applied UNCONDITIONALLY: the raw corpus rate is uniformly below the 20 Hz floor, so every lap
    needs lifting. Per-lap scalar context (lap_time, compound, stint, tyre_age, ...) is constant
    within a lap and is carried through unchanged.
    """
    t = pd.to_numeric(lap["time"], errors="coerce").to_numpy(dtype=float)
    order = np.argsort(t, kind="stable")
    t = t[order]
    finite = np.isfinite(t)
    t = t[finite]
    if t.size < 2:
        return lap
    keep = np.concatenate(([True], np.diff(t) > 0))  # np.interp needs strictly increasing x
    t = t[keep]
    if t.size < 2:
        return lap
    grid = np.arange(t[0], t[-1], 1.0 / hz, dtype=float)
    if grid.size == 0:
        return lap
    out: dict[str, Any] = {"time": grid}
    for col in _CONTINUOUS:
        if col not in lap.columns:
            continue
        vals = pd.to_numeric(lap[col], errors="coerce").to_numpy(dtype=float)[order][finite][keep]
        out[col] = np.interp(grid, t, vals)
    # Zero-order hold for discrete channels — the sample in force at each grid point.
    step_idx = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, t.size - 1)
    for col in _DISCRETE:
        if col not in lap.columns:
            continue
        vals = pd.to_numeric(lap[col], errors="coerce").to_numpy(dtype=float)[order][finite][keep]
        out[col] = vals[step_idx]
    frame = pd.DataFrame(out)
    # Per-lap constants ride along unchanged (first value — constant within the lap).
    carry = ("lap", "track_status", "compound", "pit_in", "pit_out", "lap_time", "stint",
             "tyre_age")
    for col in carry:
        if col in lap.columns:
            frame[col] = lap[col].iloc[0]
    return frame


def resample_session(frame: pd.DataFrame, hz: int) -> pd.DataFrame:
    """Resample every lap of a driver-session onto the uniform grid."""
    out = [resample_lap(g, hz) for _, g in frame.groupby("lap", sort=True)]
    return pd.concat(out, ignore_index=True) if out else frame


def _lap_distance_m(lap: pd.DataFrame) -> float | None:
    """Path length of one lap from the X/Y trace, in metres (FastF1 X/Y are in 1/10 m)."""
    if "X" not in lap.columns or "Y" not in lap.columns:
        return None
    x = pd.to_numeric(lap["X"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(lap["Y"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return None
    d = float(np.sum(np.hypot(np.diff(x[ok]), np.diff(y[ok])))) / 10.0
    return d if np.isfinite(d) and d > 0 else None


def _raw_values(
    frame: pd.DataFrame, lap_no: int, config: AppConfig, session_max_rpm: float | None,
    entity_id: str, log_path: Path,
) -> tuple[dict[str, float | None], list[str]]:
    """F03a scalars + F03b geometry for one lap, as a plain feature->value dict."""
    lap = frame.loc[frame["lap"] == lap_no]
    lt = pd.to_numeric(lap["lap_time"], errors="coerce").dropna()
    lap_time_s = float(lt.iloc[0]) if len(lt) else None
    scalar = compute_scalar_features(
        lap, lap_number=lap_no, session_max_rpm=session_max_rpm,
        lap_distance_m=_lap_distance_m(lap), config=config, lap_time_s=lap_time_s,
        log_path=log_path, entity_id=entity_id,
    )
    geo = compute_geometry_features(
        lap, lap_number=lap_no, config=config, log_path=log_path, entity_id=entity_id,
    )
    values: dict[str, float | None] = {}
    values.update(scalar.features)
    values.update(geo.features)
    return values, []


def extract_session(
    path: Path, season: int, circuit: str, driver: str, config: AppConfig,
    log_path: Path, hz: int, session_kind: str = "R",
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """One driver-circuit parquet -> per-lap RAW feature records (+ a rejection tally).

    Resample, validate, and run F03a/F03b ONCE per lap. Assembly (median imputation + the NULL
    gate) is deferred to :func:`assemble_rows`, because the median table is derived from the
    train subset of these very records. Computing features twice — once for the medians and
    again for assembly — doubled a four-season build to ~7 hours for no benefit.

    Each record carries the raw values plus the context assembly and F10/F11 need.
    """
    stats: dict[str, int] = defaultdict(int)
    frame = pd.read_parquet(path)
    if frame.empty:
        stats["empty_parquet"] += 1
        return [], stats
    frame = resample_session(frame, hz)

    vres = validate_session(
        frame, season=season, circuit=circuit, driver=driver, config=config, log_path=log_path
    )
    stats["laps_seen"] += len(vres.clean_laps) + len(vres.rejected_laps)
    stats["laps_clean"] += len(vres.clean_laps)
    for rej in vres.rejected_laps:
        stats[f"rej_{rej.error_code.value}"] += 1
    if vres.combination_excluded:
        stats["combination_excluded"] += 1
    if not vres.clean_laps:
        return [], stats

    rpm = pd.to_numeric(frame.get("RPM"), errors="coerce")
    session_max_rpm = float(rpm.max()) if rpm is not None and rpm.notna().any() else None
    circuit_type = _CIRCUIT_TYPE.get(circuit)

    # F03a/F03b per accepted lap.
    per_lap: dict[int, dict[str, float | None]] = {}
    for lv in vres.clean_laps:
        eid = f"{season}/{circuit}/{driver}/lap{lv.lap_number}"
        vals, _ = _raw_values(frame, lv.lap_number, config, session_max_rpm, eid, log_path)
        per_lap[lv.lap_number] = vals

    # Per-lap context (constant within a lap): stint/tyre-age drive the F03b slopes below, and
    # the rest are carried onto the output rows for F10/F11.
    ctx_cols = [c for c in _CONTEXT_COLS if c in frame.columns]
    absent_ctx = [c for c in _CONTEXT_COLS if c not in frame.columns]
    if absent_ctx:
        # Stage A's _CONTRACT_COLS guarantees these columns, so a parquet without one is a corpus
        # integrity fault. The session still yields features, but EVERY lap of it would carry
        # context=None into F10/F11 — degrade loudly rather than silently (non-negotiable #1).
        stats["ctx_columns_absent"] += 1
        record_dq(
            log_path, code=ErrorCode.CACHE_CORRUPT, entity_type="session",
            entity_id=f"{season}/{circuit}/{driver}",
            detail=(f"{REASON_CONTEXT_ABSENT}: {absent_ctx} absent from the parquet; "
                    "None for every lap of this session"),
            fallback_used=True,
        )
    lap_ctx = frame.drop_duplicates("lap").set_index("lap")[ctx_cols].to_dict("index")
    lap_stint = lap_ctx
    stints: dict[float, list[StintLap]] = defaultdict(list)
    for lap_no, vals in per_lap.items():
        meta = lap_stint.get(lap_no, {})
        st = meta.get("stint")
        age = meta.get("tyre_age")
        stints[float(st) if pd.notna(st) else -1.0].append(
            StintLap(
                lap_number=lap_no,
                tyre_age=int(age) if age is not None and pd.notna(age) else None,
                throttle_mean=vals.get("F01"),
                brake_peak_mean=vals.get("F06"),
            )
        )
    for sid, stint_laps in stints.items():
        dres = compute_degradation_slopes(
            sorted(stint_laps, key=lambda s: s.lap_number), stint_id=int(sid),
            config=config, log_path=log_path,
        )
        for lap_no in dres.applies_to:
            if lap_no in per_lap:
                per_lap[lap_no]["F27"] = dres.f27
                per_lap[lap_no]["F28"] = dres.f28

    # Per-lap context F10 (stint variation) and F11 (personalisation) need on top of the 28
    # features. Names mirror their _REQUIRED_COLUMNS exactly: detector wants lap_number /
    # session / stint / track_status / is_clean / has_interpolation / tyre_age, engine wants
    # lap_time_s. Every lap reaching here is an F02-accepted lap, hence is_clean=True.
    records: list[dict[str, Any]] = []
    by_lap = {lv.lap_number: lv for lv in vres.clean_laps}
    for lap_no, vals in per_lap.items():
        lv = by_lap[lap_no]
        ctx = lap_ctx.get(lap_no, {})
        records.append(
            {
                "season": season, "circuit": circuit, "driver": driver, "lap": lap_no,
                "circuit_type": circuit_type,
                "session": session_kind,
                "lap_number": lap_no,
                "stint": ctx.get("stint"),
                "tyre_age": ctx.get("tyre_age"),
                "compound": ctx.get("compound"),
                "track_status": ctx.get("track_status"),
                "lap_time_s": ctx.get("lap_time"),
                "is_clean": True,
                "has_interpolation": ErrorCode.LAP_INTERP in lv.flags,
                "_raw": vals,
                "_forced_null": lv.null_features or None,
                "_flags": list(lv.flags) or None,
            }
        )
    return records, stats


def derive_medians(records: list[dict[str, Any]], held_out: set[str]) -> FeatureMedians:
    """Train-only imputation medians (non-negotiable #2): held-out circuits never contribute."""
    train = [r for r in records if r["circuit"] not in held_out]
    raw = pd.DataFrame([r["_raw"] for r in train])
    meta = pd.DataFrame([{"driver": r["driver"], "circuit_type": r["circuit_type"]} for r in train])
    global_medians = {
        fid: float(raw[fid].median())
        for fid in FEATURE_IDS
        if fid in raw.columns and raw[fid].notna().any()
    }
    dc: dict[tuple[str, str], dict[str, float]] = {}
    joined = pd.concat([meta.reset_index(drop=True), raw.reset_index(drop=True)], axis=1)
    for (drv, ctype), grp in joined.dropna(subset=["circuit_type"]).groupby(
        ["driver", "circuit_type"]
    ):
        dc[(str(drv), str(ctype))] = {
            fid: float(grp[fid].median())
            for fid in FEATURE_IDS
            if fid in grp.columns and grp[fid].notna().any()
        }
    return FeatureMedians(global_medians=global_medians, driver_circuit_medians=dc)


def report_missing_medians(medians: FeatureMedians, log_path: Path) -> list[str]:
    """Features with no train median anywhere in the corpus, one audit row each.

    Each lap's rejection is already logged individually by ``assemble_feature_vector``
    (``FEATURE_001``), but the systemic cause — this feature is un-imputable for the WHOLE run, so
    every lap carrying it as NULL will fail — needs one persistent row of its own. Without it the
    audit trail shows thousands of symptoms and no diagnosis.
    """
    missing = [fid for fid in FEATURE_IDS if fid not in medians.global_medians]
    for fid in missing:
        record_dq(
            log_path, code=ErrorCode.FEATURE_001, entity_type="training_run", entity_id=fid,
            detail=(f"{REASON_NO_TRAIN_MEDIAN}: no train value for {fid} anywhere in the corpus; "
                    "every lap carrying it as NULL fails assembly"),
        )
    return missing


def assemble_rows(
    records: list[dict[str, Any]], medians: FeatureMedians, config: AppConfig, log_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Apply the §4.5 NULL gate + train-only imputation to every extracted record."""
    stats: dict[str, int] = defaultdict(int)
    rows: list[dict[str, Any]] = []
    for rec in records:
        lap_no = rec["lap"]
        fres = assemble_feature_vector(
            rec["_raw"], lap_number=lap_no, driver=rec["driver"],
            circuit_type=rec["circuit_type"], medians=medians, config=config,
            forced_null=rec["_forced_null"], flags=rec["_flags"], log_path=log_path,
            entity_id=f"{rec['season']}/{rec['circuit']}/{rec['driver']}/lap{lap_no}",
        )
        # FeatureResult has no `accepted`: a rejected vector is Status.FAILED (LAP_NULL_EXCESS
        # over the NULL ceiling, or FEATURE_001 when a NULL had no train median to impute from).
        if fres.status is Status.FAILED:
            stats[f"vec_{fres.error_code.value if fres.error_code else 'FAILED'}"] += 1
            continue
        row = {k: v for k, v in rec.items() if not k.startswith("_")}
        row.update({fid: fres.feature_vector.get(fid) for fid in FEATURE_IDS})
        rows.append(row)
    stats["vectors_built"] += len(rows)
    return rows, stats


def discover(
    data_dir: Path, seasons: list[int]
) -> tuple[list[tuple[int, str, str, Path]], list[int]]:
    """(season, circuit, driver, parquet_path) for the requested seasons, + the absent ones.

    A requested season with no corpus on disk is RETURNED, never quietly skipped: silently
    dropping a whole season yields a short matrix that looks legitimate (non-negotiable #1).
    """
    found = []
    missing: list[int] = []
    for season in seasons:
        sdir = data_dir / str(season)
        if not sdir.is_dir():
            missing.append(season)
            continue
        for cdir in sorted(sdir.iterdir()):
            if not cdir.is_dir():
                continue
            for pq in sorted(cdir.glob("*.parquet")):
                found.append((season, cdir.name, pq.stem, pq))
    return found, missing


def stable_core(units: list[tuple[int, str, str, Path]], seasons: list[int]) -> set[str]:
    """Drivers present in EVERY requested season (the chosen label scope)."""
    by_season: dict[int, set[str]] = defaultdict(set)
    for season, _c, driver, _p in units:
        by_season[season].add(driver)
    present = [by_season[s] for s in seasons if by_season.get(s)]
    return set.intersection(*present) if present else set()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", type=int, nargs="+", default=[2022, 2023, 2024, 2025])
    ap.add_argument("--data-dir", default="D:/COMP702/DF/fastf1_data")
    ap.add_argument("--out", default="D:/COMP702/DF/artefacts/real/training_matrix.parquet")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--limit-units", type=int, default=None,
                    help="cap driver-sessions (smoke test)")
    args = ap.parse_args()

    config = load_config(args.config)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_path.parent / "build_dq.jsonl"

    if config.ingest is None:
        # No default: silently resampling at some assumed rate would put an unstated Hz into
        # every feature in the matrix. The rate is a config value or the build does not run.
        print("config.ingest block absent; resample_target_hz is required (§3.4)",
              file=sys.stderr)
        raise SystemExit(2)
    hz = config.ingest.resample_target_hz

    units, missing_seasons = discover(Path(args.data_dir), args.seasons)
    for season in missing_seasons:
        print(f"!! season {season} has NO corpus under {args.data_dir} — excluded from the matrix",
              file=sys.stderr)
        record_dq(
            log_path, code=ErrorCode.DATA_SOURCE_UNAVAILABLE, entity_type="season",
            entity_id=str(season),
            detail=f"{REASON_SEASON_ABSENT}: no directory under {args.data_dir}; season excluded",
        )
    if not units:
        print(f"no parquet corpus under {args.data_dir} for seasons {args.seasons}",
              file=sys.stderr)
        raise SystemExit(2)

    core = stable_core(units, args.seasons)
    unmapped = sorted({c for _s, c, _d, _p in units if c not in _CIRCUIT_TYPE})
    if unmapped:
        print(f"!! circuits with NO circuit_type mapping (rows will carry None): {unmapped}",
              file=sys.stderr)
    for circuit in unmapped:
        # circuit_type=None means no driver×circuit-type imputation cell, so every NULL on this
        # circuit imputes from the GLOBAL table instead — a documented fallback, and one that
        # belongs in the audit trail rather than only on a terminal an operator may not be watching.
        record_dq(
            log_path, code=ErrorCode.NORMALISATION_FALLBACK, entity_type="circuit",
            entity_id=circuit,
            detail=(f"{REASON_CIRCUIT_TYPE_ABSENT}: '{circuit}' not in _CIRCUIT_TYPE; "
                    "laps carry circuit_type=None and impute from the global median table"),
            fallback_used=True,
        )

    units = [u for u in units if u[2] in core]
    if args.limit_units:
        units = units[: args.limit_units]
    print(f"corpus: {len(units)} driver-sessions | "
          f"stable-core drivers ({len(core)}): {sorted(core)}")
    print(f"resample: {hz} Hz (§3.4; raw corpus ~7.5 Hz < "
          f"{config.thresholds.sample_rate_floor_hz} Hz floor)")

    held_out = set(config.held_out_circuits)
    totals: dict[str, int] = defaultdict(int)

    # ---- EXTRACT: resample + validate + F03a/F03b, once per lap ------------------------------
    print(f"\n[1/3] extracting features from {len(units)} driver-sessions")
    records: list[dict[str, Any]] = []
    for i, (season, circuit, driver, pq) in enumerate(units, 1):
        try:
            recs, stats = extract_session(pq, season, circuit, driver, config, log_path, hz)
        except Exception as exc:  # noqa: BLE001 - one bad parquet must not void a multi-hour build
            # A truncated/corrupt file (e.g. an interrupted Stage A write) is a per-session fault,
            # not a build fault: count it, log it with a typed code, and keep going. Crashing here
            # would discard every session already extracted, none of it yet on disk.
            totals["session_unreadable"] += 1
            print(f"  !! session unreadable, skipped: {pq} ({type(exc).__name__}: {exc})",
                  file=sys.stderr)
            record_dq(
                log_path, code=ErrorCode.CACHE_CORRUPT, entity_type="session",
                entity_id=f"{season}/{circuit}/{driver}",
                detail=f"{REASON_SESSION_UNREADABLE}: {type(exc).__name__}: {exc}; session skipped",
            )
            continue
        records.extend(recs)
        for k, v in stats.items():
            totals[k] += v
        if i % 100 == 0:
            print(f"  {i}/{len(units)} sessions, {len(records)} clean laps", flush=True)
    if not records:
        print("no clean laps produced — refusing to write an empty matrix", file=sys.stderr)
        raise SystemExit(3)

    # ---- MEDIANS: train-only (non-negotiable #2) ----------------------------------------------
    print(f"\n[2/3] deriving train-only medians from {len(records)} laps")
    medians = derive_medians(records, held_out)
    print(f"  medians: {len(medians.global_medians)} global, "
          f"{len(medians.driver_circuit_medians)} driver×circuit-type cells")
    missing_medians = report_missing_medians(medians, log_path)
    if missing_medians:
        print(f"  !! features with NO train median (will fail every lap): {missing_medians}",
              file=sys.stderr)

    # ---- ASSEMBLE: NULL gate + imputation, held-out included as the test split ----------------
    print("\n[3/3] assembling 28-feature vectors")
    all_rows, astats = assemble_rows(records, medians, config, log_path)
    for k, v in astats.items():
        totals[k] += v

    if not all_rows:
        print("no feature vectors produced — refusing to write an empty matrix", file=sys.stderr)
        raise SystemExit(3)

    matrix = pd.DataFrame(all_rows)
    # split: held-out circuits are the quarantined test set; everything else is the train pool
    # (train_classifier carves its own disjoint calibration split from it).
    matrix["split"] = np.where(matrix["circuit"].isin(held_out), "test", "train")
    matrix.to_parquet(out_path, index=False)

    meta = {
        "seasons": args.seasons,
        "resample_hz": hz,
        "raw_rate_note": "FastF1 public telemetry ~7.5 Hz; upsampled per §3.4. "
                         "Upsampling clears the 20 Hz floor but adds no information.",
        "stable_core_drivers": sorted(core),
        "circuits": sorted(matrix["circuit"].unique().tolist()),
        "held_out_circuits": sorted(held_out),
        "unmapped_circuit_types": unmapped,
        "n_vectors": int(len(matrix)),
        "n_train": int((matrix["split"] == "train").sum()),
        "n_test": int((matrix["split"] == "test").sum()),
        "tallies": dict(sorted(totals.items())),
    }
    (out_path.parent / "training_matrix_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print(f"\nDONE: {len(matrix)} vectors -> {out_path}")
    print(f"  train={meta['n_train']} test(held-out)={meta['n_test']}  "
          f"drivers={matrix['driver'].nunique()}")
    print(f"  rejection tallies: {dict(sorted(totals.items()))}")
    print(f"  meta: {out_path.parent / 'training_matrix_meta.json'}")


if __name__ == "__main__":
    main()
