"""Stage A of real-data training: pull REAL F1 telemetry with FastF1 and write it in APEX's
F01->F02 "cached column contract" so the main pipeline can consume it.

FastF1 must stay ISOLATED from the main pipeline: it pulls in pandas 2.x / numpy 2.5, whereas APEX
runs pandas 3.x. So this script runs in a SEPARATE venv and the two halves communicate only
through Parquet files on disk (never in-process). Create the venv once:

    D:/COMP702/DF/.venv/Scripts/python.exe -m venv .ff1venv
    .ff1venv/Scripts/python.exe -m pip install fastf1 pyarrow
    .ff1venv/Scripts/python.exe scripts/fetch_fastf1.py --seasons 2022 2023 2024 2025

Output: ``<data_dir>/<season>/<circuit>/<driver>.parquet`` -- one row per ~50 Hz telemetry sample
with EXACTLY the columns app/pipeline/ingest.py documents (Speed, Throttle, Brake, RPM, Gear, DRS,
X, Y, lap; per-lap left-joined track_status, compound, pit_in, pit_out, lap_time, stint, tyre_age),
plus a ``time`` seconds-within-lap clock so F03 can derive the sample period. Unavailable sources
fill the documented null marker (NaN / "UNKNOWN") -- a column is never dropped.

Also writes ``<data_dir>/manifest.json``: what was fetched, what was SKIPPED and why, and what
FAILED with the exception type. Stage B reads it, so a partial corpus can never masquerade as a
complete one (no silent failure -- the constitution's rule #1 applies to the data layer too).

Resumable: a race whose parquet files already exist is skipped, so an interrupted multi-season
pull can simply be re-run. Pass --refetch to force.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd
from fastf1.req import RateLimitExceededError

# FastF1 `Location` (the circuit town -- stable across seasons, unlike EventName which gets
# sponsor/renaming churn) -> the circuit NAME the APEX pipeline keys on.
#
# Keyed on circuit, NOT country, deliberately. Country would merge genuinely different circuits:
# Imola and Monza are both "Italy"; Miami, Austin and Las Vegas are all "USA". F04 fits mean/std
# PER CIRCUIT, so merging them would pool incompatible distributions into one scaler -- and would
# also silently over-quarantine Imola along with the held-out Monza.
#
# Spa -> "Belgium" and Monza -> "Italy" are load-bearing: config.yaml `held_out_circuits`
# quarantines exactly those two names, and they must match for the §4.3 exclusion to bite.
_LOCATION_TO_CIRCUIT = {
    "Sakhir": "Bahrain",
    "Jeddah": "Saudi Arabia",
    "Melbourne": "Australia",
    "Baku": "Azerbaijan",
    "Miami": "Miami",
    "Imola": "Emilia-Romagna",          # Italy's SECOND circuit -- distinct from held-out Monza
    "Monaco": "Monaco",
    "Barcelona": "Spain",
    "Montreal": "Canada",
    "Montréal": "Canada",
    "Spielberg": "Austria",
    "Silverstone": "Great Britain",
    "Budapest": "Hungary",
    "Spa-Francorchamps": "Belgium",     # HELD OUT (config.yaml held_out_circuits)
    "Zandvoort": "Netherlands",
    "Monza": "Italy",                   # HELD OUT (config.yaml held_out_circuits)
    "Marina Bay": "Singapore",
    "Suzuka": "Japan",
    "Lusail": "Qatar",
    "Austin": "USA",
    "Mexico City": "Mexico",
    "São Paulo": "Brazil",
    "Sao Paulo": "Brazil",
    "Las Vegas": "Las Vegas",
    "Yas Island": "Abu Dhabi",
    "Le Castellet": "France",
    "Portimão": "Portugal",
    "Portimao": "Portugal",
    "Shanghai": "China",
    "Istanbul": "Turkey",
}

# FastF1 telemetry channel -> APEX contract column. get_telemetry() already merges car + position.
_CONTRACT_COLS = [
    "Speed", "Throttle", "Brake", "RPM", "Gear", "DRS", "X", "Y", "lap",
    "track_status", "compound", "pit_in", "pit_out", "lap_time", "stint", "tyre_age", "time",
]


def _with_rate_limit_retry(what: str, fn, *, sleep_s: int, max_waits: int):
    """Run ``fn()``, waiting out FastF1's 500-calls/hour cap instead of failing the race.

    The cap is the binding constraint on a multi-season pull (~22 races per hour), so a plain
    failure would abandon most of the corpus. Each wait is announced -- an unattended run that
    silently stalls for hours is indistinguishable from one that has hung.
    """
    for attempt in range(max_waits + 1):
        try:
            return fn()
        except RateLimitExceededError:
            if attempt >= max_waits:
                raise
            mins = sleep_s / 60.0
            print(f"  .. rate limit hit on {what}; waiting {mins:.0f} min "
                  f"(retry {attempt + 1}/{max_waits})", flush=True)
            time.sleep(sleep_s)
    raise RuntimeError("unreachable")


def _seconds(td: object) -> float:
    """A pandas Timedelta -> float seconds, or NaN when the value is missing (never raises)."""
    if td is None or pd.isna(td):
        return float("nan")
    return float(pd.Timedelta(td).total_seconds())


def circuit_name(location: str, country: str) -> tuple[str, bool]:
    """Map a FastF1 (Location, Country) to the APEX circuit name.

    Returns ``(name, mapped)``. An UNKNOWN location falls back to the country but reports
    ``mapped=False`` so the caller can warn loudly -- a silently mis-named circuit would land in
    the wrong F04 per-circuit scaler, or dodge the held-out quarantine entirely.
    """
    hit = _LOCATION_TO_CIRCUIT.get(str(location).strip())
    if hit is not None:
        return hit, True
    return str(country).strip() or str(location).strip(), False


def _as_str(v: object) -> str:
    """A per-lap categorical as text, or the documented "UNKNOWN" marker when missing."""
    return str(v) if v is not None and pd.notna(v) else "UNKNOWN"


def _as_float(v: object) -> float:
    """A per-lap numeric as float, or NaN when missing (the documented null marker)."""
    return float(v) if v is not None and pd.notna(v) else float("nan")  # type: ignore[arg-type]


def _col(tel: pd.DataFrame, name: str) -> np.ndarray:
    """A telemetry column as float, or an all-NaN column when the channel is absent."""
    if name not in tel.columns:
        return np.full(len(tel), np.nan)
    return pd.to_numeric(tel[name], errors="coerce").to_numpy(dtype=float)


def _driver_frame(lap_rows: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Build one driver's contract frame: every lap's telemetry samples, stacked.

    ONE ``get_telemetry()`` call for the whole driver, then lap numbers are assigned to samples
    VECTORISED via ``searchsorted`` on lap start times. The obvious per-lap ``lap.get_telemetry()``
    loop is ~45x slower (it re-slices the full session per lap: ~2.2 min/driver vs ~2.8 s), which
    turns a multi-season pull from hours into days.

    Telemetry is written at its RAW sample rate -- deliberately NOT upsampled. FastF1's public
    car data is ~4-10 Hz with irregular gaps, below APEX's 20 Hz `sample_rate_floor_hz`; the
    resample to a uniform grid is Stage B's job (§3.4 RESAMPLED), so the corpus on disk stays a
    faithful record and the resampling policy stays changeable without a re-download.

    Returns ``(frame, n_laps_without_telemetry)`` -- the second value is COUNTED and reported
    rather than silently dropped.
    """
    if len(lap_rows) == 0:
        return pd.DataFrame(columns=_CONTRACT_COLS), 0
    try:
        tel = lap_rows.get_telemetry()
    except Exception:  # whole-driver telemetry unavailable -- never fabricate samples
        return pd.DataFrame(columns=_CONTRACT_COLS), len(lap_rows)
    if tel is None or len(tel) == 0:
        return pd.DataFrame(columns=_CONTRACT_COLS), len(lap_rows)

    starts = lap_rows["LapStartTime"].dt.total_seconds().to_numpy(dtype=float)
    lap_no = lap_rows["LapNumber"].to_numpy(dtype=float)
    finite = np.isfinite(starts)
    if not finite.any():  # no usable lap clock -> cannot attribute samples to laps
        return pd.DataFrame(columns=_CONTRACT_COLS), len(lap_rows)
    starts, lap_no = starts[finite], lap_no[finite]
    order = np.argsort(starts)
    starts, lap_no = starts[order], lap_no[order]

    sess_s = pd.to_timedelta(tel["SessionTime"]).dt.total_seconds().to_numpy(dtype=float)
    idx = np.searchsorted(starts, sess_s, side="right") - 1
    keep = idx >= 0  # samples before the first lap start belong to no lap -> dropped, counted
    if not keep.any():
        return pd.DataFrame(columns=_CONTRACT_COLS), len(lap_rows)
    idx = idx[keep]
    tel = tel.loc[keep]
    assigned_lap = lap_no[idx]
    # Seconds WITHIN the assigned lap (F03 derives the sample period from this).
    time_s = sess_s[keep] - starts[idx]

    n = len(tel)
    brake = tel["Brake"] if "Brake" in tel.columns else None
    # Brake is boolean in FastF1 -> 0/100 so the pipeline sees a percentage-like channel.
    brake_pct = (
        (brake.astype(float) * 100.0).to_numpy() if brake is not None else np.full(n, np.nan)
    )

    # Per-lap attributes, mapped onto samples by lap number (vectorised, no per-row loop).
    def _lap_map(col: str, cast) -> np.ndarray:
        if col not in lap_rows.columns:
            return np.array([cast(None) for _ in range(1)] * n)
        m = dict(
            zip(
                lap_rows["LapNumber"].to_numpy(dtype=float),
                lap_rows[col].to_numpy(),
                strict=False,
            )
        )
        return np.array([cast(m.get(lp)) for lp in assigned_lap], dtype=object)

    frame = pd.DataFrame(
        {
            "Speed": _col(tel, "Speed"),
            "Throttle": _col(tel, "Throttle"),
            "Brake": np.asarray(brake_pct, dtype=float),
            "RPM": _col(tel, "RPM"),
            "Gear": _col(tel, "nGear"),
            "DRS": _col(tel, "DRS"),
            "X": _col(tel, "X"),
            "Y": _col(tel, "Y"),
            "lap": assigned_lap.astype(int),
            "track_status": _lap_map("TrackStatus", _as_str),
            "compound": _lap_map("Compound", _as_str),
            "pit_in": _lap_map("PitInTime", lambda v: bool(v is not None and pd.notna(v))),
            "pit_out": _lap_map("PitOutTime", lambda v: bool(v is not None and pd.notna(v))),
            "lap_time": _lap_map("LapTime", _seconds).astype(float),
            "stint": _lap_map("Stint", _as_float).astype(float),
            "tyre_age": _lap_map("TyreLife", _as_float).astype(float),
            "time": time_s,
        }
    )
    laps_with_samples = frame["lap"].nunique()
    return frame[_CONTRACT_COLS], max(0, len(lap_rows) - laps_with_samples)


def fetch_race(season: int, event_round: int, circuit: str, out_dir: Path) -> dict:
    """Write one race's per-driver parquets. Returns a manifest record for this race."""
    session = fastf1.get_session(season, event_round, "R")
    session.load(telemetry=True, laps=True, weather=False)
    dest = out_dir / str(season) / circuit
    dest.mkdir(parents=True, exist_ok=True)
    drivers = [d for d in session.laps["Driver"].dropna().unique()]
    written, samples, skipped_drivers, no_tel_total = 0, 0, [], 0
    for drv in sorted(drivers):
        lap_rows = session.laps.pick_drivers(drv)
        frame, no_tel = _driver_frame(lap_rows)
        no_tel_total += no_tel
        if frame.empty:
            skipped_drivers.append(drv)
            print(f"    {drv}: no telemetry, skipped")
            continue
        frame.to_parquet(dest / f"{drv}.parquet", index=False)
        written += 1
        samples += len(frame)
        print(f"    {drv}: laps={frame['lap'].nunique()} samples={len(frame)}")
    return {
        "season": season,
        "round": event_round,
        "circuit": circuit,
        "status": "ok",
        "drivers_written": written,
        "samples": samples,
        "drivers_without_telemetry": skipped_drivers,
        "laps_without_telemetry": no_tel_total,
    }


def season_events(season: int) -> list[tuple[int, str, str, str]]:
    """(round, location, country, event_name) for every real (non-testing) round of a season."""
    sched = fastf1.get_event_schedule(season, include_testing=False)
    out = []
    for _, ev in sched.iterrows():
        rnd = int(ev["RoundNumber"])
        if rnd <= 0:
            continue
        out.append((rnd, str(ev["Location"]), str(ev["Country"]), str(ev["EventName"])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", type=int, nargs="+", default=[2022, 2023, 2024, 2025])
    ap.add_argument("--out", default="D:/COMP702/DF/fastf1_data", help="parquet output root")
    ap.add_argument("--cache", default="D:/COMP702/DF/fastf1_cache", help="FastF1 http cache")
    ap.add_argument("--circuits", nargs="*", default=None, help="subset of APEX circuit names")
    ap.add_argument("--refetch", action="store_true", help="re-fetch races already on disk")
    ap.add_argument("--limit", type=int, default=None, help="stop after N races (smoke test)")
    ap.add_argument("--rate-sleep-s", type=int, default=900,
                    help="seconds to wait when FastF1's 500-calls/h cap is hit")
    ap.add_argument("--rate-max-waits", type=int, default=40,
                    help="max consecutive rate-limit waits before giving up on an item")
    args = ap.parse_args()

    Path(args.cache).mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(args.cache)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    unmapped: list[str] = []
    done = 0
    for season in args.seasons:
        try:
            events = _with_rate_limit_retry(
                f"{season} schedule",
                lambda s=season: season_events(s),  # bind: the retry runs inside the loop
                sleep_s=args.rate_sleep_s, max_waits=args.rate_max_waits,
            )
        except Exception as e:
            print(f"!! season {season} schedule FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            records.append({"season": season, "status": "schedule_failed",
                            "error": f"{type(e).__name__}: {e}"})
            continue
        print(f"\n=== {season}: {len(events)} rounds ===")
        for rnd, location, country, event_name in events:
            circuit, mapped = circuit_name(location, country)
            if not mapped:
                # Loud, not silent: an unmapped circuit still gets fetched under its country name,
                # but it is recorded so the mapping can be extended before it pollutes a scaler.
                print(f"  !! UNMAPPED location {location!r} ({event_name}) -> "
                      f"using {circuit!r}", file=sys.stderr)
                unmapped.append(f"{season} r{rnd} {location} ({event_name}) -> {circuit}")
            if args.circuits and circuit not in args.circuits:
                continue
            if args.limit is not None and done >= args.limit:
                break
            dest = out_dir / str(season) / circuit
            if dest.exists() and any(dest.glob("*.parquet")) and not args.refetch:
                n = len(list(dest.glob("*.parquet")))
                print(f"[{season} r{rnd}] {circuit}: {n} parquet(s) present, "
                      "skipping (--refetch to force)")
                records.append({"season": season, "round": rnd, "circuit": circuit,
                                "status": "skipped_present", "drivers_written": n})
                done += 1
                continue
            print(f"[{season} r{rnd}] {circuit} ({event_name})")
            try:
                rec = _with_rate_limit_retry(
                    f"{season} {circuit}",
                    # bind loop vars: the retry closure runs inside the loop
                    lambda s=season, r=rnd, c=circuit: fetch_race(s, r, c, out_dir),
                    sleep_s=args.rate_sleep_s, max_waits=args.rate_max_waits,
                )
                print(f"  -> {rec['drivers_written']} drivers, {rec['samples']} samples")
            except Exception as e:
                print(f"  !! FAILED: {type(e).__name__}: {e}", file=sys.stderr)
                rec = {"season": season, "round": rnd, "circuit": circuit,
                       "status": "failed", "error": f"{type(e).__name__}: {e}"}
            records.append(rec)
            done += 1

    manifest = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "seasons": args.seasons,
        "fastf1_version": fastf1.__version__,
        "races": records,
        "unmapped_locations": unmapped,
        "n_ok": sum(1 for r in records if r.get("status") == "ok"),
        "n_skipped_present": sum(1 for r in records if r.get("status") == "skipped_present"),
        "n_failed": sum(1 for r in records if r.get("status") in {"failed", "schedule_failed"}),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"\nDONE: ok={manifest['n_ok']} skipped={manifest['n_skipped_present']} "
        f"failed={manifest['n_failed']} -> {out_dir}\nmanifest: {out_dir / 'manifest.json'}"
    )
    if manifest["n_failed"]:
        print("NOTE: failures are recorded in the manifest; Stage B will refuse to treat this "
              "corpus as complete.", file=sys.stderr)


if __name__ == "__main__":
    main()
