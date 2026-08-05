"""Seed one synthetic race into the demo database so Screen 6 (Replay Monitor) and the
degradation timeline have real data to show.

The APEX demo DB (``artefacts/apex_demo.db``) ships with drivers + fingerprints but **no
sessions / laps** — enough to render the classifier, style-space and driver screens, but the
Replay Monitor and lap-by-lap degradation view have nothing to replay (every session id 404s).
This script synthesises one Bahrain race: a ~26-lap stint per demo driver whose 28-feature lap
vectors are drawn to match the deployed standardiser's scale, with the first ``baseline_laps``
laps stable and a staggered directional drift injected afterwards so the F10 detector produces a
realistic green -> amber -> red degradation trace.

It is a **demo/dev utility**, not part of the serving path. It writes only ``sessions`` / ``laps``
/ ``lap_features`` rows; it never fits a scaler, never touches the held-out circuits, and never
alters the shipped artefacts. Drift magnitudes are auto-tuned against the *real deployed*
``monitor_stint`` so the trace is verified before anything is written, and the seeded session is
re-verified end-to-end through ``replay_stint`` afterwards.

Usage (run from the repo root with the project venv):
    python scripts/seed_demo_race.py            # dry-run: generate + verify, write nothing
    python scripts/seed_demo_race.py --write     # seed the DB (replacing any prior seeded race)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

# Allow `python scripts/seed_demo_race.py` from anywhere: put the repo root on sys.path before the
# first-party imports below (which is why they carry a noqa for the import-after-statement rule).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.contracts.config import load_config  # noqa: E402
from app.db import models  # noqa: E402
from app.monitoring.detector import load_detector, monitor_stint  # noqa: E402
from app.pipeline.features import FEATURE_IDS  # noqa: E402
from app.replay.engine import replay_stint  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncEngine

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "artefacts" / "apex_demo.db"
VAR_PATH = ROOT / "artefacts" / "stint_variation.json"
CONFIG_PATH = ROOT / "config.yaml"

# One race: Bahrain (the only circuit seeded in the demo DB), a race session.
SEASON, ROUND, SESSION_TYPE = 2023, 1, "R"
CIRCUIT_NAME, CIRCUIT_TYPE = "Bahrain", "high_speed"
N_LAPS = 26  # > features.min_stint_laps (8), < replay.max_replay_laps (200)
GEN_SEED = 2026

# Per-driver scenario. Healthy drivers get no drift (NOMINAL all race); the rest drift toward an
# ALERT at a staggered target lap so the 8-driver grid shows a spread of degradation onsets.
HEALTHY: tuple[str, ...] = ("NOR", "PIA", "SAI")
TARGET_ALERT_LAP: dict[str, int] = {"LEC": 15, "RUS": 18, "HAM": 20, "VER": 22, "ALO": 24}


def _load_scale() -> tuple[dict[str, dict[str, float]], list[str], dict[str, dict[str, float]]]:
    """The deployed F10 standardiser scale + cosine feature list + circuit-type slope stats.

    Generating laps around ``feature_scaler``'s per-feature means keeps the synthetic vectors in
    the same standardised geometry the τ thresholds were calibrated on, so a stable lap sits well
    below τ and an injected drift crosses it predictably."""
    var = json.loads(VAR_PATH.read_text(encoding="utf-8"))
    fs: dict[str, dict[str, float]] = var["feature_scaler"]  # F01..F26 (cosine core)
    cosine: list[str] = list(var["cosine_features"])
    slope_stats: dict[str, Any] = var["slope_stats"]
    slope = slope_stats.get(CIRCUIT_TYPE) or slope_stats["__global__"]
    return fs, cosine, slope


# The detector derives F27/F28 as the polyfit slope of F01 (throttle_mean) / F06 (brake_peak_mean)
# vs tyre_age across the stint — it ignores any F27/F28 column. So to keep the slope component ~0
# (a "normal" tyre-age slope, |z|~0) we make F01/F06 rise at the calibration-mean slope, and we
# EXCLUDE them from the cosine drift so the drift never distorts that slope.
_SLOPE_SOURCE = ("F01", "F06")


def _make_stint(
    driver_idx: int,
    *,
    drift_from: int | None,
    drift_mag: float,
    fs: dict[str, dict[str, float]],
    cosine: list[str],
    slope: dict[str, dict[str, float]],
) -> list[tuple[int, dict[str, float]]]:
    """Build one driver's stint as ``(lap_number, {F01..F28: value})`` rows.

    Each driver gets a large, distinct base offset (a stable standardised direction) plus small
    per-lap jitter. F01/F06 additionally rise at the calibration-mean tyre-age slope so the
    detector's F27/F28 slope component stays ~0 (|z|~0). From ``drift_from`` a **uniform negative**
    ramp is added to the other cosine features, which drives the standardised vector through zero
    and out the far side — anti-parallel to the baseline — so the cosine deviation rises toward its
    ceiling of ~2.0 and crosses τ on the cosine term alone.

    The sign pattern is the load-bearing choice, and it is a geometric consequence rather than a
    tuning knob. Every feature sits at ``mean + delta``, so the standardised baseline points along
    the all-ones direction. A BIDIRECTIONAL ramp (half +, half -) drives the vector toward a
    half-plus/half-minus pattern, which is ORTHOGONAL to all-ones: cosine distance ceilings at ~1.0
    no matter how large the magnitude grows. Against the real 2022-2025 calibration (τ ~1.17-1.71
    per driver) that ceiling is unreachable — measured peaks were 0.36 (VER) and 0.70 (ALO), and
    100x more drift moved them <0.05. A UNIFORM ramp instead flips the vector once the ramp exceeds
    ``delta``, giving a ~2.0 ceiling that clears every driver's τ with room to spare (measured
    2.06-2.35). An earlier revision used the bidirectional form against the previous synthetic
    calibration; that is why the note here once claimed the one-sided ramp saturates."""
    rng = np.random.default_rng(GEN_SEED + driver_idx)
    delta = 12.0 + 6.0 * driver_idx  # distinct per-driver base offset
    f27_rate = float(slope["F27"]["mean"])  # throttle_mean (F01) slope vs tyre age
    f28_rate = float(slope["F28"]["mean"])  # brake_peak_mean (F06) slope vs tyre age
    # Uniform-negative drift over the cosine core minus the two slope-source features (see the
    # docstring: a bidirectional ramp ceilings at cosine distance ~1.0, below every real τ).
    drift_feats = [f for f in cosine if f not in _SLOPE_SOURCE]
    drift_sign = {f: -1.0 for f in drift_feats}

    rows: list[tuple[int, dict[str, float]]] = []
    for i in range(N_LAPS):
        lap = i + 1
        vec: dict[str, float] = {}
        for f in FEATURE_IDS:
            if f in _SLOPE_SOURCE:  # rise at the calibration slope -> slope_z ~ 0
                rate = f27_rate if f == "F01" else f28_rate
                vec[f] = float(fs[f]["mean"]) + delta + rate * lap + float(rng.normal(0.0, 0.12))
            elif f in fs:  # other cosine-core features: stable base + jitter
                vec[f] = float(fs[f]["mean"]) + delta + float(rng.normal(0.0, 1.2))
            else:  # F27 / F28 columns are ignored by the detector; keep them sane, not NULL
                vec[f] = (f27_rate if f == "F27" else f28_rate)
        if drift_from is not None and lap >= drift_from:
            ramp = lap - drift_from + 1
            for f in drift_feats:
                vec[f] += drift_mag * ramp * drift_sign[f]
        rows.append((lap, vec))
    return rows


def _to_frame(rows: list[tuple[int, dict[str, float]]]) -> pd.DataFrame:
    """The monitor_stint input frame (mirrors app/replay/engine.py's reconstruction)."""
    recs: list[dict[str, Any]] = []
    for lap, vec in rows:
        rec: dict[str, Any] = {
            "lap_number": lap,
            "tyre_age": lap,
            "track_status": None,  # defer to is_clean (spec §10 R1)
            "is_clean": True,
            "has_interpolation": False,
        }
        rec.update(vec)
        recs.append(rec)
    return pd.DataFrame(recs)


def _run_detector(rows: list[tuple[int, dict[str, float]]], driver: str, handle: Any, config: Any):
    """Run the real deployed detector over the full stint; return its per-lap LapDeviation list."""
    res = monitor_stint(
        _to_frame(rows),
        driver=driver,
        circuit=CIRCUIT_NAME,
        circuit_type=CIRCUIT_TYPE,
        handle=handle,
        config=config,
        style_handle=None,
        log_path=None,
    )
    return res


def _first_alert_lap(laps: Sequence[Any]) -> int | None:
    for lap in laps:
        if lap.state.value == "ALERT":
            return lap.lap_number
    return None


def _state_summary(laps: Sequence[Any]) -> str:
    counts: dict[str, int] = {}
    for lap in laps:
        counts[lap.state.value] = counts.get(lap.state.value, 0) + 1
    order = ["BASELINE", "PENDING", "NOMINAL", "WATCH", "ALERT", "SKIPPED", "HALTED", "INACTIVE"]
    return "  ".join(f"{s}:{counts[s]}" for s in order if s in counts)


def _tune_drift(
    driver: str, driver_idx: int, target_lap: int, scale: Any, handle: Any, config: Any
):
    """Sweep drift magnitude and pick the one whose first ALERT lands nearest ``target_lap``.

    Auto-tuning against the real detector means the trace is correct regardless of the exact
    standardised geometry — no hand-tuned magic number that silently stops alerting if the
    calibration changes."""
    fs, cosine, slope = scale
    delta = 12.0 + 6.0 * driver_idx
    best: tuple[float, list, float] | None = None  # (score, rows, mag)
    alerting = 0
    # The drift only rotates the vector past tau once the ramp exceeds the per-driver base (~delta),
    # so the grid runs well past 1x; below the flip point the deviation barely moves at all.
    for frac in np.linspace(0.5, 6.0, 24):
        mag = float(frac) * delta / N_LAPS
        rows = _make_stint(
            driver_idx, drift_from=BASELINE_SAFE, drift_mag=mag, fs=fs, cosine=cosine, slope=slope
        )
        res = _run_detector(rows, driver, handle, config)
        alert = _first_alert_lap(res.laps)
        alerting += alert is not None
        # Prefer an ALERT that exists and is closest to the target; penalise no-alert heavily.
        score = abs((alert if alert is not None else 999) - target_lap)
        if best is None or score < best[0]:
            best = (score, rows, mag)
    assert best is not None
    if alerting == 0:
        # Every candidate failed to alert, so `best` is just the first grid point — seeding it
        # would write a flat NOMINAL race that looks like a healthy stint rather than a broken
        # generator. Fail loudly instead of silently shipping a demo with nothing to demo.
        raise SystemExit(
            f"{driver}: no drift magnitude in the sweep produced an ALERT (target lap "
            f"{target_lap}). The detector calibration in {VAR_PATH.name} has probably moved -- "
            "check tau against the achievable cosine deviation before widening the grid; a "
            "bidirectional drift sign ceilings at ~1.0 and cannot cross a larger tau."
        )
    return best[1], best[2]


# Drift onset: after baseline (3) + rolling window (3) have established a stable NOMINAL stretch.
BASELINE_SAFE = 7


async def _seed(engine: AsyncEngine, rows_by_driver: dict[str, list], drivers: dict[str, Any],
                circuit_id: uuid.UUID) -> uuid.UUID:
    """Replace any prior seeded race and insert the new one. Returns the new session id."""
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        prior = (
            await s.execute(
                select(models.RaceSession).where(
                    models.RaceSession.season == SEASON,
                    models.RaceSession.round == ROUND,
                    models.RaceSession.session_type == SESSION_TYPE,
                )
            )
        ).scalars().all()
        for rs in prior:
            lap_ids = (
                await s.execute(select(models.Lap.id).where(models.Lap.session_id == rs.id))
            ).scalars().all()
            if lap_ids:
                await s.execute(
                    delete(models.LapFeatures).where(models.LapFeatures.lap_id.in_(lap_ids))
                )
                await s.execute(delete(models.Lap).where(models.Lap.session_id == rs.id))
            await s.execute(delete(models.RaceSession).where(models.RaceSession.id == rs.id))
        await s.flush()

        sess = models.RaceSession(
            id=uuid.uuid4(), circuit_id=circuit_id, season=SEASON, round=ROUND,
            session_type=SESSION_TYPE,
        )
        s.add(sess)
        await s.flush()

        for code, rows in rows_by_driver.items():
            drv = drivers[code]
            for lap, vec in rows:
                lap_id = uuid.uuid4()
                s.add(models.Lap(
                    id=lap_id, session_id=sess.id, driver_id=drv.id, lap_number=lap,
                    is_clean=True, error_code="OK", tyre_compound="Soft", tyre_age=lap,
                    has_interpolation=False,
                ))
                s.add(models.LapFeatures(
                    id=uuid.uuid4(), lap_id=lap_id,
                    feature_vector={f: float(vec[f]) for f in FEATURE_IDS},
                    null_count=0, imputed_features=[], corner_type=None,
                ))
        await s.commit()
        return sess.id


async def _replay_check(engine: AsyncEngine, session_id: uuid.UUID, driver_id: uuid.UUID,
                        handle: Any, config: Any):
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        return await replay_stint(
            str(session_id), str(driver_id), session=s, monitor_handle=handle,
            config=config, style_handle=None, log_path=None,
        )


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="write to the DB (default: dry-run)")
    args = ap.parse_args()

    config = load_config(CONFIG_PATH)
    handle = load_detector(config=config)
    scale = _load_scale()
    fs, cosine, slope = scale

    codes = list(HEALTHY) + list(TARGET_ALERT_LAP)
    idx_of = {c: i for i, c in enumerate(sorted(codes))}  # stable per-driver offset by code

    print(f"Generating + verifying {N_LAPS}-lap stints for {len(codes)} drivers "
          f"({CIRCUIT_NAME}, {CIRCUIT_TYPE})\n")
    rows_by_driver: dict[str, list] = {}
    for code in sorted(codes):
        di = idx_of[code]
        if code in HEALTHY:
            rows = _make_stint(
                di, drift_from=None, drift_mag=0.0, fs=fs, cosine=cosine, slope=slope
            )
            mag = 0.0
        else:
            rows, mag = _tune_drift(code, di, TARGET_ALERT_LAP[code], scale, handle, config)
        rows_by_driver[code] = rows
        res = _run_detector(rows, code, handle, config)
        tau = next((lap.threshold for lap in res.laps if lap.threshold is not None), None)
        alert = _first_alert_lap(res.laps)
        tag = "healthy" if code in HEALTHY else f"drift@{BASELINE_SAFE} mag={mag:.3f}"
        print(f"  {code}: {_state_summary(res.laps)}")
        print(f"        tau={tau:.3f}  first ALERT={alert}  [{tag}]")

    if not args.write:
        print("\nDry-run only (no DB write). Re-run with --write to seed apex_demo.db.")
        return

    engine = create_async_engine(f"sqlite+aiosqlite:///{DB_PATH.as_posix()}")
    try:
        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with factory() as s:
            circuit = (
                await s.execute(select(models.Circuit).where(models.Circuit.name == CIRCUIT_NAME))
            ).scalar_one()
            drv_rows = (await s.execute(select(models.Driver))).scalars().all()
            drivers = {d.code: d for d in drv_rows if d.code in codes}
            missing = set(codes) - set(drivers)
            if missing:
                raise SystemExit(f"drivers missing from demo DB: {sorted(missing)}")
            circuit_id = circuit.id

        session_id = await _seed(engine, rows_by_driver, drivers, circuit_id)
        print(f"\nSeeded session id: {session_id}")

        # End-to-end re-verification through the actual replay engine.
        print("\nReplay-engine verification (what the UI will fetch):")
        for code in ("VER", "NOR"):
            drv = drivers[code]
            res = await _replay_check(engine, session_id, drv.id, handle, config)
            states = [lap.alert_state for lap in res.laps]
            counts: dict[str, int] = {}
            for st in states:
                counts[st] = counts.get(st, 0) + 1
            first_alert = next((lap.lap_number for lap in res.laps if lap.alert_state == "ALERT"),
                               None)
            print(f"  {code} ({drv.id}): laps={len(res.laps)} found={res.found} "
                  f"partial={res.partial} first ALERT={first_alert}  {counts}")
    finally:
        await engine.dispose()

    print("\nDone. In the Replay Monitor paste the session id above; pick any driver, or use "
          "\"All drivers\" for the grid.")


if __name__ == "__main__":
    asyncio.run(main())
