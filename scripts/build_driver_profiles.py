"""Derive the per-driver dataset + evaluation facts the Driver Overview screen renders.

Screen 1 states four things about each driver that the serving DB cannot answer: how many laps the
fingerprint rests on, how repeatably that driver laps, how reliably the champion identifies them,
and where they sit against the rest of the grid on each of the 28 features. All four are facts
about **one training run** -- the matrix Stage B built and the champion Stage C fitted -- so they
belong beside the model artefacts, not in the serving store, and they are regenerated whenever
either is regenerated.

Sources (both read-only):

* ``artefacts/real/training_matrix.parquet`` -- Stage B's matrix, for the lap counts, the lap-time
  consistency and the per-feature grid ranks.
* ``artefacts/real/train_real_summary.json`` -- Stage C's summary, for the champion's per-driver
  held-out F1. A driver absent from it keeps ``identification_f1: null``; the number is never
  back-filled from a training-split score, which would read as an evaluation result while being
  the model grading its own homework.

**Split discipline.** Every number is computed on the ``train`` split with the configured held-out
circuits excluded, and the exclusion is asserted rather than assumed (the same tripwire Stage B
carries). Nothing here fits anything -- these are descriptive statistics -- but the screen presents
them next to model output, so they are drawn from the same circuits the model saw, and the artefact
records which those were.

**Lap-time consistency.** A best lap over the whole matrix would just be the shortest circuit on the
calendar, so there is no cross-circuit "best lap" here. Consistency is the *median within-session
standard deviation* of clean lap times: SD inside each (season, circuit, session) group with at
least ``MIN_LAPS_PER_GROUP`` clean laps, then the median across groups. Taking the SD within a
session and only then aggregating keeps circuit and fuel-load differences out of the number, so a
driver who ran more street circuits is not penalised for it.

Run:

    .venv/Scripts/python.exe scripts/build_driver_profiles.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.contracts.config import load_config  # noqa: E402
from app.pipeline.features import FEATURE_IDS  # noqa: E402

# A session's lap-time SD is only meaningful over a handful of laps; below this the group is
# dropped from the median rather than contributing a one- or two-lap "SD".
MIN_LAPS_PER_GROUP = 5

# Grouping key for the within-session SD. `session` distinguishes FP/Q/R inside one race weekend,
# which carry very different fuel loads and therefore very different lap-time spreads.
_SESSION_KEYS = ("season", "circuit", "session")

# The consistency statistic needs a real clock; a matrix without it yields `null`, not a zero.
_LAP_TIME_COL = "lap_time_s"


def load_matrix(path: Path, held_out: set[str]) -> pd.DataFrame:
    """The train split with the held-out circuits removed, or exit non-zero saying why.

    The held-out removal is a tripwire, not a filter: Stage B already assigns them a non-train
    split, so a held-out circuit surviving the split filter means the matrix upstream is wrong and
    every number derived here would inherit that. Fail loudly instead.
    """
    if not path.exists():
        raise SystemExit(f"training matrix not found: {path}")
    matrix = pd.read_parquet(path)

    missing = [c for c in ("split", "circuit", "driver", "season") if c not in matrix.columns]
    if missing:
        raise SystemExit(f"matrix is missing required columns: {missing}")

    train = matrix[matrix["split"] == "train"]
    leaked = sorted(set(train["circuit"]) & held_out)
    if leaked:
        raise SystemExit(f"held-out circuits present in the train split: {leaked}")
    if train.empty:
        raise SystemExit("no train rows in the matrix")
    return train


def lap_time_consistency(rows: pd.DataFrame) -> float | None:
    """Median within-session clean-lap-time SD, or ``None`` when no group qualifies.

    ``None`` is the honest answer for a driver whose sessions are all shorter than
    ``MIN_LAPS_PER_GROUP`` clean laps, or for a matrix carrying no lap clock at all. The screen
    renders that as an explicit dash; it is never collapsed to 0.0, which would read as a driver
    who laps identically every time.
    """
    if _LAP_TIME_COL not in rows.columns:
        return None
    clean = rows[rows["is_clean"]] if "is_clean" in rows.columns else rows
    clean = clean[np.isfinite(clean[_LAP_TIME_COL])]
    if clean.empty:
        return None

    grouped = clean.groupby(list(_SESSION_KEYS), observed=True)[_LAP_TIME_COL]
    # ddof=1: these are samples of a driver's lap times, not the whole population of them.
    sds = grouped.std(ddof=1)[grouped.count() >= MIN_LAPS_PER_GROUP]
    sds = sds[np.isfinite(sds)]
    if sds.empty:
        return None
    return float(sds.median())


def feature_percentiles(train: pd.DataFrame, drivers: list[str]) -> dict[str, dict[str, float]]:
    """Each driver's rank on each feature, as a fraction of the grid, keyed ``[driver][feature]``.

    The rank is over per-driver **means**, so 1.0 is "highest mean of the 13 drivers on this
    feature" and 0.0 is "lowest" -- a relative statement about this grid, which is what the screen
    claims. Features whose column is absent or all-NaN are omitted for every driver rather than
    ranked against nothing.
    """
    present = [f for f in FEATURE_IDS if f in train.columns]
    means = train.groupby("driver", observed=True)[present].mean()
    means = means.reindex(drivers)

    # A feature every driver is NaN on carries no ordering; drop it rather than emit a flat 0.5.
    usable = [f for f in present if means[f].notna().sum() >= 2]
    if not usable:
        return {d: {} for d in drivers}

    # pct=True maps ranks onto (0,1]; subtracting the minimum and rescaling puts the grid's lowest
    # driver at exactly 0.0 and its highest at exactly 1.0, so "top of the grid" is unambiguous.
    ranks = means[usable].rank(pct=True)
    lo = ranks.min()
    span = (ranks.max() - lo).replace(0.0, np.nan)
    scaled = (ranks - lo) / span

    out: dict[str, dict[str, float]] = {}
    for driver in drivers:
        row = scaled.loc[driver]
        out[driver] = {f: float(row[f]) for f in usable if np.isfinite(row[f])}
    return out


def per_driver_f1(summary_path: Path) -> dict[str, float]:
    """The champion's held-out per-driver F1, or ``{}`` when the summary is absent/unreadable.

    An absent summary means the screen shows "identification accuracy unavailable" rather than a
    made-up score, so this degrades to empty instead of raising -- but it says so on stderr, because
    a profiles artefact silently missing its evaluation half is exactly the kind of quiet gap the
    house rules exist to prevent.
    """
    def warn(detail: str) -> None:
        print(f"WARNING: {detail} — identification_f1 will be null", file=sys.stderr)

    if not summary_path.exists():
        warn(f"{summary_path} not found")
        return {}
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        warn(f"{summary_path} unreadable ({exc})")
        return {}
    scores = summary.get("classifier", {}).get("held_out_per_driver_f1", {})
    if not isinstance(scores, dict):
        warn("held_out_per_driver_f1 is not a mapping")
        return {}
    return {str(k): float(v) for k, v in scores.items() if isinstance(v, (int, float))}


def build_profiles(train: pd.DataFrame, f1_by_driver: dict[str, float]) -> dict[str, dict]:
    drivers = sorted(train["driver"].unique().tolist())
    percentiles = feature_percentiles(train, drivers)

    profiles: dict[str, dict] = {}
    for driver in drivers:
        rows = train[train["driver"] == driver]
        profiles[driver] = {
            "laps_analysed": int(len(rows)),
            "seasons": sorted(int(s) for s in rows["season"].unique()),
            "n_circuits": int(rows["circuit"].nunique()),
            "lap_time_consistency_s": lap_time_consistency(rows),
            "identification_f1": f1_by_driver.get(driver),
            "feature_percentile": percentiles.get(driver, {}),
        }
    return profiles


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--matrix", type=Path, default=Path("artefacts/real/training_matrix.parquet")
    )
    parser.add_argument(
        "--summary", type=Path, default=Path("artefacts/real/train_real_summary.json")
    )
    parser.add_argument("--out", type=Path, default=Path("artefacts/driver_profiles.json"))
    args = parser.parse_args()

    config = load_config(args.config)
    held_out = set(config.held_out_circuits)

    train = load_matrix(args.matrix, held_out)
    profiles = build_profiles(train, per_driver_f1(args.summary))

    payload = {
        "source_matrix": str(args.matrix).replace("\\", "/"),
        "split": "train",
        "held_out_circuits": sorted(held_out),
        "n_laps": int(len(train)),
        "min_laps_per_consistency_group": MIN_LAPS_PER_GROUP,
        "drivers": profiles,
    }
    # Hash the content, then attach it — the same self-describing pattern normalisation.json uses,
    # so a stale artefact served against a regenerated matrix is detectable rather than plausible.
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload = {"content_hash": hashlib.sha256(body).hexdigest(), **payload}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    missing_f1 = [d for d, p in profiles.items() if p["identification_f1"] is None]
    missing_consistency = [d for d, p in profiles.items() if p["lap_time_consistency_s"] is None]
    print(f"wrote {args.out} — {len(profiles)} drivers over {len(train)} train laps")
    if missing_f1:
        print(f"  no identification_f1 for: {', '.join(missing_f1)}")
    if missing_consistency:
        print(f"  no lap-time consistency for: {', '.join(missing_consistency)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
