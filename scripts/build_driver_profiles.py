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

# --- Tyre-wear derivation constants ----------------------------------------------------------
# Dry slicks only: wet-weather laps answer a different question (track state, not tyre wear).
DRY_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
# A stint shorter than this cannot anchor a baseline AND show drift beyond it.
MIN_STINT_CLEAN_LAPS = 8
# A curve point needs this many laps across stints, or it is noise presented as signal.
MIN_LAPS_PER_AGE = 8
# The stint's pace baseline = its first N observed tyre ages AT OR ABOVE this age. Age-1 laps
# carry out-lap/warm-up time and belong in the curve as data, never in the baseline.
BASELINE_AGES = 2
BASELINE_MIN_AGE = 2
# A race needs at least this many laps (across the field) to estimate its fuel slope.
MIN_RACE_LAPS_FOR_FUEL = 60
# Cliff detection: the linear reference is fitted on this leading fraction of the curve...
CLIFF_FIT_FRACTION = 0.6
# ...and a cliff needs the median delta to exceed the linear prediction by at least this many
# seconds (or 4 robust SDs of the fit residuals, whichever is larger) for 2 consecutive ages.
CLIFF_MIN_EXCESS_S = 0.30
CLIFF_MAD_MULTIPLIER = 4.0
CLIFF_CONSECUTIVE = 2

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


def signed_shap_means(beeswarm_path: Path) -> dict[str, dict[str, float]]:
    """Mean **signed** SHAP per (driver, feature) from the F07 beeswarm artefact, or ``{}``.

    The DB's ``shap_global`` stores mean |SHAP| — magnitude only. The beeswarm artefact keeps the
    per-lap signed values, so their mean per feature says which way the feature pushes the
    champion's log-odds for that driver's own laps: positive = toward the driver (confirms),
    negative = away (argues against). This is the only place the sign is computed; it is never
    inferred from the magnitude. Absent/malformed artefact degrades to empty with a warning —
    the profile screen then omits the signed panel instead of faking directions.
    """
    def warn(detail: str) -> None:
        print(f"WARNING: {detail} — shap_signed_mean omitted", file=sys.stderr)

    if not beeswarm_path.exists():
        warn(f"{beeswarm_path} not found")
        return {}
    try:
        beeswarm = json.loads(beeswarm_path.read_text(encoding="utf-8"))
        points = beeswarm["points"]
        frame = pd.DataFrame(points, columns=["driver", "feature_id", "shap"])
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        warn(f"{beeswarm_path} unreadable ({exc})")
        return {}
    if frame.empty:
        warn(f"{beeswarm_path} carries no points")
        return {}
    means = frame.groupby(["driver", "feature_id"], observed=True)["shap"].mean()
    out: dict[str, dict[str, float]] = {}
    for (driver, feature), value in means.items():
        if np.isfinite(value):
            out.setdefault(str(driver), {})[str(feature)] = float(value)
    return out


def _detect_cliff(ages: list[int], deltas: list[float]) -> int | None:
    """First tyre age where the aggregated delta curve accelerates clearly past its linear trend.

    The reference line is fitted on the leading ``CLIFF_FIT_FRACTION`` of the curve; a cliff is
    called only when the median delta exceeds the line's prediction by ``max(CLIFF_MIN_EXCESS_S,
    CLIFF_MAD_MULTIPLIER × residual MAD)`` for ``CLIFF_CONSECUTIVE`` consecutive ages. Returns
    ``None`` when the criterion never fires — most driver/compound curves are near-linear and an
    "estimated cliff" on those would be decoration, not detection.
    """
    n_fit = int(len(ages) * CLIFF_FIT_FRACTION)
    if n_fit < 3 or len(ages) - n_fit < CLIFF_CONSECUTIVE:
        return None
    x = np.asarray(ages[:n_fit], dtype=float)
    y = np.asarray(deltas[:n_fit], dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    residuals = y - (slope * x + intercept)
    mad = float(np.median(np.abs(residuals - np.median(residuals))))
    threshold = max(CLIFF_MIN_EXCESS_S, CLIFF_MAD_MULTIPLIER * 1.4826 * mad)

    run = 0
    for age, delta in zip(ages[n_fit:], deltas[n_fit:], strict=True):
        predicted = slope * age + intercept
        if delta - predicted > threshold:
            run += 1
            if run >= CLIFF_CONSECUTIVE:
                return int(ages[n_fit:][ages[n_fit:].index(age) - CLIFF_CONSECUTIVE + 1])
        else:
            run = 0
    return None


def race_fuel_slopes(laps: pd.DataFrame) -> dict[tuple[int, str], float]:
    """Per-race fuel slope (s per race lap), identified from the multi-stint structure.

    Within one stint, tyre age and race lap rise in lockstep — fuel and tyre are perfectly
    collinear and cannot be separated. Across a WHOLE race they decouple: pit stops reset tyre
    age while the race lap keeps counting, so a regression over all of a race's clean dry laps
    (per-driver demeaned, so car/driver pace differences drop out) of

        lap_time ~ b_fuel * lap_number + b_tyre * tyre_age

    identifies the fuel term from exactly those resets. ``b_fuel`` is pooled across the field —
    fuel burn is a property of the race, not the driver. Races with too few laps or a singular
    design (nobody pitted) are omitted; their laps are then excluded from the tyre curves rather
    than corrected with a number from a different circuit.
    """
    slopes: dict[tuple[int, str], float] = {}
    for (season, circuit), race in laps.groupby(["season", "circuit"], observed=True):
        if len(race) < MIN_RACE_LAPS_FOR_FUEL:
            continue
        # Demean per driver: absorbs car pace so the fit sees only within-driver evolution.
        demeaned = race[["lap_time_s", "lap_number", "tyre_age"]].astype(float).copy()
        for col in demeaned.columns:
            demeaned[col] = demeaned[col] - demeaned.groupby(race["driver"], observed=True)[
                col
            ].transform("mean")
        x = demeaned[["lap_number", "tyre_age"]].to_numpy()
        # A race where lap_number and tyre_age stay collinear (no pit stops in the clean laps)
        # cannot separate the two — detectable as a near-degenerate design matrix.
        if np.linalg.matrix_rank(x) < 2 or np.linalg.cond(x.T @ x) > 1e6:
            continue
        coef, *_ = np.linalg.lstsq(x, demeaned["lap_time_s"].to_numpy(), rcond=None)
        slopes[(int(season), str(circuit))] = float(coef[0])
    return slopes


def tyre_wear(train: pd.DataFrame) -> dict[str, dict[str, dict]]:
    """Per (driver, compound) stint-degradation statistics from clean RACE laps.

    The naive within-stint pace slope is a fuel/tyre blend — and on this data the blend is
    dominated by fuel (net curves FALL as the car burns ~1.5 kg/lap). So the model corrects it:
    :func:`race_fuel_slopes` identifies each race's fuel term from its pit-stop structure, the
    correction ``lap_time − b_fuel × lap_number`` strips it, and everything below (baselines,
    curves, slopes, cliffs) runs on the corrected times. Races whose fuel term cannot be
    identified are excluded, not guessed. Age-1 laps stay in the curve (the out-lap/warm-up cost
    is real data) but never anchor a baseline.

    F27/F28 (the pipeline's own behavioural tyre-age slopes: throttle and brake adaptation) are
    aggregated per compound as medians — the behavioural half of the tyre story, and the part
    APEX is actually about.
    """
    required = {"compound", "stint", "tyre_age", "lap_time_s", "lap_number", "session"}
    if not required.issubset(train.columns):
        print(f"WARNING: matrix lacks {sorted(required - set(train.columns))} — tyre_wear omitted",
              file=sys.stderr)
        return {}

    laps = train[
        (train["session"] == "R")  # races only: multi-stint fuel structure + long stints
        & train["compound"].isin(DRY_COMPOUNDS)
        & np.isfinite(train["lap_time_s"])
        & train["tyre_age"].notna()
        & train["lap_number"].notna()
    ].copy()
    laps["tyre_age"] = laps["tyre_age"].astype(int)

    fuel = race_fuel_slopes(laps)
    covered = np.array(
        [(int(s), str(c)) in fuel for s, c in laps[["season", "circuit"]].to_numpy()]
    )
    dropped = int((~covered).sum())
    if dropped:
        print(f"tyre_wear: {dropped} laps dropped (fuel slope unidentifiable)", file=sys.stderr)
    laps = laps[covered]
    if laps.empty:
        print("WARNING: no fuel-corrected race laps — tyre_wear omitted", file=sys.stderr)
        return {}
    fuel_of = laps.apply(lambda r: fuel[(int(r["season"]), str(r["circuit"]))], axis=1)
    laps["corrected_s"] = laps["lap_time_s"] - fuel_of * laps["lap_number"].astype(float)

    out: dict[str, dict[str, dict]] = {}
    stint_keys = ["season", "circuit", "session", "stint"]
    for (driver, compound), group in laps.groupby(["driver", "compound"], observed=True):
        stint_slopes: list[float] = []
        stint_lengths: list[int] = []
        delta_rows: list[pd.DataFrame] = []
        for _, stint in group.groupby(stint_keys, observed=True):
            if len(stint) < MIN_STINT_CLEAN_LAPS:
                continue
            stint = stint.sort_values("tyre_age")
            # Baseline from the earliest ages past warm-up; a stint with no post-warm-up laps
            # cannot anchor a delta and is skipped.
            eligible = sorted(a for a in stint["tyre_age"].unique() if a >= BASELINE_MIN_AGE)
            if not eligible:
                continue
            baseline_ages = eligible[:BASELINE_AGES]
            baseline = float(stint[stint["tyre_age"].isin(baseline_ages)]["corrected_s"].median())
            body = stint[stint["tyre_age"] >= BASELINE_MIN_AGE]  # slope past warm-up only
            slope = float(np.polyfit(body["tyre_age"], body["corrected_s"], 1)[0])
            stint_slopes.append(slope)
            stint_lengths.append(int(len(stint)))
            delta_rows.append(
                pd.DataFrame(
                    {"tyre_age": stint["tyre_age"], "delta": stint["corrected_s"] - baseline}
                )
            )
        if not stint_slopes:
            continue

        deltas = pd.concat(delta_rows, ignore_index=True)
        per_age = deltas.groupby("tyre_age")["delta"]
        counts = per_age.count()
        ages = sorted(int(a) for a in counts[counts >= MIN_LAPS_PER_AGE].index)
        curve = [
            {
                "age": age,
                "delta_s": float(per_age.median()[age]),
                "q25": float(per_age.quantile(0.25)[age]),
                "q75": float(per_age.quantile(0.75)[age]),
                "n": int(counts[age]),
            }
            for age in ages
        ]

        median_len = int(np.median(stint_lengths))
        slope_s_per_lap = float(np.median(stint_slopes))
        group_fuel = [
            fuel[(int(s), str(c))] for s, c in group[["season", "circuit"]].drop_duplicates().values
        ]
        stats: dict = {
            "n_laps": int(len(group)),
            "n_stints": len(stint_slopes),
            "median_stint_len": median_len,
            "max_stint_len": int(max(stint_lengths)),
            "deg_slope_s_per_lap": slope_s_per_lap,
            "total_delta_s": slope_s_per_lap * max(0, median_len - 1),
            "fuel_slope_s_per_lap": float(np.median(group_fuel)) if group_fuel else None,
            "cliff_age": _detect_cliff([c["age"] for c in curve], [c["delta_s"] for c in curve]),
            "curve": curve,
        }
        for fid, key in (("F27", "f27_median"), ("F28", "f28_median")):
            stats[key] = (
                float(group[fid].median())
                if fid in group.columns and group[fid].notna().any()
                else None
            )
        out.setdefault(str(driver), {})[str(compound)] = stats
    return out


def sample_segments(matrix_path: Path, held_out: set[str], seed: int) -> list[dict]:
    """One real held-out clean lap per driver — the Prediction screen's segment picker.

    Held-out laps (Belgium/Italy — circuits no model ever trained on) are the honest choice for a
    "watch the classifier work" demo: the prediction is a genuine generalisation test, and the
    matrix records the true driver so the screen can grade the answer. Selection is seeded, so the
    artefact — and therefore the screen — is reproducible from the repo. Raw features go out as
    written (NaN → null); /predict applies the promoted normalisation exactly as evaluation does.
    """
    matrix = pd.read_parquet(matrix_path)
    pool = matrix[(matrix["split"] != "train") & matrix["is_clean"]]
    pool = pool[pool["circuit"].isin(held_out)]
    if pool.empty:
        print("WARNING: no held-out clean laps — segments omitted", file=sys.stderr)
        return []

    session_names = {"R": "Race", "Q": "Qualifying", "S": "Sprint"}
    rng = np.random.default_rng(seed)
    segments: list[dict] = []
    for driver, group in pool.groupby("driver", sort=True):
        row = group.iloc[int(rng.integers(len(group)))]
        features = {
            fid: (None if pd.isna(row[fid]) else float(row[fid]))
            for fid in FEATURE_IDS
            if fid in row.index
        }
        session = session_names.get(str(row["session"]), str(row["session"]))
        circuit, season, lap = str(row["circuit"]), int(row["season"]), int(row["lap_number"])
        segments.append(
            {
                "id": f"{circuit}-{season}-{driver}-{lap}",
                "label": f"{circuit} {season} · {session} · Lap {lap}",
                "circuit": str(row["circuit"]),
                "season": int(row["season"]),
                "session": session,
                "lap_number": int(row["lap_number"]),
                "true_driver": str(driver),
                "features": features,
            }
        )
    return segments


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


def build_profiles(
    train: pd.DataFrame,
    f1_by_driver: dict[str, float],
    shap_signed: dict[str, dict[str, float]] | None = None,
) -> dict[str, dict]:
    drivers = sorted(train["driver"].unique().tolist())
    percentiles = feature_percentiles(train, drivers)
    signed = shap_signed or {}
    wear = tyre_wear(train)

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
            "shap_signed_mean": signed.get(driver, {}),
            "tyre_wear": wear.get(driver, {}),
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
    parser.add_argument("--beeswarm", type=Path, default=Path("artefacts/shap_beeswarm.json"))
    parser.add_argument("--out", type=Path, default=Path("artefacts/driver_profiles.json"))
    args = parser.parse_args()

    config = load_config(args.config)
    held_out = set(config.held_out_circuits)

    train = load_matrix(args.matrix, held_out)
    profiles = build_profiles(
        train, per_driver_f1(args.summary), signed_shap_means(args.beeswarm)
    )
    segments = sample_segments(args.matrix, held_out, seed=config.seeds.numpy)

    payload = {
        "source_matrix": str(args.matrix).replace("\\", "/"),
        "split": "train",
        "held_out_circuits": sorted(held_out),
        "n_laps": int(len(train)),
        "min_laps_per_consistency_group": MIN_LAPS_PER_GROUP,
        "drivers": profiles,
        "segments": segments,
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
