"""``scripts/build_driver_profiles.py`` — the Driver Overview facts artefact.

Load-bearing cases: the held-out tripwire in :func:`load_matrix` (a leaked circuit must kill the
build, not tint the statistics), the ``None``-vs-``0.0`` discipline in
:func:`lap_time_consistency`, and the exact 0→1 grid scaling in :func:`feature_percentiles`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import build_driver_profiles as bdp

HELD_OUT = {"Belgium", "Italy"}


def matrix_rows(
    *,
    driver: str = "VER",
    circuit: str = "Bahrain",
    season: int = 2024,
    session: str = "R",
    lap_times: list[float] | None = None,
    split: str = "train",
    **features: float,
) -> pd.DataFrame:
    times = lap_times if lap_times is not None else [90.0]
    return pd.DataFrame(
        [
            {
                "season": season,
                "circuit": circuit,
                "driver": driver,
                "session": session,
                "split": split,
                "is_clean": True,
                "lap_time_s": t,
                **features,
            }
            for t in times
        ]
    )


# --- load_matrix: the held-out tripwire ------------------------------------------------------


def test_load_matrix_dies_on_a_leaked_held_out_circuit(tmp_path: Path) -> None:
    """A held-out circuit inside the train split is an upstream-matrix bug — every statistic
    derived from it would inherit the leak, so the build must die, not filter."""
    leaked = pd.concat(
        [matrix_rows(circuit="Bahrain"), matrix_rows(circuit="Belgium")], ignore_index=True
    )
    path = tmp_path / "m.parquet"
    leaked.to_parquet(path)
    with pytest.raises(SystemExit, match="Belgium"):
        bdp.load_matrix(path, HELD_OUT)


def test_load_matrix_dies_on_missing_file_and_missing_columns(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="not found"):
        bdp.load_matrix(tmp_path / "absent.parquet", HELD_OUT)
    bad = matrix_rows().drop(columns=["split"])
    path = tmp_path / "bad.parquet"
    bad.to_parquet(path)
    with pytest.raises(SystemExit, match="split"):
        bdp.load_matrix(path, HELD_OUT)


def test_load_matrix_keeps_only_the_train_split(tmp_path: Path) -> None:
    m = pd.concat(
        [matrix_rows(circuit="Bahrain"), matrix_rows(circuit="Italy", split="held_out")],
        ignore_index=True,
    )
    path = tmp_path / "m.parquet"
    m.to_parquet(path)
    out = bdp.load_matrix(path, HELD_OUT)
    assert set(out["circuit"]) == {"Bahrain"}  # held-out rows carry a non-train split → dropped


# --- lap_time_consistency: None is a state, not zero -----------------------------------------


def test_consistency_is_the_median_of_within_session_sds() -> None:
    """Two qualifying sessions with different spreads → the median of the two SDs, computed
    within each session first so circuit/fuel offsets never enter the number."""
    tight = matrix_rows(session="R", circuit="Bahrain", lap_times=[90.0, 90.1, 90.2, 90.1, 90.0])
    wide = matrix_rows(session="Q", circuit="Monaco", lap_times=[70.0, 71.0, 72.0, 71.0, 70.0])
    rows = pd.concat([tight, wide], ignore_index=True)
    expected = float(
        np.median(
            [
                np.std([90.0, 90.1, 90.2, 90.1, 90.0], ddof=1),
                np.std([70.0, 71.0, 72.0, 71.0, 70.0], ddof=1),
            ]
        )
    )
    assert bdp.lap_time_consistency(rows) == pytest.approx(expected)


def test_consistency_is_none_when_no_session_reaches_the_lap_floor() -> None:
    rows = matrix_rows(lap_times=[90.0, 90.1])  # 2 laps < MIN_LAPS_PER_GROUP
    assert bdp.lap_time_consistency(rows) is None


def test_consistency_ignores_dirty_laps() -> None:
    rows = matrix_rows(lap_times=[90.0, 90.1, 90.2, 90.1, 90.0, 300.0])
    rows.loc[rows["lap_time_s"] == 300.0, "is_clean"] = False  # the outlier is a flagged lap
    clean_sd = float(np.std([90.0, 90.1, 90.2, 90.1, 90.0], ddof=1))
    assert bdp.lap_time_consistency(rows) == pytest.approx(clean_sd)


def test_consistency_is_none_without_a_lap_clock() -> None:
    rows = matrix_rows().drop(columns=["lap_time_s"])
    assert bdp.lap_time_consistency(rows) is None


# --- feature_percentiles: exact 0→1 grid scaling ---------------------------------------------


def test_percentiles_span_exactly_zero_to_one() -> None:
    train = pd.concat(
        [
            matrix_rows(driver="LOW", F01=1.0),
            matrix_rows(driver="MID", F01=2.0),
            matrix_rows(driver="TOP", F01=3.0),
        ],
        ignore_index=True,
    )
    out = bdp.feature_percentiles(train, ["LOW", "MID", "TOP"])
    assert out["LOW"]["F01"] == 0.0  # the grid's lowest is exactly 0
    assert out["MID"]["F01"] == pytest.approx(0.5)
    assert out["TOP"]["F01"] == 1.0  # the grid's highest is exactly 1


def test_percentiles_omit_features_nobody_measures() -> None:
    """An all-NaN feature carries no ordering — omitted for everyone, never a flat 0.5."""
    train = pd.concat(
        [matrix_rows(driver="AAA", F01=1.0), matrix_rows(driver="BBB", F01=2.0)],
        ignore_index=True,
    )
    train["F02"] = np.nan
    out = bdp.feature_percentiles(train, ["AAA", "BBB"])
    assert "F02" not in out["AAA"]
    assert "F02" not in out["BBB"]


# --- per_driver_f1 + build_profiles ----------------------------------------------------------


def test_per_driver_f1_degrades_to_empty_when_summary_absent(tmp_path: Path, capsys) -> None:
    assert bdp.per_driver_f1(tmp_path / "absent.json") == {}
    assert "identification_f1 will be null" in capsys.readouterr().err


def test_build_profiles_keeps_unscored_drivers_null(tmp_path: Path) -> None:
    """A driver missing from the champion summary keeps identification_f1 None — the score is
    never back-filled from anywhere else."""
    train = pd.concat(
        [matrix_rows(driver="VER", F01=1.0), matrix_rows(driver="UNK", F01=2.0)],
        ignore_index=True,
    )
    profiles = bdp.build_profiles(train, {"VER": 0.53})
    assert profiles["VER"]["identification_f1"] == 0.53
    assert profiles["UNK"]["identification_f1"] is None
    assert profiles["UNK"]["laps_analysed"] == 1


# --- signed_shap_means: sign comes ONLY from the beeswarm -------------------------------------


def test_signed_means_average_per_driver_feature(tmp_path: Path) -> None:
    import json

    beeswarm = {
        "drivers": ["VER"],
        "features": ["F01"],
        "points": [
            {"driver": "VER", "feature_id": "F01", "feature_value": 1.0, "shap": 0.4},
            {"driver": "VER", "feature_id": "F01", "feature_value": 2.0, "shap": -0.2},
            {"driver": "LEC", "feature_id": "F02", "feature_value": 3.0, "shap": -0.3},
        ],
    }
    path = tmp_path / "bees.json"
    path.write_text(json.dumps(beeswarm), encoding="utf-8")
    out = bdp.signed_shap_means(path)
    assert out["VER"]["F01"] == pytest.approx(0.1)  # mean of +0.4 and -0.2 — sign preserved
    assert out["LEC"]["F02"] == pytest.approx(-0.3)


def test_signed_means_degrade_to_empty_with_warning(tmp_path: Path, capsys) -> None:
    """Missing or malformed beeswarm → {} + stderr warning; the screen then omits the signed
    panel instead of faking directions."""
    assert bdp.signed_shap_means(tmp_path / "absent.json") == {}
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    assert bdp.signed_shap_means(tmp_path / "bad.json") == {}
    err = capsys.readouterr().err
    assert err.count("shap_signed_mean omitted") == 2


def test_build_profiles_carries_signed_means_per_driver(tmp_path: Path) -> None:
    train = pd.concat(
        [matrix_rows(driver="VER", F01=1.0), matrix_rows(driver="UNK", F01=2.0)],
        ignore_index=True,
    )
    profiles = bdp.build_profiles(train, {}, {"VER": {"F01": 0.25}})
    assert profiles["VER"]["shap_signed_mean"] == {"F01": 0.25}
    assert profiles["UNK"]["shap_signed_mean"] == {}  # absent driver -> empty, never borrowed


# --- sample_segments: real held-out laps for the Prediction screen ----------------------------


def test_segments_come_only_from_clean_held_out_laps(tmp_path: Path) -> None:
    m = pd.concat(
        [
            matrix_rows(driver="VER", circuit="Bahrain", split="train", F01=1.0),
            matrix_rows(driver="VER", circuit="Belgium", split="test", F01=2.0),
            matrix_rows(driver="LEC", circuit="Italy", split="test", F01=3.0),
        ],
        ignore_index=True,
    )
    m["lap_number"] = [10, 20, 30]
    dirty = matrix_rows(driver="LEC", circuit="Belgium", split="test", F01=9.0)
    dirty["is_clean"] = False
    dirty["lap_number"] = 40
    m = pd.concat([m, dirty], ignore_index=True)
    path = tmp_path / "m.parquet"
    m.to_parquet(path)

    segments = bdp.sample_segments(path, HELD_OUT, seed=42)
    by_driver = {s["true_driver"]: s for s in segments}
    assert set(by_driver) == {"VER", "LEC"}
    assert by_driver["VER"]["circuit"] == "Belgium"  # never the train lap
    assert by_driver["LEC"]["lap_number"] == 30  # never the dirty lap
    assert by_driver["VER"]["features"]["F01"] == 2.0  # raw value carried as-is


def test_segments_selection_is_seeded_deterministic(tmp_path: Path) -> None:
    rows = []
    for lap in range(1, 9):
        r = matrix_rows(driver="VER", circuit="Belgium", split="test", F01=float(lap))
        r["lap_number"] = lap
        rows.append(r)
    path = tmp_path / "m.parquet"
    pd.concat(rows, ignore_index=True).to_parquet(path)
    first = bdp.sample_segments(path, HELD_OUT, seed=42)
    second = bdp.sample_segments(path, HELD_OUT, seed=42)
    assert first == second  # reproducibility — the artefact is stable for a given matrix + seed


def test_segments_nan_features_become_null_not_zero(tmp_path: Path) -> None:
    m = matrix_rows(driver="VER", circuit="Belgium", split="test", F01=np.nan)
    m["lap_number"] = 5
    path = tmp_path / "m.parquet"
    m.to_parquet(path)
    segments = bdp.sample_segments(path, HELD_OUT, seed=42)
    assert segments[0]["features"]["F01"] is None  # a genuine NULL, never a fabricated 0.0


def test_segments_empty_pool_warns_and_returns_empty(tmp_path: Path, capsys) -> None:
    path = tmp_path / "m.parquet"
    matrix_rows(split="train").assign(lap_number=1).to_parquet(path)
    assert bdp.sample_segments(path, HELD_OUT, seed=42) == []
    assert "segments omitted" in capsys.readouterr().err


# --- tyre wear: the fuel-corrected degradation model ------------------------------------------


def race_laps(
    *,
    driver: str = "VER",
    compound: str = "MEDIUM",
    stint: int = 1,
    start_lap: int = 1,
    n: int = 10,
    base: float = 90.0,
    fuel: float = -0.06,
    tyre: float = 0.05,
    circuit: str = "Bahrain",
    season: int = 2024,
) -> pd.DataFrame:
    """Clean race laps whose lap time is base + fuel*lap_number + tyre*tyre_age exactly."""
    rows = []
    for i in range(n):
        lap_number = start_lap + i
        age = i + 1
        rows.append(
            {
                "season": season,
                "circuit": circuit,
                "driver": driver,
                "session": "R",
                "split": "train",
                "is_clean": True,
                "compound": compound,
                "stint": stint,
                "tyre_age": age,
                "lap_number": lap_number,
                "lap_time_s": base + fuel * lap_number + tyre * age,
            }
        )
    return pd.DataFrame(rows)


def two_stint_race(driver: str = "VER", tyre: float = 0.05, fuel: float = -0.06) -> pd.DataFrame:
    """One driver, two stints — the pit stop resets tyre age, which identifies the fuel term."""
    return pd.concat(
        [
            race_laps(driver=driver, stint=1, start_lap=1, n=18, tyre=tyre, fuel=fuel),
            race_laps(driver=driver, stint=2, start_lap=19, n=18, tyre=tyre, fuel=fuel),
        ],
        ignore_index=True,
    )


def test_fuel_slope_identified_from_pit_stop_structure() -> None:
    """With two stints, lap_number and tyre_age decouple and the fuel term is recovered."""
    race = pd.concat([two_stint_race("VER"), two_stint_race("LEC")], ignore_index=True)
    slopes = bdp.race_fuel_slopes(race)
    assert slopes[(2024, "Bahrain")] == pytest.approx(-0.06, abs=1e-6)


def test_single_stint_race_is_skipped_not_guessed() -> None:
    """One stint → tyre age and race lap are collinear; the race must be omitted, not fitted."""
    race = pd.concat(
        [
            race_laps(driver="VER", stint=1, n=30),
            race_laps(driver="LEC", stint=1, n=30),
        ],
        ignore_index=True,
    )
    assert bdp.race_fuel_slopes(race) == {}


def test_tyre_wear_recovers_the_true_tyre_slope() -> None:
    """The fuel-corrected within-stint slope equals the constructed tyre term, not the net."""
    train = pd.concat([two_stint_race("VER"), two_stint_race("LEC")], ignore_index=True)
    out = bdp.tyre_wear(train)
    stats = out["VER"]["MEDIUM"]
    assert stats["deg_slope_s_per_lap"] == pytest.approx(0.05, abs=1e-6)
    assert stats["fuel_slope_s_per_lap"] == pytest.approx(-0.06, abs=1e-6)
    assert stats["n_stints"] == 2


def test_tyre_wear_baseline_excludes_the_warm_up_lap() -> None:
    """Age-1 carries out-lap time; it must appear in the curve but never anchor the baseline.

    Counts are per (driver, compound): four two-stint races per driver → 8 laps per tyre age,
    clearing MIN_LAPS_PER_AGE so the curve exists.
    """
    circuits = ("Bahrain", "Austria", "Japan", "Brazil")
    race = pd.concat(
        [
            pd.concat(
                [
                    race_laps(driver=d, stint=1, start_lap=1, n=18, circuit=c),
                    race_laps(driver=d, stint=2, start_lap=19, n=18, circuit=c),
                ],
                ignore_index=True,
            )
            for d in ("VER", "LEC")
            for c in circuits
        ],
        ignore_index=True,
    )
    race.loc[race["tyre_age"] == 1, "lap_time_s"] += 3.0  # warm-up cost on every age-1 lap
    out = bdp.tyre_wear(race)
    curve = {c["age"]: c["delta_s"] for c in out["VER"]["MEDIUM"]["curve"]}
    assert curve[1] == pytest.approx(3.0 + 0.05 * (1 - 2.5), abs=0.05)  # warm-up visible in curve
    assert abs(curve[2]) < 0.1  # baseline anchored past the warm-up → age-2 delta ≈ 0


def test_tyre_wear_excludes_wet_and_non_race_laps() -> None:
    train = pd.concat(
        [
            two_stint_race("VER"),
            two_stint_race("LEC"),
            race_laps(driver="VER", compound="INTERMEDIATE", stint=3, start_lap=40, n=10),
        ],
        ignore_index=True,
    )
    quali = race_laps(driver="VER", stint=1, n=10)
    quali["session"] = "Q"
    train = pd.concat([train, quali], ignore_index=True)
    out = bdp.tyre_wear(train)
    assert set(out["VER"]) == {"MEDIUM"}  # no INTERMEDIATE block, no quali contamination


def test_detect_cliff_fires_only_on_real_acceleration() -> None:
    ages = list(range(2, 22))
    linear = [0.05 * a for a in ages]
    assert bdp._detect_cliff(ages, linear) is None  # near-linear → no decorative cliff
    cliffed = [0.05 * a + (1.5 if a >= 17 else 0.0) for a in ages]
    assert bdp._detect_cliff(ages, cliffed) == 17
