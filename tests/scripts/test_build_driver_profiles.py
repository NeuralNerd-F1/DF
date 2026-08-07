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
