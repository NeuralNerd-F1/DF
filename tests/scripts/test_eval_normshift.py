"""Evaluation A/B mechanisms — the leakage-critical machinery, on synthetic frames.

The real curve lives in the content-hashed artefact; these tests pin what MUST hold for the
protocol to be defensible: calibration/scoring disjointness, chronological-clean selection, the
degenerate-stats fallback, and the transform's NULL honesty.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts import eval_normshift as ns


def _frame() -> pd.DataFrame:
    rows = []
    for season in (2022, 2023):
        for lap in range(1, 7):
            rows.append(
                {
                    "circuit": "Belgium",
                    "season": season,
                    "session": "R",
                    "lap_number": lap,
                    "is_clean": lap != 3,  # lap 3 dirty each season
                    "F01": float(lap),
                    "F09": float(season * 10 + lap),
                    "driver": "VER" if lap % 2 else "HAM",
                }
            )
    return pd.DataFrame(rows)


def test_calibration_split_is_disjoint_chronological_and_clean_only() -> None:
    df = _frame()
    calib, remainder = ns.calibration_split(df, "Belgium", 4)
    assert len(calib) == 4
    assert set(calib.index).isdisjoint(set(remainder.index))  # the leakage-critical property
    assert len(calib) + len(remainder) == len(df)
    assert calib["is_clean"].all()  # dirty laps never calibrate
    # Chronological: all four come from the EARLIER season (2022 has 5 clean laps).
    assert (calib["season"] == 2022).all()
    # The dirty 2022 lap is not in calibration but IS still scored.
    assert ((remainder["season"] == 2022) & (~remainder["is_clean"])).any()


def test_calibration_split_short_circuit_takes_what_exists() -> None:
    calib, remainder = ns.calibration_split(_frame(), "Belgium", 999)
    assert len(calib) == 10  # all clean laps
    assert len(remainder) == 2  # only the dirty ones remain to score


def test_circuit_stats_and_degenerate_fallback() -> None:
    calib, _ = ns.calibration_split(_frame(), "Belgium", 4)
    global_stats = {"F09": (0.0, 1.0), "F13": (5.0, 2.0)}
    stats = ns.circuit_stats(
        calib, ["F09", "F13"], ddof=0, min_std=1e-8, global_stats=global_stats
    )
    vals = pd.to_numeric(calib["F09"], errors="coerce")
    assert stats["F09"][0] == float(vals.mean())  # computed from calibration laps only
    assert stats["F13"] == (5.0, 2.0)  # feature absent in the frame → global fallback, no crash


def test_circuit_stats_zero_variance_falls_back() -> None:
    calib = pd.DataFrame({"F09": [3.0, 3.0, 3.0]})
    stats = ns.circuit_stats(
        calib, ["F09"], ddof=0, min_std=1e-8, global_stats={"F09": (1.0, 9.0)}
    )
    assert stats["F09"] == (1.0, 9.0)  # mirrors the serving normaliser's fallback semantics


def test_normalise_frame_transforms_only_named_features_and_keeps_nans() -> None:
    frame = pd.DataFrame({"F01": [2.0, np.nan], "F09": [10.0, 20.0]})
    x = ns.normalise_frame(frame, ["F09"], ["F01", "F09"], {"F09": (10.0, 5.0)})
    assert x[0, 0] == 2.0  # non-normalised feature verbatim
    assert np.isnan(x[1, 0])  # NULL stays NaN — never imputed to a number here
    assert x[0, 1] == 0.0 and x[1, 1] == 2.0  # z-scored by the given stats


def test_macro_f1_maps_driver_names_to_model_indices() -> None:
    class Stub:
        def predict(self, x: np.ndarray) -> np.ndarray:
            return np.zeros(len(x), dtype=int)  # always class 0

    f1 = ns.macro_f1(Stub(), ["VER", "HAM"], np.zeros((4, 1)), pd.Series(["VER"] * 4))
    assert f1 == 1.0  # perfect when truth is always the predicted class
    f1_half = ns.macro_f1(
        Stub(), ["VER", "HAM"], np.zeros((4, 1)), pd.Series(["VER", "VER", "HAM", "HAM"])
    )
    assert 0.0 < f1_half < 1.0
