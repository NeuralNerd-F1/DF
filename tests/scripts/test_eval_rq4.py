"""RQ4 evaluation mechanisms — stint summarisation and aggregation, on constructed sequences.

The real distributions live in the content-hashed artefact; these tests pin the MACHINERY:
lead/lag derivation from served fields, the recovered-alert (false-positive proxy) rule, the
aggregation quantiles, and the modal-compound bookkeeping.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from scripts import eval_rq4 as rq4


def _alert(alert_lap: int = 10, first_crossing: int = 9, confidence: str = "HIGH"):
    return SimpleNamespace(
        alert_lap=alert_lap, first_crossing_lap=first_crossing, alert_confidence=confidence
    )


def test_stint_summary_no_alert() -> None:
    s = rq4.stint_summary(["BASELINE"] * 3 + ["NOMINAL"] * 7, None)
    assert s["alerted"] is False
    assert s["watched"] is False
    assert "laps_remaining_after_alert" not in s


def test_stint_summary_alert_lead_lag_and_recovery() -> None:
    states = ["BASELINE"] * 3 + ["NOMINAL"] * 4 + ["WATCH", "ALERT", "ALERT"] + ["NOMINAL"] * 5
    s = rq4.stint_summary(states, _alert(alert_lap=9, first_crossing=8))
    assert s["alerted"] is True
    assert s["confirmation_lag_laps"] == 1  # served lap numbers, not recomputed
    # First ALERT at index 8 in a 15-lap sequence → 6 laps remained after the alert.
    assert s["laps_remaining_after_alert"] == 6
    assert s["recovered"] is True  # NOMINAL after ALERT → the self-labelled FP proxy


def test_stint_summary_terminal_alert_is_not_recovered() -> None:
    states = ["BASELINE"] * 3 + ["NOMINAL"] * 5 + ["WATCH", "ALERT", "ALERT"]
    s = rq4.stint_summary(states, _alert(alert_lap=10, first_crossing=9))
    assert s["recovered"] is False  # alert ran to the stint's end — degradation-consistent
    assert s["laps_remaining_after_alert"] == 1


def test_aggregate_rates_and_quantiles() -> None:
    summaries = [
        rq4.stint_summary(["NOMINAL"] * 10, None),
        rq4.stint_summary(["NOMINAL"] * 8 + ["WATCH", "NOMINAL"], None),
        rq4.stint_summary(
            ["NOMINAL"] * 6 + ["ALERT"] * 2 + ["NOMINAL"] * 2, _alert(alert_lap=7, first_crossing=6)
        ),
        rq4.stint_summary(["NOMINAL"] * 8 + ["ALERT"] * 2, _alert(alert_lap=9, first_crossing=8)),
    ]
    m = rq4.aggregate(summaries)
    assert m["n_stints"] == 4
    assert m["alert_rate"] == 0.5
    assert m["watch_only_rate"] == 0.25
    assert m["recovered_alert_fraction"] == 0.5  # one of the two alerts recovered
    assert m["laps_remaining_after_alert"]["n"] == 2
    assert m["alert_confidence_counts"] == {"HIGH": 2}


def test_aggregate_empty_group_is_honest_nulls() -> None:
    m = rq4.aggregate([])
    assert m["n_stints"] == 0
    assert m["alert_rate"] is None
    assert m["laps_remaining_after_alert"] is None


def test_modal_compound_filters_placeholders() -> None:
    assert rq4.modal_compound(pd.Series(["SOFT", "SOFT", "MEDIUM"])) == "SOFT"
    assert rq4.modal_compound(pd.Series(["nan", "None", None])) == "UNKNOWN"
    assert rq4.modal_compound(pd.Series([], dtype=object)) == "UNKNOWN"
