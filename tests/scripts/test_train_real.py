"""Stage C — pre-flight and run-outcome tests.

Two decisions are under test. ``preflight`` refuses to train on a matrix that cannot support the
claim the run would make (no held-out circuit, one class, a driver absent from train), rather than
emitting a model whose headline number is meaningless. ``failed_stages`` decides the process exit
code, and specifically that DEGRADED is NOT failure — a model that trained and missed its target is
a research finding (§11.1), while a stage that produced no model must never read as success.
"""

from __future__ import annotations

import pandas as pd
import pytest

from scripts import train_real as tr

HELD_OUT = {"Belgium", "Italy"}


def matrix(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    """(driver, circuit, split) triples -> the columns preflight reads."""
    return pd.DataFrame(
        [{"driver": d, "circuit": c, "split": s} for d, c, s in rows]
    )


def _healthy() -> list[tuple[str, str, str]]:
    return [
        ("VER", "Bahrain", "train"), ("VER", "Monaco", "train"), ("VER", "Belgium", "test"),
        ("HAM", "Bahrain", "train"), ("HAM", "Monaco", "train"), ("HAM", "Belgium", "test"),
    ]


# --- preflight ------------------------------------------------------------------------------


def test_preflight_passes_a_well_formed_matrix() -> None:
    assert tr.preflight(matrix(_healthy()), HELD_OUT) == []


def test_preflight_rejects_a_single_driver() -> None:
    rows = [r for r in _healthy() if r[0] == "VER"]

    problems = tr.preflight(matrix(rows), HELD_OUT)

    assert any("driver" in p for p in problems)


def test_preflight_rejects_a_matrix_with_no_held_out_circuit() -> None:
    """Without Belgium/Italy present, ``held_out_macro_f1`` would be computed on an empty test
    set — a number that looks like a score but measures nothing."""
    rows = [r for r in _healthy() if r[1] != "Belgium"]

    problems = tr.preflight(matrix(rows), HELD_OUT)

    assert any("held-out" in p for p in problems)


def test_preflight_rejects_an_empty_train_split() -> None:
    rows = [(d, c, "test") for d, c, _ in _healthy()]

    problems = tr.preflight(matrix(rows), HELD_OUT)

    assert any("train split" in p for p in problems)


def test_preflight_rejects_a_single_training_circuit() -> None:
    """StratifiedGroupKFold groups by circuit, so one training circuit cannot be split."""
    rows = [r for r in _healthy() if r[1] != "Monaco"]

    problems = tr.preflight(matrix(rows), HELD_OUT)

    assert any("StratifiedGroupKFold" in p for p in problems)


def test_preflight_rejects_a_driver_missing_from_train() -> None:
    """A driver present only in the held-out split is a class the model was never taught but
    will be scored on."""
    rows = [*_healthy(), ("LEC", "Belgium", "test")]

    problems = tr.preflight(matrix(rows), HELD_OUT)

    assert any("absent from the train split" in p and "LEC" in p for p in problems)


def test_preflight_reports_every_problem_at_once() -> None:
    """All structural problems surface in one pass — an operator fixes the matrix once, not
    once per re-run."""
    problems = tr.preflight(matrix([("VER", "Bahrain", "train")]), HELD_OUT)

    assert len(problems) >= 2


# --- failed_stages / exit code ----------------------------------------------------------------


def test_failed_stages_lists_only_failed_stages() -> None:
    summary = {
        "matrix": "m.parquet",
        "n_vectors": 100,
        "classifier": {"status": "DEGRADED"},
        "stylespace": {"status": "OK"},
        "degradation": {"status": "FAILED", "error": "boom"},
        "personalisation": {"status": "FAILED", "untyped": True},
    }

    assert tr.failed_stages(summary) == ["degradation", "personalisation"]


def test_failed_stages_treats_degraded_as_success() -> None:
    """The real 2022-2025 run ended DEGRADED on the classifier and personalisation with every
    model on disk. That must exit 0: DEGRADED is a reported finding, not a failed run."""
    summary = {
        "matrix": "m.parquet",
        "n_vectors": 53928,
        "classifier": {"status": "DEGRADED"},
        "stylespace": {"status": "OK"},
        "degradation": {"status": "OK"},
        "personalisation": {"status": "DEGRADED"},
    }

    assert tr.failed_stages(summary) == []


def test_failed_stages_ignores_non_stage_entries() -> None:
    """``matrix`` and ``n_vectors`` are scalars, not stage results — they must not be mistaken
    for a stage with no status."""
    assert tr.failed_stages({"matrix": "m.parquet", "n_vectors": 0}) == []


def test_failed_stages_is_sorted_for_a_stable_message() -> None:
    summary = {
        "stylespace": {"status": "FAILED"},
        "degradation": {"status": "FAILED"},
        "classifier": {"status": "FAILED"},
    }

    assert tr.failed_stages(summary) == ["classifier", "degradation", "stylespace"]


@pytest.mark.parametrize("status", ["OK", "DEGRADED", "PARTIAL"])
def test_failed_stages_only_reacts_to_the_failed_literal(status: str) -> None:
    assert tr.failed_stages({"classifier": {"status": status}}) == []
