"""Stage A — circuit naming and per-lap coercion tests.

The circuit-name mapping is the quietest leakage risk in the whole build. ``config.yaml``
quarantines the literal strings "Belgium" and "Italy"; Stage B's median filter tests laps with
``circuit not in held_out``. If Stage A ever labelled Spa as "Spa", every Spa lap would pass that
membership test and be fitted on, with nothing raising — so the mapping is pinned here against the
real config rather than trusted.

Stage A runs in its own venv (FastF1 is not installed in the main one), so ``fastf1`` is stubbed
before import. Nothing tested here touches the FastF1 API — these are the module's pure helpers.
"""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import pandas as pd
import pytest
from app.contracts.config import AppConfig

if "fastf1" not in sys.modules:  # pragma: no cover - exercised only in the main venv
    _fastf1 = types.ModuleType("fastf1")
    _req = types.ModuleType("fastf1.req")

    class _RateLimitExceededError(Exception):
        """Stand-in for fastf1.req.RateLimitExceededError."""

    _req.RateLimitExceededError = _RateLimitExceededError  # type: ignore[attr-defined]
    _fastf1.req = _req  # type: ignore[attr-defined]
    sys.modules["fastf1"] = _fastf1
    sys.modules["fastf1.req"] = _req

from scripts import build_training_matrix as btm  # noqa: E402
from scripts import fetch_fastf1 as ff  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


# --- circuit naming: the held-out quarantine depends on these strings ------------------------


def test_held_out_circuits_are_produced_under_the_config_spelling(
    scripts_config: AppConfig,
) -> None:
    """Spa -> "Belgium" and Monza -> "Italy" must match ``held_out_circuits`` character for
    character. A mismatch would not raise anywhere: the laps would simply be treated as train
    data and fitted on."""
    held_out = set(scripts_config.held_out_circuits)

    assert ff.circuit_name("Spa-Francorchamps", "Belgium")[0] in held_out
    assert ff.circuit_name("Monza", "Italy")[0] in held_out
    assert {ff.circuit_name("Spa-Francorchamps", "Belgium")[0],
            ff.circuit_name("Monza", "Italy")[0]} == held_out


def test_imola_is_not_quarantined_with_monza(scripts_config: AppConfig) -> None:
    """Italy's second circuit maps to "Emilia-Romagna", not "Italy" — keying on country would
    silently over-quarantine Imola along with the held-out Monza, and pool two incompatible
    distributions into one F04 scaler."""
    name, mapped = ff.circuit_name("Imola", "Italy")

    assert mapped
    assert name == "Emilia-Romagna"
    assert name not in set(scripts_config.held_out_circuits)


def test_renamed_location_maps_to_the_existing_circuit() -> None:
    """FastF1 renamed Miami's Location to "Miami Gardens" for 2025. Both must resolve to the
    same circuit, or 2025 would land as a separate circuit with its own scaler and no history."""
    assert ff.circuit_name("Miami Gardens", "United States")[0] == "Miami"
    assert ff.circuit_name("Miami", "United States")[0] == "Miami"


def test_unmapped_location_falls_back_to_country_and_reports_it() -> None:
    """The fallback is deliberate but must be REPORTED — a silently mis-named circuit lands in
    the wrong per-circuit scaler or dodges the quarantine."""
    name, mapped = ff.circuit_name("Some New Track", "Ruritania")

    assert name == "Ruritania"
    assert mapped is False


def test_unmapped_location_with_no_country_keeps_the_location() -> None:
    name, mapped = ff.circuit_name("Some New Track", "")

    assert name == "Some New Track"
    assert mapped is False


def test_location_is_matched_after_stripping_whitespace() -> None:
    assert ff.circuit_name("  Monza  ", "Italy") == ("Italy", True)


def test_stage_a_and_stage_b_circuit_vocabularies_agree() -> None:
    """Every circuit Stage A can emit must have a Stage B circuit-type, and vice versa.

    A circuit in only one map degrades silently in a different way at each end: an unmapped type
    yields ``circuit_type=None`` (no driver×circuit-type imputation cell), while a type for a
    circuit that is never fetched is dead config that hides the real gap.
    """
    src = (REPO_ROOT / "scripts" / "fetch_fastf1.py").read_text(encoding="utf-8")
    block = src.split("_LOCATION_TO_CIRCUIT = {")[1].split("}")[0]
    emitted = set(re.findall(r':\s*"([^"]+)"', block))

    assert emitted == set(btm._CIRCUIT_TYPE)


# --- per-lap coercion: documented null markers, never a raise --------------------------------


def test_seconds_converts_a_timedelta() -> None:
    assert ff._seconds(pd.Timedelta(seconds=92.5)) == pytest.approx(92.5)


@pytest.mark.parametrize("missing", [None, pd.NaT])
def test_seconds_returns_nan_for_a_missing_value(missing: object) -> None:
    """NaN is the documented null marker; raising here would abort a whole race's pull over one
    absent lap time."""
    assert pd.isna(ff._seconds(missing))


def test_as_str_marks_a_missing_categorical() -> None:
    assert ff._as_str("SOFT") == "SOFT"
    assert ff._as_str(None) == "UNKNOWN"
    assert ff._as_str(float("nan")) == "UNKNOWN"


def test_as_float_marks_a_missing_numeric() -> None:
    assert ff._as_float(3) == 3.0
    assert pd.isna(ff._as_float(None))


# --- rate-limit retry: waits are announced, exhaustion is loud --------------------------------


def test_retry_returns_immediately_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ff.time, "sleep", lambda _s: pytest.fail("should not have waited"))

    assert ff._with_rate_limit_retry("x", lambda: "ok", sleep_s=1, max_waits=3) == "ok"


def test_retry_waits_then_succeeds(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The cap is the binding constraint on a multi-season pull, so a wait is correct — but an
    unattended run that stalls silently is indistinguishable from one that has hung."""
    waited: list[int] = []
    monkeypatch.setattr(ff.time, "sleep", lambda s: waited.append(s))
    calls = {"n": 0}

    def _flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ff.RateLimitExceededError("429")
        return "ok"

    assert ff._with_rate_limit_retry("race 5", _flaky, sleep_s=60, max_waits=5) == "ok"
    assert waited == [60, 60]
    assert "rate limit hit on race 5" in capsys.readouterr().out


def test_retry_reraises_once_the_waits_are_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exhaustion re-raises rather than returning None — the caller records the failure in the
    manifest instead of writing a silently empty race."""
    monkeypatch.setattr(ff.time, "sleep", lambda _s: None)

    def _always() -> str:
        raise ff.RateLimitExceededError("429")

    with pytest.raises(ff.RateLimitExceededError):
        ff._with_rate_limit_retry("x", _always, sleep_s=1, max_waits=2)
