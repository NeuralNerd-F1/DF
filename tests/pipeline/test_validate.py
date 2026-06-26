"""F02 lap filter chain — one test per acceptance criterion AND per failure mode.

The acceptance criteria (spec §7) and the failure-mode table (spec §6) are the
definition of done; each is exercised below, including the ``data_quality_log`` row it
must write. Boundaries are pinned to a concrete expected value on each side.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from app.contracts.config import AppConfig, ValidateConfig
from app.contracts.enums import LapStatus, Status
from app.contracts.errors import ErrorCode
from app.pipeline import validate as validate_mod
from app.pipeline.validate import LapValidationResult, ValidationResult, validate_session
from pydantic import ValidationError

from tests.pipeline.conftest import lap_frame, read_dq, session_frame

# --- helpers -----------------------------------------------------------------------


def run(df: pd.DataFrame, cfg: AppConfig, *, driver: str = "VER") -> ValidationResult:
    return validate_session(df, season=2024, circuit="Bahrain", driver=driver, config=cfg)


def lap(res: ValidationResult, number: int) -> LapValidationResult:
    """The single LapValidationResult for ``number`` (clean or rejected)."""
    for r in (*res.clean_laps, *res.rejected_laps):
        if r.lap_number == number:
            return r
    raise AssertionError(f"lap {number} not in result")


# --- happy path --------------------------------------------------------------------


def test_clean_session_all_accepted(validate_config: AppConfig) -> None:
    """A green, well-sampled, plausibly-timed 30-lap session is fully accepted."""
    res = run(session_frame(30), validate_config)
    assert res.status is Status.OK
    assert res.clean_lap_count == 30
    assert res.combination_excluded is False
    sample = res.clean_laps[0]
    assert sample.status is LapStatus.ACCEPTED
    assert sample.error_code is ErrorCode.OK
    assert sample.flags == []
    # A clean lap writes nothing to the data_quality_log.
    assert read_dq(validate_config.paths.data_quality_log) == []


# --- ordering ----------------------------------------------------------------------


def test_first_failing_filter_wins(validate_config: AppConfig) -> None:
    """A lap failing both LAP_001 and a later filter is coded with the earlier one."""
    # SC status (LAP_001) AND a pit event (LAP_002) AND an implausible time (LAP_006).
    target = lap_frame(lap=5, track_status="4", pit_out=True, lap_time=9.9, n_samples=60)
    df = pd.concat([session_frame(4), target], ignore_index=True)
    res = run(df, validate_config)
    assert lap(res, 5).error_code is ErrorCode.LAP_001


# --- LAP_001 green flag ------------------------------------------------------------


def test_lap_001_non_green(validate_config: AppConfig) -> None:
    res = run(lap_frame(track_status="4"), validate_config)  # 4 = SC deployed
    rej = lap(res, 1)
    assert rej.status is LapStatus.REJECTED
    assert rej.error_code is ErrorCode.LAP_001


def test_lap_001_missing_status_is_rejected_not_assumed_green(validate_config: AppConfig) -> None:
    """Unknown/missing track status must be REJECTED under LAP_001, never green."""
    res = run(lap_frame(track_status=None), validate_config)
    assert lap(res, 1).error_code is ErrorCode.LAP_001
    assert lap(res, 1).detail["reason"] == "track status unavailable"


def test_lap_001_multi_token_partial_green(validate_config: AppConfig) -> None:
    """A lap that is green for part and yellow for part ("12") is not entirely green."""
    res = run(lap_frame(track_status="12"), validate_config)
    assert lap(res, 1).error_code is ErrorCode.LAP_001


# --- LAP_002 lap type --------------------------------------------------------------


def test_lap_002_pit_out(validate_config: AppConfig) -> None:
    res = run(lap_frame(pit_out=True), validate_config)
    assert lap(res, 1).error_code is ErrorCode.LAP_002


def test_lap_002_pit_in(validate_config: AppConfig) -> None:
    res = run(lap_frame(pit_in=True), validate_config)
    assert lap(res, 1).error_code is ErrorCode.LAP_002


# --- LAP_003 telemetry completeness (major) ----------------------------------------


def test_lap_003_consecutive_dropout(validate_config: AppConfig) -> None:
    # 6 consecutive NaN in 40 samples = 15% > 10%.
    res = run(lap_frame(n_samples=40, nan_gap=("Throttle", 6)), validate_config)
    rej = lap(res, 1)
    assert rej.error_code is ErrorCode.LAP_003
    assert rej.detail["channel"] == "Throttle"
    assert rej.detail["dropout_pct"] == pytest.approx(15.0)


def test_lap_003_boundary_exactly_10pct_passes(validate_config: AppConfig) -> None:
    # 20 consecutive NaN in 200 samples = exactly 10% -> NOT LAP_003 (gap > 5 so not
    # interpolated either; lap is accepted with the NaNs left for downstream NULLing).
    res = run(lap_frame(n_samples=200, nan_gap=("Brake", 20)), validate_config)
    accepted = lap(res, 1)
    assert accepted.status is LapStatus.ACCEPTED
    assert accepted.error_code is ErrorCode.OK
    assert ErrorCode.LAP_INTERP not in accepted.flags


def test_lap_003_boundary_just_over_10pct_rejects(validate_config: AppConfig) -> None:
    # 21 consecutive NaN in 200 samples = 10.5% > 10% -> LAP_003.
    res = run(lap_frame(n_samples=200, nan_gap=("Brake", 21)), validate_config)
    assert lap(res, 1).error_code is ErrorCode.LAP_003


def test_lap_003_short_lap_six_gap_exceeds_pct(validate_config: AppConfig) -> None:
    # On a short lap a 6-sample gap is 15% > 10% -> LAP_003 (the middle-zone fork).
    res = run(lap_frame(n_samples=40, nan_gap=("Gear", 6)), validate_config)
    assert lap(res, 1).error_code is ErrorCode.LAP_003


# --- LAP_INTERP telemetry completeness (minor) -------------------------------------


def test_lap_interp_gap_of_five_is_interpolated(validate_config: AppConfig) -> None:
    # 5-sample gap in a 100-sample lap = 5% (passes LAP_003), <=5 -> interpolated.
    res = run(lap_frame(n_samples=100, nan_gap=("Brake", 5)), validate_config)
    accepted = lap(res, 1)
    assert accepted.status is LapStatus.ACCEPTED
    assert ErrorCode.LAP_INTERP in accepted.flags
    assert accepted.detail["max_gap_filled"] == 5


def test_lap_interp_gap_of_six_not_interpolated(validate_config: AppConfig) -> None:
    # 6-sample gap in a 100-sample lap = 6% (passes LAP_003), >5 -> NOT interpolated.
    res = run(lap_frame(n_samples=100, nan_gap=("Brake", 6)), validate_config)
    accepted = lap(res, 1)
    assert accepted.status is LapStatus.ACCEPTED
    assert ErrorCode.LAP_INTERP not in accepted.flags


# --- LAP_004 sampling rate ---------------------------------------------------------


def test_lap_004_below_floor(validate_config: AppConfig) -> None:
    # 39 samples / 2.0 s = 19.5 Hz < 20 Hz.
    res = run(lap_frame(n_samples=39, lap_time=2.0), validate_config)
    rej = lap(res, 1)
    assert rej.error_code is ErrorCode.LAP_004
    assert rej.detail["sampling_rate_hz"] == pytest.approx(19.5)


def test_lap_004_boundary_exactly_20hz_passes(validate_config: AppConfig) -> None:
    # 40 samples / 2.0 s = 20.0 Hz -> accepted.
    res = run(lap_frame(n_samples=40, lap_time=2.0), validate_config)
    assert lap(res, 1).status is LapStatus.ACCEPTED


# --- LAP_005 speed/stationary anomaly ----------------------------------------------


def test_lap_005_stationary_at_one_percent(validate_config: AppConfig) -> None:
    # 2 stationary samples in 200 = 1.0% >= 1% -> LAP_005.
    res = run(lap_frame(n_samples=200, stationary=2), validate_config)
    rej = lap(res, 1)
    assert rej.error_code is ErrorCode.LAP_005
    assert rej.detail["stationary_fraction"] == pytest.approx(0.01)


def test_lap_005_boundary_half_percent_passes(validate_config: AppConfig) -> None:
    # 1 stationary sample in 200 = 0.5% < 1% -> accepted.
    res = run(lap_frame(n_samples=200, stationary=1), validate_config)
    assert lap(res, 1).status is LapStatus.ACCEPTED


# --- LAP_006 lap-time plausibility -------------------------------------------------


def _session_with_target(target_lap_time: float, n_samples: int = 60) -> pd.DataFrame:
    """4 normal laps (lap_time=2.0 -> median 2.0) plus one target lap (#5)."""
    return pd.concat(
        [session_frame(4), lap_frame(lap=5, lap_time=target_lap_time, n_samples=n_samples)],
        ignore_index=True,
    )


def test_lap_006_outlier(validate_config: AppConfig) -> None:
    # target 2.6 s vs median 2.0 s = 130% > 115%.
    res = run(_session_with_target(2.6), validate_config)
    rej = lap(res, 5)
    assert rej.error_code is ErrorCode.LAP_006
    assert rej.detail["lap_time_s"] == pytest.approx(2.6)
    assert rej.detail["session_median_s"] == pytest.approx(2.0)
    # exactly one lap-entity DQ row (the rejected target).
    rows = read_dq(validate_config.paths.data_quality_log)
    lap_rows = [r for r in rows if r["entity_type"] == "lap"]
    assert len(lap_rows) == 1 and lap_rows[0]["error_code"] == ErrorCode.LAP_006.value


def test_lap_006_median_excludes_in_out_and_sc_laps(validate_config: AppConfig) -> None:
    """The LAP_006 median is taken over racing laps only; slow pit/SC laps must not
    inflate it (else a genuinely-anomalous lap would slip through)."""
    racing = session_frame(4)  # 4 green non-pit laps at lap_time=2.0 -> racing median 2.0
    target = lap_frame(lap=5, lap_time=2.4, n_samples=60)  # green, 2.4 = 120% of 2.0
    pit_laps = pd.concat(
        [
            lap_frame(lap=6, lap_time=10.0, pit_out=True),
            lap_frame(lap=7, lap_time=10.0, pit_in=True),
        ],
        ignore_index=True,
    )
    res = run(pd.concat([racing, target, pit_laps], ignore_index=True), validate_config)
    # If the pit laps (10.0 s) had inflated the median, 2.4 s would pass; they must not.
    assert lap(res, 5).error_code is ErrorCode.LAP_006


def test_lap_006_boundary_exactly_115pct_passes(validate_config: AppConfig) -> None:
    # 2.3 s = exactly 115% of median 2.0 s -> "within 115%" -> accepted.
    res = run(_session_with_target(2.3, n_samples=50), validate_config)
    assert lap(res, 5).status is LapStatus.ACCEPTED


def test_lap_006_boundary_just_over_115pct_rejects(validate_config: AppConfig) -> None:
    # 2.301 s > 115% of 2.0 s -> LAP_006.
    res = run(_session_with_target(2.301, n_samples=50), validate_config)
    assert lap(res, 5).error_code is ErrorCode.LAP_006


# --- LAP_PARTIAL tyre age ----------------------------------------------------------


def test_lap_partial_null_tyre_age(validate_config: AppConfig) -> None:
    res = run(lap_frame(tyre_age=None), validate_config)
    accepted = lap(res, 1)
    assert accepted.status is LapStatus.ACCEPTED
    assert ErrorCode.LAP_PARTIAL in accepted.flags
    assert accepted.null_features == ["F27", "F28"]
    rows = read_dq(validate_config.paths.data_quality_log)
    lap_rows = [r for r in rows if r["entity_type"] == "lap"]
    assert len(lap_rows) == 1 and lap_rows[0]["error_code"] == ErrorCode.LAP_PARTIAL.value


# --- COMBO_001 minimum clean-lap count ---------------------------------------------


def test_combo_excluded_below_minimum(validate_config: AppConfig) -> None:
    res = run(session_frame(12), validate_config)
    assert res.combination_excluded is True
    assert res.error_code is ErrorCode.COMBO_001
    assert res.status is Status.FAILED
    rows = read_dq(validate_config.paths.data_quality_log)
    combo_rows = [r for r in rows if r["error_code"] == ErrorCode.DATA_INSUFFICIENT.value]
    assert len(combo_rows) == 1
    assert combo_rows[0]["entity_type"] == "combination"


def test_combo_boundary_30_not_excluded_29_excluded(validate_config: AppConfig) -> None:
    assert run(session_frame(30), validate_config).combination_excluded is False
    assert run(session_frame(29), validate_config).combination_excluded is True


# --- failsafe: validator never raises ----------------------------------------------


def test_all_nan_lap_is_coded_not_raised(validate_config: AppConfig) -> None:
    res = run(lap_frame(all_nan=True), validate_config)
    assert lap(res, 1).status is LapStatus.REJECTED
    assert lap(res, 1).error_code is ErrorCode.LAP_003  # 100% consecutive NaN


def test_empty_session_excluded_no_raise(validate_config: AppConfig) -> None:
    empty = lap_frame().iloc[0:0]  # F01-shaped columns, zero rows
    res = run(empty, validate_config)
    assert res.clean_lap_count == 0
    assert res.combination_excluded is True
    assert res.status is Status.FAILED


def test_missing_config_block_returns_typed_failure(validate_config: AppConfig) -> None:
    validate_config.lap_validation = None
    res = run(session_frame(30), validate_config)
    assert res.status is Status.FAILED
    assert res.error_code is ErrorCode.CONFIG_001
    assert res.logged is True


# --- data_quality_log discipline ---------------------------------------------------


def test_one_dq_row_per_rejection(validate_config: AppConfig) -> None:
    run(lap_frame(track_status="4"), validate_config)
    rows = read_dq(validate_config.paths.data_quality_log)
    # one lap rejected (LAP_001) + the combination exclusion (1 clean < 30).
    lap_rows = [r for r in rows if r["entity_type"] == "lap"]
    assert len(lap_rows) == 1
    assert lap_rows[0]["error_code"] == ErrorCode.LAP_001.value


def test_one_dq_row_per_interp_tag(validate_config: AppConfig) -> None:
    run(lap_frame(n_samples=100, nan_gap=("Brake", 5)), validate_config)
    rows = read_dq(validate_config.paths.data_quality_log)
    interp = [r for r in rows if r["error_code"] == ErrorCode.LAP_INTERP.value]
    assert len(interp) == 1
    assert interp[0]["entity_type"] == "lap"


# --- thresholds are config-driven (no magic numbers) -------------------------------


def test_laptime_bound_is_config_driven(validate_config: AppConfig) -> None:
    """Raising the configured bound flips a previously-rejected LAP_006 lap to accepted."""
    df = _session_with_target(2.6)  # 130% of median
    assert lap(run(df, validate_config), 5).error_code is ErrorCode.LAP_006
    validate_config.thresholds.laptime_bound_pct = 200
    assert lap(run(df, validate_config), 5).status is LapStatus.ACCEPTED


def test_min_clean_laps_is_config_driven(validate_config: AppConfig) -> None:
    validate_config.thresholds.min_clean_laps = 10
    assert run(session_frame(12), validate_config).combination_excluded is False


def test_stationary_fraction_is_config_driven(validate_config: AppConfig) -> None:
    df = lap_frame(n_samples=200, stationary=2)  # 1% stationary
    assert lap(run(df, validate_config), 1).error_code is ErrorCode.LAP_005
    validate_config.lap_validation.max_stationary_fraction = 0.05
    assert lap(run(df, validate_config), 1).status is LapStatus.ACCEPTED


# --- exactly-one DQ row per rejection (all single-lap codes) ------------------------


@pytest.mark.parametrize(
    ("frame_kwargs", "code"),
    [
        (dict(track_status="4"), ErrorCode.LAP_001),
        (dict(pit_out=True), ErrorCode.LAP_002),
        (dict(n_samples=40, nan_gap=("Throttle", 6)), ErrorCode.LAP_003),
        (dict(n_samples=39, lap_time=2.0), ErrorCode.LAP_004),
        (dict(n_samples=200, stationary=2), ErrorCode.LAP_005),
    ],
)
def test_each_rejection_writes_exactly_one_lap_dq_row(
    validate_config: AppConfig, frame_kwargs: dict, code: ErrorCode
) -> None:
    run(lap_frame(**frame_kwargs), validate_config)
    lap_rows = [
        r for r in read_dq(validate_config.paths.data_quality_log) if r["entity_type"] == "lap"
    ]
    assert len(lap_rows) == 1
    assert lap_rows[0]["error_code"] == code.value


# --- post-review hardening (code-review high / failsafe-auditor findings) -----------


def test_lap_004_missing_lap_time(validate_config: AppConfig) -> None:
    """F01 leaving lap_time null is caught as LAP_004 (rate uncomputable), not a raise."""
    res = run(lap_frame(lap_time=None), validate_config)
    rej = lap(res, 1)
    assert rej.error_code is ErrorCode.LAP_004
    assert rej.detail["sampling_rate_hz"] is None


def test_speed_channel_must_be_configured() -> None:
    """ValidateConfig rejects a required_channels list without 'Speed' (LAP_005 needs it),
    so a bad config is a build-breaking error, not a silent per-lap KeyError."""
    with pytest.raises(ValidationError):
        ValidateConfig(
            required_channels=["Throttle", "Brake", "RPM", "Gear"],
            green_track_status_code="1",
            max_consecutive_nan_pct=10,
            interp_gap_max_samples=5,
            min_moving_speed_kmh=0,
            max_stationary_fraction=0.01,
            degradation_features=["F27", "F28"],
        )


def test_missing_lap_column_is_validation_error(validate_config: AppConfig) -> None:
    """A structurally-malformed frame (no 'lap' column) is VALIDATION_ERROR, not the
    misleading DATA_INSUFFICIENT 'driver lacked laps' reading."""
    df = session_frame(30).drop(columns=["lap"])
    res = run(df, validate_config)
    assert res.status is Status.FAILED
    assert res.error_code is ErrorCode.VALIDATION_ERROR
    rows = read_dq(validate_config.paths.data_quality_log)
    assert any(r["error_code"] == ErrorCode.VALIDATION_ERROR.value for r in rows)


def test_orphan_samples_are_logged_not_silently_dropped(validate_config: AppConfig) -> None:
    """Samples with a null lap key are dropped by groupby — but only after an audit row."""
    df = session_frame(30)
    df.loc[0:3, "lap"] = np.nan  # 4 orphan samples
    run(df, validate_config)
    rows = read_dq(validate_config.paths.data_quality_log)
    orphan = [r for r in rows if r["error_code"] == ErrorCode.ORPHAN_SAMPLES.value]
    assert len(orphan) == 1
    assert "4" in orphan[0]["error_detail"]


def test_backstop_codes_validation_error_and_never_raises(
    validate_config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unexpected internal error is coded VALIDATION_ERROR (not LAP_003) and the
    session does not crash."""
    def boom(*_args: object, **_kwargs: object) -> LapValidationResult:
        raise RuntimeError("synthetic internal failure")

    monkeypatch.setattr(validate_mod, "_validate_lap", boom)
    res = run(lap_frame(), validate_config)  # must not raise
    assert lap(res, 1).error_code is ErrorCode.VALIDATION_ERROR
    rows = read_dq(validate_config.paths.data_quality_log)
    assert any(r["error_code"] == ErrorCode.VALIDATION_ERROR.value for r in rows)


def test_string_pit_flag_false_not_treated_as_pit(validate_config: AppConfig) -> None:
    """A pit flag arriving as the string 'False' must not be read as a pit event."""
    res = run(lap_frame(pit_out="False"), validate_config)
    assert lap(res, 1).status is LapStatus.ACCEPTED


def test_numeric_string_track_status_is_green(validate_config: AppConfig) -> None:
    """A green status stringified as '1.0' upstream is still recognised as green."""
    res = run(lap_frame(track_status="1.0"), validate_config)
    assert lap(res, 1).status is LapStatus.ACCEPTED
