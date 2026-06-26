"""F03a — scalar-channel feature extraction + vector assembly tests.

Covers the spec's acceptance criteria and every failure mode: one happy-path assertion per
feature, one NULL/failure trigger per feature, the F10 cap fallback, the §4.5 null-ceiling
boundary (6 accept+impute / 7 reject), imputation source resolution, the unexpected-error
path, data-quality row counts, the no-normalisation / no-per-row-loop source guards,
config-driven thresholds, and rate-feature resample invariance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from app.contracts.config import GEOMETRY_FEATURE_IDS, SCALAR_FEATURE_IDS, AppConfig, load_config
from app.contracts.enums import Status
from app.contracts.errors import ErrorCode
from app.pipeline import features as feat
from app.pipeline.features import (
    FEATURE_IDS,
    FeatureComputationError,
    FeatureMedians,
    StintLap,
    assemble_feature_vector,
    compute_degradation_slopes,
    compute_geometry_features,
    compute_scalar_features,
)

from tests.pipeline.conftest import read_dq

DT = 0.05  # 20 Hz canonical sample period (matches features.sample_period_ms = 50)
SESSION_MAX_RPM = 12000.0
LAP_DISTANCE_M = 6000.0


@pytest.fixture
def feat_cfg(repo_config_path: Path, tmp_path: Path) -> AppConfig:
    """Real repo config (shipped feature values) with the DQ log redirected to tmp."""
    cfg = load_config(repo_config_path)
    cfg.paths.data_quality_log = str(tmp_path / "logs" / "dq.jsonl")
    return cfg


# --- Canonical good lap ------------------------------------------------------------
#
# Six identical 20-sample "corner" blocks (approach / braking / apex / exit) engineered so
# every one of the 22 scalar features computes IN RANGE with a hand-checkable value.

_SPEED_BLOCK = [280.0] * 6 + [260.0, 220.0, 180.0, 140.0] + [120.0, 120.0] + [
    140.0, 160.0, 180.0, 200.0, 220.0, 240.0, 260.0, 280.0
]
_THROTTLE_BLOCK = [100.0] * 6 + [0.0] * 4 + [0.0] * 2 + [
    12.5, 25.0, 37.5, 50.0, 62.5, 75.0, 87.5, 100.0
]
_BRAKE_BLOCK = [0.0] * 6 + [80.0] * 4 + [0.0] * 2 + [0.0] * 8
_RPM_BLOCK = [12000.0] * 6 + [10000.0, 9000.0, 8000.0, 7000.0] + [7000.0, 7000.0] + [
    7500.0, 8000.0, 8500.0, 9000.0, 9500.0, 10000.0, 10500.0, 11000.0
]
_GEAR_BLOCK = [7.0] * 6 + [6.0, 5.0, 4.0, 3.0] + [3.0, 3.0] + [
    3.0, 4.0, 5.0, 6.0, 7.0, 7.0, 7.0, 7.0
]
_N_BLOCKS = 6


def good_lap() -> pd.DataFrame:
    speed = np.tile(_SPEED_BLOCK, _N_BLOCKS)
    throttle = np.tile(_THROTTLE_BLOCK, _N_BLOCKS)
    brake = np.tile(_BRAKE_BLOCK, _N_BLOCKS)
    rpm = np.tile(_RPM_BLOCK, _N_BLOCKS)
    gear = np.tile(_GEAR_BLOCK, _N_BLOCKS)
    n = speed.size
    return pd.DataFrame(
        {
            "time": np.arange(n) * DT,
            "Speed": speed,
            "Throttle": throttle,
            "Brake": brake,
            "RPM": rpm,
            "Gear": gear,
            "lap": 3,
        }
    )


def compute_good(cfg: AppConfig, **overrides: object) -> feat.ScalarFeatureResult:
    kwargs: dict[str, object] = {
        "lap_number": 3,
        "session_max_rpm": SESSION_MAX_RPM,
        "lap_distance_m": LAP_DISTANCE_M,
        "config": cfg,
    }
    kwargs.update(overrides)
    return compute_scalar_features(good_lap(), **kwargs)  # type: ignore[arg-type]


# ----------------------------------------------------------------------------------
# Happy path
# ----------------------------------------------------------------------------------


def test_good_lap_all_features_in_range_status_ok(feat_cfg: AppConfig) -> None:
    res = compute_good(feat_cfg)
    assert res.status is Status.OK
    assert res.null_features == []
    assert res.capped_features == []
    for fid in SCALAR_FEATURE_IDS:
        lo, hi = feat_cfg.features.valid_ranges[fid]  # type: ignore[union-attr]
        val = res.features[fid]
        assert val is not None, f"{fid} unexpectedly NULL"
        assert lo <= val <= hi, f"{fid}={val} out of [{lo},{hi}]"


def test_scalar_result_keys_are_exactly_the_22(feat_cfg: AppConfig) -> None:
    res = compute_good(feat_cfg)
    assert set(res.features) == set(SCALAR_FEATURE_IDS)
    for fid in ("F19", "F20", "F21", "F25", "F27", "F28"):
        assert fid not in res.features


def test_happy_path_expected_values(feat_cfg: AppConfig) -> None:
    """One asserted value per feature: aggregates re-derived via the §4.2 formula from the
    same arrays; event features against the engineered design."""
    df = good_lap()
    thr = df["Throttle"].to_numpy(float)
    spd = df["Speed"].to_numpy(float)
    rpm = df["RPM"].to_numpy(float)
    gear = df["Gear"].to_numpy(float)
    f = compute_good(feat_cfg).features

    # Aggregates — independent numpy re-derivation of the spec formula.
    assert f["F01"] == pytest.approx(np.mean(thr))
    assert f["F02"] == pytest.approx(np.std(thr))
    pos = np.diff(thr)[np.diff(thr) > 0]
    assert f["F03"] == pytest.approx(np.mean(pos))  # dt == sample_period => factor 1
    assert f["F04"] == pytest.approx(np.mean(thr >= 100))
    assert f["F05"] == pytest.approx(np.mean(thr < 5))
    assert f["F10"] == pytest.approx(0.0)  # throttle and brake never overlap
    assert f["F12"] == pytest.approx(np.var(spd))
    assert f["F13"] == pytest.approx(np.percentile(spd, 10))
    assert f["F14"] == pytest.approx(np.percentile(spd, 90))
    assert f["F18"] == pytest.approx(np.std(rpm))
    assert f["F22"] == pytest.approx(np.corrcoef(thr, spd)[0, 1])

    # Event/context features — engineered values.
    assert f["F06"] == pytest.approx(80.0)  # every braking-event peak is 80 bar
    assert f["F07"] == pytest.approx(200.0)  # 4 samples * 50 ms
    assert f["F08"] == pytest.approx(80.0)  # single -80 release step per block
    assert f["F09"] == pytest.approx(260.0)  # speed at each braking-event start
    assert f["F11"] == pytest.approx(120.0)  # apex minimum speed
    n_changes = int(np.count_nonzero(np.diff(gear) != 0))
    assert f["F15"] == pytest.approx(n_changes / (LAP_DISTANCE_M / 1000.0))
    assert f["F17"] == pytest.approx(np.mean(rpm > 0.97 * SESSION_MAX_RPM))
    assert f["F24"] == pytest.approx(150.0)  # 3 samples * 50 ms apex->throttle
    assert f["F26"] == pytest.approx(0.0, abs=1e-9)  # symmetric halves


def test_assembled_vector_has_all_28_keys_status_ok(feat_cfg: AppConfig) -> None:
    res = compute_good(feat_cfg)
    values = dict(res.features)
    for fid in ("F19", "F20", "F21", "F25", "F27", "F28"):
        values[fid] = 1.0  # F03b's six, supplied valid
    medians = FeatureMedians(global_medians={fid: 0.0 for fid in FEATURE_IDS})
    out = assemble_feature_vector(
        values, lap_number=3, driver="VER", circuit_type="high_speed",
        medians=medians, config=feat_cfg,
    )
    assert set(out.feature_vector) == set(FEATURE_IDS)
    assert len(out.feature_vector) == 28
    assert out.status is Status.OK
    assert out.null_count == 0
    assert out.imputed_features == []
    assert out.error_code is None


# ----------------------------------------------------------------------------------
# NULL triggers (one per documented §4.2 trigger)
# ----------------------------------------------------------------------------------


def _null_reason(res: feat.ScalarFeatureResult, fid: str) -> str:
    return str(res.detail[fid]["reason"])


def test_f02_constant_throttle_nulls(feat_cfg: AppConfig) -> None:
    df = good_lap()
    df["Throttle"] = 50.0
    res = compute_scalar_features(
        df, lap_number=3, session_max_rpm=SESSION_MAX_RPM,
        lap_distance_m=LAP_DISTANCE_M, config=feat_cfg,
    )
    assert res.features["F02"] is None
    assert "F02" in res.null_features
    assert _null_reason(res, "F02") == feat.REASON_NULL_TRIGGER
    assert res.status is Status.DEGRADED


def test_f03_no_rising_throttle_nulls(feat_cfg: AppConfig) -> None:
    df = good_lap()
    df["Throttle"] = np.linspace(100.0, 0.0, len(df))  # monotonically falling
    res = compute_scalar_features(
        df, lap_number=3, session_max_rpm=SESSION_MAX_RPM,
        lap_distance_m=LAP_DISTANCE_M, config=feat_cfg,
    )
    assert res.features["F03"] is None
    assert _null_reason(res, "F03") == feat.REASON_NULL_TRIGGER


@pytest.mark.parametrize(
    "fid,mutate",
    [
        ("F06", lambda df: df.assign(Brake=0.0)),  # 0 braking events < 5
        ("F07", lambda df: df.assign(Brake=0.0)),  # no braking events
        ("F08", lambda df: df.assign(Brake=0.0)),  # no release events
        ("F12", lambda df: df.assign(Speed=200.0)),  # identical speed
        ("F18", lambda df: df.assign(RPM=9000.0)),  # constant RPM
        ("F23", lambda df: df.assign(Brake=0.0)),  # no braking events
    ],
)
def test_channel_null_triggers(feat_cfg: AppConfig, fid: str, mutate) -> None:
    df = mutate(good_lap())
    res = compute_scalar_features(
        df, lap_number=3, session_max_rpm=SESSION_MAX_RPM,
        lap_distance_m=LAP_DISTANCE_M, config=feat_cfg,
    )
    assert res.features[fid] is None
    assert fid in res.null_features
    assert _null_reason(res, fid) == feat.REASON_NULL_TRIGGER


def test_f09_f11_no_corners_null(feat_cfg: AppConfig) -> None:
    df = good_lap()
    df["Speed"] = 250.0  # flat speed => no corner minima
    res = compute_scalar_features(
        df, lap_number=3, session_max_rpm=SESSION_MAX_RPM,
        lap_distance_m=LAP_DISTANCE_M, config=feat_cfg,
    )
    assert res.features["F09"] is None
    assert res.features["F11"] is None
    assert _null_reason(res, "F09") == feat.REASON_NULL_TRIGGER
    assert _null_reason(res, "F11") == feat.REASON_NULL_TRIGGER


def test_f09_no_braking_anchor_nulls(feat_cfg: AppConfig) -> None:
    """F09 second trigger: corners exist but no braking event anchors a corner entry."""
    df = good_lap()
    df["Brake"] = 0.0  # corners (speed minima) remain, but no braking events
    res = compute_scalar_features(
        df, lap_number=3, session_max_rpm=SESSION_MAX_RPM,
        lap_distance_m=LAP_DISTANCE_M, config=feat_cfg,
    )
    assert res.features["F09"] is None
    assert _null_reason(res, "F09") == feat.REASON_NULL_TRIGGER


def test_f15_no_lap_distance_nulls(feat_cfg: AppConfig) -> None:
    res = compute_good(feat_cfg, lap_distance_m=None)
    assert res.features["F15"] is None
    assert _null_reason(res, "F15") == feat.REASON_NULL_TRIGGER


def test_f16_too_few_gear_changes_nulls(feat_cfg: AppConfig) -> None:
    df = good_lap()
    df["Gear"] = 6.0  # zero gear changes < 10
    res = compute_scalar_features(
        df, lap_number=3, session_max_rpm=SESSION_MAX_RPM,
        lap_distance_m=LAP_DISTANCE_M, config=feat_cfg,
    )
    assert res.features["F16"] is None
    assert _null_reason(res, "F16") == feat.REASON_NULL_TRIGGER


def test_f17_no_session_max_rpm_nulls(feat_cfg: AppConfig) -> None:
    res = compute_good(feat_cfg, session_max_rpm=None)
    assert res.features["F17"] is None
    assert _null_reason(res, "F17") == feat.REASON_NULL_TRIGGER


def test_f24_too_few_transitions_nulls(feat_cfg: AppConfig) -> None:
    # One short braking event, no throttle re-application after it => 0 transitions < 5.
    n = 60
    df = pd.DataFrame(
        {
            "time": np.arange(n) * DT,
            "Speed": np.linspace(250, 100, n),
            "Throttle": 0.0,
            "Brake": np.where((np.arange(n) >= 10) & (np.arange(n) < 14), 80.0, 0.0),
            "RPM": np.linspace(11000, 7000, n),
            "Gear": 5.0,
            "lap": 1,
        }
    )
    res = compute_scalar_features(
        df, lap_number=1, session_max_rpm=SESSION_MAX_RPM,
        lap_distance_m=LAP_DISTANCE_M, config=feat_cfg,
    )
    assert res.features["F24"] is None
    assert _null_reason(res, "F24") == feat.REASON_NULL_TRIGGER


# ----------------------------------------------------------------------------------
# Out-of-range, capping, unexpected
# ----------------------------------------------------------------------------------


def test_out_of_range_raises_and_nulls_only_that_feature(feat_cfg: AppConfig) -> None:
    """F14 (speed_p90 range [100,380]): drive p90 below 100 -> FeatureComputationError ->
    caught -> NULL -> OUT_OF_RANGE logged, while the other 21 still compute."""
    df = good_lap()
    df["Speed"] = 50.0 + np.tile([0.0, 1.0], len(df) // 2)  # p90 ~ 51 << 100
    res = compute_scalar_features(
        df, lap_number=3, session_max_rpm=SESSION_MAX_RPM,
        lap_distance_m=LAP_DISTANCE_M, config=feat_cfg,
    )
    assert res.features["F14"] is None
    assert _null_reason(res, "F14") == feat.REASON_OUT_OF_RANGE
    assert res.features["F01"] is not None  # other features unaffected


def test_feature_function_raises_on_out_of_range() -> None:
    """A feature function must RAISE, never return the raw out-of-range value (§4.1)."""
    cfg = load_config().features
    assert cfg is not None
    with pytest.raises(FeatureComputationError) as ei:
        feat._checked("F14", 90.0, cfg)
    assert ei.value.reason == feat.REASON_OUT_OF_RANGE
    assert ei.value.error_code is ErrorCode.FEATURE_001


def test_f10_overlap_capped_not_nulled(feat_cfg: AppConfig) -> None:
    df = good_lap()
    df["Brake"] = 80.0  # full overlap with the high-throttle samples
    df["Throttle"] = 80.0
    res = compute_scalar_features(
        df, lap_number=3, session_max_rpm=SESSION_MAX_RPM,
        lap_distance_m=LAP_DISTANCE_M, config=feat_cfg,
    )
    assert res.features["F10"] == pytest.approx(feat_cfg.features.overlap_cap)  # type: ignore[union-attr]
    assert "F10" in res.capped_features
    assert "F10" not in res.null_features
    assert _null_reason(res, "F10") == feat.REASON_OVERLAP_CAPPED
    assert res.status is Status.DEGRADED
    rows = [r for r in read_dq(feat_cfg.paths.data_quality_log) if "F10" in r["error_detail"]]
    assert len(rows) == 1 and rows[0]["fallback_used"] is True


def test_f10_at_cap_not_capped(feat_cfg: AppConfig) -> None:
    """Raw overlap exactly == cap is in range (not capped); strictly above is capped."""
    cap = feat_cfg.features.overlap_cap  # type: ignore[union-attr]
    n = 100
    k = int(round(cap * n))  # exactly cap fraction overlapping
    overlap = np.zeros(n, dtype=bool)
    overlap[:k] = True
    df = pd.DataFrame(
        {
            "time": np.arange(n) * DT,
            "Speed": 200.0,
            "Throttle": np.where(overlap, 80.0, 0.0),
            "Brake": np.where(overlap, 80.0, 0.0),
            "RPM": 9000.0,
            "Gear": 5.0,
            "lap": 1,
        }
    )
    res = compute_scalar_features(
        df, lap_number=1, session_max_rpm=SESSION_MAX_RPM,
        lap_distance_m=LAP_DISTANCE_M, config=feat_cfg,
    )
    assert res.features["F10"] == pytest.approx(cap)
    assert "F10" not in res.capped_features


def test_unexpected_error_is_caught_and_nulled(feat_cfg: AppConfig, monkeypatch) -> None:
    def boom(*_a, **_k):
        raise ValueError("synthetic internal fault")

    monkeypatch.setitem(feat._FEATURE_FUNCS, "F01", boom)
    res = compute_good(feat_cfg)
    assert res.features["F01"] is None
    assert _null_reason(res, "F01") == feat.REASON_UNEXPECTED
    assert res.features["F02"] is not None  # others unaffected


def test_all_nan_lap_never_raises(feat_cfg: AppConfig) -> None:
    n = 40
    df = pd.DataFrame(
        {
            "time": np.arange(n) * DT,
            "Speed": np.nan, "Throttle": np.nan, "Brake": np.nan,
            "RPM": np.nan, "Gear": np.nan, "lap": 1,
        }
    )
    res = compute_scalar_features(
        df, lap_number=1, session_max_rpm=None, lap_distance_m=None, config=feat_cfg,
    )
    # Every feature NULL, no exception escaped.
    assert all(res.features[fid] is None for fid in SCALAR_FEATURE_IDS)
    assert res.status is Status.DEGRADED


# ----------------------------------------------------------------------------------
# Vector assembly: null-ceiling gate + imputation (§4.5)
# ----------------------------------------------------------------------------------


def _vector_with_nulls(n_null: int) -> dict[str, float | None]:
    vals: dict[str, float | None] = {fid: 1.0 for fid in FEATURE_IDS}
    for fid in list(FEATURE_IDS)[:n_null]:
        vals[fid] = None
    return vals


def _medians() -> FeatureMedians:
    return FeatureMedians(
        global_medians={fid: 7.0 for fid in FEATURE_IDS},
        driver_circuit_medians={("VER", "high_speed"): {fid: 3.0 for fid in FEATURE_IDS}},
    )


def test_null_ceiling_six_accepted_and_imputed(feat_cfg: AppConfig) -> None:
    out = assemble_feature_vector(
        _vector_with_nulls(6), lap_number=1, driver="VER", circuit_type="high_speed",
        medians=_medians(), config=feat_cfg,
    )
    assert out.status is Status.DEGRADED
    assert out.error_code is None
    assert out.null_count == 6
    assert len(out.imputed_features) == 6
    assert ErrorCode.IMPUTED in out.flags
    assert all(out.feature_vector[fid] is not None for fid in FEATURE_IDS)


def test_null_ceiling_seven_rejected(feat_cfg: AppConfig) -> None:
    out = assemble_feature_vector(
        _vector_with_nulls(7), lap_number=1, driver="VER", circuit_type="high_speed",
        medians=_medians(), config=feat_cfg,
    )
    assert out.status is Status.FAILED
    assert out.error_code is ErrorCode.LAP_NULL_EXCESS
    assert out.null_count == 7
    assert out.imputed_features == []
    # No imputation happened: the NULLs are still NULL.
    assert any(out.feature_vector[fid] is None for fid in FEATURE_IDS)


def test_impute_prefers_driver_circuit_then_global(feat_cfg: AppConfig) -> None:
    # Known driver/circuit -> driver-circuit median (3.0).
    out = assemble_feature_vector(
        _vector_with_nulls(2), lap_number=1, driver="VER", circuit_type="high_speed",
        medians=_medians(), config=feat_cfg,
    )
    imputed_vals = [out.feature_vector[fid] for fid in out.imputed_features]
    assert imputed_vals == [3.0, 3.0]
    # Unknown circuit -> global median (7.0).
    out2 = assemble_feature_vector(
        _vector_with_nulls(2), lap_number=1, driver="VER", circuit_type="unseen",
        medians=_medians(), config=feat_cfg,
    )
    assert [out2.feature_vector[fid] for fid in out2.imputed_features] == [7.0, 7.0]


def test_forced_null_from_f02_is_applied(feat_cfg: AppConfig) -> None:
    vals: dict[str, float | None] = {fid: 1.0 for fid in FEATURE_IDS}
    out = assemble_feature_vector(
        vals, lap_number=1, driver="VER", circuit_type="high_speed", medians=_medians(),
        config=feat_cfg, forced_null=["F27", "F28"], flags=[ErrorCode.LAP_PARTIAL],
    )
    assert out.null_count == 2
    assert set(out.imputed_features) == {"F27", "F28"}
    assert ErrorCode.LAP_PARTIAL in out.flags  # carried through from F02


# ----------------------------------------------------------------------------------
# Data-quality logging row counts
# ----------------------------------------------------------------------------------


def test_one_dq_row_per_null_feature(feat_cfg: AppConfig) -> None:
    df = good_lap()
    df["Brake"] = 0.0  # nulls F06, F07, F08, F09, F23, F24 (braking-dependent)
    res = compute_scalar_features(
        df, lap_number=3, session_max_rpm=SESSION_MAX_RPM,
        lap_distance_m=LAP_DISTANCE_M, config=feat_cfg,
    )
    rows = read_dq(feat_cfg.paths.data_quality_log)
    assert len(rows) == len(res.null_features) + len(res.capped_features)
    assert all(r["error_code"] == ErrorCode.FEATURE_001.value for r in rows)


def test_one_dq_row_per_imputed_feature_and_reject(feat_cfg: AppConfig) -> None:
    assemble_feature_vector(
        _vector_with_nulls(6), lap_number=1, driver="VER", circuit_type="high_speed",
        medians=_medians(), config=feat_cfg,
    )
    rows = read_dq(feat_cfg.paths.data_quality_log)
    assert len(rows) == 6
    assert all(r["error_code"] == ErrorCode.IMPUTED.value for r in rows)
    assert all(r["fallback_used"] is True for r in rows)


def test_reject_logs_single_lap_null_excess_row(feat_cfg: AppConfig) -> None:
    assemble_feature_vector(
        _vector_with_nulls(8), lap_number=1, driver="VER", circuit_type="high_speed",
        medians=_medians(), config=feat_cfg,
    )
    rows = read_dq(feat_cfg.paths.data_quality_log)
    assert len(rows) == 1
    assert rows[0]["error_code"] == ErrorCode.LAP_NULL_EXCESS.value


# ----------------------------------------------------------------------------------
# Contract guards: no normalisation, no per-row loops, config-driven
# ----------------------------------------------------------------------------------

_SRC = Path(feat.__file__).read_text(encoding="utf-8")


def test_no_normalisation_in_source() -> None:
    for token in ("StandardScaler", "MinMaxScaler", "RobustScaler", ".fit(", "zscore"):
        assert token not in _SRC, f"normalisation token {token!r} found in features.py"


def test_no_per_row_loops_in_source() -> None:
    for token in ("iterrows", "itertuples", ".apply("):
        assert token not in _SRC, f"per-row pandas iteration {token!r} found in features.py"


def test_overlap_cap_is_config_driven(feat_cfg: AppConfig) -> None:
    """Changing overlap_cap (and its tied F10 range) changes the cap (no hardcoded 0.25)."""
    df = good_lap()
    df["Brake"] = 80.0
    df["Throttle"] = 80.0
    feat_cfg.features.overlap_cap = 0.5  # type: ignore[union-attr]
    feat_cfg.features.valid_ranges["F10"] = (0.0, 0.5)  # type: ignore[union-attr]
    res = compute_scalar_features(
        df, lap_number=3, session_max_rpm=SESSION_MAX_RPM,
        lap_distance_m=LAP_DISTANCE_M, config=feat_cfg,
    )
    assert res.features["F10"] == pytest.approx(0.5)  # capped to the NEW config value


def test_min_braking_events_is_config_driven(feat_cfg: AppConfig) -> None:
    """Raising min_braking_events above the lap's event count NULLs F06 (no hardcoded 5)."""
    feat_cfg.features.min_braking_events = 99  # type: ignore[union-attr]
    res = compute_good(feat_cfg)
    assert res.features["F06"] is None
    assert _null_reason(res, "F06") == feat.REASON_NULL_TRIGGER


def test_null_ceiling_is_config_driven(feat_cfg: AppConfig) -> None:
    """Lowering null_ceiling to 5 makes a 6-NULL vector reject instead of impute."""
    feat_cfg.thresholds.null_ceiling = 5
    out = assemble_feature_vector(
        _vector_with_nulls(6), lap_number=1, driver="VER", circuit_type="high_speed",
        medians=_medians(), config=feat_cfg,
    )
    assert out.status is Status.FAILED
    assert out.error_code is ErrorCode.LAP_NULL_EXCESS


# ----------------------------------------------------------------------------------
# Time base: lap_time proxy, no-clock NULLing, and gear-NaN robustness
# ----------------------------------------------------------------------------------


def test_dt_derived_from_lap_time_when_no_time_column(feat_cfg: AppConfig) -> None:
    """With no per-sample time column, the sample period is derived from lap_time/n (the
    documented F02 proxy), so rate features still compute rather than NULL."""
    df = good_lap().drop(columns=["time"])  # remove the real clock
    n = len(df)
    res = compute_scalar_features(
        df, lap_number=3, session_max_rpm=SESSION_MAX_RPM, lap_distance_m=LAP_DISTANCE_M,
        config=feat_cfg, lap_time_s=n * DT,  # same 50 ms/sample as the dropped clock
    )
    # Rate/duration features are computed (not NULL) and match the explicit-clock values.
    ref = compute_good(feat_cfg).features
    for fid in ("F03", "F07", "F08", "F16", "F24"):
        assert res.features[fid] is not None
        assert res.features[fid] == pytest.approx(ref[fid])


def test_no_time_base_nulls_rate_features(feat_cfg: AppConfig) -> None:
    """No per-sample clock AND no lap_time -> time-dependent features NULL (never fabricated)."""
    df = good_lap().drop(columns=["time"])
    res = compute_scalar_features(
        df, lap_number=3, session_max_rpm=SESSION_MAX_RPM, lap_distance_m=LAP_DISTANCE_M,
        config=feat_cfg, lap_time_s=None,  # no clock at all
    )
    for fid in ("F03", "F07", "F08", "F16", "F24"):
        assert res.features[fid] is None
        assert _null_reason(res, fid) == feat.REASON_NULL_TRIGGER
    # Non-time features are unaffected.
    assert res.features["F01"] is not None


def test_f15_ignores_gear_nan_dropout(feat_cfg: AppConfig) -> None:
    """A NaN gear dropout must not be miscounted as gear changes (F15/F16 finite-filter)."""
    clean = good_lap()
    clean_f15 = _feat(clean, feat_cfg, "F15")
    dropped = good_lap()
    g = dropped["Gear"].to_numpy(float).copy()
    g[10:13] = np.nan  # inject a dropout inside a constant-gear run
    dropped["Gear"] = g
    assert _feat(dropped, feat_cfg, "F15") == pytest.approx(clean_f15)  # no spurious changes


# ----------------------------------------------------------------------------------
# Assembly: a NULL with no train median cannot be returned as accepted
# ----------------------------------------------------------------------------------


def test_unimputable_null_fails_not_accepted(feat_cfg: AppConfig) -> None:
    """A <=6-NULL vector whose NULL has no train median is FAILED, not a surviving-None accept."""
    medians = FeatureMedians(global_medians={})  # no medians at all
    out = assemble_feature_vector(
        _vector_with_nulls(1), lap_number=1, driver="VER", circuit_type="high_speed",
        medians=medians, config=feat_cfg,
    )
    assert out.status is Status.FAILED
    assert out.error_code is ErrorCode.FEATURE_001
    assert out.imputed_features == []
    assert any(v is None for v in out.feature_vector.values())  # not silently filled


# ----------------------------------------------------------------------------------
# Rate-feature resample invariance (F03, F08)
# ----------------------------------------------------------------------------------


def _ramp_lap(n: int, dt: float) -> pd.DataFrame:
    half = n // 2
    brake = np.concatenate([np.linspace(0, 80, half), np.linspace(80, 0, n - half)])
    return pd.DataFrame(
        {
            "time": np.arange(n) * dt,
            "Speed": np.linspace(250, 150, n),
            "Throttle": np.linspace(0, 100, n),  # strictly rising
            "Brake": brake,
            "RPM": np.linspace(8000, 11000, n),
            "Gear": 5.0,
            "lap": 1,
        }
    )


def test_config_features_absent_nulls_all_with_config_001(feat_cfg: AppConfig) -> None:
    """If the features: block is missing, every feature NULLs and CONFIG_001 is logged —
    a typed failure, not a bare raise (non-negotiable #1)."""
    feat_cfg.features = None
    res = compute_scalar_features(
        good_lap(), lap_number=3, session_max_rpm=SESSION_MAX_RPM,
        lap_distance_m=LAP_DISTANCE_M, config=feat_cfg,
    )
    assert res.error_code is ErrorCode.CONFIG_001
    assert all(res.features[fid] is None for fid in SCALAR_FEATURE_IDS)
    rows = read_dq(feat_cfg.paths.data_quality_log)
    assert len(rows) == 1 and rows[0]["error_code"] == ErrorCode.CONFIG_001.value


# ----------------------------------------------------------------------------------
# Spec §8 boundary pins (both sides of each threshold)
# ----------------------------------------------------------------------------------


def _flat_lap(n: int = 40, **channels: object) -> pd.DataFrame:
    base: dict[str, object] = {
        "Speed": 200.0, "Throttle": 50.0, "Brake": 0.0, "RPM": 9000.0, "Gear": 5.0,
    }
    base.update(channels)
    data: dict[str, object] = {"time": np.arange(n) * DT, "lap": 1}
    for ch, val in base.items():
        data[ch] = np.full(n, val, dtype=float) if np.isscalar(val) else np.asarray(val, float)
    return pd.DataFrame(data)


def _feat(df: pd.DataFrame, cfg: AppConfig, fid: str) -> float | None:
    res = compute_scalar_features(
        df, lap_number=1, session_max_rpm=SESSION_MAX_RPM,
        lap_distance_m=LAP_DISTANCE_M, config=cfg,
    )
    return res.features[fid]


def test_f04_full_throttle_boundary(feat_cfg: AppConfig) -> None:
    # Half the samples exactly 100 (full), half 99.9 (not) -> fraction 0.5.
    thr = np.tile([100.0, 99.9], 20)
    assert _feat(_flat_lap(40, Throttle=thr), feat_cfg, "F04") == pytest.approx(0.5)


def test_f05_lift_throttle_boundary(feat_cfg: AppConfig) -> None:
    # 5.0 is NOT a lift (strict <5); 4.9 is -> only the 4.9s count -> fraction 0.5.
    thr = np.tile([5.0, 4.9], 20)
    assert _feat(_flat_lap(40, Throttle=thr), feat_cfg, "F05") == pytest.approx(0.5)


def _brake_pulses(k: int, width: int = 3, gap: int = 3) -> np.ndarray:
    unit = [80.0] * width + [0.0] * gap
    return np.array(unit * k, dtype=float)


def test_f06_braking_event_count_boundary(feat_cfg: AppConfig) -> None:
    # Exactly min_braking_events (5) -> computed; one fewer (4) -> NULL.
    five = _brake_pulses(5)
    assert _feat(_flat_lap(five.size, Brake=five), feat_cfg, "F06") == pytest.approx(80.0)
    four = _brake_pulses(4)
    assert _feat(_flat_lap(four.size, Brake=four), feat_cfg, "F06") is None


def _gear_k_changes(k: int, seg: int = 3) -> np.ndarray:
    vals: list[float] = []
    for i in range(k + 1):
        vals += [5.0 if i % 2 == 0 else 6.0] * seg
    return np.array(vals, dtype=float)


def test_f16_gear_change_count_boundary(feat_cfg: AppConfig) -> None:
    ten = _gear_k_changes(10)
    assert _feat(_flat_lap(ten.size, Gear=ten), feat_cfg, "F16") is not None
    nine = _gear_k_changes(9)
    assert _feat(_flat_lap(nine.size, Gear=nine), feat_cfg, "F16") is None


def _transition_lap(k: int) -> pd.DataFrame:
    brake: list[float] = []
    throttle: list[float] = []
    for _ in range(k):
        brake += [80.0, 80.0, 0.0, 0.0]  # braking event ...
        throttle += [0.0, 0.0, 50.0, 50.0]  # ... then throttle re-applied
    n = len(brake)
    return _flat_lap(n, Brake=np.array(brake), Throttle=np.array(throttle))


def test_f24_transition_count_boundary(feat_cfg: AppConfig) -> None:
    assert _feat(_transition_lap(5), feat_cfg, "F24") is not None
    assert _feat(_transition_lap(4), feat_cfg, "F24") is None


def test_f14_p90_range_boundary(feat_cfg: AppConfig) -> None:
    # p90 == 100 is the inclusive lower bound (accepted); 99.99 is out of range (NULL).
    assert _feat(_flat_lap(40, Speed=100.0), feat_cfg, "F14") == pytest.approx(100.0)
    assert _feat(_flat_lap(40, Speed=99.99), feat_cfg, "F14") is None


def test_f10_strictly_above_cap_is_capped(feat_cfg: AppConfig) -> None:
    # Overlap fraction just above the cap (0.2501 > 0.25) -> clamped to the cap.
    n = 10000
    k = 2501  # 0.2501
    overlap = np.zeros(n, dtype=bool)
    overlap[:k] = True
    df = _flat_lap(
        n,
        Throttle=np.where(overlap, 80.0, 0.0),
        Brake=np.where(overlap, 80.0, 0.0),
    )
    res = compute_scalar_features(
        df, lap_number=1, session_max_rpm=SESSION_MAX_RPM,
        lap_distance_m=LAP_DISTANCE_M, config=feat_cfg,
    )
    assert res.features["F10"] == pytest.approx(feat_cfg.features.overlap_cap)  # type: ignore[union-attr]
    assert "F10" in res.capped_features


def test_rate_features_resample_invariant(feat_cfg: AppConfig) -> None:
    slow = compute_scalar_features(
        _ramp_lap(100, 0.05), lap_number=1, session_max_rpm=SESSION_MAX_RPM,
        lap_distance_m=LAP_DISTANCE_M, config=feat_cfg,
    )
    fast = compute_scalar_features(
        _ramp_lap(250, 0.02), lap_number=1, session_max_rpm=SESSION_MAX_RPM,
        lap_distance_m=LAP_DISTANCE_M, config=feat_cfg,
    )
    # Rescaling to sample_period_ms makes these match within discretisation noise; WITHOUT
    # it the two rates would differ by 2.5x (the dt ratio), so 2% proves invariance.
    assert slow.features["F03"] == pytest.approx(fast.features["F03"], rel=2e-2)
    assert slow.features["F08"] == pytest.approx(fast.features["F08"], rel=2e-2)


# ==================================================================================
# F03b — geometry (F19-F21, F25) + degradation slopes (F27, F28)
# ==================================================================================

GRAVITY = 9.81


def _arc_lap(*, radius_m: float = 50.0, v_kmh: float = 108.0, n: int = 240) -> pd.DataFrame:
    """A constant-radius circular arc at constant speed -> analytic lateral G = v^2/(R*g)."""
    v = v_kmh / 3.6
    dt = (2 * np.pi * radius_m / v) / n
    theta = np.linspace(0.0, 2 * np.pi, n)
    return pd.DataFrame(
        {
            "time": np.arange(n) * dt,
            "X": radius_m * np.cos(theta),
            "Y": radius_m * np.sin(theta),
            "Speed": v_kmh,
            "Throttle": 50.0, "Brake": 0.0, "RPM": 9000.0, "Gear": 5.0, "lap": 1,
        }
    )


def _slalom_lap(
    *, cycles: int = 3, n: int = 360, amp: float = 25.0, v_kmh: float = 150.0
) -> pd.DataFrame:
    """Sinusoidal lateral path: amp*sin gives 2*cycles alternating-sign curvature peaks ->
    exercises F20 (peaks) and F21 (direction reversals)."""
    i = np.arange(n)
    v = v_kmh / 3.6
    return pd.DataFrame(
        {
            "time": i * (3.0 / v), "X": i * 3.0, "Y": amp * np.sin(2 * np.pi * cycles * i / n),
            "Speed": v_kmh, "Throttle": 50.0, "Brake": 0.0, "RPM": 9000.0, "Gear": 5.0, "lap": 1,
        }
    )


def _corner_brake_lap() -> pd.DataFrame:
    """Straight X/Y (X/Y available) with two speed valleys (corners) and braking only on the
    ENTRY side of each -> F25 ratio ~ 1.0 (all in-corner braking before the apex)."""
    block = [280.0, 260.0, 220.0, 180.0, 140.0, 120.0, 140.0, 180.0, 220.0, 260.0]
    speed = np.array(block * 2 + [280.0], dtype=float)
    n = speed.size
    brake = np.zeros(n)
    brake[1:5] = 80.0   # descending side of valley 1 (apex at idx 5)
    brake[11:15] = 80.0  # descending side of valley 2 (apex at idx 15)
    return pd.DataFrame(
        {
            "time": np.arange(n) * DT, "X": np.arange(n, dtype=float), "Y": 0.0,
            "Speed": speed, "Throttle": 0.0, "Brake": brake, "RPM": 9000.0, "Gear": 5.0, "lap": 1,
        }
    )


def _geom(df: pd.DataFrame, cfg: AppConfig) -> feat.GeometryFeatureResult:
    return compute_geometry_features(df, lap_number=1, config=cfg)


def _stint(n: int, *, throttle_slope: float = -0.5, brake_slope: float = 1.0,
           tyre_age_known: bool = True) -> list[StintLap]:
    return [
        StintLap(
            lap_number=i,
            tyre_age=(i if tyre_age_known else None),
            throttle_mean=80.0 + throttle_slope * i,
            brake_peak_mean=100.0 + brake_slope * i,
        )
        for i in range(n)
    ]


# --- happy path ---


def test_geometry_result_keys_exactly_four(feat_cfg: AppConfig) -> None:
    res = _geom(_slalom_lap(), feat_cfg)
    assert set(res.features) == set(GEOMETRY_FEATURE_IDS)
    for fid in ("F27", "F28"):
        assert fid not in res.features


def test_f19_lateral_g_matches_analytic_arc(feat_cfg: AppConfig) -> None:
    res = _geom(_arc_lap(radius_m=50.0, v_kmh=108.0), feat_cfg)
    expected = (108.0 / 3.6) ** 2 / (50.0 * GRAVITY)  # v^2/(R*g) ~ 1.835 G
    assert res.features["F19"] == pytest.approx(expected, rel=0.05)
    assert 0.0 <= res.features["F19"] <= 6.0


def test_f20_peaks_on_slalom(feat_cfg: AppConfig) -> None:
    res = _geom(_slalom_lap(cycles=3), feat_cfg)  # 6 curvature peaks >= min 5
    assert res.features["F20"] is not None
    assert 0.0 <= res.features["F20"] <= 6.0


def test_f21_reversals_slalom_vs_arc(feat_cfg: AppConfig) -> None:
    slalom = _geom(_slalom_lap(cycles=3), feat_cfg)
    arc = _geom(_arc_lap(), feat_cfg)
    assert slalom.features["F21"] > 0.0  # alternating turns -> reversals
    assert arc.features["F21"] == pytest.approx(0.0)  # single-direction arc -> none


def test_f25_corner_balance_entry_biased(feat_cfg: AppConfig) -> None:
    res = _geom(_corner_brake_lap(), feat_cfg)
    assert res.features["F25"] == pytest.approx(1.0)  # all in-corner braking before the apex


def test_degradation_slopes_known_trend(feat_cfg: AppConfig) -> None:
    res = compute_degradation_slopes(_stint(10), stint_id=1, config=feat_cfg)
    assert res.f27 == pytest.approx(-0.5)
    assert res.f28 == pytest.approx(1.0)
    assert res.applies_to == list(range(10))
    assert res.status is Status.OK


# --- X/Y unavailable: CORNER_001, four NULL, lap survives ---


def test_xy_unavailable_nulls_four_and_logs_corner_001(feat_cfg: AppConfig) -> None:
    df = _slalom_lap().drop(columns=["X", "Y"])
    res = _geom(df, feat_cfg)
    assert all(res.features[fid] is None for fid in GEOMETRY_FEATURE_IDS)
    assert set(res.null_features) == set(GEOMETRY_FEATURE_IDS)
    assert res.error_code is ErrorCode.CORNER_001
    assert res.status is Status.DEGRADED  # lap continues, not FAILED
    rows = read_dq(feat_cfg.paths.data_quality_log)
    assert len(rows) == 1 and rows[0]["error_code"] == ErrorCode.CORNER_001.value


def test_xy_all_nan_logs_corner_001(feat_cfg: AppConfig) -> None:
    df = _slalom_lap()
    df["X"] = np.nan
    df["Y"] = np.nan
    res = _geom(df, feat_cfg)
    assert res.error_code is ErrorCode.CORNER_001
    assert all(res.features[fid] is None for fid in GEOMETRY_FEATURE_IDS)


def test_xy_mostly_nan_below_fraction_logs_corner_001(feat_cfg: AppConfig) -> None:
    """A near-total X/Y dropout (finite fraction < min_xy_finite_fraction) is unavailable, not
    fabricated from the few finite samples."""
    df = _slalom_lap()
    xy = df["X"].to_numpy(float).copy()
    xy[5:] = np.nan  # only 5 finite of 360 -> ~1.4% << 90%
    df["X"] = xy
    df["Y"] = xy
    res = _geom(df, feat_cfg)
    assert res.error_code is ErrorCode.CORNER_001
    assert all(res.features[fid] is None for fid in GEOMETRY_FEATURE_IDS)


def test_geometry_context_build_error_is_caught(feat_cfg: AppConfig, monkeypatch) -> None:
    """An unexpected error while building the geometry context is caught: 4 NULL, no raise."""
    def boom(*_a, **_k):
        raise ValueError("synthetic gradient failure")

    monkeypatch.setattr(feat, "_signed_lateral_g", boom)
    res = _geom(_slalom_lap(), feat_cfg)
    assert all(res.features[fid] is None for fid in GEOMETRY_FEATURE_IDS)
    assert res.status is Status.DEGRADED
    assert str(res.detail["F19"]["reason"]) == feat.REASON_UNEXPECTED


# --- per-feature NULL / out-of-range (X/Y present) ---


def test_f19_out_of_range_raises_no_clamp(feat_cfg: AppConfig) -> None:
    res = _geom(_arc_lap(radius_m=10.0, v_kmh=200.0), feat_cfg)  # G >> 6
    assert res.features["F19"] is None
    assert str(res.detail["F19"]["reason"]) == feat.REASON_OUT_OF_RANGE


def test_checked_raises_on_out_of_range_geometry() -> None:
    cfg = load_config().features
    assert cfg is not None
    with pytest.raises(FeatureComputationError) as ei:
        feat._checked("F19", 7.0, cfg)  # range [0, 6]
    assert ei.value.reason == feat.REASON_OUT_OF_RANGE


def test_f20_too_few_peaks_on_arc_nulls(feat_cfg: AppConfig) -> None:
    res = _geom(_arc_lap(), feat_cfg)  # constant curvature -> no distinct peaks
    assert res.features["F20"] is None
    assert str(res.detail["F20"]["reason"]) == feat.REASON_NULL_TRIGGER


def test_f25_no_corners_nulls(feat_cfg: AppConfig) -> None:
    res = _geom(_slalom_lap(), feat_cfg)  # constant speed -> no speed-minimum corners
    assert res.features["F25"] is None
    assert str(res.detail["F25"]["reason"]) == feat.REASON_NULL_TRIGGER


def test_f21_no_time_base_nulls(feat_cfg: AppConfig) -> None:
    """F21 (X/Y present) NULLs when there is no clock to give a per-second rate -> the
    documented 'lap duration unavailable' trigger."""
    df = _slalom_lap().drop(columns=["time"])  # no per-sample clock and no lap_time column
    res = _geom(df, feat_cfg)
    assert res.features["F21"] is None
    assert str(res.detail["F21"]["reason"]) == feat.REASON_NULL_TRIGGER
    # X/Y is still present, so this is a per-feature FEATURE_001 NULL, not CORNER_001.
    assert res.error_code is not ErrorCode.CORNER_001


def test_f21_lateral_g_not_computable_nulls(feat_cfg: AppConfig) -> None:
    """F21 second trigger: X/Y present but speed all-NaN -> lateral G undefined -> NULL."""
    df = _slalom_lap()
    df["Speed"] = np.nan
    res = _geom(df, feat_cfg)
    assert res.features["F21"] is None
    assert "lateral G not computable" in str(res.detail["F21"]["detail"])


def test_f25_no_in_corner_braking_nulls(feat_cfg: AppConfig) -> None:
    """F25 second trigger: corners exist but no braking inside any corner window -> NULL."""
    df = _corner_brake_lap()
    df["Brake"] = 0.0  # corners (speed valleys) remain, but no in-corner braking
    res = _geom(df, feat_cfg)
    assert res.features["F25"] is None
    assert "no corner has braking" in str(res.detail["F25"]["detail"])


# --- F27/F28 stint gate ---


def test_stint_too_short_nulls_f27_f28(feat_cfg: AppConfig) -> None:
    res = compute_degradation_slopes(_stint(7), stint_id=1, config=feat_cfg)  # 7 < 8
    assert res.f27 is None and res.f28 is None
    assert res.status is Status.DEGRADED
    rows = read_dq(feat_cfg.paths.data_quality_log)  # 7 laps * 2 NULL slopes
    assert len(rows) == 14
    assert all(r["error_code"] == ErrorCode.FEATURE_001.value for r in rows)


def test_stint_exactly_min_computes(feat_cfg: AppConfig) -> None:
    res = compute_degradation_slopes(_stint(8), stint_id=1, config=feat_cfg)  # boundary: 8
    assert res.f27 is not None and res.f28 is not None


def test_tyre_age_unknown_nulls_f27_f28(feat_cfg: AppConfig) -> None:
    res = compute_degradation_slopes(_stint(10, tyre_age_known=False), stint_id=1, config=feat_cfg)
    assert res.f27 is None and res.f28 is None
    assert "tyre age UNKNOWN" in (res.null_reason or "")


def test_f27_f28_are_unconstrained(feat_cfg: AppConfig) -> None:
    res = compute_degradation_slopes(
        _stint(10, throttle_slope=1e6, brake_slope=-1e6), stint_id=1, config=feat_cfg
    )
    assert res.f27 == pytest.approx(1e6)
    assert res.f28 == pytest.approx(-1e6)


def test_degradation_never_raises_on_polyfit_error(feat_cfg: AppConfig, monkeypatch) -> None:
    """A numpy failure in the slope fit is caught -> both slopes NULL, no exception escapes."""
    def boom(*_a, **_k):
        raise np.linalg.LinAlgError("synthetic LAPACK failure")

    monkeypatch.setattr(feat.np, "polyfit", boom)
    res = compute_degradation_slopes(_stint(10), stint_id=1, config=feat_cfg)
    assert res.f27 is None and res.f28 is None
    assert res.status is Status.DEGRADED


def test_stint_known_age_but_all_values_null_reports_usable_laps(feat_cfg: AppConfig) -> None:
    """Reason is '0 usable laps < N' (not the misleading 'tyre age UNKNOWN') when tyre ages
    are known but every F01/F06 value is NULL."""
    stint = [StintLap(lap_number=i, tyre_age=i, throttle_mean=None, brake_peak_mean=None)
             for i in range(10)]
    res = compute_degradation_slopes(stint, stint_id=1, config=feat_cfg)
    assert res.f27 is None
    assert "usable laps" in (res.null_reason or "")
    assert "tyre age UNKNOWN" not in (res.null_reason or "")


def test_empty_stint_is_logged(feat_cfg: AppConfig) -> None:
    """A degenerate empty stint NULLs F27/F28 but is never unaudited (one stint-level row)."""
    res = compute_degradation_slopes([], stint_id=9, config=feat_cfg)
    assert res.f27 is None and res.f28 is None
    assert res.logged is True
    rows = read_dq(feat_cfg.paths.data_quality_log)
    assert len(rows) == 1 and rows[0]["entity_id"] == "stint9"


# --- config-driven ---


def test_min_stint_laps_is_config_driven(feat_cfg: AppConfig) -> None:
    feat_cfg.features.min_stint_laps = 12  # type: ignore[union-attr]
    res = compute_degradation_slopes(_stint(10), stint_id=1, config=feat_cfg)  # 10 < new 12
    assert res.f27 is None


def test_f19_range_is_config_driven(feat_cfg: AppConfig) -> None:
    feat_cfg.features.valid_ranges["F19"] = (0.0, 1.0)  # type: ignore[union-attr]
    res = _geom(_arc_lap(), feat_cfg)  # arc ~1.8 G now out of the tightened range
    assert res.features["F19"] is None
    assert str(res.detail["F19"]["reason"]) == feat.REASON_OUT_OF_RANGE


# --- integration with F03a's assemble_feature_vector (unchanged) ---


def _full_values() -> dict[str, float | None]:
    return {fid: 1.0 for fid in FEATURE_IDS}


def test_xy_loss_plus_short_stint_sits_at_ceiling(feat_cfg: AppConfig) -> None:
    values = _full_values()
    for fid in (*GEOMETRY_FEATURE_IDS, "F27", "F28"):  # 4 + 2 = 6 NULL == ceiling
        values[fid] = None
    medians = FeatureMedians(global_medians={fid: 5.0 for fid in FEATURE_IDS})
    out = assemble_feature_vector(
        values, lap_number=1, driver="VER", circuit_type="high_speed",
        medians=medians, config=feat_cfg,
    )
    assert out.status is Status.DEGRADED
    assert out.error_code is None
    assert out.null_count == 6
    assert len(out.imputed_features) == 6


def test_one_more_null_over_ceiling_rejects(feat_cfg: AppConfig) -> None:
    values = _full_values()
    for fid in (*GEOMETRY_FEATURE_IDS, "F27", "F28", "F01"):  # 7 NULL
        values[fid] = None
    medians = FeatureMedians(global_medians={fid: 5.0 for fid in FEATURE_IDS})
    out = assemble_feature_vector(
        values, lap_number=1, driver="VER", circuit_type="high_speed",
        medians=medians, config=feat_cfg,
    )
    assert out.status is Status.FAILED
    assert out.error_code is ErrorCode.LAP_NULL_EXCESS
