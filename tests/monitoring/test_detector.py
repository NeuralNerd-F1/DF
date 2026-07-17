"""F10 degradation-detector tests (§7 acceptance criteria + §6 failure rows).

Every §6 failure row has a test that triggers it and asserts the typed result AND the code
placement; ``monitor_stint`` / ``lead_time_report`` never-raise is proven by fault injection;
calibration leakage + determinism + the train-only standardiser are proven behaviourally.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from app.contracts.enums import Status
from app.contracts.errors import ErrorCode
from app.monitoring import detector as det
from app.monitoring.detector import (
    AlertConfidence,
    DetectorCalibrationError,
    DetectorLoadError,
    MonitorHandle,
    StintMonitorState,
    _AlertOutcome,
    _build_alert,
    _Cell,
    _LapCore,
    _StintCore,
    calibrate_stint_variation,
    health,
    lead_time_report,
    load_detector,
    monitor_stint,
)
from app.pipeline.normalise import fit_normaliser

from tests.monitoring.conftest import (
    _make_config,
    build_eval_matrix,
    build_training_matrix,
    make_stint,
)


def _codes(read_dq: Any, cfg: Any) -> set[str]:
    return {r["error_code"] for r in read_dq(cfg.paths.data_quality_log)}


def _with(handle: MonitorHandle, **kw: Any) -> MonitorHandle:
    return dataclasses.replace(handle, **kw)


def _drift(**kw: Any) -> Any:
    d: dict[str, Any] = dict(
        driver="VER", circuit="Bahrain", circuit_type="high_speed", n_laps=16, drift_from=8,
        drift_mag=2.0, seed=999,
    )
    d.update(kw)
    return make_stint(**d)


# ==================================================================================
# Calibration happy path + determinism + leakage + standardiser (§7-14/15, §4.1)
# ==================================================================================


def test_calibration_status_and_shape(calibrated: dict[str, Any]) -> None:
    fit = calibrated["fit"]
    art_dir = calibrated["art_dir"]
    assert fit.status is Status.OK
    assert fit.n_stints_used == 16
    assert fit.n_entries == 4  # 2 drivers × 2 circuit types
    assert fit.n_driver_global == 2
    assert fit.coverage == 1.0
    assert fit.mlflow_run_id is not None  # MLflow logged the calibration run (§7-19 happy path)
    assert (art_dir / "stint_variation.json").is_file()


def test_calibration_deterministic_byte_identical(
    tmp_path: Path, make_config: Any
) -> None:
    matrix = build_training_matrix()

    def _one(sub: str) -> bytes:
        d = tmp_path / sub
        cfg = make_config(d)
        art = Path(cfg.paths.artefacts_dir)
        fit_normaliser(matrix, config=cfg)
        calibrate_stint_variation(matrix, config=cfg, artefact_dir=art)
        return (art / "stint_variation.json").read_bytes()

    assert _one("run_a") == _one("run_b")  # no timestamp, no RNG → byte-identical


def test_assert_no_leakage_raises_on_held_out(monitoring_config: Any) -> None:
    frame = build_training_matrix()
    frame.loc[0, "circuit"] = "Belgium"  # a held-out circuit survives into an unfiltered frame
    with pytest.raises(DetectorCalibrationError) as exc:
        det._assert_no_leakage(frame, monitoring_config)
    assert exc.value.error_code is ErrorCode.TRAIN_LEAKAGE


def test_calibrate_leakage_logs_and_writes_no_artefact(
    monitoring_config: Any, read_dq: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = monitoring_config
    art = Path(cfg.paths.artefacts_dir)
    matrix = build_training_matrix()

    def _boom(_train: Any, _config: Any) -> None:
        raise DetectorCalibrationError("held-out leaked", error_code=ErrorCode.TRAIN_LEAKAGE)

    monkeypatch.setattr(det, "_assert_no_leakage", _boom)
    with pytest.raises(DetectorCalibrationError) as exc:
        calibrate_stint_variation(matrix, config=cfg, artefact_dir=art)
    assert exc.value.error_code is ErrorCode.TRAIN_LEAKAGE
    assert not (art / "stint_variation.json").is_file()
    assert "TRAIN_LEAKAGE" in _codes(read_dq, cfg)


def test_calibrate_quarantines_held_out_train_row(monitoring_config: Any) -> None:
    """The real train_mask excludes a held-out circuit even when it is mislabeled split=="train":
    calibration succeeds (no TRAIN_LEAKAGE) and the held-out stint is NOT among the kept stints."""
    import pandas as pd

    cfg = monitoring_config
    art = Path(cfg.paths.artefacts_dir)
    clean = build_training_matrix()
    poison = make_stint(driver="VER", circuit="Belgium", circuit_type="street",
                        session="Belgium_x", split="train", n_laps=12)  # held-out, mislabeled train
    matrix = pd.concat([clean, poison], ignore_index=True)
    fit_normaliser(matrix, config=cfg)
    fit = calibrate_stint_variation(matrix, config=cfg, artefact_dir=art)
    assert fit.status in (Status.OK, Status.DEGRADED)  # did not raise TRAIN_LEAKAGE
    assert fit.n_stints_used == 16  # the Belgium stint was quarantined, not counted


def test_calibrate_empty_post_quarantine_fails(monitoring_config: Any) -> None:
    cfg = monitoring_config
    matrix = build_training_matrix()
    matrix["split"] = "val"  # nothing left after quarantining to split=="train"
    fit_normaliser(build_training_matrix(), config=cfg)
    with pytest.raises(DetectorCalibrationError) as exc:
        calibrate_stint_variation(matrix, config=cfg, artefact_dir=Path(cfg.paths.artefacts_dir))
    assert exc.value.error_code is ErrorCode.MODEL_TRAINING_FAILED


def test_calibrate_write_fault_typed_and_logged(monitoring_config: Any, read_dq: Any) -> None:
    """A filesystem write fault is a typed MODEL_TRAINING_FAILED + DQ row, never a raw OSError
    (§6 "write fault" row, Finding 3)."""
    cfg = monitoring_config
    art = Path(cfg.paths.artefacts_dir)
    matrix = build_training_matrix()
    fit_normaliser(matrix, config=cfg)
    art.mkdir(parents=True, exist_ok=True)
    (art / "stint_variation.json").mkdir()  # a DIRECTORY where the file must be written → OSError
    with pytest.raises(DetectorCalibrationError) as exc:
        calibrate_stint_variation(matrix, config=cfg, artefact_dir=art)
    assert exc.value.error_code is ErrorCode.MODEL_TRAINING_FAILED
    assert "MODEL_TRAINING_FAILED" in _codes(read_dq, cfg)


def test_standardiser_fitted_on_quarantined_train_only(calibrated: dict[str, Any]) -> None:
    """The persisted feature_scaler mean equals the hand-recomputed mean of the quarantined,
    F04-normalised training rows (behavioural proof, F08 precedent)."""
    from app.pipeline.normalise import apply_normalisation, load_normalisation_params

    art_dir = calibrated["art_dir"]
    cfg = calibrated["config"]
    matrix = calibrated["matrix"]
    handle = calibrated["handle"]
    params = load_normalisation_params(art_dir / "normalisation.json")
    fid = handle.cosine_features[0]
    vals: list[float] = []
    for rec in matrix.to_dict("records"):
        vec = {f: float(rec[f]) for f in det.FEATURE_IDS}
        res = apply_normalisation(vec, circuit=str(rec["circuit"]), params=params, config=cfg)
        nv = res.normalised_vector[fid]
        if nv is not None:
            vals.append(float(nv))
    assert handle.feature_mean[fid] == pytest.approx(float(np.mean(vals)))


def test_mixed_scale_standardiser_absorbs_unit_scale(
    tmp_path: Path, make_config: Any
) -> None:
    """×10³ on one raw-unit feature column → its feature_std scales ~×10³, so the standardiser
    cancels the unit scale (the cosine core is scale-invariant — the §4.0-e / F08-#6 defence)."""
    base = build_training_matrix()

    def _std_of(matrix: Any, sub: str) -> float:
        cfg = make_config(tmp_path / sub)
        art = Path(cfg.paths.artefacts_dir)
        fit_normaliser(matrix, config=cfg)
        calibrate_stint_variation(matrix, config=cfg, artefact_dir=art)
        artefact = json.loads((art / "stint_variation.json").read_text(encoding="utf-8"))
        return artefact["feature_scaler"]["F03"]["std"]

    scaled = base.copy()
    scaled["F03"] = scaled["F03"] * 1000.0
    ratio = _std_of(scaled, "scaled") / _std_of(base, "base")
    assert 900.0 < ratio < 1100.0  # std tracks the unit scale → cosine attribution invariant


# ==================================================================================
# Serving happy path (§7-1/2/4/5/8/9)
# ==================================================================================


def test_happy_state_progression_and_alert(calibrated: dict[str, Any]) -> None:
    handle, cfg = calibrated["handle"], calibrated["config"]
    res = monitor_stint(_drift(), driver="VER", circuit="Bahrain", circuit_type="high_speed",
                        handle=handle, config=cfg)
    assert len(res.laps) == 16  # one LapDeviation per input lap — no lap dropped
    states = [lp.state for lp in res.laps]
    assert states[:3] == [StintMonitorState.BASELINE] * 3
    assert StintMonitorState.PENDING in states
    assert StintMonitorState.NOMINAL in states
    assert StintMonitorState.WATCH in states
    assert StintMonitorState.ALERT in states
    assert res.alert is not None
    assert res.baseline_lap_numbers == [1, 2, 3]
    assert res.baseline_extended is False
    assert res.status is Status.OK  # clean stint: no fallback, no degradation


def test_alert_payload_fields(calibrated: dict[str, Any]) -> None:
    handle, cfg = calibrated["handle"], calibrated["config"]
    res = monitor_stint(_drift(), driver="VER", circuit="Bahrain", circuit_type="high_speed",
                        handle=handle, config=cfg)
    alert = res.alert
    assert alert is not None
    assert alert.first_crossing_lap < alert.alert_lap
    assert alert.laps_since_threshold_crossed == alert.alert_lap - alert.first_crossing_lap == 1
    assert set(alert.feature_attribution) == set(det.FEATURE_IDS)  # all 28 keys
    assert len(alert.top_features) == cfg.monitoring.attribution_top_k
    assert alert.alert_confidence in set(AlertConfidence)


def test_attribution_sums_to_deviation_score(calibrated: dict[str, Any]) -> None:
    handle, cfg = calibrated["handle"], calibrated["config"]
    res = monitor_stint(_drift(drift_mag=4.0), driver="VER", circuit="Bahrain",
                        circuit_type="high_speed", handle=handle, config=cfg)
    alert = res.alert
    assert alert is not None
    assert sum(alert.feature_attribution.values()) == pytest.approx(alert.deviation_score, abs=1e-9)


def test_pending_before_rolling_window_fills(calibrated: dict[str, Any]) -> None:
    handle, cfg = calibrated["handle"], calibrated["config"]
    res = monitor_stint(_drift(drift_mag=0.0), driver="VER", circuit="Bahrain",
                        circuit_type="high_speed", handle=handle, config=cfg)
    # laps 4 & 5 (0-indexed 3,4) are PENDING (rolling_window=3 not yet filled).
    assert res.laps[3].state is StintMonitorState.PENDING
    assert res.laps[3].deviation_score is None
    assert res.laps[5].state is StintMonitorState.NOMINAL  # lap 6 = 3rd valid distance
    assert res.laps[5].deviation_score is not None


# ==================================================================================
# Activation + zero-composite (§7-5/7, §6 rows)
# ==================================================================================


def test_short_stint_inactive(calibrated: dict[str, Any], read_dq: Any) -> None:
    handle, cfg = calibrated["handle"], calibrated["config"]
    res = monitor_stint(make_stint(n_laps=7), driver="VER", circuit="Bahrain",
                        circuit_type="high_speed", handle=handle, config=cfg)
    assert res.status is Status.DEGRADED
    assert res.error_code is ErrorCode.MONITOR_INACTIVE
    assert res.monitor_state is StintMonitorState.INACTIVE
    assert res.laps == []
    assert "MONITOR_INACTIVE" in _codes(read_dq, cfg)


def test_exactly_eight_laps_activates(calibrated: dict[str, Any]) -> None:
    handle, cfg = calibrated["handle"], calibrated["config"]
    res = monitor_stint(make_stint(n_laps=8), driver="VER", circuit="Bahrain",
                        circuit_type="high_speed", handle=handle, config=cfg)
    assert res.error_code is not ErrorCode.MONITOR_INACTIVE
    assert res.monitor_state is not StintMonitorState.INACTIVE


def test_zero_composite_degraded_pending(calibrated: dict[str, Any], read_dq: Any) -> None:
    handle, cfg = calibrated["handle"], calibrated["config"]
    # baseline laps 1-3 clean; every post-baseline lap is a safety-car SKIP → no composite score.
    stint = make_stint(n_laps=12, sc_laps=frozenset(range(4, 13)))
    res = monitor_stint(stint, driver="VER", circuit="Bahrain", circuit_type="high_speed",
                        handle=handle, config=cfg)
    assert res.status is Status.DEGRADED
    assert res.monitor_state is StintMonitorState.PENDING
    assert res.alert is None
    assert res.n_laps_scored == 0
    rows = read_dq(cfg.paths.data_quality_log)
    assert any("ZERO_COMPOSITE" in r["error_detail"] for r in rows)  # the one stint-level row


# ==================================================================================
# NaN handling + halt (§6 rows 6/7, §7 property tests)
# ==================================================================================


def test_two_nans_no_halt(calibrated: dict[str, Any]) -> None:
    handle, cfg = calibrated["handle"], calibrated["config"]
    stint = make_stint(n_laps=14, nan_laps=frozenset({10, 11}))
    res = monitor_stint(stint, driver="VER", circuit="Bahrain", circuit_type="high_speed",
                        handle=handle, config=cfg)
    assert res.error_code is not ErrorCode.MONITOR_ERROR
    nan_rows = [lp for lp in res.laps if ErrorCode.DISTANCE_NAN in lp.flags]
    assert len(nan_rows) == 2
    assert all(lp.deviation_raw is None for lp in nan_rows)


def test_three_consecutive_nans_halt(calibrated: dict[str, Any], read_dq: Any) -> None:
    handle, cfg = calibrated["handle"], calibrated["config"]
    stint = make_stint(n_laps=16, nan_laps=frozenset({10, 11, 12}))
    res = monitor_stint(stint, driver="VER", circuit="Bahrain", circuit_type="high_speed",
                        handle=handle, config=cfg)
    assert res.status is Status.FAILED
    assert res.error_code is ErrorCode.MONITOR_ERROR
    assert res.monitor_state is StintMonitorState.HALTED
    assert res.laps[-1].lap_number == 12  # honest partial series up to the halt
    assert "MONITOR_ERROR" in _codes(read_dq, cfg)


def test_skip_freezes_nan_run_halts(calibrated: dict[str, Any]) -> None:
    # NaN, SKIP, NaN, NaN → the SKIP freezes (does not reset) the run → the 3rd NaN halts.
    handle, cfg = calibrated["handle"], calibrated["config"]
    stint = make_stint(n_laps=16, nan_laps=frozenset({10, 12, 13}), sc_laps=frozenset({11}))
    res = monitor_stint(stint, driver="VER", circuit="Bahrain", circuit_type="high_speed",
                        handle=handle, config=cfg)
    assert res.error_code is ErrorCode.MONITOR_ERROR


def test_skip_freezes_nan_run_no_halt_on_reset(calibrated: dict[str, Any]) -> None:
    # NaN, SKIP, NaN, VALID → the valid distance resets → never reaches 3 consecutive NaN.
    handle, cfg = calibrated["handle"], calibrated["config"]
    stint = make_stint(n_laps=16, nan_laps=frozenset({10, 12}), sc_laps=frozenset({11}))
    res = monitor_stint(stint, driver="VER", circuit="Bahrain", circuit_type="high_speed",
                        handle=handle, config=cfg)
    assert res.error_code is not ErrorCode.MONITOR_ERROR


def test_nan_in_one_baseline_lap_does_not_poison_stint(calibrated: dict[str, Any]) -> None:
    """A NULL/NaN feature in ONE baseline lap must not poison the baseline dim for the whole
    stint (NaN-safe per-dim mean): the other baseline laps cover it and scoring proceeds."""
    handle, cfg = calibrated["handle"], calibrated["config"]
    stint = _drift(nan_laps=frozenset({1}))  # lap 1 (a baseline lap) has a NaN cosine feature
    res = monitor_stint(stint, driver="VER", circuit="Bahrain", circuit_type="high_speed",
                        handle=handle, config=cfg)
    assert res.error_code is not ErrorCode.MONITOR_ERROR  # not a cascade halt
    assert res.n_laps_scored > 0  # valid distances still computed
    assert res.alert is not None


def test_safety_car_train_does_not_halt(calibrated: dict[str, Any]) -> None:
    handle, cfg = calibrated["handle"], calibrated["config"]
    stint = make_stint(n_laps=16, sc_laps=frozenset({9, 10, 11}))  # 3 SKIPs ≠ 3 NaN
    res = monitor_stint(stint, driver="VER", circuit="Bahrain", circuit_type="high_speed",
                        handle=handle, config=cfg)
    assert res.error_code is not ErrorCode.MONITOR_ERROR
    skipped = [lp for lp in res.laps if lp.state is StintMonitorState.SKIPPED]
    assert len(skipped) == 3
    assert all(ErrorCode.LAP_001 in lp.flags for lp in skipped)


# ==================================================================================
# Baseline selection (§7-1, §6 rows 2/3)
# ==================================================================================


def test_baseline_extension_lap5_scored_lap3_skipped(calibrated: dict[str, Any]) -> None:
    handle, cfg = calibrated["handle"], calibrated["config"]
    stint = make_stint(n_laps=16, sc_laps=frozenset({3}))  # lap 3 ineligible in the first 3
    res = monitor_stint(stint, driver="VER", circuit="Bahrain", circuit_type="high_speed",
                        handle=handle, config=cfg)
    assert res.baseline_lap_numbers == [1, 2, 4]  # extended to first 5, earliest 3 eligible
    assert res.baseline_extended is True
    by_lap = {lp.lap_number: lp for lp in res.laps}
    assert by_lap[3].state is StintMonitorState.SKIPPED and ErrorCode.LAP_001 in by_lap[3].flags
    assert by_lap[5].state is not StintMonitorState.SKIPPED  # lap 5 is scored, never dropped


def test_baseline_insufficient(calibrated: dict[str, Any], read_dq: Any) -> None:
    handle, cfg = calibrated["handle"], calibrated["config"]
    stint = make_stint(n_laps=12, sc_laps=frozenset({1, 2, 3, 4, 5}))  # no clean lap in first 5
    res = monitor_stint(stint, driver="VER", circuit="Bahrain", circuit_type="high_speed",
                        handle=handle, config=cfg)
    assert res.status is Status.FAILED
    assert res.error_code is ErrorCode.BASELINE_INSUFFICIENT
    codes = read_dq(cfg.paths.data_quality_log)
    assert any(r["error_code"] == "BASELINE_INSUFFICIENT" for r in codes)
    # §4.2-3 is unconditional: every excluded ineligible candidate is LAP_001-logged, even here.
    assert sum(1 for r in codes if r["error_code"] == "LAP_001") >= 5


def test_all_interpolated_baseline_caps_confidence(calibrated: dict[str, Any]) -> None:
    handle, cfg = calibrated["handle"], calibrated["config"]
    stint = _drift(drift_mag=10.0, interp_laps=frozenset({1, 2, 3}))  # huge drift → HIGH-magnitude
    res = monitor_stint(stint, driver="VER", circuit="Bahrain", circuit_type="high_speed",
                        handle=handle, config=cfg)
    assert res.degraded_quality is True
    assert ErrorCode.DEGRADED_QUALITY in res.flags
    assert res.alert is not None
    assert res.alert.alert_confidence is AlertConfidence.LOW  # capped despite huge deviation


# ==================================================================================
# Alert state machine unit tests (§7-8) — strict >, confirmation, SKIP freeze
# ==================================================================================


def _cores(scores: list[float | None], *, skip_at: set[int] = frozenset()) -> list[_LapCore]:
    out = []
    for i, s in enumerate(scores):
        state = StintMonitorState.SKIPPED if i in skip_at else (
            StintMonitorState.PENDING if s is None else StintMonitorState.NOMINAL
        )
        out.append(_LapCore(lap_number=i + 1, state=state, deviation_score=s))
    return out


def test_state_machine_strict_greater_than() -> None:
    laps = _cores([2.0, 2.0, 2.0])
    out = det._run_alert_machine(laps, tau=2.0, confirm_laps=2)
    assert out.alert_lap is None  # equality is NOT a crossing (strict >)
    assert all(lp.state is StintMonitorState.NOMINAL for lp in laps)


def test_state_machine_confirms_on_two_consecutive() -> None:
    laps = _cores([3.0, 3.0])
    out = det._run_alert_machine(laps, tau=2.0, confirm_laps=2)
    assert laps[0].state is StintMonitorState.WATCH
    assert laps[1].state is StintMonitorState.ALERT
    assert out.alert_lap == 2 and out.first_crossing_lap == 1


def test_state_machine_subthreshold_resets_run() -> None:
    laps = _cores([3.0, 1.0, 3.0])  # cross, drop (reset), cross → never confirms
    out = det._run_alert_machine(laps, tau=2.0, confirm_laps=2)
    assert out.alert_lap is None


def test_state_machine_skip_freezes_run() -> None:
    laps = _cores([3.0, None, 3.0], skip_at={1})  # cross, SKIP (freeze), cross → confirms
    out = det._run_alert_machine(laps, tau=2.0, confirm_laps=2)
    assert out.alert_lap == 3 and out.first_crossing_lap == 1


def test_single_final_crossing_watch_no_alert() -> None:
    # A crossing at the final scored lap with no room to confirm → WATCH, no alert object (§8).
    laps = _cores([1.0, 1.0, 3.0])
    out = det._run_alert_machine(laps, tau=2.0, confirm_laps=2)
    assert out.alert_lap is None
    assert laps[-1].state is StintMonitorState.WATCH


# ==================================================================================
# Confidence bands + pit window (§7-9, §4.2-11) — _build_alert unit tests
# ==================================================================================


def _alert_at(score: float, cell: _Cell, mon: Any, cosine: tuple[str, ...], *,
              degraded: bool = False, slope: float = 0.0, pit_valid: bool = True) -> Any:
    attr = np.zeros(len(cosine))
    attr[0] = score - slope  # so Σ attribution == score
    lap = _LapCore(lap_number=10, state=StintMonitorState.ALERT, deviation_score=score,
                   slope_component=slope, cosine_attribution=attr)
    core = _StintCore(outcome="ACTIVE", laps=[lap])
    out = _AlertOutcome(alert_lap=10, first_crossing_lap=9)
    tau = cell.mean + mon.alert_threshold_sd * cell.std
    return _build_alert(core, out, cell=cell, tau=tau, mon=mon, cosine_features=cosine,
                        degraded_quality=degraded, pit_window_valid=pit_valid)


def test_confidence_bands(calibrated: dict[str, Any]) -> None:
    mon = calibrated["config"].monitoring
    cosine = calibrated["handle"].cosine_features
    cell = _Cell(0.0, 1.0, 5, None)  # μ=0, σ=1 → LOW<2.5, MEDIUM∈[2.5,3), HIGH≥3
    assert _alert_at(2.2, cell, mon, cosine).alert_confidence is AlertConfidence.LOW
    assert _alert_at(2.7, cell, mon, cosine).alert_confidence is AlertConfidence.MEDIUM
    assert _alert_at(3.1, cell, mon, cosine).alert_confidence is AlertConfidence.HIGH
    assert _alert_at(3.1, cell, mon, cosine, degraded=True).alert_confidence is AlertConfidence.LOW


def test_build_alert_attribution_sums_with_slope(calibrated: dict[str, Any]) -> None:
    mon = calibrated["config"].monitoring
    cosine = calibrated["handle"].cosine_features
    alert = _alert_at(4.0, _Cell(0.0, 1.0, 5, None), mon, cosine, slope=1.2)
    assert alert.feature_attribution["F27"] == pytest.approx(0.6)  # half the slope component
    assert alert.feature_attribution["F28"] == pytest.approx(0.6)
    assert sum(alert.feature_attribution.values()) == pytest.approx(4.0, abs=1e-9)


def test_pit_window_recommendation(calibrated: dict[str, Any]) -> None:
    mon = calibrated["config"].monitoring
    cosine = calibrated["handle"].cosine_features
    cell = _Cell(0.0, 1.0, 5, 8.0)  # median crossing→pit gap = 8; laps_since = 10−9 = 1
    assert _alert_at(3.0, cell, mon, cosine).optimal_window_remaining_laps == 7  # max(0, 8−1)
    assert _alert_at(3.0, cell, mon, cosine, pit_valid=False).optimal_window_remaining_laps is None
    no_median = _Cell(0.0, 1.0, 5, None)
    assert _alert_at(3.0, no_median, mon, cosine).optimal_window_remaining_laps is None


# ==================================================================================
# Threshold resolution + fallback tiers (§7-6/12)
# ==================================================================================


def test_threshold_exact_and_source(calibrated: dict[str, Any]) -> None:
    handle, cfg = calibrated["handle"], calibrated["config"]
    h = _with(handle, entries={("VER", "high_speed"): _Cell(0.1, 0.1, 5, None)})
    res = monitor_stint(_drift(), driver="VER", circuit="Bahrain", circuit_type="high_speed",
                        handle=h, config=cfg)
    assert res.threshold_source == "driver_circuit_type"
    assert res.laps[-1].threshold == pytest.approx(0.3)  # 0.1 + 2.0·0.1
    assert ErrorCode.FALLBACK_THRESHOLD not in res.flags


def test_circuit_type_none_uses_driver_global(calibrated: dict[str, Any], read_dq: Any) -> None:
    handle, cfg = calibrated["handle"], calibrated["config"]
    h = _with(handle, driver_global={"VER": _Cell(0.2, 0.05, 4, None)})
    res = monitor_stint(_drift(circuit_type=None), driver="VER", circuit="Bahrain",
                        circuit_type=None, handle=h, config=cfg)
    assert res.threshold_source == "driver_global"
    assert ErrorCode.FALLBACK_THRESHOLD in res.flags
    assert res.status is Status.DEGRADED
    assert "FALLBACK_THRESHOLD" in _codes(read_dq, cfg)


def test_unknown_driver_uses_global(calibrated: dict[str, Any]) -> None:
    handle, cfg = calibrated["handle"], calibrated["config"]
    res = monitor_stint(_drift(driver="ZZZ"), driver="ZZZ", circuit="Bahrain",
                        circuit_type="high_speed", handle=handle, config=cfg)
    assert res.threshold_source == "global"
    assert ErrorCode.FALLBACK_THRESHOLD in res.flags


def test_thin_cell_falls_back(calibrated: dict[str, Any]) -> None:
    handle, cfg = calibrated["handle"], calibrated["config"]
    h = _with(handle, entries={("VER", "high_speed"): _Cell(0.1, 0.1, 2, None)},  # n_stints < 3
              driver_global={"VER": _Cell(0.2, 0.1, 9, None)})
    res = monitor_stint(_drift(), driver="VER", circuit="Bahrain", circuit_type="high_speed",
                        handle=h, config=cfg)
    assert res.threshold_source == "driver_global"
    assert ErrorCode.FALLBACK_THRESHOLD in res.flags


# ==================================================================================
# Slope component (§7-11)
# ==================================================================================


def test_no_tyre_age_cosine_only_one_lap_partial_zero_feature001(
    calibrated: dict[str, Any], read_dq: Any
) -> None:
    handle, cfg = calibrated["handle"], calibrated["config"]
    stint = _drift(tyre_age_known=False)
    res = monitor_stint(stint, driver="VER", circuit="Bahrain", circuit_type="high_speed",
                        handle=handle, config=cfg)
    assert all(not lp.slope_component_active for lp in res.laps)
    assert all(lp.slope_component == 0.0 for lp in res.laps)
    rows = read_dq(cfg.paths.data_quality_log)
    assert sum(1 for r in rows if r["error_code"] == "FEATURE_001") == 0  # F03b never invoked
    partial = [r for r in rows if r["error_code"] == "LAP_PARTIAL"]
    assert len(partial) == 1  # exactly one F10 slope-inactive row
    assert res.alert is not None  # cosine-only deviation still scored + alerted


def test_slope_stats_missing_serves_cosine_only(calibrated: dict[str, Any], read_dq: Any) -> None:
    handle, cfg = calibrated["handle"], calibrated["config"]
    h = _with(handle, slope_stats={})
    res = monitor_stint(_drift(), driver="VER", circuit="Bahrain", circuit_type="high_speed",
                        handle=h, config=cfg)
    assert all(not lp.slope_component_active for lp in res.laps)
    assert "LAP_PARTIAL" in _codes(read_dq, cfg)


def test_broadcast_f27_f28_columns_ignored(calibrated: dict[str, Any]) -> None:
    handle, cfg = calibrated["handle"], calibrated["config"]
    stint = _drift()
    res1 = monitor_stint(stint, driver="VER", circuit="Bahrain", circuit_type="high_speed",
                         handle=handle, config=cfg)
    mutated = stint.copy()
    mutated["F27"] = mutated["F27"] + 999.0  # the broadcast slope columns are never read
    mutated["F28"] = mutated["F28"] - 999.0
    res2 = monitor_stint(mutated, driver="VER", circuit="Bahrain", circuit_type="high_speed",
                         handle=handle, config=cfg)
    s1 = [lp.deviation_score for lp in res1.laps]
    s2 = [lp.deviation_score for lp in res2.laps]
    assert s1 == s2


def test_calibration_serving_core_parity(calibrated: dict[str, Any], tmp_path: Path) -> None:
    """The shared `_score_stint_core` (used by calibration Pass B AND serving) produces IDENTICAL
    composite scores for the same stint — train/serve parity by construction, incl. the live-window
    slopes (§7-11 parity test)."""
    handle, cfg = calibrated["handle"], calibrated["config"]
    stint = _drift()
    records = stint.to_dict("records")
    core = det._score_stint_core(
        records, circuit="Bahrain", circuit_type="high_speed",
        cosine_features=handle.cosine_features, feature_mean=handle.feature_mean,
        feature_std=handle.feature_std, slope_stats=handle.slope_stats,
        norm_params=handle.normalisation_params, config=cfg, min_std=handle.min_std,
        log=tmp_path / "dq.jsonl", entity_prefix="VER/Bahrain",
    )
    res = monitor_stint(stint, driver="VER", circuit="Bahrain", circuit_type="high_speed",
                        handle=handle, config=cfg)
    assert [lc.deviation_score for lc in core.laps] == [lp.deviation_score for lp in res.laps]
    assert [lc.slope_component for lc in core.laps] == [lp.slope_component for lp in res.laps]


def test_stint_lap_adapter_feeds_raw() -> None:
    stint = make_stint(n_laps=8)
    rec = stint.to_dict("records")[2]
    laps = det._stint_laps([rec])
    assert laps[0].throttle_mean == pytest.approx(float(rec["F01"]))
    assert laps[0].brake_peak_mean == pytest.approx(float(rec["F06"]))
    assert laps[0].tyre_age == int(round(float(rec["tyre_age"])))


# ==================================================================================
# Never-raises + input gate + normalisation fallback (§7-13)
# ==================================================================================


def test_missing_column_invalid_input(calibrated: dict[str, Any]) -> None:
    handle, cfg = calibrated["handle"], calibrated["config"]
    stint = make_stint(n_laps=10).drop(columns=["F05"])
    res = monitor_stint(stint, driver="VER", circuit="Bahrain", circuit_type="high_speed",
                        handle=handle, config=cfg)
    assert res.status is Status.FAILED and res.error_code is ErrorCode.INVALID_INPUT


def test_empty_frame_invalid_input(calibrated: dict[str, Any]) -> None:
    import pandas as pd

    handle, cfg = calibrated["handle"], calibrated["config"]
    res = monitor_stint(pd.DataFrame(), driver="VER", circuit="Bahrain",
                        circuit_type="high_speed", handle=handle, config=cfg)
    assert res.error_code is ErrorCode.INVALID_INPUT


def test_nonmonotonic_laps_invalid_input(calibrated: dict[str, Any]) -> None:
    handle, cfg = calibrated["handle"], calibrated["config"]
    stint = make_stint(n_laps=10)
    stint.loc[5, "lap_number"] = 2  # duplicate / out-of-order
    res = monitor_stint(stint, driver="VER", circuit="Bahrain", circuit_type="high_speed",
                        handle=handle, config=cfg)
    assert res.error_code is ErrorCode.INVALID_INPUT


def test_backstop_never_raises(calibrated: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    handle, cfg = calibrated["handle"], calibrated["config"]

    def _boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("injected internal fault")

    monkeypatch.setattr(det, "apply_normalisation", _boom)
    res = monitor_stint(_drift(), driver="VER", circuit="Bahrain", circuit_type="high_speed",
                        handle=handle, config=cfg)  # must not raise
    assert res.status is Status.FAILED
    assert res.error_code is ErrorCode.MONITOR_ERROR
    assert res.monitor_state is StintMonitorState.HALTED


def test_unwritable_dq_flag_no_raise(calibrated: dict[str, Any], tmp_path: Path) -> None:
    handle, cfg = calibrated["handle"], calibrated["config"]
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")  # a FILE where a dir parent is needed → OSError
    res = monitor_stint(make_stint(n_laps=7), driver="VER", circuit="Bahrain",
                        circuit_type="high_speed", handle=handle, config=cfg,
                        log_path=blocker / "dq.jsonl")
    assert ErrorCode.DATA_QUALITY_LOG_UNWRITABLE in res.flags


def test_unknown_circuit_normalisation_fallback(calibrated: dict[str, Any]) -> None:
    handle, cfg = calibrated["handle"], calibrated["config"]
    res = monitor_stint(_drift(circuit="Nowhere"), driver="VER", circuit="Nowhere",
                        circuit_type="high_speed", handle=handle, config=cfg)
    assert ErrorCode.NORMALISATION_FALLBACK in res.flags
    assert res.status is Status.DEGRADED


def test_none_handle_never_raises(calibrated: dict[str, Any]) -> None:
    # A None handle (e.g. F13 didn't gate on a failed load) must not escape the never-raises
    # contract via the pre-try shell construction (Finding 2).
    cfg = calibrated["config"]
    res = monitor_stint(_drift(), driver="VER", circuit="Bahrain", circuit_type="high_speed",
                        handle=None, config=cfg)  # type: ignore[arg-type]
    assert res.status is Status.FAILED
    assert res.error_code is ErrorCode.MONITOR_ERROR


# ==================================================================================
# Load + health (§7-17, §4.3)
# ==================================================================================


def test_load_missing_artefact_refuses(monitoring_config: Any, read_dq: Any) -> None:
    cfg = monitoring_config
    with pytest.raises(DetectorLoadError) as exc:
        load_detector(config=cfg, artefact_dir=Path(cfg.paths.artefacts_dir))
    assert exc.value.error_code is ErrorCode.MODEL_NOT_FOUND
    assert "MODEL_NOT_FOUND" in _codes(read_dq, cfg)


def test_load_corrupt_artefact_refuses(monitoring_config: Any) -> None:
    cfg = monitoring_config
    art = Path(cfg.paths.artefacts_dir)
    art.mkdir(parents=True, exist_ok=True)
    (art / "stint_variation.json").write_text("{ not json", encoding="utf-8")
    with pytest.raises(DetectorLoadError):
        load_detector(config=cfg, artefact_dir=art)


def test_load_hash_tampered_refuses(calibrated: dict[str, Any], tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    art = Path(cfg.paths.artefacts_dir)
    art.mkdir(parents=True, exist_ok=True)
    src = calibrated["art_dir"]
    (art / "normalisation.json").write_text(
        (src / "normalisation.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    data = json.loads((src / "stint_variation.json").read_text(encoding="utf-8"))
    data["content_hash"] = "deadbeef"  # tamper
    (art / "stint_variation.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(DetectorLoadError):
        load_detector(config=cfg, artefact_dir=art)


def _write_tampered_artefact(art: Path, base_json: dict[str, Any], mutate: Any) -> None:
    """Apply `mutate` to a copy of the artefact dict, recompute the content_hash so the tamper
    survives the hash check, and write it — isolating the specific downstream refusal under test."""
    data = json.loads(json.dumps(base_json))
    mutate(data)
    art_model = det._StintVariationArtefact(**{**data, "content_hash": ""})
    data["content_hash"] = det._hash_artefact(art_model)
    art.mkdir(parents=True, exist_ok=True)
    (art / "stint_variation.json").write_text(json.dumps(data), encoding="utf-8")


def test_load_empty_threshold_table_refuses(calibrated: dict[str, Any], tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    art = Path(cfg.paths.artefacts_dir)
    base = json.loads((calibrated["art_dir"] / "stint_variation.json").read_text(encoding="utf-8"))

    def _empty(d: dict[str, Any]) -> None:
        d["entries"] = []
        d["driver_global"] = []
        d["global_"]["n_stints"] = 0

    _write_tampered_artefact(art, base, _empty)
    with pytest.raises(DetectorLoadError):
        load_detector(config=cfg, artefact_dir=art)


def test_load_wrong_width_scaler_refuses(calibrated: dict[str, Any], tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    art = Path(cfg.paths.artefacts_dir)
    base = json.loads((calibrated["art_dir"] / "stint_variation.json").read_text(encoding="utf-8"))

    def _narrow(d: dict[str, Any]) -> None:
        dropped = d["cosine_features"][0]
        d["cosine_features"] = d["cosine_features"][1:]  # 25 dims, config expects 26
        d["feature_scaler"].pop(dropped, None)

    _write_tampered_artefact(art, base, _narrow)
    with pytest.raises(DetectorLoadError):
        load_detector(config=cfg, artefact_dir=art)


def test_load_schema_mismatch_refuses(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    art = Path(cfg.paths.artefacts_dir)
    art.mkdir(parents=True, exist_ok=True)
    # Well-formed JSON object, wrong shape (missing required `config`) → Pydantic ValidationError
    # branch, distinct from the JSONDecodeError branch tested by test_load_corrupt_artefact_refuses.
    (art / "stint_variation.json").write_text(
        json.dumps({"schema_version": "1.0", "cosine_features": []}), encoding="utf-8"
    )
    with pytest.raises(DetectorLoadError):
        load_detector(config=cfg, artefact_dir=art)


def test_load_empty_slope_stats_serves_cosine_only(
    calibrated: dict[str, Any], tmp_path: Path, read_dq: Any
) -> None:
    cfg = _make_config(tmp_path)
    art = Path(cfg.paths.artefacts_dir)
    (art / "normalisation.json").parent.mkdir(parents=True, exist_ok=True)
    (art / "normalisation.json").write_text(
        (calibrated["art_dir"] / "normalisation.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    base = json.loads((calibrated["art_dir"] / "stint_variation.json").read_text(encoding="utf-8"))
    _write_tampered_artefact(art, base, lambda d: d.__setitem__("slope_stats", {}))
    handle = load_detector(config=cfg, artefact_dir=art)  # loads (degrade, don't die)
    assert not handle.slope_stats
    assert "LAP_PARTIAL" in _codes(read_dq, cfg)


def test_structural_knob_mismatch_refuses(calibrated: dict[str, Any]) -> None:
    cfg = _make_config(Path(calibrated["art_dir"]).parent)  # reuse same artefacts dir
    cfg.paths.artefacts_dir = str(calibrated["art_dir"])
    assert cfg.monitoring is not None
    cfg.monitoring.rolling_window = 4  # structural knob differs from calibration (was 3)
    with pytest.raises(DetectorLoadError):
        load_detector(config=cfg, artefact_dir=calibrated["art_dir"])


def test_crossing_knob_mismatch_serves_pit_invalid(
    calibrated: dict[str, Any], read_dq: Any
) -> None:
    cfg = _make_config(Path(calibrated["art_dir"]).parent)
    cfg.paths.artefacts_dir = str(calibrated["art_dir"])
    assert cfg.monitoring is not None
    cfg.monitoring.alert_threshold_sd = 1.5  # crossing knob differs (was 2.0)
    handle = load_detector(config=cfg, artefact_dir=calibrated["art_dir"])
    assert handle.pit_window_valid is False


def test_health(calibrated: dict[str, Any]) -> None:
    assert health(None).error_code is ErrorCode.HEALTH_FAIL
    assert health(None).reason is ErrorCode.MODEL_NOT_FOUND
    assert health(None).ready is False
    ok = health(calibrated["handle"])
    assert ok.ready is True and ok.status is Status.OK and ok.model_version


# ==================================================================================
# MLflow degrade-don't-die (§7-16)
# ==================================================================================


def test_mlflow_fault_degrades_not_aborts(
    monitoring_config: Any, read_dq: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = monitoring_config
    art = Path(cfg.paths.artefacts_dir)
    matrix = build_training_matrix()
    fit_normaliser(matrix, config=cfg)

    def _boom(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("mlflow down")

    monkeypatch.setattr(det.mlflow, "set_tracking_uri", _boom)
    fit = calibrate_stint_variation(matrix, config=cfg, artefact_dir=art)
    assert fit.status is Status.DEGRADED
    assert fit.mlflow_run_id is None
    assert (art / "stint_variation.json").is_file()  # artefact still written
    codes = _codes(read_dq, cfg)
    assert "MODEL_TRAINING_FAILED" in codes


# ==================================================================================
# Static no-fit + no-magic-literal scans (§7-15)
# ==================================================================================


_SOURCE = Path(det.__file__).read_text(encoding="utf-8")
_SERVING = (
    det.monitor_stint, det.load_detector, det.health, det.lead_time_report,
    det._score_stint_core, det._standardise_lap, det._run_alert_machine, det._resolve_cell,
    det._build_alert, det._cosine_distance, det._evaluate_stint, det._alert_lap_for,
    det._sensitivity_sweep,
)


def test_no_fit_in_serving_bodies() -> None:
    for fn in _SERVING:
        src = inspect.getsource(fn)
        assert ".fit(" not in src
        assert ".fit_transform(" not in src


def test_no_normalisation_refit_anywhere() -> None:
    assert "fit_normaliser" not in _SOURCE  # F10 reuses F04 params, never refits normalisation


def test_no_magic_threshold_literals() -> None:
    body = "\n".join(
        ln for ln in _SOURCE.splitlines()
        if "_MIN_STD" not in ln and not ln.strip().startswith("#")
    )
    for literal in ("0.25", "2.5", "3.0"):  # slope_weight / confidence bands come from config
        assert literal not in body, f"threshold {literal!r} hardcoded"
    for key in (
        "baseline_laps", "rolling_window", "alert_threshold_sd", "min_stint_laps",
        "min_calibration_stints", "degradation_features",
    ):
        assert key in _SOURCE, f"config key {key!r} must drive the module"


# ==================================================================================
# Evaluation — lead_time_report (§7-18) + Week-11 deliverable (§7-19)
# ==================================================================================


def test_lead_time_report_full(calibrated: dict[str, Any]) -> None:
    handle, cfg = calibrated["handle"], calibrated["config"]
    report = lead_time_report(build_eval_matrix(), handle=handle, config=cfg)
    assert report.status is Status.OK
    assert report.n_stints > 0
    assert report.per_stint
    assert report.tp_rate is not None and report.fp_rate is not None
    assert "UNKNOWN" not in report.per_compound  # excluded from the stratified view only
    assert set(report.per_compound) <= {"Soft", "Medium", "Hard", "Intermediate"}
    assert set(report.sensitivity) == {"1.0", "1.5", "2.0", "2.5"}  # the §11.3 grid
    assert report.optimal_threshold_per_circuit_type


def test_lead_time_report_unknown_present_in_totals(calibrated: dict[str, Any]) -> None:
    handle, cfg = calibrated["handle"], calibrated["config"]
    report = lead_time_report(build_eval_matrix(), handle=handle, config=cfg)
    assert any(s.compound == "UNKNOWN" for s in report.per_stint)  # present in totals


def test_lead_time_report_malformed_invalid_input(calibrated: dict[str, Any]) -> None:
    import pandas as pd

    handle, cfg = calibrated["handle"], calibrated["config"]
    report = lead_time_report(pd.DataFrame(), handle=handle, config=cfg)
    assert report.status is Status.FAILED and report.error_code is ErrorCode.INVALID_INPUT


def test_week11_report_logged(monitoring_config: Any) -> None:
    cfg = monitoring_config
    art = Path(cfg.paths.artefacts_dir)
    train = build_training_matrix()
    val = build_eval_matrix()  # split=="val", carries compound
    matrix = __import__("pandas").concat([train, val], ignore_index=True)
    fit_normaliser(matrix, config=cfg)
    calibrate_stint_variation(matrix, config=cfg, artefact_dir=art)
    assert (art / "week11_lead_time_report.json").is_file()


# ==================================================================================
# F08 fingerprint embed + seam parity (§8) — needs a real StyleSpaceHandle
# ==================================================================================


@pytest.fixture(scope="module")
def style_handle(tmp_path_factory: pytest.TempPathFactory) -> Any:
    from app.ml.stylespace import build_style_space, load_style_space

    from tests.ml.conftest import build_matrix

    tmp = tmp_path_factory.mktemp("f10_ss")
    cfg = _make_config(tmp)
    art = Path(cfg.paths.artefacts_dir)
    drivers = ["VER", "HAM", "LEC", "NOR", "PIA", "RUS", "ALO", "SAI"]  # ≥ n5 + 1
    matrix = build_matrix(drivers=drivers, train_circuits=["Bahrain", "Spain", "Japan"])
    fit_normaliser(matrix, config=cfg)
    build_style_space(matrix, config=cfg, artefact_dir=art)
    return load_style_space(config=cfg, artefact_dir=art)


def test_fingerprint_available_and_sparse_by_design(
    calibrated: dict[str, Any], style_handle: Any
) -> None:
    handle, cfg = calibrated["handle"], calibrated["config"]
    res = monitor_stint(_drift(n_laps=10, drift_from=8), driver="VER", circuit="Bahrain",
                        circuit_type="high_speed", handle=handle, config=cfg,
                        style_handle=style_handle)
    assert res.fingerprint_snapshot is not None
    assert res.fingerprint_available is True  # coords exist
    snap = res.fingerprint_snapshot
    assert snap.sparse or ErrorCode.SPARSE_PROJECTION in snap.flags  # < 30 laps → sparse BY DESIGN


def test_inactive_still_attempts_fingerprint(
    calibrated: dict[str, Any], style_handle: Any
) -> None:
    handle, cfg = calibrated["handle"], calibrated["config"]
    res = monitor_stint(make_stint(n_laps=7), driver="VER", circuit="Bahrain",
                        circuit_type="high_speed", handle=handle, config=cfg,
                        style_handle=style_handle)
    assert res.error_code is ErrorCode.MONITOR_INACTIVE
    assert res.fingerprint_snapshot is not None  # "Fingerprint displayed but no score computed"


def test_fingerprint_coords_none_not_available(
    calibrated: dict[str, Any], style_handle: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.ml.stylespace import ProjectionResult

    handle, cfg = calibrated["handle"], calibrated["config"]

    def _degraded(*_a: Any, **_k: Any) -> ProjectionResult:
        return ProjectionResult(status=Status.DEGRADED, coords_5d=None)

    monkeypatch.setattr(det, "project_unknown", _degraded)
    res = monitor_stint(_drift(n_laps=10, drift_from=8), driver="VER", circuit="Bahrain",
                        circuit_type="high_speed", handle=handle, config=cfg,
                        style_handle=style_handle)
    assert res.fingerprint_snapshot is not None
    assert res.fingerprint_available is False


def test_fingerprint_exception_never_blocks_scoring(
    calibrated: dict[str, Any], style_handle: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    handle, cfg = calibrated["handle"], calibrated["config"]

    def _boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("projection blew up")

    monkeypatch.setattr(det, "project_unknown", _boom)
    res = monitor_stint(_drift(), driver="VER", circuit="Bahrain", circuit_type="high_speed",
                        handle=handle, config=cfg, style_handle=style_handle)
    assert res.fingerprint_available is False
    assert res.alert is not None  # scoring unaffected


def test_fingerprint_seam_parity_raw_prefix(
    calibrated: dict[str, Any], style_handle: Any
) -> None:
    from app.ml.stylespace import project_unknown

    handle, cfg = calibrated["handle"], calibrated["config"]
    stint = _drift(n_laps=10, drift_from=8)
    res = monitor_stint(stint, driver="VER", circuit="Bahrain", circuit_type="high_speed",
                        handle=handle, config=cfg, style_handle=style_handle)
    # The embed must equal a direct project_unknown on the SAME RAW prefix (F08 normalises
    # internally — handing it the standardised frame would double-normalise at the seam).
    direct = project_unknown(stint.reindex(columns=list(det.FEATURE_IDS)), circuit="Bahrain",
                             handle=style_handle, config=cfg)
    assert res.fingerprint_snapshot is not None
    assert res.fingerprint_snapshot.coords_5d == direct.coords_5d
    assert res.fingerprint_snapshot.flags == direct.flags
