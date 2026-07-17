"""F11 personalisation-engine tests (§7 acceptance criteria + §6 failure rows).

Every §6 failure row has a test that triggers it and asserts the typed result AND code placement;
``recommend_setup`` / ``onboard_unknown`` never-raise is proven by fault injection; leakage +
reproducibility + the honest reliability finding are proven behaviourally.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import joblib
import pytest
from app.contracts.enums import Status
from app.contracts.errors import ErrorCode
from app.personalisation import engine as eng
from app.personalisation.engine import (
    _PARAM_NAMES,
    APPROX_DISCLAIMER,
    PersonalisationLoadError,
    PersonalisationTrainingError,
    SetupConfidence,
    health,
    load_personalisation,
    onboard_unknown,
    recommend_setup,
    train_personalisation,
)
from app.pipeline.normalise import fit_normaliser

from tests.personalisation.conftest import (
    DRIVERS,
    _make_config,
    build_matrix,
    full_predictions,
    make_stub_handle,
)


def _codes(read_dq: Any, cfg: Any) -> set[str]:
    return {r["error_code"] for r in read_dq(cfg.paths.data_quality_log)}


def _param(rec: Any, name: str) -> Any:
    return next(p for p in rec.parameters if p.name == name)


# ==================================================================================
# Training (§7 training criteria, §4.4)
# ==================================================================================


def test_fit_status_and_shape(trained: dict[str, Any]) -> None:
    fit = trained["fit"]
    assert fit.status is Status.OK
    assert fit.n_train_cells == 48  # 8 drivers × 6 non-held-out circuits
    assert fit.parameters_fitted == 10
    assert set(fit.reliability) == set(_PARAM_NAMES)
    assert fit.mlflow_run_id is not None
    assert (trained["art_dir"] / "personalisation.pkl").is_file()


def test_reliability_finding_counts_reliable(trained: dict[str, Any]) -> None:
    """The §14 finding: n_reliable = the count of parameters with reliability ≥ confidence_medium
    (reported honestly, never used to silently drop parameters — all 10 are always returned)."""
    fit, cfg = trained["fit"], trained["config"]
    medium = cfg.personalisation.confidence_medium
    expected = sum(1 for r in fit.reliability.values() if r >= medium)
    assert fit.n_reliable == expected


def test_training_reproducible_metric_equal(trained: dict[str, Any]) -> None:
    """Two training runs on the SAME style handle + matrix produce equal per-parameter reliability
    (metric-equal, not byte-identical pkl — XGBoost bytes aren't guaranteed identical)."""
    sh, cfg, art_dir, matrix = (trained[k] for k in ("style_handle", "config", "art_dir", "matrix"))
    a = train_personalisation(matrix, style_handle=sh, config=cfg, artefact_dir=art_dir)
    b = train_personalisation(matrix, style_handle=sh, config=cfg, artefact_dir=art_dir)
    for name in _PARAM_NAMES:
        assert a.reliability[name] == pytest.approx(b.reliability[name], rel=1e-9, abs=1e-9)


def test_assert_no_leakage_raises_on_held_out(personalisation_config: Any) -> None:
    frame = build_matrix()
    frame.loc[0, "circuit"] = "Belgium"  # a held-out circuit survives into an unfiltered frame
    with pytest.raises(PersonalisationTrainingError) as exc:
        eng._assert_no_leakage(frame, personalisation_config)
    assert exc.value.error_code is ErrorCode.TRAIN_LEAKAGE


def test_train_leakage_logs_and_writes_no_artefact(
    personalisation_config: Any, read_dq: Any, monkeypatch: pytest.MonkeyPatch,
    trained: dict[str, Any],
) -> None:
    cfg = personalisation_config
    art = Path(cfg.paths.artefacts_dir)

    def _boom(_t: Any, _c: Any) -> None:
        raise PersonalisationTrainingError("leak", error_code=ErrorCode.TRAIN_LEAKAGE)

    monkeypatch.setattr(eng, "_assert_no_leakage", _boom)
    with pytest.raises(PersonalisationTrainingError) as exc:
        train_personalisation(build_matrix(), style_handle=trained["style_handle"], config=cfg,
                              artefact_dir=art)
    assert exc.value.error_code is ErrorCode.TRAIN_LEAKAGE
    assert not (art / "personalisation.pkl").is_file()
    assert "TRAIN_LEAKAGE" in _codes(read_dq, cfg)


def test_train_quarantines_held_out_train_row(trained: dict[str, Any], tmp_path: Path) -> None:
    """A held-out circuit mislabeled split=="train" is quarantined by the mask (not counted), and
    calibration succeeds without TRAIN_LEAKAGE."""
    cfg = _make_config(tmp_path)
    art = Path(cfg.paths.artefacts_dir)
    matrix = build_matrix()
    matrix.loc[matrix["circuit"] == "Belgium", "split"] = "train"  # held-out, mislabeled train
    fit_normaliser(matrix, config=cfg)
    fit = train_personalisation(matrix, style_handle=trained["style_handle"], config=cfg,
                                artefact_dir=art)
    assert fit.status in (Status.OK, Status.DEGRADED)
    assert fit.n_train_cells == 48  # Belgium quarantined, not counted


def test_train_empty_post_quarantine_fails(trained: dict[str, Any], tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    matrix = build_matrix()
    matrix["split"] = "val"
    fit_normaliser(build_matrix(), config=cfg)
    with pytest.raises(PersonalisationTrainingError) as exc:
        train_personalisation(matrix, style_handle=trained["style_handle"], config=cfg,
                              artefact_dir=Path(cfg.paths.artefacts_dir))
    assert exc.value.error_code is ErrorCode.MODEL_TRAINING_FAILED


def test_train_missing_columns_fails(trained: dict[str, Any], tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    matrix = build_matrix().drop(columns=["F05"])  # a required feature column is absent
    with pytest.raises(PersonalisationTrainingError) as exc:
        train_personalisation(matrix, style_handle=trained["style_handle"], config=cfg,
                              artefact_dir=Path(cfg.paths.artefacts_dir))
    assert exc.value.error_code is ErrorCode.MODEL_TRAINING_FAILED


def test_train_write_fault_typed(trained: dict[str, Any], tmp_path: Path, read_dq: Any) -> None:
    """A filesystem write fault is a typed MODEL_TRAINING_FAILED, never a raw OSError (§6 row)."""
    cfg = _make_config(tmp_path)
    art = Path(cfg.paths.artefacts_dir)
    matrix = build_matrix()
    fit_normaliser(matrix, config=cfg)
    art.mkdir(parents=True, exist_ok=True)
    (art / "personalisation.pkl").mkdir()  # a DIRECTORY where the pkl must be written → OSError
    with pytest.raises(PersonalisationTrainingError) as exc:
        train_personalisation(matrix, style_handle=trained["style_handle"], config=cfg,
                              artefact_dir=art)
    assert exc.value.error_code is ErrorCode.MODEL_TRAINING_FAILED
    assert "MODEL_TRAINING_FAILED" in _codes(read_dq, cfg)


def test_train_thin_cells_dropped(trained: dict[str, Any], tmp_path: Path, read_dq: Any) -> None:
    cfg = _make_config(tmp_path)
    art = Path(cfg.paths.artefacts_dir)
    matrix = build_matrix(laps_per_cell=5)  # every cell < min_clean_laps (30)
    fit_normaliser(matrix, config=cfg)
    with pytest.raises(PersonalisationTrainingError) as exc:
        train_personalisation(matrix, style_handle=trained["style_handle"], config=cfg,
                              artefact_dir=art)
    assert exc.value.error_code is ErrorCode.MODEL_TRAINING_FAILED
    assert "DATA_INSUFFICIENT" in _codes(read_dq, cfg)


def test_train_mlflow_fault_degrades(
    trained: dict[str, Any], tmp_path: Path, read_dq: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _make_config(tmp_path)
    art = Path(cfg.paths.artefacts_dir)
    matrix = build_matrix()
    fit_normaliser(matrix, config=cfg)

    def _boom(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("mlflow down")

    monkeypatch.setattr(eng.mlflow, "set_tracking_uri", _boom)
    fit = train_personalisation(matrix, style_handle=trained["style_handle"], config=cfg,
                                artefact_dir=art)
    assert fit.status is Status.DEGRADED
    assert fit.mlflow_run_id is None
    assert (art / "personalisation.pkl").is_file()
    assert "MODEL_TRAINING_FAILED" in _codes(read_dq, cfg)


# ==================================================================================
# Serving happy path (§7 layer criteria)
# ==================================================================================


def test_ten_parameters_canonical_order(trained: dict[str, Any]) -> None:
    handle, sh, cfg = trained["handle"], trained["style_handle"], trained["config"]
    res = recommend_setup(driver=DRIVERS[0], circuit="Bahrain", circuit_type="high_speed",
                          handle=handle, style_handle=sh, config=cfg)
    assert [p.name for p in res.parameters] == list(_PARAM_NAMES)  # exactly 10, canonical order
    assert res.approximation_disclaimer == APPROX_DISCLAIMER
    for p in res.parameters:
        assert isinstance(p.confidence, float)  # never NULL (§8.4)
        assert p.rating in set(SetupConfidence)


def test_shap_present_on_happy_path(trained: dict[str, Any]) -> None:
    handle, sh, cfg = trained["handle"], trained["style_handle"], trained["config"]
    res = recommend_setup(driver=DRIVERS[0], circuit="Bahrain", circuit_type="high_speed",
                          handle=handle, style_handle=sh, config=cfg)
    for p in res.parameters:
        assert p.shap_available is True
        assert len(p.shap_top_features) == cfg.personalisation.shap_top_k
        assert all(
            lbl.startswith("fp") or lbl.startswith("circuit:") for lbl in p.shap_top_features
        )


def test_categorical_suspension_is_a_category(trained: dict[str, Any]) -> None:
    handle, sh, cfg = trained["handle"], trained["style_handle"], trained["config"]
    res = recommend_setup(driver=DRIVERS[0], circuit="Bahrain", circuit_type="high_speed",
                          handle=handle, style_handle=sh, config=cfg)
    susp = _param(res, "suspension_stiffness_front")
    if susp.rating is not SetupConfidence.WIDE:
        assert susp.category in ("Soft", "Medium", "Stiff")


# ==================================================================================
# Layer 2 constraints (§7.2) — exact assertions via stub regressors
# ==================================================================================


def test_fia_clip(trained: dict[str, Any], read_dq: Any) -> None:
    cfg, sh = trained["config"], trained["style_handle"]
    handle = make_stub_handle(cfg, predictions=full_predictions(front_wing_angle=99.0))
    res = recommend_setup(driver=DRIVERS[0], circuit="Bahrain", circuit_type="high_speed",
                          handle=handle, style_handle=sh, config=cfg)
    fw = _param(res, "front_wing_angle")
    assert fw.value == 12.0  # clipped to fia_max
    assert ErrorCode.CONSTRAINT_CLIPPED in fw.constraint_flags
    assert res.status is Status.DEGRADED
    assert "CONSTRAINT_CLIPPED" in _codes(read_dq, cfg)


def test_monaco_override(trained: dict[str, Any], read_dq: Any) -> None:
    cfg, sh = trained["config"], trained["style_handle"]
    handle = make_stub_handle(cfg, predictions=full_predictions(front_wing_angle=1.0,
                                                                rear_wing_angle=1.0))
    res = recommend_setup(driver=DRIVERS[0], circuit="Monaco", circuit_type="street",
                          handle=handle, style_handle=sh, config=cfg)
    assert res.monaco_override is True
    assert _param(res, "front_wing_angle").value == 12.0  # fia_max
    assert _param(res, "rear_wing_angle").value == 15.0
    assert ErrorCode.CIRCUIT_OVERRIDE_MONACO in _param(res, "front_wing_angle").constraint_flags
    assert "CIRCUIT_OVERRIDE_MONACO" in _codes(read_dq, cfg)
    fw = _param(res, "front_wing_angle")  # a hard rule → HIGH rating AND confidence 1.0
    assert fw.rating is SetupConfidence.HIGH and fw.confidence == 1.0
    assert fw.value_range is None  # a point value, never a WIDE range


def test_consistency_warning_no_value_change(trained: dict[str, Any], read_dq: Any) -> None:
    cfg, sh = trained["config"], trained["style_handle"]
    # front wing > 10 AND brake bias < 54 → CONSISTENCY_WARNING (flag only, values unchanged).
    handle = make_stub_handle(cfg, predictions=full_predictions(front_wing_angle=11.0,
                                                                brake_bias=53.0))
    res = recommend_setup(driver=DRIVERS[0], circuit="Bahrain", circuit_type="high_speed",
                          handle=handle, style_handle=sh, config=cfg)
    assert res.consistency_warning is True
    assert ErrorCode.CONSISTENCY_WARNING in res.flags
    assert _param(res, "front_wing_angle").value == 11.0  # unchanged
    assert _param(res, "brake_bias").value == 53.0
    assert "CONSISTENCY_WARNING" in _codes(read_dq, cfg)


def test_wide_confidence_range(trained: dict[str, Any], read_dq: Any) -> None:
    cfg, sh = trained["config"], trained["style_handle"]
    # brake_duct_aperture [30, 100]: ±15% of 60 = (51, 69) stays within the band → exact range.
    handle = make_stub_handle(cfg, predictions=full_predictions(brake_duct_aperture=60.0),
                              reliability={"brake_duct_aperture": 0.3})  # < confidence_floor 0.5
    res = recommend_setup(driver=DRIVERS[0], circuit="Bahrain", circuit_type="high_speed",
                          handle=handle, style_handle=sh, config=cfg)
    bd = _param(res, "brake_duct_aperture")
    assert bd.rating is SetupConfidence.WIDE
    assert bd.value is None
    assert bd.value_range == pytest.approx((60.0 * 0.85, 60.0 * 1.15))  # ±15%
    assert ErrorCode.WIDE_CONFIDENCE in bd.constraint_flags
    assert "WIDE_CONFIDENCE" in _codes(read_dq, cfg)


def test_wide_confidence_range_clipped_to_fia(trained: dict[str, Any]) -> None:
    """A WIDE range that would exceed the FIA band is clipped to it (§4.2-4c)."""
    cfg, sh = trained["config"], trained["style_handle"]
    handle = make_stub_handle(cfg, predictions=full_predictions(brake_bias=57.0),
                              reliability={"brake_bias": 0.3})  # ±15% = (48.4, 65.6) ⊄ [52, 62]
    res = recommend_setup(driver=DRIVERS[0], circuit="Bahrain", circuit_type="high_speed",
                          handle=handle, style_handle=sh, config=cfg)
    bb = _param(res, "brake_bias")
    assert bb.value_range == (52.0, 62.0)  # clipped to the FIA bounds


def test_confidence_bands(trained: dict[str, Any]) -> None:
    cfg, sh = trained["config"], trained["style_handle"]
    handle = make_stub_handle(
        cfg, predictions=full_predictions(),
        reliability={"front_wing_angle": 0.85, "rear_wing_angle": 0.70, "brake_bias": 0.55,
                     "differential_ramp_entry": 0.30},
    )
    res = recommend_setup(driver=DRIVERS[0], circuit="Bahrain", circuit_type="high_speed",
                          handle=handle, style_handle=sh, config=cfg)
    assert _param(res, "front_wing_angle").rating is SetupConfidence.HIGH
    assert _param(res, "rear_wing_angle").rating is SetupConfidence.MEDIUM
    assert _param(res, "brake_bias").rating is SetupConfidence.LOW
    assert _param(res, "differential_ramp_entry").rating is SetupConfidence.WIDE


def test_confidence_never_null_even_wide(trained: dict[str, Any]) -> None:
    cfg, sh = trained["config"], trained["style_handle"]
    handle = make_stub_handle(cfg, predictions=full_predictions(),
                              reliability={n: 0.1 for n in _PARAM_NAMES})  # all WIDE
    res = recommend_setup(driver=DRIVERS[0], circuit="Bahrain", circuit_type="high_speed",
                          handle=handle, style_handle=sh, config=cfg)
    assert all(isinstance(p.confidence, float) for p in res.parameters)
    assert all(p.rating is SetupConfidence.WIDE for p in res.parameters)


# ==================================================================================
# Never-raises + input gates (§7 never-raises criterion)
# ==================================================================================


def test_unknown_driver_invalid_input(trained: dict[str, Any]) -> None:
    handle, sh, cfg = trained["handle"], trained["style_handle"], trained["config"]
    res = recommend_setup(driver="ZZZ", circuit="Bahrain", circuit_type="high_speed",
                          handle=handle, style_handle=sh, config=cfg)
    assert res.status is Status.FAILED and res.error_code is ErrorCode.INVALID_INPUT


def test_unknown_circuit_type_fallback(trained: dict[str, Any], read_dq: Any) -> None:
    handle, sh, cfg = trained["handle"], trained["style_handle"], trained["config"]
    res = recommend_setup(driver=DRIVERS[0], circuit="Nowhere", circuit_type="oval",
                          handle=handle, style_handle=sh, config=cfg)
    assert ErrorCode.FALLBACK_THRESHOLD in res.flags
    assert res.status is Status.DEGRADED
    assert "FALLBACK_THRESHOLD" in _codes(read_dq, cfg)


def test_recommend_backstop_never_raises(trained: dict[str, Any]) -> None:
    cfg, sh = trained["config"], trained["style_handle"]
    handle = make_stub_handle(cfg, predictions=full_predictions(), boom=True)  # predict raises
    res = recommend_setup(driver=DRIVERS[0], circuit="Bahrain", circuit_type="high_speed",
                          handle=handle, style_handle=sh, config=cfg)  # must not raise
    assert res.status is Status.FAILED and res.error_code is ErrorCode.INFERENCE_ERROR
    assert len(res.parameters) == 10
    assert all(isinstance(p.confidence, float) for p in res.parameters)  # never NULL


def test_none_handle_never_raises(trained: dict[str, Any]) -> None:
    cfg, sh = trained["config"], trained["style_handle"]
    res = recommend_setup(driver=DRIVERS[0], circuit="Bahrain", circuit_type="high_speed",
                          handle=None, style_handle=sh, config=cfg)  # type: ignore[arg-type]
    assert res.status is Status.FAILED and res.error_code is ErrorCode.INFERENCE_ERROR
    assert len(res.parameters) == 10  # always 10, sourced from config bounds


def test_none_config_never_raises(trained: dict[str, Any]) -> None:
    """A None/malformed config must not escape the never-raises contract before the try."""
    handle, sh = trained["handle"], trained["style_handle"]
    res = recommend_setup(driver=DRIVERS[0], circuit="Bahrain", circuit_type="high_speed",
                          handle=handle, style_handle=sh, config=None)  # type: ignore[arg-type]
    assert res.status is Status.FAILED and res.error_code is ErrorCode.INFERENCE_ERROR
    assert len(res.parameters) == 10  # bounds sourced from the handle even with no config
    assert all(isinstance(p.confidence, float) for p in res.parameters)


def test_malformed_input_invalid(trained: dict[str, Any]) -> None:
    handle, sh, cfg = trained["handle"], trained["style_handle"], trained["config"]
    res = recommend_setup(driver="", circuit="Bahrain", circuit_type="high_speed",
                          handle=handle, style_handle=sh, config=cfg)
    assert res.status is Status.FAILED and res.error_code is ErrorCode.INVALID_INPUT


def test_circuit_type_none_flags_fallback(trained: dict[str, Any], read_dq: Any) -> None:
    handle, sh, cfg = trained["handle"], trained["style_handle"], trained["config"]
    res = recommend_setup(driver=DRIVERS[0], circuit="Bahrain", circuit_type=None,
                          handle=handle, style_handle=sh, config=cfg)
    assert ErrorCode.FALLBACK_THRESHOLD in res.flags  # None circuit type → generic mapping, flagged
    assert res.status is Status.DEGRADED
    assert "FALLBACK_THRESHOLD" in _codes(read_dq, cfg)


def test_shap_fault_degrades_not_blocks(trained: dict[str, Any], read_dq: Any) -> None:
    cfg, sh = trained["config"], trained["style_handle"]
    handle = make_stub_handle(cfg, predictions=full_predictions(), boom_shap=True)
    res = recommend_setup(driver=DRIVERS[0], circuit="Bahrain", circuit_type="high_speed",
                          handle=handle, style_handle=sh, config=cfg)
    assert len(res.parameters) == 10  # recommendation still returned
    assert all(p.shap_available is False for p in res.parameters)
    assert "SHAP_UNAVAILABLE" in _codes(read_dq, cfg)


def test_unwritable_dq_flag_no_raise(trained: dict[str, Any], tmp_path: Path) -> None:
    cfg, sh = trained["config"], trained["style_handle"]
    handle = make_stub_handle(cfg, predictions=full_predictions(front_wing_angle=99.0))
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    res = recommend_setup(driver=DRIVERS[0], circuit="Bahrain", circuit_type="high_speed",
                          handle=handle, style_handle=sh, config=cfg, log_path=blocker / "dq.jsonl")
    assert ErrorCode.DATA_QUALITY_LOG_UNWRITABLE in res.flags


# ==================================================================================
# §7.4 onboarding
# ==================================================================================


def test_onboard_real(trained: dict[str, Any]) -> None:
    handle, sh, cfg = trained["handle"], trained["style_handle"], trained["config"]
    laps = build_matrix().query("driver == 'VER' and circuit == 'Bahrain'").head(10)
    pkg = onboard_unknown(laps, circuit="Bahrain", circuit_type="high_speed", handle=handle,
                          style_handle=sh, config=cfg)
    assert pkg.fingerprint_available is True
    assert len(pkg.style_match) == 3  # top-3 nearest known drivers
    assert pkg.setup is not None and len(pkg.setup.parameters) == 10


def test_onboard_coords_none_wide_heavy(
    trained: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.ml.stylespace import ProjectionResult

    handle, sh, cfg = trained["handle"], trained["style_handle"], trained["config"]

    def _no_coord(*_a: Any, **_k: Any) -> ProjectionResult:
        return ProjectionResult(status=Status.DEGRADED, coords_5d=None)

    monkeypatch.setattr(eng, "project_unknown", _no_coord)
    laps = build_matrix().head(10)
    pkg = onboard_unknown(laps, circuit="Bahrain", circuit_type="high_speed", handle=handle,
                          style_handle=sh, config=cfg)
    assert pkg.fingerprint_available is False
    assert pkg.setup is not None
    assert all(p.rating is SetupConfidence.WIDE for p in pkg.setup.parameters)  # WIDE-heavy sheet


def test_onboard_projection_raises_still_returns(
    trained: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    handle, sh, cfg = trained["handle"], trained["style_handle"], trained["config"]

    def _boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("projection blew up")

    monkeypatch.setattr(eng, "project_unknown", _boom)
    pkg = onboard_unknown(build_matrix().head(10), circuit="Bahrain", circuit_type="high_speed",
                          handle=handle, style_handle=sh, config=cfg)  # must not raise
    assert pkg.status is Status.FAILED and pkg.error_code is ErrorCode.INFERENCE_ERROR


# ==================================================================================
# Load + health (§7 load criterion)
# ==================================================================================


def test_load_missing_artefact_refuses(personalisation_config: Any, read_dq: Any) -> None:
    cfg = personalisation_config
    with pytest.raises(PersonalisationLoadError) as exc:
        load_personalisation(config=cfg, artefact_dir=Path(cfg.paths.artefacts_dir))
    assert exc.value.error_code is ErrorCode.MODEL_NOT_FOUND
    assert "MODEL_NOT_FOUND" in _codes(read_dq, cfg)


def test_load_corrupt_artefact_refuses(personalisation_config: Any) -> None:
    cfg = personalisation_config
    art = Path(cfg.paths.artefacts_dir)
    art.mkdir(parents=True, exist_ok=True)
    (art / "personalisation.pkl").write_bytes(b"not a pickle")
    with pytest.raises(PersonalisationLoadError):
        load_personalisation(config=cfg, artefact_dir=art)


def test_load_wrong_regressor_count_refuses(trained: dict[str, Any], tmp_path: Path) -> None:
    from hashlib import sha256

    cfg = _make_config(tmp_path)
    art = Path(cfg.paths.artefacts_dir)
    art.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(trained["art_dir"] / "personalisation.pkl")
    bundle["regressors"].pop("brake_bias")  # now 9 regressors
    pkl = art / "personalisation.pkl"
    joblib.dump(bundle, pkl)
    # Write a MATCHING sidecar so load passes the integrity check and reaches the count check.
    (art / "personalisation.pkl.sha256").write_text(sha256(pkl.read_bytes()).hexdigest(),
                                                     encoding="utf-8")
    with pytest.raises(PersonalisationLoadError):
        load_personalisation(config=cfg, artefact_dir=art)


def test_load_content_hash_mismatch_refuses(trained: dict[str, Any], tmp_path: Path) -> None:
    """A tampered-but-valid pickle (sidecar hash no longer matches the bytes) → MODEL_NOT_FOUND."""
    cfg = _make_config(tmp_path)
    art = Path(cfg.paths.artefacts_dir)
    art.mkdir(parents=True, exist_ok=True)
    src = trained["art_dir"]
    (art / "personalisation.pkl").write_bytes((src / "personalisation.pkl").read_bytes())
    (art / "personalisation.pkl.sha256").write_text("deadbeef", encoding="utf-8")  # wrong hash
    with pytest.raises(PersonalisationLoadError):
        load_personalisation(config=cfg, artefact_dir=art)


def test_load_missing_sidecar_refuses(trained: dict[str, Any], tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    art = Path(cfg.paths.artefacts_dir)
    art.mkdir(parents=True, exist_ok=True)
    src_pkl = (trained["art_dir"] / "personalisation.pkl").read_bytes()
    (art / "personalisation.pkl").write_bytes(src_pkl)
    with pytest.raises(PersonalisationLoadError):  # no .sha256 sidecar
        load_personalisation(config=cfg, artefact_dir=art)


def test_partial_thin_cells_degrades(trained: dict[str, Any], tmp_path: Path, read_dq: Any) -> None:
    """Some (not all) cells dropped → training succeeds, status=DEGRADED + DATA_INSUFFICIENT."""
    import pandas as pd

    cfg = _make_config(tmp_path)
    art = Path(cfg.paths.artefacts_dir)
    matrix = build_matrix()
    # Thin ONE cell (VER/Bahrain) below the floor by dropping most of its laps.
    ver_bah = matrix.query("driver == 'VER' and circuit == 'Bahrain'").head(5)
    rest = matrix.query("not (driver == 'VER' and circuit == 'Bahrain')")
    thinned = pd.concat([rest, ver_bah], ignore_index=True)
    fit_normaliser(thinned, config=cfg)
    fit = train_personalisation(thinned, style_handle=trained["style_handle"], config=cfg,
                                artefact_dir=art)
    assert fit.status is Status.DEGRADED
    assert ErrorCode.DATA_INSUFFICIENT in fit.flags
    assert fit.n_train_cells == 47  # the thinned cell was dropped
    assert "DATA_INSUFFICIENT" in _codes(read_dq, cfg)


def test_load_fia_mismatch_refuses(trained: dict[str, Any], tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    cfg.personalisation.fia_bounds["brake_bias"] = (50.0, 60.0)  # differ from the artefact
    with pytest.raises(PersonalisationLoadError):
        load_personalisation(config=cfg, artefact_dir=trained["art_dir"])


def test_load_vocab_mismatch_refuses(trained: dict[str, Any], tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    cfg.circuit_types = ["street", "high_speed", "technical"]  # dropped one → vocab mismatch
    with pytest.raises(PersonalisationLoadError):
        load_personalisation(config=cfg, artefact_dir=trained["art_dir"])


def test_health(trained: dict[str, Any]) -> None:
    assert health(None).error_code is ErrorCode.HEALTH_FAIL
    assert health(None).reason is ErrorCode.MODEL_NOT_FOUND
    assert health(None).ready is False
    ok = health(trained["handle"])
    assert ok.ready is True and ok.status is Status.OK and ok.model_version


# ==================================================================================
# Static no-fit scan (§7 leakage criterion)
# ==================================================================================


_SOURCE = Path(eng.__file__).read_text(encoding="utf-8")
_SERVING = (
    eng.recommend_setup, eng.onboard_unknown, eng.load_personalisation, eng.health,
    eng._recommend_from_coord, eng._attach_shap, eng._assemble_parameter,
    eng._wide_only_recommendation, eng._backstop_recommendation, eng._wide_sheet_from_bounds,
    eng._full_bounds_wide_params, eng._resolve_fia_bounds,
)


def test_no_fit_in_serving_bodies() -> None:
    for fn in _SERVING:
        src = inspect.getsource(fn)
        assert ".fit(" not in src
        assert ".fit_transform(" not in src


def test_no_normalisation_refit_anywhere() -> None:
    assert "fit_normaliser" not in _SOURCE  # F11 reuses F04 params (train aggregates), never refits


def test_no_magic_threshold_literals() -> None:
    body = "\n".join(ln for ln in _SOURCE.splitlines() if not ln.strip().startswith("#"))
    for literal in ("0.15", "0.65", "0.80", "0.25"):  # wide/bands/quartile come from config
        assert literal not in body, f"threshold {literal!r} hardcoded"
    for key in ("confidence_floor", "wide_range_pct", "top_quartile_pct", "fia_bounds", "cv_folds"):
        assert key in _SOURCE, f"config key {key!r} must drive the module"
