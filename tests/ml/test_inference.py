"""F06b classifier-inference tests — one per acceptance criterion + per failure mode.

Gate boundaries are pinned with a `StubModel` (caller-supplied `predict_proba`) so the exact
max-prob / entropy / argmax is deterministic; the load/serve path is exercised against the real
F06a champion produced by the module-scoped `trained` fixture.
"""

from __future__ import annotations

import io
import json
import tokenize
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from app.contracts.config import AppConfig, ClassifierConfig, load_config
from app.contracts.enums import ConfidenceState, Status
from app.contracts.errors import ErrorCode
from app.ml import inference as inf
from app.ml.inference import ModelLoadError, health, load_model, predict

if TYPE_CHECKING:
    from collections.abc import Callable

SOURCE = Path(inf.__file__).read_text(encoding="utf-8")


def _code_only(src: str) -> str:
    out: list[str] = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type not in (tokenize.STRING, tokenize.COMMENT):
            out.append(tok.string)
    return " ".join(out)


CODE_ONLY = _code_only(SOURCE)
REPO_CONFIG = Path(__file__).resolve().parents[2] / "config.yaml"


# --------------------------------------------------------------------------------------------
# Load + happy path (real champion)
# --------------------------------------------------------------------------------------------


def test_load_model_and_health(trained: tuple[Any, AppConfig]) -> None:
    _result, cfg = trained
    handle = load_model(config=cfg, artefact_dir=Path(cfg.paths.artefacts_dir))
    assert handle.ready and handle.model_version
    assert handle.model.n_features_in_ == 28
    assert list(handle.model.classes_) == list(range(len(handle.drivers)))
    assert health(handle).ready is True
    assert health(handle).status is Status.OK
    h = health(None)
    assert h.ready is False and h.error_code is ErrorCode.HEALTH_FAIL


def test_predict_happy_real_champion(
    trained: tuple[Any, AppConfig], clean_vector: Callable[[], dict[str, float]]
) -> None:
    _result, cfg = trained
    handle = load_model(config=cfg, artefact_dir=Path(cfg.paths.artefacts_dir))
    r = predict(clean_vector(), circuit="Bahrain", handle=handle, config=cfg)
    assert r.status in (Status.OK, Status.DEGRADED)
    assert set(r.probabilities) == set(handle.drivers)
    assert abs(sum(r.probabilities.values()) - 1.0) < 1e-6
    assert r.confidence_state is not None
    assert r.shap_values is None and r.shap_available is False
    assert r.model_version == handle.model_version


def test_model_version_deterministic(trained: tuple[Any, AppConfig]) -> None:
    _result, cfg = trained
    v1 = load_model(config=cfg, artefact_dir=Path(cfg.paths.artefacts_dir)).model_version
    v2 = load_model(config=cfg, artefact_dir=Path(cfg.paths.artefacts_dir)).model_version
    assert v1 and v1 == v2


# --------------------------------------------------------------------------------------------
# Label-order fidelity (the silent-mislabel guard)
# --------------------------------------------------------------------------------------------


def test_label_order_fidelity(
    make_handle: Callable[..., Any], clean_vector: Callable[[], dict[str, float]], tmp_path: Path
) -> None:
    handle = make_handle([0.1, 0.7, 0.2], drivers=["ALO", "VER", "HAM"])
    r = predict(clean_vector(), circuit="Bahrain", handle=handle, config=_cfg(tmp_path))
    assert r.top_prediction == "VER"  # argmax column -> correct driver name
    assert r.probabilities == {"ALO": 0.1, "VER": 0.7, "HAM": 0.2}
    assert r.top_3[0] == ("VER", 0.7)


# --------------------------------------------------------------------------------------------
# Confidence / entropy / HIGH-MEDIUM gate boundaries
# --------------------------------------------------------------------------------------------


def _cfg(tmp_path: Path) -> AppConfig:
    cfg = load_config(REPO_CONFIG)
    cfg.paths.data_quality_log = str(tmp_path / "dq.jsonl")
    return cfg


def _state(make_handle: Callable[..., Any], row: list[float], tmp_path: Path) -> Any:
    handle = make_handle(row)
    return predict(
        {f: 1.0 for f in inf.FEATURE_IDS}, circuit="Bahrain", handle=handle, config=_cfg(tmp_path)
    )


def test_confidence_breakdown_composes_served_signals(
    make_handle: Callable[..., Any], tmp_path: Path
) -> None:
    """System input #6: the breakdown is a product of the three served signals, with denominators
    taken from the EXISTING config boundaries — computed after the gate, never read by it."""
    cfg = _cfg(tmp_path)
    th = cfg.thresholds
    r = _state(make_handle, [0.9, 0.1], tmp_path)
    bd = r.confidence_breakdown
    assert bd is not None
    assert bd.feature_quality == pytest.approx(1.0)  # no NULLs → far from the ceiling
    h = float(-(0.9 * np.log2(0.9) + 0.1 * np.log2(0.1)))
    assert bd.distribution == pytest.approx(1.0 - h / th.entropy_ambiguous_bits)
    assert bd.model_confidence == pytest.approx(0.9)
    assert bd.overall == pytest.approx(bd.feature_quality * bd.distribution * bd.model_confidence)


def test_confidence_breakdown_scales_with_nulls(
    make_handle: Callable[..., Any],
    clean_vector: Callable[[], dict[str, float]],
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    th = cfg.thresholds
    handle = make_handle([0.9, 0.1])
    fv: dict[str, float | None] = dict(clean_vector())
    for fid in list(fv)[:3]:
        fv[fid] = None  # 3 of the 6-NULL tolerance
    r = predict(fv, circuit="Bahrain", handle=handle, config=cfg)
    bd = r.confidence_breakdown
    assert bd is not None
    assert bd.feature_quality == pytest.approx(1.0 - 3 / th.null_ceiling)
    assert bd.overall < bd.model_confidence  # a weak factor drags the product down


def test_confidence_breakdown_present_on_gated_results_but_never_gates(
    make_handle: Callable[..., Any], tmp_path: Path
) -> None:
    # A LOW_CONFIDENCE result still carries the breakdown (it EXPLAINS the uncertainty) — and the
    # gate outcome is exactly what it was before the breakdown existed (state stays authoritative).
    low = _state(make_handle, [0.5, 0.5], tmp_path)
    assert low.confidence_state is ConfidenceState.LOW_CONFIDENCE
    assert low.top_prediction is None
    assert low.confidence_breakdown is not None
    assert low.confidence_breakdown.model_confidence == pytest.approx(0.5)


def test_confidence_breakdown_none_on_failed(
    make_handle: Callable[..., Any],
    clean_vector: Callable[[], dict[str, float]],
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    handle = make_handle([0.9, 0.1])
    fv: dict[str, float | None] = dict(clean_vector())
    for fid in list(fv)[:7]:  # over the 6-NULL ceiling → FAILED, model never called
        fv[fid] = None
    r = predict(fv, circuit="Bahrain", handle=handle, config=cfg)
    assert r.status is Status.FAILED
    assert r.confidence_breakdown is None  # no distribution to score — mirrors confidence_state


def test_confidence_gate_boundary(make_handle: Callable[..., Any], tmp_path: Path) -> None:
    below = _state(make_handle, [0.649, 0.351], tmp_path)
    assert below.confidence_state is ConfidenceState.LOW_CONFIDENCE
    assert below.top_prediction is None and below.error_code is ErrorCode.LOW_CONFIDENCE
    assert below.top_3  # candidates still surfaced
    above = _state(make_handle, [0.651, 0.349], tmp_path)
    assert above.confidence_state is ConfidenceState.MEDIUM
    assert above.top_prediction is not None


def test_high_medium_split(make_handle: Callable[..., Any], tmp_path: Path) -> None:
    assert _state(make_handle, [0.85, 0.15], tmp_path).confidence_state is ConfidenceState.HIGH
    assert _state(make_handle, [0.84, 0.16], tmp_path).confidence_state is ConfidenceState.MEDIUM


def test_entropy_gate_and_precedence(make_handle: Callable[..., Any], tmp_path: Path) -> None:
    # 8-class uniform -> entropy 3.0 bits > 2.9 AND max 0.125 < 0.65 -> AMBIGUOUS wins (precedence).
    amb = _state(make_handle, [0.125] * 8, tmp_path)
    assert amb.confidence_state is ConfidenceState.AMBIGUOUS
    assert amb.top_prediction is None and amb.error_code is ErrorCode.AMBIGUOUS
    assert len(amb.top_3) == 3  # candidates still surfaced on the AMBIGUOUS path (§7)
    # entropy under 2.9 but max < 0.65 -> NOT ambiguous, falls to LOW_CONFIDENCE.
    low = _state(make_handle, [0.5, *([0.5 / 7] * 7)], tmp_path)
    assert low.confidence_state is ConfidenceState.LOW_CONFIDENCE


# --------------------------------------------------------------------------------------------
# NULL ceiling (model not called)
# --------------------------------------------------------------------------------------------


def test_null_ceiling(
    make_handle: Callable[..., Any], clean_vector: Callable[[], dict[str, float]], tmp_path: Path
) -> None:
    handle = make_handle([0.9, 0.1])
    six = clean_vector()
    for fid in list(inf.FEATURE_IDS)[:6]:
        six[fid] = None
    r6 = predict(six, circuit="Bahrain", handle=handle, config=_cfg(tmp_path))
    assert r6.error_code is not ErrorCode.INVALID_INPUT  # accepted
    assert handle.model.calls == 1  # model WAS called

    handle7 = make_handle([0.9, 0.1])
    seven = clean_vector()
    for fid in list(inf.FEATURE_IDS)[:7]:
        seven[fid] = None
    r7 = predict(seven, circuit="Bahrain", handle=handle7, config=_cfg(tmp_path))
    assert r7.status is Status.FAILED and r7.error_code is ErrorCode.INVALID_INPUT
    assert handle7.model.calls == 0  # model NOT called
    assert len(r7.null_features) == 7
    assert r7.confidence == 0.0 and r7.confidence_state is None


# --------------------------------------------------------------------------------------------
# confidence/state contract + status mapping across every path
# --------------------------------------------------------------------------------------------


def test_confidence_never_null_and_status_mapping(
    make_handle: Callable[..., Any], clean_vector: Callable[[], dict[str, float]], tmp_path: Path
) -> None:
    cases = {
        "HIGH": ([0.9, 0.1], Status.OK, True),
        "MEDIUM": ([0.7, 0.3], Status.OK, True),
        "LOW": ([0.6, 0.4], Status.DEGRADED, True),
        "AMBIGUOUS": ([0.125] * 8, Status.DEGRADED, True),
    }
    for row, status, state_set in cases.values():
        r = _state(make_handle, row, tmp_path)
        assert isinstance(r.confidence, float)
        assert r.status is status
        assert (r.confidence_state is not None) is state_set
    # FAILED paths: confidence 0.0 (never NULL), state None
    handle = make_handle([0.9, 0.1], raises=True)
    err = predict(
        {f: 1.0 for f in inf.FEATURE_IDS}, circuit="Bahrain", handle=handle, config=_cfg(tmp_path)
    )
    assert err.status is Status.FAILED and err.confidence == 0.0 and err.confidence_state is None


# --------------------------------------------------------------------------------------------
# Never-raises / INFERENCE_ERROR
# --------------------------------------------------------------------------------------------


def test_inference_error_on_model_fault(
    make_handle: Callable[..., Any], clean_vector: Callable[[], dict[str, float]], tmp_path: Path
) -> None:
    handle = make_handle([0.9, 0.1], raises=True)
    r = predict(clean_vector(), circuit="Bahrain", handle=handle, config=_cfg(tmp_path))  # no raise
    assert r.status is Status.FAILED and r.error_code is ErrorCode.INFERENCE_ERROR
    assert r.logged is True


# --------------------------------------------------------------------------------------------
# Normalisation fallback (unknown circuit)
# --------------------------------------------------------------------------------------------


def test_normalisation_fallback_unknown_circuit(
    make_handle: Callable[..., Any], clean_vector: Callable[[], dict[str, float]], tmp_path: Path
) -> None:
    handle = make_handle([0.9, 0.1])
    r = predict(clean_vector(), circuit="Atlantis", handle=handle, config=_cfg(tmp_path))
    assert ErrorCode.NORMALISATION_FALLBACK in r.flags
    assert r.status is Status.DEGRADED
    assert r.top_prediction is not None  # still predicts
    assert isinstance(r.confidence, float)
    assert r.logged is True  # F04 wrote the fallback DQ row, propagated to the result


# --------------------------------------------------------------------------------------------
# Logged + DQ row
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("proba", "nulls", "circuit", "raises", "code"),
    [
        ([0.6, 0.4], 0, "Bahrain", False, ErrorCode.LOW_CONFIDENCE),
        ([0.125] * 8, 0, "Bahrain", False, ErrorCode.AMBIGUOUS),
        ([0.9, 0.1], 7, "Bahrain", False, ErrorCode.INVALID_INPUT),
        ([0.9, 0.1], 0, "Bahrain", True, ErrorCode.INFERENCE_ERROR),
        ([0.9, 0.1], 0, "Atlantis", False, ErrorCode.NORMALISATION_FALLBACK),
    ],
)
def test_logged_modes_write_dq_row(
    make_handle: Callable[..., Any],
    read_dq: Callable[[str | Path], list[dict]],
    tmp_path: Path,
    proba: list[float],
    nulls: int,
    circuit: str,
    raises: bool,
    code: ErrorCode,
) -> None:
    # Every logged failure mode sets result.logged AND writes a DQ row with the matching code (§7).
    cfg = _cfg(tmp_path)
    handle = make_handle(proba, raises=raises)
    vec: dict[str, float | None] = {f: 1.0 for f in inf.FEATURE_IDS}
    for fid in list(inf.FEATURE_IDS)[:nulls]:
        vec[fid] = None
    r = predict(vec, circuit=circuit, handle=handle, config=cfg)
    assert r.logged is True
    rows = read_dq(cfg.paths.data_quality_log)
    assert any(row["error_code"] == code.value for row in rows)


# --------------------------------------------------------------------------------------------
# Latency / PERF_WARN (both sides)
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("delta_s", "expect_warn"), [(0.201, True), (0.199, False)])
def test_latency_perf_warn(
    make_handle: Callable[..., Any],
    read_dq: Callable[[str | Path], list[dict]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    delta_s: float,
    expect_warn: bool,
) -> None:
    calls = {"n": 0}

    def fake_perf() -> float:
        calls["n"] += 1
        return 0.0 if calls["n"] == 1 else delta_s

    monkeypatch.setattr(inf.time, "perf_counter", fake_perf)
    cfg = _cfg(tmp_path)
    handle = make_handle([0.9, 0.1])
    full = {f: 1.0 for f in inf.FEATURE_IDS}
    r = predict(full, circuit="Bahrain", handle=handle, config=cfg)
    assert (ErrorCode.PERF_WARN in r.flags) is expect_warn
    assert r.status is Status.OK  # latency alone never degrades status
    # the logged PERF_WARN mode writes a matching DQ row (§7 logged-modes criterion)
    rows = read_dq(cfg.paths.data_quality_log)
    has_row = any(row["error_code"] == ErrorCode.PERF_WARN.value for row in rows)
    assert has_row is expect_warn


# --------------------------------------------------------------------------------------------
# Fail-loud load: model-not-found, stale labels
# --------------------------------------------------------------------------------------------


def test_missing_champion_raises(
    bundle_dir: Path, read_dq: Callable[[str | Path], list[dict]], tmp_path: Path
) -> None:
    (bundle_dir / "classifier.pkl").unlink()
    log = tmp_path / "dq.jsonl"
    with pytest.raises(ModelLoadError) as exc:
        load_model(config=load_config(REPO_CONFIG), artefact_dir=bundle_dir, log_path=log)
    assert exc.value.error_code is ErrorCode.MODEL_NOT_FOUND
    assert any(r["error_code"] == ErrorCode.MODEL_NOT_FOUND.value for r in read_dq(log))


def test_missing_normalisation_raises(
    bundle_dir: Path, read_dq: Callable[[str | Path], list[dict]], tmp_path: Path
) -> None:
    (bundle_dir / "normalisation.json").unlink()
    log = tmp_path / "dq.jsonl"
    with pytest.raises(ModelLoadError) as exc:
        load_model(config=load_config(REPO_CONFIG), artefact_dir=bundle_dir, log_path=log)
    assert exc.value.error_code is ErrorCode.MODEL_NOT_FOUND
    assert any(r["error_code"] == ErrorCode.MODEL_NOT_FOUND.value for r in read_dq(log))


def test_corrupt_champion_raises(bundle_dir: Path, tmp_path: Path) -> None:
    (bundle_dir / "classifier.pkl").write_bytes(b"not a pickle")
    with pytest.raises(ModelLoadError) as exc:
        load_model(
            config=load_config(REPO_CONFIG), artefact_dir=bundle_dir, log_path=tmp_path / "d"
        )
    assert exc.value.error_code is ErrorCode.MODEL_NOT_FOUND


def test_config_block_absent_raises(bundle_dir: Path, tmp_path: Path) -> None:
    cfg = load_config(REPO_CONFIG)
    cfg.classifier = None
    with pytest.raises(ModelLoadError) as exc:
        load_model(config=cfg, artefact_dir=bundle_dir, log_path=tmp_path / "d")
    assert exc.value.error_code is ErrorCode.MODEL_NOT_FOUND


def test_non_integer_classes_refuse_to_serve(bundle_dir: Path, tmp_path: Path) -> None:
    # A champion whose classes_ are not the contiguous range [0..n-1] must refuse to serve.
    import joblib
    from tests.ml.conftest import StubModel

    stub = StubModel([0.5, 0.5])
    stub.classes_ = np.array([1, 2])  # non-contiguous / non-zero-based
    joblib.dump(stub, bundle_dir / "classifier.pkl")
    (bundle_dir / "labels.json").write_text(json.dumps(["A", "B"]), encoding="utf-8")
    with pytest.raises(ModelLoadError) as exc:
        load_model(
            config=load_config(REPO_CONFIG), artefact_dir=bundle_dir, log_path=tmp_path / "d"
        )
    assert exc.value.error_code is ErrorCode.MODEL_NOT_FOUND


def test_stale_labels_refuse_to_serve(
    bundle_dir: Path, read_dq: Callable[[str | Path], list[dict]], tmp_path: Path
) -> None:
    # A labels.json whose length != the champion's class count must refuse to serve (no mislabel).
    (bundle_dir / "labels.json").write_text(json.dumps(["ONLY_ONE"]), encoding="utf-8")
    log = tmp_path / "dq.jsonl"
    with pytest.raises(ModelLoadError) as exc:
        load_model(config=load_config(REPO_CONFIG), artefact_dir=bundle_dir, log_path=log)
    assert exc.value.error_code is ErrorCode.MODEL_NOT_FOUND
    assert any("LABELS_MISMATCH" in r["error_detail"] for r in read_dq(log))


# --------------------------------------------------------------------------------------------
# Config validators
# --------------------------------------------------------------------------------------------


def _classifier_kwargs() -> dict[str, Any]:
    return {
        "cv_folds": 5, "cv_strategy": "stratified_group_kfold_by_circuit",
        "cv_stratify_by": "driver", "cv_group_by": "circuit", "optuna_trials": 200,
        "optuna_objective": "macro_f1", "optuna_storage": "sqlite:///x.db",
        "calibration_method": "isotonic", "calibration_holdout_frac": 0.2,
        "baselines": ["majority"], "target_macro_f1": 0.78, "per_driver_floor": 0.70,
        "mlflow_tracking_uri": "sqlite:///m.db", "mlflow_experiment": "e",
        "mlflow_registered_model": "m", "model_artefact_filename": "c.pkl",
        "labels_filename": "labels.json", "confidence_high": 0.85, "latency_warn_ms": 200.0,
    }


@pytest.mark.parametrize(("field", "bad"), [("confidence_high", 1.5), ("latency_warn_ms", 0.0)])
def test_classifier_config_rejects_bad_serving_scalars(field: str, bad: Any) -> None:
    kwargs = _classifier_kwargs()
    kwargs[field] = bad
    with pytest.raises(ValueError):
        ClassifierConfig(**kwargs)


def test_appconfig_rejects_confidence_high_below_gate() -> None:
    cfg = load_config(REPO_CONFIG)
    data = cfg.model_dump(by_alias=True)
    data["classifier"]["confidence_high"] = cfg.thresholds.confidence_gate  # not strictly above
    with pytest.raises(ValueError):
        AppConfig.model_validate(data)


# --------------------------------------------------------------------------------------------
# Static scans: no fit / no inference-of-training / config-driven
# --------------------------------------------------------------------------------------------


def test_no_training_symbols_in_source() -> None:
    banned_symbols = (
        ".fit(", "fit_normaliser", "optuna", "train_classifier", "CalibratedClassifierCV("
    )
    for banned in banned_symbols:
        assert banned not in CODE_ONLY, f"inference.py must not contain training symbol {banned!r}"


def test_no_magic_thresholds_in_source() -> None:
    # The gate constants are read from config, never hardcoded in executable code.
    for literal in ("0.65", "2.9", "0.85"):
        assert literal not in CODE_ONLY, f"threshold {literal!r} is hardcoded in inference.py"


def test_inference_error_code_exists() -> None:
    assert ErrorCode.INFERENCE_ERROR.value == "INFERENCE_ERROR"


# --------------------------------------------------------------------------------------------
# Code-review hardening: omitted keys, provenance, config-absent, known-circuit NULL, dup labels
# --------------------------------------------------------------------------------------------


def test_omitted_key_treated_as_null_not_error(
    make_handle: Callable[..., Any], tmp_path: Path
) -> None:
    # A caller may OMIT a key rather than set None — must count as NULL, not KeyError->error.
    handle = make_handle([0.9, 0.1])
    partial = {f: 1.0 for f in inf.FEATURE_IDS if f != "F03"}  # F03 omitted entirely
    r = predict(partial, circuit="Bahrain", handle=handle, config=_cfg(tmp_path))
    assert r.error_code is not ErrorCode.INFERENCE_ERROR  # served, not spuriously failed
    assert r.status in (Status.OK, Status.DEGRADED)
    assert handle.model.calls == 1


def test_null_features_provenance_on_accepted(
    make_handle: Callable[..., Any], clean_vector: Callable[[], dict[str, float]], tmp_path: Path
) -> None:
    # An accepted prediction with tolerated NULLs still reports which features were missing.
    handle = make_handle([0.9, 0.1])
    vec: dict[str, float | None] = dict(clean_vector())
    vec["F01"] = None
    vec["F02"] = None
    r = predict(vec, circuit="Bahrain", handle=handle, config=_cfg(tmp_path))
    assert r.status in (Status.OK, Status.DEGRADED)
    assert set(r.null_features) == {"F01", "F02"}


def test_predict_never_raises_on_config_absent(
    make_handle: Callable[..., Any], tmp_path: Path
) -> None:
    # config.classifier absent must yield a coded result, not an AssertionError (never-raises).
    cfg = _cfg(tmp_path)
    cfg.classifier = None
    handle = make_handle([0.9, 0.1])
    r = predict({f: 1.0 for f in inf.FEATURE_IDS}, circuit="Bahrain", handle=handle, config=cfg)
    assert r.status is Status.FAILED and r.error_code is ErrorCode.INFERENCE_ERROR


def test_known_circuit_null_not_flagged_as_circuit_fallback(
    make_handle: Callable[..., Any], clean_vector: Callable[[], dict[str, float]], tmp_path: Path
) -> None:
    # A NULL among the six normalised features on a KNOWN circuit is tolerated passthrough, not a
    # NORMALISATION_FALLBACK (which means unknown/skipped circuit).
    handle = make_handle([0.9, 0.1])
    vec: dict[str, float | None] = dict(clean_vector())
    vec["F09"] = None  # a normalised feature, known circuit
    r = predict(vec, circuit="Bahrain", handle=handle, config=_cfg(tmp_path))
    assert ErrorCode.NORMALISATION_FALLBACK not in r.flags
    assert "F09" in r.null_features


def test_flags_deduped_when_sink_unwritable(
    make_handle: Callable[..., Any], tmp_path: Path
) -> None:
    # When the DQ sink is down, BOTH F04's apply-ledger and predict's own ledger emit
    # DATA_QUALITY_LOG_UNWRITABLE — the merged flags must not contain it twice.
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file", encoding="utf-8")  # parent-is-a-file -> writes fail
    cfg = load_config(REPO_CONFIG)
    cfg.paths.data_quality_log = str(blocker / "dq.jsonl")
    handle = make_handle([0.6, 0.4])  # LOW gate (predict logs) on an unknown circuit (F04 logs)
    r = predict({f: 1.0 for f in inf.FEATURE_IDS}, circuit="Atlantis", handle=handle, config=cfg)
    assert r.flags.count(ErrorCode.DATA_QUALITY_LOG_UNWRITABLE) == 1
    assert len(r.flags) == len(set(r.flags))  # no duplicate codes at all


def test_duplicate_labels_refuse_to_serve(bundle_dir: Path, tmp_path: Path) -> None:
    import joblib
    from tests.ml.conftest import StubModel

    joblib.dump(StubModel([0.5, 0.5]), bundle_dir / "classifier.pkl")
    (bundle_dir / "labels.json").write_text(json.dumps(["VER", "VER"]), encoding="utf-8")
    with pytest.raises(ModelLoadError) as exc:
        load_model(
            config=load_config(REPO_CONFIG), artefact_dir=bundle_dir, log_path=tmp_path / "d"
        )
    assert exc.value.error_code is ErrorCode.MODEL_NOT_FOUND
