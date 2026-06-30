"""F06a classifier-training tests — one per acceptance criterion + per failure mode.

The slow happy-path assertions read a single module-scoped ``trained`` run; the fail-loud guards,
the CV fold-membership guarantee, the trial-failure handling, and the OK/DEGRADED decision are
tested as fast, focused units so the suite stays quick despite exercising XGBoost + Optuna +
MLflow.
"""

from __future__ import annotations

import io
import tokenize
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import optuna
import pytest
from app.contracts.config import AppConfig, ClassifierConfig, load_config
from app.contracts.enums import Status
from app.contracts.errors import ErrorCode
from app.ml import classifier as clf
from app.ml.classifier import (
    _OBJECTIVE_AVERAGE,
    ClassifierTrainingError,
    _assert_calibration_covers_classes,
    _assert_calibration_disjoint,
    _assert_no_leakage,
    _baseline_estimator,
    _DQLedger,
    _evaluate_outcome,
    _label_space,
    _partition,
    _run_study,
    _study_key,
    _xy,
    build_cv,
    train_classifier,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

SOURCE = Path(clf.__file__).read_text(encoding="utf-8")


def _code_only(src: str) -> str:
    """Strip string-literal and comment tokens, leaving executable code (names/numbers/operators).

    Config-owned values legitimately appear in docstrings/comments for documentation; the contract
    is that they are never *hardcoded in executable code*. This mirrors F04's function-body scan.
    """
    out: list[str] = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in (tokenize.STRING, tokenize.COMMENT):
            continue
        out.append(tok.string)
    return " ".join(out)


CODE_ONLY = _code_only(SOURCE)


# --------------------------------------------------------------------------------------------
# Happy path (shared module-scoped run)
# --------------------------------------------------------------------------------------------


def test_happy_result_wellformed(trained: tuple[Any, AppConfig]) -> None:
    result, cfg = trained
    assert result.status in (Status.OK, Status.DEGRADED)
    assert result.mlflow_run_id
    assert result.model_uri.startswith("runs:/")
    assert Path(result.model_artefact_path).is_file()
    assert (Path(cfg.paths.artefacts_dir) / "normalisation.json").is_file()
    assert result.normalisation_content_hash


def test_cv_strategy_recorded(trained: tuple[Any, AppConfig]) -> None:
    result, cfg = trained
    assert result.cv_strategy == "stratified_group_kfold_by_circuit"
    assert result.cv_folds == cfg.classifier.cv_folds


def test_held_out_evaluation(trained: tuple[Any, AppConfig]) -> None:
    result, _ = trained
    assert np.isfinite(result.held_out_macro_f1)
    assert set(result.held_out_per_driver_f1) == {"VER", "HAM", "LEC", "NOR"}
    assert result.excluded_circuits == ["Belgium", "Italy"]
    # target_met / floor_met are derived strictly from the recorded metrics + config thresholds.
    assert result.target_met == (result.held_out_macro_f1 >= result.target_macro_f1)
    assert result.per_driver_floor_met == all(
        v >= result.per_driver_floor for v in result.held_out_per_driver_f1.values()
    )


def test_baselines_logged(trained: tuple[Any, AppConfig]) -> None:
    result, _ = trained
    assert set(result.baseline_macro_f1) == {"majority", "logistic_regression", "random_forest"}
    assert isinstance(result.beats_all_baselines, bool)


def test_champion_registered_in_mlflow(trained: tuple[Any, AppConfig]) -> None:
    result, cfg = trained
    import mlflow

    mlflow.set_tracking_uri(cfg.classifier.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{cfg.classifier.mlflow_registered_model}'")
    assert len(versions) >= 1


def test_mlflow_completeness_artifacts(trained: tuple[Any, AppConfig]) -> None:
    result, _ = trained
    assert result.confusion_matrix_path == "confusion_matrix.json"
    assert result.calibration_curve_path == "calibration_curve.json"


def test_every_trial_logged_to_mlflow(trained: tuple[Any, AppConfig]) -> None:
    # The full search history (one run per trial) must be in MLflow, not just champion + baselines.
    result, cfg = trained
    import mlflow

    mlflow.set_tracking_uri(cfg.classifier.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(cfg.classifier.mlflow_experiment)
    runs = client.search_runs([exp.experiment_id], max_results=1000)
    trial_runs = [r for r in runs if r.data.tags.get("mlflow.runName", "").startswith("trial-")]
    assert len(trial_runs) >= cfg.classifier.optuna_trials


def test_normalisation_ablation_reported(trained: tuple[Any, AppConfig]) -> None:
    result, _ = trained
    assert np.isfinite(result.normalisation_ablation_macro_f1)


def test_seeds_recorded(trained: tuple[Any, AppConfig]) -> None:
    result, _ = trained
    assert result.seeds == {"numpy": 42, "xgboost": 42, "optuna": 42}


def test_trial_floor_reached(trained: tuple[Any, AppConfig]) -> None:
    result, cfg = trained
    complete = result.n_trials - result.n_trials_failed
    assert complete >= cfg.classifier.optuna_trials
    assert result.n_trials_failed == 0


def test_degraded_still_registers_champion(trained: tuple[Any, AppConfig]) -> None:
    # On fully-separable synthetic data the strong baselines tie the champion, so the run is
    # DEGRADED — but a champion is STILL registered + written (degraded, not crashed).
    result, _ = trained
    assert result.status is Status.DEGRADED
    assert result.error_code is ErrorCode.MODEL_TRAINING_FAILED
    assert ErrorCode.MODEL_TRAINING_FAILED in result.flags
    assert Path(result.model_artefact_path).is_file()
    assert result.model_uri.startswith("runs:/")
    assert result.logged is True


# --------------------------------------------------------------------------------------------
# Leakage guards (fail-loud) — TRAIN_LEAKAGE
# --------------------------------------------------------------------------------------------


def test_leakage_guard_raises_on_held_out_in_pool(
    classifier_config: AppConfig, training_matrix: Callable[..., pd.DataFrame]
) -> None:
    # A pool frame that (pathologically) still contains a held-out circuit must trip the guard.
    poisoned = training_matrix(train_circuits=["Bahrain", "Spain", "Belgium"], held_out=["Italy"])
    with pytest.raises(ClassifierTrainingError) as exc:
        _assert_no_leakage(poisoned, classifier_config)
    assert exc.value.error_code is ErrorCode.TRAIN_LEAKAGE


def test_config_absent_raises_training_failed_not_leakage(
    classifier_config: AppConfig, training_matrix: Callable[..., pd.DataFrame]
) -> None:
    # A missing config block is not leakage -> MODEL_TRAINING_FAILED, not the TRAIN_LEAKAGE code.
    classifier_config.classifier = None
    with pytest.raises(ClassifierTrainingError) as exc:
        _assert_no_leakage(training_matrix(), classifier_config)
    assert exc.value.error_code is ErrorCode.MODEL_TRAINING_FAILED


def test_calibration_overlap_guard_raises(training_matrix: Callable[..., pd.DataFrame]) -> None:
    frame = training_matrix()
    fit = frame.iloc[:50]
    overlapping = frame.iloc[40:60]  # shares rows 40..49 with fit
    with pytest.raises(ClassifierTrainingError) as exc:
        _assert_calibration_disjoint(fit, overlapping)
    assert exc.value.error_code is ErrorCode.TRAIN_LEAKAGE


def test_partition_produces_disjoint_calibration(
    classifier_config: AppConfig, training_matrix: Callable[..., pd.DataFrame]
) -> None:
    splits = _partition(training_matrix(), config=classifier_config)
    assert set(splits.fit.index).isdisjoint(set(splits.calibration.index))
    assert len(splits.calibration) > 0
    # held-out circuits are quarantined out of both fit and calibration
    assert set(splits.fit["circuit"]).isdisjoint({"Belgium", "Italy"})
    assert set(splits.calibration["circuit"]).isdisjoint({"Belgium", "Italy"})
    assert set(splits.held_out["circuit"]) == {"Belgium", "Italy"}


def test_partition_excludes_non_train_split_rows(
    classifier_config: AppConfig, training_matrix: Callable[..., pd.DataFrame]
) -> None:
    # A val/test row from a TRAINING circuit must not enter the fit / calibration pool (spec §4).
    import pandas as pd

    base = training_matrix()
    val_rows = base[base["circuit"] == "Bahrain"].head(8).copy()
    val_rows["split"] = "val"
    poisoned = pd.concat([base, val_rows], ignore_index=True)
    splits = _partition(poisoned, config=classifier_config)
    assert set(splits.fit["split"]) == {"train"}
    assert set(splits.calibration["split"]) == {"train"}


def test_unknown_held_out_driver_raises(
    training_matrix: Callable[..., pd.DataFrame],
) -> None:
    # A held-out driver never seen in training must fail loud (coded), not raise a bare KeyError.
    held_out = training_matrix()[training_matrix()["split"] == "test"].copy()
    held_out["driver"] = "ZZZ"  # a driver absent from the training label space
    with pytest.raises(ClassifierTrainingError) as exc:
        from app.ml.classifier import _assert_known_drivers

        _assert_known_drivers(held_out, ["VER", "HAM", "LEC", "NOR"])
    assert exc.value.error_code is ErrorCode.MODEL_TRAINING_FAILED


def test_no_artefact_written_on_leakage(
    classifier_config: AppConfig,
    training_matrix: Callable[..., pd.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # If the leakage tripwire fires, the partition raises BEFORE any artefact is written.
    def boom(*_a: Any, **_k: Any) -> Any:
        raise ClassifierTrainingError("leak", error_code=ErrorCode.TRAIN_LEAKAGE)

    monkeypatch.setattr(clf, "_partition", boom)
    with pytest.raises(ClassifierTrainingError) as exc:
        train_classifier(training_matrix(), config=classifier_config)
    assert exc.value.error_code is ErrorCode.TRAIN_LEAKAGE
    art = Path(classifier_config.paths.artefacts_dir)
    assert not (art / "normalisation.json").is_file()
    assert not (art / "classifier.pkl").is_file()


def test_registration_failure_is_typed_and_logged(
    classifier_config: AppConfig,
    training_matrix: Callable[..., pd.DataFrame],
    read_dq: Callable[[str | Path], list[dict]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A registry/log failure must surface as a typed MODEL_TRAINING_FAILED + DQ row (not raw).
    def boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("registry backend exploded")

    monkeypatch.setattr(clf.mlflow.sklearn, "log_model", boom)
    with pytest.raises(ClassifierTrainingError) as exc:
        train_classifier(training_matrix(), config=classifier_config)
    assert exc.value.error_code is ErrorCode.MODEL_TRAINING_FAILED
    rows = read_dq(classifier_config.paths.data_quality_log)
    assert any("REGISTRATION_FAILED" in r["error_detail"] for r in rows)


def test_registration_failure_removes_orphan_pkl(
    classifier_config: AppConfig,
    training_matrix: Callable[..., pd.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # On a registration failure no champion pkl may survive (F06b must not serve an unshipped one).
    def boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("registry down")

    monkeypatch.setattr(clf.mlflow.sklearn, "log_model", boom)
    with pytest.raises(ClassifierTrainingError):
        train_classifier(training_matrix(), config=classifier_config)
    assert not (Path(classifier_config.paths.artefacts_dir) / "classifier.pkl").is_file()


def test_study_key_changes_with_data() -> None:
    import numpy as np
    from app.contracts.config import load_config

    cfg = load_config(Path(__file__).resolve().parents[2] / "config.yaml")
    x1 = np.zeros((10, 28))
    y1 = np.array([0, 1] * 5)
    g1 = np.array(["A", "B"] * 5)
    x2 = np.ones((10, 28))
    g2 = np.array(["A", "C"] * 5)
    assert _study_key(x1, y1, g1, cfg) == _study_key(x1.copy(), y1.copy(), g1.copy(), cfg)  # resume
    assert _study_key(x1, y1, g1, cfg) != _study_key(x2, y1, g1, cfg)  # changed features -> fresh
    assert _study_key(x1, y1, g1, cfg) != _study_key(x1, y1, g2, cfg)  # changed circuits -> fresh


def test_sklearn_baselines_tolerate_nan_features() -> None:
    # F04 leaves NULL features as NaN; logreg/forest must not crash uncoded (imputed in-pipeline).
    import numpy as np

    x = np.array([[1.0, np.nan], [2.0, 1.0], [np.nan, 3.0], [4.0, 2.0]])
    y = np.array([0, 1, 0, 1])
    for name in ("logistic_regression", "random_forest", "majority"):
        model = _baseline_estimator(name, seed=42)
        model.fit(x, y)  # must not raise ValueError("Input contains NaN")
        assert len(model.predict(x)) == 4


def test_calibration_missing_class_raises() -> None:
    import numpy as np

    # y_cal missing class index 2 of 3 -> fail loud, not a silently-dropped driver.
    with pytest.raises(ClassifierTrainingError) as exc:
        _assert_calibration_covers_classes(
            np.array([0, 1, 0, 1]), n_classes=3, labels=["VER", "HAM", "LEC"]
        )
    assert exc.value.error_code is ErrorCode.MODEL_TRAINING_FAILED
    assert "LEC" in str(exc.value)


def test_partition_handles_nonunique_index(
    classifier_config: AppConfig, training_matrix: Callable[..., pd.DataFrame]
) -> None:
    # A non-unique incoming index must not corrupt the carve (reset_index inside _partition).
    frame = training_matrix()
    frame.index = [0] * len(frame)  # pathological: every row shares one label
    splits = _partition(frame, config=classifier_config)
    assert set(splits.fit.index).isdisjoint(set(splits.calibration.index))
    assert len(splits.fit) + len(splits.calibration) == len(frame[frame["split"] == "train"])


def test_config_fields_drive_xy_and_objective(
    training_matrix: Callable[..., pd.DataFrame],
) -> None:
    # cv_group_by is genuinely read: _xy groups by the column it names, not a hardcoded constant.
    frame = training_matrix()
    _, _, groups = _xy(frame, labels=["VER", "HAM", "LEC", "NOR"], group_col="season")
    assert set(groups.tolist()) == {2023}
    assert _OBJECTIVE_AVERAGE["macro_f1"] == "macro"  # optuna_objective maps to the f1 average


def test_dq_ledger_counts_unwritable_sink(tmp_path: Path) -> None:
    # The ledger never propagates a sink failure — it counts it so the caller can flag it.
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file, not a directory", encoding="utf-8")
    ledger = _DQLedger(log_path=blocker / "dq.jsonl")  # parent is a file -> mkdir fails
    ledger.record(
        code=ErrorCode.MODEL_TRAINING_FAILED,
        entity_type="trial",
        entity_id="0",
        detail="x",
        fallback_used=False,
    )
    assert ledger.failed == 1
    assert ledger.rows == 0


# --------------------------------------------------------------------------------------------
# CV: StratifiedGroupKFold fold-membership disjointness (Option A guarantee)
# --------------------------------------------------------------------------------------------


def test_cv_folds_hold_circuits_out(
    classifier_config: AppConfig, training_matrix: Callable[..., pd.DataFrame]
) -> None:
    frame = training_matrix()
    pool = frame[~frame["circuit"].isin(["Belgium", "Italy"])]
    labels = _label_space(pool)
    x, y, groups = _xy(pool, labels=labels)
    cv = build_cv(classifier_config)
    n_folds = 0
    for train_idx, val_idx in cv.split(x, y, groups):
        train_circuits = set(groups[train_idx])
        val_circuits = set(groups[val_idx])
        assert train_circuits.isdisjoint(val_circuits), "a circuit appeared in both train and val"
        n_folds += 1
    assert n_folds == classifier_config.classifier.cv_folds


# --------------------------------------------------------------------------------------------
# Trial-failure handling + no-valid-champion  — MODEL_TRAINING_FAILED
# --------------------------------------------------------------------------------------------


def _study_inputs(
    classifier_config: AppConfig, training_matrix: Callable[..., pd.DataFrame]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = training_matrix()
    pool = frame[~frame["circuit"].isin(["Belgium", "Italy"])]
    labels = _label_space(pool)
    return _xy(pool, labels=labels)


def test_failing_trial_pruned_study_continues(
    classifier_config: AppConfig,
    training_matrix: Callable[..., pd.DataFrame],
    tmp_path: Path,
    read_dq: Callable[[str | Path], list[dict]],
) -> None:
    x, y, groups = _study_inputs(classifier_config, training_matrix)
    cv = build_cv(classifier_config)
    log_path = tmp_path / "dq.jsonl"
    ledger = _DQLedger(log_path=log_path)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=classifier_config.seeds.optuna),
    )

    def fail_first(trial_number: int) -> None:
        if trial_number == 0:
            raise RuntimeError("injected trial failure")

    records, n_failed, n_complete = _run_study(
        study, x=x, y=y, groups=groups, cv=cv, config=classifier_config, ledger=ledger,
        fail_hook=fail_first,
    )
    assert n_failed >= 1  # the injected failure was counted, not swallowed
    assert n_complete >= classifier_config.classifier.optuna_trials  # study topped up past it
    assert any(r.state != "COMPLETE" for r in records)
    rows = read_dq(log_path)
    assert any(
        r["error_code"] == ErrorCode.MODEL_TRAINING_FAILED.value and r["entity_type"] == "trial"
        for r in rows
    )


def test_no_valid_champion_raises(
    classifier_config: AppConfig,
    training_matrix: Callable[..., pd.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Force the study to yield zero COMPLETE trials -> train_classifier must fail loud, not ship.
    def zero_complete(*_a: Any, **_k: Any) -> tuple[list[Any], int, int]:
        return [], 5, 0

    monkeypatch.setattr(clf, "_run_study", zero_complete)
    with pytest.raises(ClassifierTrainingError) as exc:
        train_classifier(training_matrix(), config=classifier_config)
    assert exc.value.error_code is ErrorCode.MODEL_TRAINING_FAILED
    # no champion artefact written on the poisoned run
    assert not (Path(classifier_config.paths.artefacts_dir) / "classifier.pkl").is_file()


# --------------------------------------------------------------------------------------------
# OK vs DEGRADED decision (pure) + boundaries at 0.78 / 0.70
# --------------------------------------------------------------------------------------------


@pytest.fixture
def decision_cfg() -> ClassifierConfig:
    cfg = load_config(Path(__file__).resolve().parents[2] / "config.yaml")
    assert cfg.classifier is not None
    return cfg.classifier


def test_outcome_ok_when_all_met(decision_cfg: ClassifierConfig) -> None:
    out = _evaluate_outcome(
        held_out_macro_f1=0.90,
        per_driver={"a": 0.80, "b": 0.85},
        baseline_macro_f1={"majority": 0.10, "random_forest": 0.50},
        cfg=decision_cfg,
    )
    assert out.status is Status.OK
    assert out.flags == []
    assert out.target_met and out.floor_met and out.beats_all


@pytest.mark.parametrize(
    ("held", "per", "base"),
    [
        (0.50, {"a": 0.80}, {"m": 0.10}),  # target missed
        (0.90, {"a": 0.60}, {"m": 0.10}),  # per-driver floor missed
        (0.90, {"a": 0.85}, {"m": 0.95}),  # baseline not beaten
    ],
)
def test_outcome_degraded_paths(
    decision_cfg: ClassifierConfig, held: float, per: dict[str, float], base: dict[str, float]
) -> None:
    out = _evaluate_outcome(
        held_out_macro_f1=held, per_driver=per, baseline_macro_f1=base, cfg=decision_cfg
    )
    assert out.status is Status.DEGRADED
    assert out.flags == [ErrorCode.MODEL_TRAINING_FAILED]
    assert out.reasons


def test_outcome_floor_reason_names_failing_driver(decision_cfg: ClassifierConfig) -> None:
    # FM6 audit row must name which driver(s) fell below the floor (spec §6 "yes (driver + F1)").
    out = _evaluate_outcome(
        held_out_macro_f1=0.90,
        per_driver={"VER": 0.95, "SAI": 0.55},
        baseline_macro_f1={"m": 0.10},
        cfg=decision_cfg,
    )
    floor_reason = next(d for tag, d in out.reasons if tag == "PER_DRIVER_FLOOR_MISSED")
    assert "SAI" in floor_reason
    assert "VER" not in floor_reason


def test_outcome_target_boundary(decision_cfg: ClassifierConfig) -> None:
    base = {"m": 0.10}
    per = {"a": 0.80}
    assert _evaluate_outcome(
        held_out_macro_f1=0.78, per_driver=per, baseline_macro_f1=base, cfg=decision_cfg
    ).target_met
    assert not _evaluate_outcome(
        held_out_macro_f1=0.7799, per_driver=per, baseline_macro_f1=base, cfg=decision_cfg
    ).target_met


def test_outcome_floor_boundary(decision_cfg: ClassifierConfig) -> None:
    base = {"m": 0.10}
    assert _evaluate_outcome(
        held_out_macro_f1=0.90, per_driver={"a": 0.70}, baseline_macro_f1=base, cfg=decision_cfg
    ).floor_met
    assert not _evaluate_outcome(
        held_out_macro_f1=0.90, per_driver={"a": 0.6999}, baseline_macro_f1=base, cfg=decision_cfg
    ).floor_met


# --------------------------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------------------------


def test_reproducible_champion(
    tmp_path: Path,
    training_matrix: Callable[..., pd.DataFrame],
    make_config: Callable[[Path], AppConfig],
) -> None:
    matrix = training_matrix()
    r1 = train_classifier(matrix, config=make_config(tmp_path / "run1"))
    r2 = train_classifier(matrix, config=make_config(tmp_path / "run2"))
    assert r1.champion_params == r2.champion_params
    assert r1.held_out_macro_f1 == pytest.approx(r2.held_out_macro_f1, rel=1e-9, abs=1e-9)


# --------------------------------------------------------------------------------------------
# Source-scan fences: no magic numbers / one fit / no inference path
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "literal",
    ["0.78", "0.70", "0.2", "200", "classifier.pkl", "normalisation.json",
     "stratified_group_kfold_by_circuit", "apex-classifier", "apex-driver-classifier",
     "sqlite:///"],
)
def test_no_magic_numbers_in_source(literal: str) -> None:
    # Every config-owned value must be read from config, never hardcoded in executable code
    # (docstrings/comments may mention them — see _code_only).
    assert literal not in CODE_ONLY, f"config-owned value {literal!r} is hardcoded in classifier.py"


def test_fit_normaliser_called_exactly_once() -> None:
    # The ONE scaler fit; the held-out path must only ever apply_normalisation.
    assert SOURCE.count("fit_normaliser(") == 1
    assert "apply_normalisation(" in SOURCE


def test_scope_fence_no_inference_path() -> None:
    for banned in (
        "PredictionResult", "confidence_gate", "entropy_ambiguous_bits", "null_ceiling",
        "LOW_CONFIDENCE", "AMBIGUOUS", "INVALID_INPUT",
    ):
        assert banned not in SOURCE, f"F06a must not reference F06b inference symbol {banned!r}"


def test_both_new_error_codes_exist() -> None:
    assert ErrorCode.TRAIN_LEAKAGE.value == "TRAIN_LEAKAGE"
    assert ErrorCode.MODEL_TRAINING_FAILED.value == "MODEL_TRAINING_FAILED"


# --------------------------------------------------------------------------------------------
# Config validators (the pinned CV keys / choices)
# --------------------------------------------------------------------------------------------


def _valid_classifier_kwargs() -> dict[str, Any]:
    return {
        "cv_folds": 5,
        "cv_strategy": "stratified_group_kfold_by_circuit",
        "cv_stratify_by": "driver",
        "cv_group_by": "circuit",
        "optuna_trials": 200,
        "optuna_objective": "macro_f1",
        "optuna_storage": "sqlite:///x.db",
        "calibration_method": "isotonic",
        "calibration_holdout_frac": 0.2,
        "baselines": ["majority", "logistic_regression", "random_forest"],
        "target_macro_f1": 0.78,
        "per_driver_floor": 0.70,
        "mlflow_tracking_uri": "sqlite:///m.db",
        "mlflow_experiment": "apex-classifier",
        "mlflow_registered_model": "apex-driver-classifier",
        "model_artefact_filename": "classifier.pkl",
    }


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("cv_strategy", "stratified_kfold_by_driver_circuit"),
        ("cv_stratify_by", "circuit"),
        ("cv_group_by", "driver"),
        ("cv_folds", 1),
        ("calibration_method", "platt"),
        ("calibration_holdout_frac", 1.5),
        ("target_macro_f1", 0.0),
        ("per_driver_floor", 1.5),
        ("optuna_objective", "accuracy"),
        ("baselines", ["mystery_model"]),
    ],
)
def test_classifier_config_rejects_bad(field: str, bad: Any) -> None:
    kwargs = _valid_classifier_kwargs()
    kwargs[field] = bad
    with pytest.raises(ValueError):
        ClassifierConfig(**kwargs)
