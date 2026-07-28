"""F20 — tests for the §11 evaluation orchestrator.

Covers the offline mechanism (failsafe runs; the three data/artefact pillars skip), the classifier
harvest gating (above/below/mixed/None), the stability/degradation/ablation call+gate+finding
wiring (via injected fakes / tiny matrices — F15 pattern, no artefacts), the RQ verdict rules, the
content-hash stability + no-pollution + never-abort contracts, and the config validators. The real
numbers are ``@pytest.mark.artefacts``-gated (skipped offline, mirroring F15's rate tests).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml
from app.contracts.config import AppConfig, EvalConfig, load_config
from app.eval.evaluate import (
    PillarResult,
    _gate,
    _held_out_matrix,
    _pillar_ablation,
    _pillar_degradation,
    _pillar_stability,
    _rq_verdicts,
    evaluate_classifier,
    run_full_evaluation,
)
from app.ml.classifier import ClassifierTrainingResult

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config.yaml"
EVALUATE_SRC = Path(__file__).resolve().parents[2] / "app" / "eval" / "evaluate.py"


@pytest.fixture
def cfg() -> AppConfig:
    return load_config(REPO_CONFIG)


def _result(**over: Any) -> ClassifierTrainingResult:
    """A synthetic champion ClassifierTrainingResult above target (override to fail a metric)."""
    base: dict[str, Any] = dict(
        mlflow_run_id="r", model_uri="u", model_artefact_path="p",
        held_out_macro_f1=0.85,
        held_out_per_driver_f1={f"D{i}": 0.80 for i in range(8)},
        beats_all_baselines=True,
        normalisation_ablation_macro_f1=0.60,
        baseline_macro_f1={"majority": 0.2, "logistic_regression": 0.5, "random_forest": 0.6},
    )
    base.update(over)
    return ClassifierTrainingResult(**base)


# --- offline mechanism (no artefacts / matrix=None) ----------------------------------------------
async def test_offline_report_shape(cfg: AppConfig, tmp_path: Path) -> None:
    """AC: async, never raises, exactly 5 pillars + 5 RQVerdicts; failsafe runs (mechanism), the
    three data/artefact pillars skip; RQ5 answered, RQ1/RQ3/RQ4 INSUFFICIENT_DATA, RQ2 PARTIAL."""
    report = await run_full_evaluation(None, config=cfg, artefact_dir=tmp_path)
    assert [p.pillar for p in report.pillars] == [
        "classifier", "stability", "degradation", "failsafe", "ablation",
    ]
    assert [v.rq for v in report.rq_verdicts] == ["RQ1", "RQ2", "RQ3", "RQ4", "RQ5"]
    verdicts = {v.rq: v.verdict for v in report.rq_verdicts}
    assert verdicts["RQ2"] == "PARTIAL"
    assert verdicts["RQ5"] == "SUPPORTED"  # mechanism: all 7 §11.4 tests pass offline
    assert verdicts["RQ1"] == verdicts["RQ3"] == verdicts["RQ4"] == "INSUFFICIENT_DATA"

    failsafe = next(p for p in report.pillars if p.pillar == "failsafe")
    assert not failsafe.skipped
    assert failsafe.native["failsafe_report"]["total"] == 7
    for name in ("classifier", "stability", "degradation", "ablation"):
        pillar = next(p for p in report.pillars if p.pillar == name)
        assert pillar.skipped and pillar.skip_reason
    assert report.n_pillars == 5 and report.n_pillars_run == 1 and report.n_pillars_skipped == 4
    assert report.all_targets_met is False
    assert (tmp_path / "evaluation_report.json").is_file()


async def test_rate_extensions_skip_offline(cfg: AppConfig, tmp_path: Path) -> None:
    """The two §11.4 rate extensions are artefact-gated: offline → value=None (never fabricated)."""
    report = await run_full_evaluation(None, config=cfg, artefact_dir=tmp_path)
    failsafe = next(p for p in report.pillars if p.pillar == "failsafe")
    rate_metrics = {m.name: m for m in failsafe.metrics if m.name.startswith("test")}
    assert set(rate_metrics) == {"test1_low_confidence_rate", "test6_ood_5sd_detection"}
    for m in rate_metrics.values():
        assert m.value is None and m.passes is None and m.finding


# --- §11.1 classifier harvest (above / below / mixed / None) -------------------------------------
def test_classifier_harvest_above_target(cfg: AppConfig) -> None:
    pillar = evaluate_classifier(_result(), config=cfg)
    assert not pillar.skipped
    assert all(m.passes for m in pillar.metrics)
    names = {m.name for m in pillar.metrics}
    assert {"held_out_macro_f1", "held_out_per_driver_f1_min", "circuit_generalisation_f1_min",
            "beats_all_baselines", "normalisation_ablation_drop"} <= names


def test_classifier_reads_config_targets_not_magic_numbers(cfg: AppConfig) -> None:
    """The macro-F1 gate must read classifier.target_macro_f1 (0.78) from config, not a hardcode."""
    macro = next(
        m for m in evaluate_classifier(_result(held_out_macro_f1=0.79), config=cfg).metrics
        if m.name == "held_out_macro_f1"
    )
    assert macro.target == cfg.classifier.target_macro_f1 == 0.78  # type: ignore[union-attr]
    assert macro.passes is True  # 0.79 > 0.78


def test_classifier_harvest_below_target(cfg: AppConfig) -> None:
    bad = _result(
        held_out_macro_f1=0.50,
        held_out_per_driver_f1={f"D{i}": 0.40 for i in range(8)},
        beats_all_baselines=False,
        normalisation_ablation_macro_f1=0.60,  # 0.60 < 0.50 is False → no drop → fail
    )
    pillar = evaluate_classifier(bad, config=cfg)
    assert pillar.metrics[0].passes is False
    verdict = _rq1(pillar)
    assert verdict == "NOT_SUPPORTED"


def test_classifier_harvest_mixed_is_partial(cfg: AppConfig) -> None:
    mixed = _result(
        held_out_macro_f1=0.75,  # below 0.78 → fail
        held_out_per_driver_f1={f"D{i}": 0.80 for i in range(8)},  # pass
        normalisation_ablation_macro_f1=0.50,  # 0.50 < 0.75 → drop present → pass
    )
    assert _rq1(evaluate_classifier(mixed, config=cfg)) == "PARTIAL"


def test_classifier_none_skips(cfg: AppConfig) -> None:
    pillar = evaluate_classifier(None, config=cfg)
    assert pillar.skipped and pillar.skip_reason and not pillar.metrics


async def test_injected_classifier_result_drives_rq1(cfg: AppConfig, tmp_path: Path) -> None:
    report = await run_full_evaluation(
        None, config=cfg, artefact_dir=tmp_path, classifier_result=_result(),
    )
    rq1 = next(v for v in report.rq_verdicts if v.rq == "RQ1")
    assert rq1.verdict == "SUPPORTED"


_OTHER_PILLARS = ("stability", "degradation", "failsafe", "ablation")
_ALL_PILLARS = ("classifier", *_OTHER_PILLARS)


def _pillar(name: str, *, skipped: bool = False) -> PillarResult:
    return PillarResult(pillar=name, skipped=skipped, skip_reason="x" if skipped else None)  # type: ignore[arg-type]


def _rq1(classifier_pillar: PillarResult) -> str:
    """Helper: the RQ1 verdict given only a classifier pillar (others skipped)."""
    pillars = [classifier_pillar] + [_pillar(p, skipped=True) for p in _OTHER_PILLARS]
    return next(v.verdict for v in _rq_verdicts(pillars) if v.rq == "RQ1")


# --- §11.2 stability / §11.3 degradation / F09 ablation wiring (injected fakes) ------------------
def _fake_handle(drivers: list[str]) -> Any:
    class _H:
        pass

    h = _H()
    h.drivers = drivers  # type: ignore[attr-defined]
    return h


def _tiny_matrix(drivers: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {"driver": drivers, "circuit": ["Bahrain"] * len(drivers), "F01": [0.5] * len(drivers)}
    )


def test_stability_pillar_wiring(cfg: AppConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    import app.ml.stylespace as ss

    drivers = ["VER", "HAM"]
    monkeypatch.setattr(
        ss, "stability_report",
        lambda m, **k: ss.StabilityReport(procrustes_variance=0.02, passes=True),
    )
    monkeypatch.setattr(
        ss, "teammate_proximity",
        lambda h, t, **k: ss.TeammateProximityResult(
            mean_intra_team=1.0, mean_inter_team=2.0, passes=True,
        ),
    )
    monkeypatch.setattr(
        ss, "project_unknown",
        lambda laps, **k: ss.ProjectionResult(
            nearest_neighbours=[ss.Neighbour(
                driver=str(laps["driver"].iloc[0]), cosine_similarity=0.9, distance=0.1,
            )],
        ),
    )
    m = _tiny_matrix(drivers)
    pillar = _pillar_stability(
        _fake_handle(drivers), m, m, {"VER": "RB", "HAM": "MER"}, cfg,
        classifier_result=None,
    )
    metrics = {mr.name: mr for mr in pillar.metrics}
    assert metrics["umap_procrustes_variance"].passes is True
    assert metrics["teammate_proximity"].passes is True
    # both drivers re-project to themselves → 2/2 ≥ target is honoured (target is 6, so fails, but
    # the metric is COMPUTED, not skipped — the wiring is what we assert here)
    assert metrics["unknown_driver_nn_correct"].value == 2


def test_degradation_pillar_wiring(cfg: AppConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    import app.monitoring.detector as det

    monkeypatch.setattr(
        det, "lead_time_report",
        lambda m, **k: det.LeadTimeReport(
            n_stints=5, mean_lead_time_laps=3.0, tp_rate=0.80, fp_rate=0.10,
        ),
    )
    # held_out slice is passed positionally (the tiny matrix stands in for the held-out rows)
    pillar = _pillar_degradation(_fake_handle([]), _tiny_matrix(["VER"]), cfg)
    metrics = {m.name: m for m in pillar.metrics}
    assert metrics["mean_lead_time_laps"].passes is True  # 3 in [2, 5]
    assert metrics["true_positive_rate"].passes is True  # 0.80 ≥ 0.70
    assert metrics["false_positive_rate"].passes is True  # 0.10 ≤ 0.20


def test_ablation_pillar_wiring(cfg: AppConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    import app.ml.corners as corners

    monkeypatch.setattr(
        corners, "run_ablation",
        lambda frame, **k: corners.AblationReport(
            base_macro_f1=0.70, expanded_macro_f1=0.72, delta=0.02, finding="IMPROVES",
        ),
    )
    pillar = _pillar_ablation(_tiny_matrix(["VER"]), cfg, classifier_result=_result())
    # harvested normalisation-ablation metric present + the IMPROVES finding surfaced
    assert any(m.name == "normalisation_ablation_macro_f1" for m in pillar.metrics)
    assert any("IMPROVES" in f for f in pillar.findings)


def test_degradation_skips_without_matrix(cfg: AppConfig) -> None:
    pillar = _pillar_degradation(_fake_handle([]), None, cfg)
    assert pillar.skipped and "matrix" in (pillar.skip_reason or "")


# --- RQ verdict rules ----------------------------------------------------------------------------
def test_rq2_always_partial() -> None:
    verdicts = _rq_verdicts([_pillar(p) for p in _ALL_PILLARS])  # not skipped, no metrics
    rq2 = next(v for v in verdicts if v.rq == "RQ2")
    assert rq2.verdict == "PARTIAL" and rq2.findings  # under-instrumentation finding present


def test_rq_skipped_evidence_is_insufficient_data() -> None:
    verdicts = {
        v.rq: v.verdict
        for v in _rq_verdicts([_pillar(p, skipped=True) for p in _ALL_PILLARS])
    }
    rqs = (verdicts["RQ1"], verdicts["RQ3"], verdicts["RQ4"], verdicts["RQ5"])
    assert all(v == "INSUFFICIENT_DATA" for v in rqs)


def test_rq3_rq4_supported_when_targets_met() -> None:
    degradation = PillarResult(pillar="degradation", metrics=[
        _gate("mean_lead_time_laps", 3.0, [2.0, 5.0], "in_range"),
        _gate("true_positive_rate", 0.8, 0.7, ">="),
        _gate("false_positive_rate", 0.1, 0.2, "<="),
    ])
    stability = PillarResult(pillar="stability", metrics=[
        _gate("unknown_driver_nn_correct", 7, 6, ">="),
        _gate("umap_procrustes_variance", 0.02, 0.05, "<="),
    ])
    verdicts = {v.rq: v.verdict for v in _rq_verdicts([
        PillarResult(pillar="classifier"), stability, degradation,
        PillarResult(pillar="failsafe"), PillarResult(pillar="ablation"),
    ])}
    assert verdicts["RQ3"] == "SUPPORTED"
    assert verdicts["RQ4"] == "SUPPORTED"


# --- report: hash stability, no pollution, never-abort, render, value=None -----------------------
async def test_content_hash_stable_across_runs(cfg: AppConfig, tmp_path: Path) -> None:
    r1 = await run_full_evaluation(None, config=cfg, artefact_dir=tmp_path)
    r2 = await run_full_evaluation(None, config=cfg, artefact_dir=tmp_path)
    assert r1.content_hash == r2.content_hash and r1.content_hash


async def test_run_leaves_real_cache_and_dq_log_untouched(cfg: AppConfig, tmp_path: Path) -> None:
    cfg.paths.cache_dir = str(tmp_path / "real_cache")
    cfg.paths.data_quality_log = str(tmp_path / "real_dq.jsonl")
    await run_full_evaluation(None, config=cfg, artefact_dir=tmp_path / "art")
    assert not (tmp_path / "real_cache").exists()
    assert not (tmp_path / "real_dq.jsonl").exists()


async def test_pillar_fault_skips_not_aborts(
    cfg: AppConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pillar-internal fault → skipped + finding, and the run still returns 5 pillars (§4.0-g)."""
    import app.eval.evaluate as ev

    def boom(*a: Any, **k: Any) -> PillarResult:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(ev, "evaluate_classifier", boom)
    report = await run_full_evaluation(None, config=cfg, artefact_dir=tmp_path)
    assert len(report.pillars) == 5
    classifier = next(p for p in report.pillars if p.pillar == "classifier")
    assert classifier.skipped and any("kaboom" in f for f in classifier.findings)


async def test_render_report_has_all_tables(cfg: AppConfig, tmp_path: Path) -> None:
    report = await run_full_evaluation(None, config=cfg, artefact_dir=tmp_path)
    md = report.render_report()
    for token in ("Classifier pillar", "Stability pillar", "Degradation pillar",
                  "Failsafe pillar", "Ablation pillar", "Research questions",
                  "Failures & skips roll-up", "RQ5"):
        assert token in md, token


def test_gate_nan_and_none_do_not_fabricate_a_pass() -> None:
    assert _gate("m", None, 0.5, ">").passes is None
    assert _gate("m", float("nan"), 0.5, ">").passes is None  # all-NaN metric → not a pass
    assert _gate("m", 0.9, 0.5, ">").passes is True
    assert _gate("m", 0.1, 0.5, ">").passes is False


def test_gate_nan_or_none_target_is_skip_not_fabricated_fail() -> None:
    """A NaN target (e.g. a harvested held_out_macro_f1 from an empty test split) or an absent
    config target must yield passes=None — never a spurious FAIL (IEEE-754 `x < nan` is False)."""
    assert _gate("ablation_drop", 0.60, float("nan"), "<").passes is None
    assert _gate("year_on_year_js", 0.10, None, "<").passes is None


def test_classifier_nan_held_out_macro_f1_does_not_fabricate_ablation_fail(cfg: AppConfig) -> None:
    """Finding B: when held_out_macro_f1 is NaN (empty test split) it is used as the ablation-drop
    TARGET — the metric must skip (passes=None), not report a fabricated FAIL."""
    pillar = evaluate_classifier(_result(held_out_macro_f1=float("nan")), config=cfg)
    drop = next(m for m in pillar.metrics if m.name == "normalisation_ablation_drop")
    assert drop.passes is None


# --- §4.0-h: held-out (Belgium/Italy) isolation — never silently score on training data ----------
def test_held_out_matrix_isolates_belgium_italy(cfg: AppConfig) -> None:
    m = pd.DataFrame({
        "driver": ["VER", "HAM", "LEC"],
        "circuit": ["Bahrain", "Belgium", "Italy"],
        "F01": [0.1, 0.2, 0.3],
    })
    sliced, reason = _held_out_matrix(m, cfg)
    assert reason is None and sliced is not None
    assert set(sliced["circuit"]) == {"Belgium", "Italy"}  # training circuit excluded


def test_matrix_with_no_held_out_rows_yields_reason(cfg: AppConfig) -> None:
    m = pd.DataFrame({"driver": ["VER"], "circuit": ["Bahrain"], "F01": [0.5]})  # no Belgium/Italy
    sliced, reason = _held_out_matrix(m, cfg)
    assert sliced is None and reason and "held-out" in reason


def test_matrix_without_circuit_or_split_column_yields_reason(cfg: AppConfig) -> None:
    sliced, reason = _held_out_matrix(pd.DataFrame({"driver": ["VER"], "F01": [0.5]}), cfg)
    assert sliced is None and reason and "circuit" in reason


def test_degradation_skips_when_matrix_has_no_held_out_rows(
    cfg: AppConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A detector handle IS present but the matrix has no held-out rows → degradation skips with the
    held-out reason (never runs lead_time_report on training data). Finding A."""
    import app.monitoring.detector as det

    called = {"n": 0}

    def _boom(*a: Any, **k: Any) -> Any:
        called["n"] += 1
        raise AssertionError("lead_time_report must NOT be called on a no-held-out matrix")

    monkeypatch.setattr(det, "lead_time_report", _boom)
    _, reason = _held_out_matrix(
        pd.DataFrame({"driver": ["VER"], "circuit": ["Bahrain"], "F01": [0.5]}), cfg,
    )
    pillar = _pillar_degradation(_fake_handle([]), None, cfg, held_out_reason=reason)
    assert pillar.skipped and "held-out" in (pillar.skip_reason or "")
    assert called["n"] == 0


async def test_full_run_with_no_held_out_matrix_skips_data_pillars(
    cfg: AppConfig, tmp_path: Path,
) -> None:
    """End-to-end: a non-None matrix with only training circuits → the data pillars skip (they never
    score on training data), while the failsafe mechanism still runs. Finding A, orchestrated."""
    m = pd.DataFrame(
        {"driver": ["VER", "HAM"], "circuit": ["Bahrain", "Bahrain"], "F01": [0.4, 0.6]}
    )
    report = await run_full_evaluation(m, config=cfg, artefact_dir=tmp_path)
    degradation = next(p for p in report.pillars if p.pillar == "degradation")
    assert degradation.skipped
    assert next(p for p in report.pillars if p.pillar == "failsafe").skipped is False


# --- §4.0-g: run_full_evaluation never raises, even if persistence fails -------------------------
async def test_run_never_raises_on_report_write_failure(cfg: AppConfig, tmp_path: Path) -> None:
    """If the report write fails (artefact_dir collides with a file), the run must still return a
    typed report rather than raise — the report OBJECT is the deliverable (§4.0-g). Finding C."""
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file, not a directory", encoding="utf-8")
    # artefact_dir points INSIDE a regular file → mkdir/write raises OSError inside _write_report
    report = await run_full_evaluation(None, config=cfg, artefact_dir=blocker / "sub")
    assert len(report.pillars) == 5  # returned despite the write failure
    assert not (blocker / "sub" / "evaluation_report.json").exists()


# --- config validators ---------------------------------------------------------------------------
def test_evalconfig_rejects_bad_lead_time_order() -> None:
    with pytest.raises(ValueError, match="lead_time_min_laps"):
        EvalConfig(
            stability_js_target=0.15, unknown_nn_correct_min=6, lead_time_min_laps=5,
            lead_time_max_laps=2, tp_rate_target=0.7, fp_rate_max=0.2, report_filename="x.json",
        )


def test_evalconfig_rejects_tp_rate_over_one() -> None:
    with pytest.raises(ValueError, match="tp_rate_target"):
        EvalConfig(
            stability_js_target=0.15, unknown_nn_correct_min=6, lead_time_min_laps=2,
            lead_time_max_laps=5, tp_rate_target=1.5, fp_rate_max=0.2, report_filename="x.json",
        )


def test_eval_block_requires_classifier_sibling() -> None:
    raw = yaml.safe_load(REPO_CONFIG.read_text(encoding="utf-8"))
    raw.pop("classifier")
    with pytest.raises(ValueError, match="eval:"):
        AppConfig(**raw)


# --- no re-implementation of metric maths (§4.0-a / §4.0-i) --------------------------------------
def test_no_fit_or_metric_maths_in_evaluate() -> None:
    src = EVALUATE_SRC.read_text(encoding="utf-8")
    assert ".fit(" not in src  # F20 refits nothing
    assert "groupby" not in src  # no aggregation maths — it harvests/calls only


# --- real numbers: artefact-gated (skipped offline, mirrors F15's rate tests) --------------------
@pytest.mark.artefacts
async def test_real_full_run(cfg: AppConfig, tmp_path: Path) -> None:
    """The real §11 numbers require the committed champion / umap / detector / normalisation
    artefacts AND a held-out matrix — none are committed, so this skips offline (the honest deferred
    state, divergence #54). When present, it asserts real macro-F1 vs 0.78, Procrustes var vs 0.05,
    lead time ∈ [2, 5], and the two §11.4 rate extensions."""
    assert cfg.classifier is not None and cfg.stylespace is not None
    art = Path(cfg.paths.artefacts_dir)
    champion = art / cfg.classifier.model_artefact_filename
    umap = art / cfg.stylespace.umap_model_filename
    if not (champion.is_file() and umap.is_file()):
        pytest.skip("model artefacts absent — real §11 numbers need the committed artefacts (F20)")
    raise AssertionError(  # pragma: no cover - only reached with committed artefacts + a matrix
        "real full-run assertions need the held-out evaluation matrix (supply it here when built)"
    )
