"""F07 SHAP-explainability tests — one per acceptance criterion + per failure mode.

The happy end-to-end path is exercised against the REAL F06a champion (the module-scoped
``trained`` fixture → ``load_model`` → ``build_explainer``). Gate/consistency boundaries are pinned
with a ``StubExplainer`` (caller-supplied SHAP values + margin) so the exact additivity delta and
domain scores are deterministic, independent of the real model's output.
"""

from __future__ import annotations

import io
import json
import tokenize
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from app.contracts.config import AppConfig, ShapConfig, load_config
from app.contracts.enums import Status
from app.contracts.errors import ErrorCode
from app.ml import explain as xai
from app.ml.explain import (
    DOMAIN_FEATURES,
    FEATURE_NAMES,
    RADAR_DOMAINS,
    ShapExplainer,
    aggregate_domains,
    build_explainer,
    build_global_explainability,
    describe_style,
    explain_prediction,
    season_stability,
)
from app.ml.inference import load_model
from app.pipeline.features import FEATURE_IDS

if TYPE_CHECKING:
    from collections.abc import Callable

SOURCE = Path(xai.__file__).read_text(encoding="utf-8")
REPO_CONFIG = Path(__file__).resolve().parents[2] / "config.yaml"


def _code_only(src: str) -> str:
    """Strip strings + comments so a static scan sees only executable tokens (F06b convention)."""
    out: list[str] = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type not in (tokenize.STRING, tokenize.COMMENT):
            out.append(tok.string)
    return " ".join(out)


CODE_ONLY = _code_only(SOURCE)


def _cfg(tmp_path: Path) -> AppConfig:
    cfg = load_config(REPO_CONFIG)
    cfg.paths.data_quality_log = str(tmp_path / "logs" / "dq.jsonl")
    return cfg


# --------------------------------------------------------------------------------------------
# Stub explainer for deterministic boundary tests
# --------------------------------------------------------------------------------------------


class _Expl:
    def __init__(self, values: np.ndarray, base: np.ndarray) -> None:
        self.values = values
        self.base_values = base


class _CallableExpl:
    def __init__(self, values: np.ndarray, base: np.ndarray, *, raises: bool = False) -> None:
        self._v = values
        self._b = base
        self.raises = raises

    def __call__(self, x: np.ndarray) -> _Expl:
        if self.raises:
            raise RuntimeError("stub SHAP boom")
        return _Expl(self._v, self._b)


class _StubBase:
    def __init__(self, margins: np.ndarray) -> None:
        self._m = margins

    def predict(self, x: np.ndarray, output_margin: bool = False) -> np.ndarray:
        return self._m


def _stub_explainer(
    *,
    drivers: list[str],
    k: int,
    shap_vec: np.ndarray,
    base_val: float,
    margin: float | None = None,
    raises: bool = False,
) -> ShapExplainer:
    """A ShapExplainer whose predicted-class-k SHAP row and base tree margin are caller-set, so the
    additivity delta is exact. ``margin=None`` makes it consistent (base+Σshap); a supplied margin
    injects an additivity mismatch."""
    kk = len(drivers)
    values = np.zeros((1, len(FEATURE_IDS), kk), dtype=float)
    values[0, :, k] = shap_vec
    base = np.zeros((1, kk), dtype=float)
    base[0, k] = base_val
    m = base_val + float(shap_vec.sum()) if margin is None else margin
    margins = np.zeros((1, kk), dtype=float)
    margins[0, k] = m
    # explain_prediction reads the base from the stored base_values[predicted_class]; set class k's.
    return ShapExplainer(
        explainer=_CallableExpl(values, base, raises=raises),
        base_model=_StubBase(margins),
        base_values={d: (base_val if i == k else 0.0) for i, d in enumerate(drivers)},
        drivers=drivers,
        ready=True,
    )


# --------------------------------------------------------------------------------------------
# 1. Exact tree SHAP over the extracted BASE XGBClassifier (not the calibrated wrapper)
# --------------------------------------------------------------------------------------------


def test_build_explainer_uses_base_xgbclassifier(trained: tuple[Any, AppConfig]) -> None:
    import xgboost as xgb

    _result, cfg = trained
    handle = load_model(config=cfg, artefact_dir=Path(cfg.paths.artefacts_dir))
    ex = build_explainer(handle, config=cfg, log_path=cfg.paths.data_quality_log)
    assert ex.ready is True
    assert isinstance(ex.base_model, xgb.XGBClassifier)  # not the calibrated wrapper
    assert set(ex.base_values) == set(handle.drivers)


# --------------------------------------------------------------------------------------------
# 2. 28-feature local SHAP + ordered force with toward/away directions (happy, real champion)
# --------------------------------------------------------------------------------------------


def test_explain_prediction_happy_real_champion(
    trained: tuple[Any, AppConfig], clean_vector: Callable[[], dict[str, float]]
) -> None:
    _result, cfg = trained
    handle = load_model(config=cfg, artefact_dir=Path(cfg.paths.artefacts_dir))
    ex = build_explainer(handle, config=cfg, log_path=cfg.paths.data_quality_log)
    predicted = handle.drivers[0]
    r = explain_prediction(
        clean_vector(), circuit="Bahrain", predicted_class=predicted, explainer=ex,
        handle=handle, config=cfg,
    )
    assert r.shap_available is True
    assert r.shap_values is not None and set(r.shap_values) == set(FEATURE_IDS)
    assert len(r.force) == len(FEATURE_IDS)
    # force ordered by |shap| desc
    mags = [abs(c.shap) for c in r.force]
    assert mags == sorted(mags, reverse=True)
    for c in r.force:
        assert c.direction == ("toward" if c.shap > 0 else "away")
        assert c.name == FEATURE_NAMES[c.feature_id]
    assert set(r.radar) == set(RADAR_DOMAINS)
    assert r.nl_description
    assert r.consistency_ok is True  # base-tree additivity holds exactly
    assert r.status is Status.OK


# --------------------------------------------------------------------------------------------
# 3. SHAP raises -> SHAP_UNAVAILABLE, no exception, prediction preserved (§6.2 FM1)
# --------------------------------------------------------------------------------------------


def test_shap_raises_degrades_never_blocks(
    make_handle: Callable[..., Any], clean_vector: Callable[[], dict[str, float]], tmp_path: Path
) -> None:
    cfg = _cfg(tmp_path)
    handle = make_handle([0.7, 0.2, 0.1], drivers=["VER", "HAM", "LEC"])
    ex = _stub_explainer(
        drivers=["VER", "HAM", "LEC"], k=0, shap_vec=np.ones(len(FEATURE_IDS)), base_val=0.0,
        raises=True,
    )
    r = explain_prediction(
        clean_vector(), circuit="Bahrain", predicted_class="VER", explainer=ex, handle=handle,
        config=cfg,
    )
    assert r.error_code is ErrorCode.SHAP_UNAVAILABLE
    assert r.status is Status.DEGRADED
    assert r.shap_values is None and r.shap_available is False
    assert r.force == [] and r.radar == {} and r.nl_description == ""
    assert r.logged is True


# --------------------------------------------------------------------------------------------
# 4. Not-ready explainer degrades cleanly (build failure)
# --------------------------------------------------------------------------------------------


def test_build_explainer_not_ready_on_bad_champion(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)

    class _BadHandle:
        model = object()  # no calibrated_classifiers_
        drivers = ["A", "B"]
        normalisation_params = None

    ex = build_explainer(_BadHandle(), config=cfg, log_path=cfg.paths.data_quality_log)
    assert ex.ready is False
    rows = [json.loads(x) for x in Path(cfg.paths.data_quality_log).read_text().splitlines() if x]
    assert any(row["error_code"] == ErrorCode.SHAP_UNAVAILABLE for row in rows)


def test_not_ready_explainer_degrades(
    make_handle: Callable[..., Any], clean_vector: Callable[[], dict[str, float]], tmp_path: Path
) -> None:
    cfg = _cfg(tmp_path)
    handle = make_handle([0.7, 0.3], drivers=["VER", "HAM"])
    ex = ShapExplainer(explainer=None, base_model=None, drivers=["VER", "HAM"], ready=False)
    r = explain_prediction(
        clean_vector(), circuit="Bahrain", predicted_class="VER", explainer=ex, handle=handle,
        config=cfg,
    )
    assert r.error_code is ErrorCode.SHAP_UNAVAILABLE and r.shap_available is False


def test_unknown_predicted_class_degrades(
    make_handle: Callable[..., Any], clean_vector: Callable[[], dict[str, float]], tmp_path: Path
) -> None:
    cfg = _cfg(tmp_path)
    handle = make_handle([0.7, 0.3], drivers=["VER", "HAM"])
    ex = _stub_explainer(
        drivers=["VER", "HAM"], k=0, shap_vec=np.ones(len(FEATURE_IDS)), base_val=0.0
    )
    r = explain_prediction(
        clean_vector(), circuit="Bahrain", predicted_class="ZZZ", explainer=ex, handle=handle,
        config=cfg,
    )
    assert r.error_code is ErrorCode.SHAP_UNAVAILABLE and r.shap_available is False


# --------------------------------------------------------------------------------------------
# 5. Consistency (additivity) check both sides of tolerance (§6.2 FM2)
# --------------------------------------------------------------------------------------------


def test_consistency_within_tolerance(
    make_handle: Callable[..., Any], clean_vector: Callable[[], dict[str, float]], tmp_path: Path
) -> None:
    cfg = _cfg(tmp_path)
    handle = make_handle([0.7, 0.3], drivers=["VER", "HAM"])
    shap_vec = np.linspace(-1.0, 1.0, len(FEATURE_IDS))
    ex = _stub_explainer(drivers=["VER", "HAM"], k=0, shap_vec=shap_vec, base_val=0.5)
    r = explain_prediction(
        clean_vector(), circuit="Bahrain", predicted_class="VER", explainer=ex, handle=handle,
        config=cfg,
    )
    assert r.consistency_ok is True
    assert ErrorCode.SHAP_CONSISTENCY_WARN not in r.flags
    assert r.shap_available is True and r.status is Status.OK


def test_consistency_warn_returns_shap_with_flag(
    make_handle: Callable[..., Any], clean_vector: Callable[[], dict[str, float]], read_dq: Any,
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    assert cfg.shap is not None
    handle = make_handle([0.7, 0.3], drivers=["VER", "HAM"])
    shap_vec = np.ones(len(FEATURE_IDS))
    # inject a margin off by 10x the tolerance -> additivity mismatch
    bad_margin = float(shap_vec.sum()) + 0.5 + 10 * cfg.shap.consistency_tolerance
    ex = _stub_explainer(
        drivers=["VER", "HAM"], k=0, shap_vec=shap_vec, base_val=0.5, margin=bad_margin
    )
    r = explain_prediction(
        clean_vector(), circuit="Bahrain", predicted_class="VER", explainer=ex, handle=handle,
        config=cfg,
    )
    assert r.consistency_ok is False
    assert ErrorCode.SHAP_CONSISTENCY_WARN in r.flags
    assert r.status is Status.DEGRADED
    assert r.shap_available is True and r.shap_values is not None  # values STILL returned (§6.2)
    dq = read_dq(cfg.paths.data_quality_log)
    assert any(row["error_code"] == ErrorCode.SHAP_CONSISTENCY_WARN for row in dq)
    assert r.logged is True


# --------------------------------------------------------------------------------------------
# 6-7. Radar over the 6 named domains + DOMAIN_FEATURES partition integrity
# --------------------------------------------------------------------------------------------


def test_aggregate_domains_six_axes() -> None:
    sv = {fid: 0.0 for fid in FEATURE_IDS}
    sv["F06"], sv["F07"] = 2.0, -1.5  # braking; abs summed
    sv["F22"], sv["F28"] = 9.0, 9.0  # interaction / degradation -> NO radar axis
    radar = aggregate_domains(sv)
    assert set(radar) == set(RADAR_DOMAINS)
    assert radar["braking"] == pytest.approx(3.5)
    assert sum(radar.values()) == pytest.approx(3.5)  # F22/F28 contribute to no axis


def test_domain_features_partition_f01_f21() -> None:
    seen = [fid for ids in DOMAIN_FEATURES.values() for fid in ids]
    assert tuple(sorted(seen)) == tuple(f"F{n:02d}" for n in range(1, 22))
    assert len(seen) == len(set(seen)) == 21
    # F22..F28 must be in NO domain
    for n in range(22, 29):
        assert all(f"F{n:02d}" not in ids for ids in DOMAIN_FEATURES.values())


# --------------------------------------------------------------------------------------------
# 8-9. NL top-2 vs generic fallback (§6.2 FM3)
# --------------------------------------------------------------------------------------------


def test_describe_style_top_two(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    radar = {
        "throttle": 0.1, "braking": 5.0, "speed": 0.2, "gear": 0.1, "engine": 0.1, "cornering": 3.0
    }
    style = describe_style(radar, config=cfg)
    assert style.generic is False
    assert style.dominant_domains == ["braking", "cornering"]
    assert "braking" in style.sentence and "cornering" in style.sentence


def test_describe_style_generic_when_no_dominant(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    radar = {d: 1.0 for d in RADAR_DOMAINS}  # all equal -> within 10%
    style = describe_style(radar, config=cfg)
    assert style.generic is True
    assert style.dominant_domains == []
    assert style.sentence == xai._GENERIC_STYLE  # verbatim; never fabricate a trait


def test_describe_style_generic_when_all_zero(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    style = describe_style({d: 0.0 for d in RADAR_DOMAINS}, config=cfg)
    assert style.generic is True and style.dominant_domains == []


# --------------------------------------------------------------------------------------------
# 10-12. Global beeswarm artefact + per-driver shap/radar; never raises; held-out contract
# --------------------------------------------------------------------------------------------


def _background(drivers: list[str], *, per: int = 6, seed: int = 1) -> Any:
    import pandas as pd

    rng = np.random.default_rng(seed)
    rows = []
    off = {d: i * 3.0 for i, d in enumerate(drivers)}
    for d in drivers:
        for _ in range(per):
            vec = {fid: float(rng.normal(off[d], 1.0)) for fid in FEATURE_IDS}
            rows.append({"driver": d, **vec})
    return pd.DataFrame(rows)


def test_global_beeswarm_deterministic(trained: tuple[Any, AppConfig], tmp_path: Path) -> None:
    _result, cfg = trained
    handle = load_model(config=cfg, artefact_dir=Path(cfg.paths.artefacts_dir))
    ex = build_explainer(handle, config=cfg, log_path=str(tmp_path / "dq.jsonl"))
    bg = _background(handle.drivers)
    dq = str(tmp_path / "dq.jsonl")
    out1 = build_global_explainability(
        ex, bg, handle.drivers, config=cfg, artefact_dir=tmp_path / "a", log_path=dq
    )
    out2 = build_global_explainability(
        ex, bg, handle.drivers, config=cfg, artefact_dir=tmp_path / "b", log_path=dq
    )
    assert out1.status is Status.OK and out1.shap_available is True
    assert set(out1.per_driver_shap) == set(handle.drivers)
    assert all(set(v) == set(FEATURE_IDS) for v in out1.per_driver_shap.values())
    assert all(set(v) == set(RADAR_DOMAINS) for v in out1.per_driver_radar.values())
    assert out1.n_background == len(bg)
    a = Path(out1.beeswarm_artefact_path).read_bytes()
    b = Path(out2.beeswarm_artefact_path).read_bytes()
    assert a == b  # byte-identical on re-run for the same background
    payload = json.loads(a)
    assert payload["features"] == list(FEATURE_IDS)


def test_global_never_raises_on_shap_fault(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    ex = ShapExplainer(explainer=None, base_model=None, drivers=["VER", "HAM"], ready=False)
    out = build_global_explainability(
        ex, _background(["VER", "HAM"]), ["VER", "HAM"], config=cfg, artefact_dir=tmp_path
    )
    assert out.status is Status.DEGRADED and out.error_code is ErrorCode.SHAP_UNAVAILABLE
    assert out.per_driver_shap == {} and out.logged is True


def test_global_background_excludes_held_out(
    trained: tuple[Any, AppConfig], training_matrix: Callable[..., Any]
) -> None:
    # Caller contract (spec §7 crit 12): the beeswarm background is the F06a training-fit split, so
    # Belgium/Italy never reach F07. Prove the caller's split filter is non-vacuous: the held-out
    # circuits DO exist in the full matrix, and are ALL gone after filtering to split=="train".
    _result, cfg = trained
    full = training_matrix()  # has a 'circuit' column incl. Belgium/Italy in the held-out split
    held = set(cfg.held_out_circuits)
    assert held & set(full["circuit"])  # guard: held-out circuits are present in the full matrix
    train_bg = full[full["split"] == "train"]
    assert not held & set(train_bg["circuit"])  # ...and removed by the split filter before F07


def test_global_batch_perf_warn(
    trained: tuple[Any, AppConfig], monkeypatch: pytest.MonkeyPatch, read_dq: Any, tmp_path: Path
) -> None:
    # §7: batch SHAP over batch_warn_ms -> PERF_WARN (flag + logged), status stays OK.
    _result, cfg = trained
    assert cfg.shap is not None
    handle = load_model(config=cfg, artefact_dir=Path(cfg.paths.artefacts_dir))
    ex = build_explainer(handle, config=cfg, log_path=str(tmp_path / "dq.jsonl"))
    dq_log = str(tmp_path / "dq.jsonl")
    ticks = iter([0.0, (cfg.shap.batch_warn_ms + 1.0) / 1000.0])
    monkeypatch.setattr(xai.time, "perf_counter", lambda: next(ticks))
    out = build_global_explainability(
        ex, _background(handle.drivers), handle.drivers, config=cfg, artefact_dir=tmp_path,
        log_path=dq_log,
    )
    assert ErrorCode.PERF_WARN in out.flags and out.status is Status.OK
    assert any(row["error_code"] == ErrorCode.PERF_WARN for row in read_dq(dq_log))


def test_dq_unwritable_surfaced(
    make_handle: Callable[..., Any], clean_vector: Callable[[], dict[str, float]], tmp_path: Path
) -> None:
    # An unwritable DQ sink must surface DATA_QUALITY_LOG_UNWRITABLE in flags, never crash.
    cfg = _cfg(tmp_path)
    # point the log at a path whose parent is a FILE, so every append raises DataQualityLogError
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    cfg.paths.data_quality_log = str(blocker / "dq.jsonl")
    handle = make_handle([0.7, 0.3], drivers=["VER", "HAM"])
    ex = _stub_explainer(
        drivers=["VER", "HAM"], k=0, shap_vec=np.ones(len(FEATURE_IDS)), base_val=0.0, raises=True
    )
    r = explain_prediction(
        clean_vector(), circuit="Bahrain", predicted_class="VER", explainer=ex, handle=handle,
        config=cfg,
    )
    assert ErrorCode.DATA_QUALITY_LOG_UNWRITABLE in r.flags  # sink failure surfaced, not swallowed


# --------------------------------------------------------------------------------------------
# 13. Static scan: no fit / no retrain in the module
# --------------------------------------------------------------------------------------------


def test_no_fit_or_training_symbols() -> None:
    for banned in (".fit(", "fit_normaliser", "optuna", "CalibratedClassifierCV("):
        assert banned not in CODE_ONLY, f"F07 must not contain {banned!r} (read-only)"


# --------------------------------------------------------------------------------------------
# 14. Latency flags (single-lap) both sides; latency alone does not degrade status
# --------------------------------------------------------------------------------------------


def test_latency_perf_warn(
    make_handle: Callable[..., Any], clean_vector: Callable[[], dict[str, float]],
    monkeypatch: pytest.MonkeyPatch, read_dq: Any, tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    assert cfg.shap is not None
    handle = make_handle([0.7, 0.3], drivers=["VER", "HAM"])
    ex = _stub_explainer(
        drivers=["VER", "HAM"], k=0, shap_vec=np.ones(len(FEATURE_IDS)), base_val=0.0
    )

    ticks = iter([0.0, (cfg.shap.single_lap_warn_ms + 1.0) / 1000.0])
    monkeypatch.setattr(xai.time, "perf_counter", lambda: next(ticks))
    r = explain_prediction(
        clean_vector(), circuit="Bahrain", predicted_class="VER", explainer=ex, handle=handle,
        config=cfg,
    )
    assert ErrorCode.PERF_WARN in r.flags
    assert r.status is Status.OK  # latency alone does not degrade
    assert r.logged is True  # §7: PERF_WARN is a logged mode
    dq = read_dq(cfg.paths.data_quality_log)
    assert any(row["error_code"] == ErrorCode.PERF_WARN for row in dq)


def test_latency_no_warn_under_budget(
    make_handle: Callable[..., Any], clean_vector: Callable[[], dict[str, float]],
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    assert cfg.shap is not None
    handle = make_handle([0.7, 0.3], drivers=["VER", "HAM"])
    ex = _stub_explainer(
        drivers=["VER", "HAM"], k=0, shap_vec=np.ones(len(FEATURE_IDS)), base_val=0.0
    )
    ticks = iter([0.0, (cfg.shap.single_lap_warn_ms - 1.0) / 1000.0])
    monkeypatch.setattr(xai.time, "perf_counter", lambda: next(ticks))
    r = explain_prediction(
        clean_vector(), circuit="Bahrain", predicted_class="VER", explainer=ex, handle=handle,
        config=cfg,
    )
    assert ErrorCode.PERF_WARN not in r.flags


# --------------------------------------------------------------------------------------------
# 15. Logged + DQ row per logged mode
# --------------------------------------------------------------------------------------------


def test_logged_dq_rows(
    make_handle: Callable[..., Any], clean_vector: Callable[[], dict[str, float]], read_dq: Any,
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    handle = make_handle([0.7, 0.3], drivers=["VER", "HAM"])
    ex = _stub_explainer(
        drivers=["VER", "HAM"], k=0, shap_vec=np.ones(len(FEATURE_IDS)), base_val=0.0, raises=True
    )
    r = explain_prediction(
        clean_vector(), circuit="Bahrain", predicted_class="VER", explainer=ex, handle=handle,
        config=cfg,
    )
    assert r.logged is True
    rows = read_dq(cfg.paths.data_quality_log)
    assert any(row["error_code"] == ErrorCode.SHAP_UNAVAILABLE for row in rows)


# --------------------------------------------------------------------------------------------
# 16. Config-driven — no threshold literals in the module body
# --------------------------------------------------------------------------------------------


def test_no_threshold_literals_in_source() -> None:
    for lit in ("0.01", "0.10", "2000"):
        assert lit not in CODE_ONLY, f"threshold literal {lit} must come from config.shap"
    # 50 could appear inside an index/range; assert the specific budget isn't hardcoded as a float
    assert "50.0" not in CODE_ONLY


# --------------------------------------------------------------------------------------------
# 17. Config validator rejects bad shap: values
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field,value",
    [
        ("nl_dominance_margin", 0.0),
        ("nl_dominance_margin", 1.0),
        ("nl_top_domains", 0),
        ("nl_top_domains", 7),
        ("consistency_tolerance", 0.0),
        ("single_lap_warn_ms", 0.0),
        ("batch_warn_ms", -1.0),
        ("stability_js_bins", 1),
        ("stability_min_seasons", 1),
        ("beeswarm_filename", ""),
    ],
)
def test_shap_config_validator_rejects(field: str, value: Any) -> None:
    good = {
        "consistency_tolerance": 0.01, "nl_dominance_margin": 0.10, "nl_top_domains": 2,
        "single_lap_warn_ms": 50.0, "batch_warn_ms": 2000.0, "beeswarm_filename": "b.json",
        "stability_js_bins": 20, "stability_min_seasons": 3,
    }
    good[field] = value
    with pytest.raises(Exception):  # noqa: B017,PT011 - pydantic ValidationError
        ShapConfig(**good)


# --------------------------------------------------------------------------------------------
# 18. Multi-season stability (Jensen-Shannon drift)
# --------------------------------------------------------------------------------------------


def _season(shift: float, *, n: int = 200, seed: int = 0) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    return {fid: (rng.normal(shift, 1.0, n)).tolist() for fid in FEATURE_IDS[:5]}


def test_stability_identical_seasons_zero(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    s = _season(0.0, seed=7)
    per = {"2022": s, "2023": s, "2024": s}
    r = season_stability(per, config=cfg)
    assert r.stability_score == pytest.approx(0.0, abs=1e-9)
    assert r.partial is False and r.seasons_available == 3
    assert set(r.pairwise_js) == {"2022|2023", "2022|2024", "2023|2024"}


def test_stability_divergent_seasons_positive(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    per = {"2022": _season(0.0, seed=1), "2023": _season(0.0, seed=1), "2024": _season(6.0, seed=2)}
    r = season_stability(per, config=cfg)
    assert r.stability_score is not None and r.stability_score > 0.0
    assert 0.0 <= r.stability_score <= 1.0


def test_stability_partial_two_seasons(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    per = {"2022": _season(0.0, seed=1), "2023": _season(1.0, seed=2)}
    r = season_stability(per, config=cfg)
    assert r.partial is True and r.seasons_available == 2
    assert r.stability_score is not None and 0.0 <= r.stability_score <= 1.0
    assert r.status is Status.DEGRADED


def test_stability_one_season_none(tmp_path: Path, read_dq: Any) -> None:
    cfg = _cfg(tmp_path)
    r = season_stability({"2024": _season(0.0)}, config=cfg)
    assert r.stability_score is None  # never a fabricated 0.0
    assert r.partial is True and r.seasons_available == 1
    dq = read_dq(cfg.paths.data_quality_log)
    assert any(row["error_code"] == ErrorCode.SHAP_UNAVAILABLE for row in dq)


# --------------------------------------------------------------------------------------------
# 19. never-raises backstop on garbage input
# --------------------------------------------------------------------------------------------


def test_explain_never_raises_on_garbage(
    make_handle: Callable[..., Any], tmp_path: Path
) -> None:
    cfg = _cfg(tmp_path)
    handle = make_handle([0.7, 0.3], drivers=["VER", "HAM"])
    ex = _stub_explainer(
        drivers=["VER", "HAM"], k=0, shap_vec=np.ones(len(FEATURE_IDS)), base_val=0.0
    )
    # a vector full of NaN / wrong types must not raise
    garbage = {fid: (float("nan") if i % 2 else None) for i, fid in enumerate(FEATURE_IDS)}
    r = explain_prediction(
        garbage, circuit="Bahrain", predicted_class="VER", explainer=ex, handle=handle, config=cfg
    )
    assert isinstance(r.status, Status)  # returned a coded result, no exception


# --------------------------------------------------------------------------------------------
# Regression tests for code-review findings
# --------------------------------------------------------------------------------------------


def test_global_class_index_from_model_order(
    trained: tuple[Any, AppConfig], tmp_path: Path
) -> None:
    """build_global must key the SHAP class column off explainer.drivers (the model's class order),
    NOT the caller's driver_labels — re-ordering the label roster cannot change any fingerprint."""
    _result, cfg = trained
    handle = load_model(config=cfg, artefact_dir=Path(cfg.paths.artefacts_dir))
    ex = build_explainer(handle, config=cfg, log_path=str(tmp_path / "dq.jsonl"))
    bg = _background(handle.drivers)
    dq = str(tmp_path / "dq.jsonl")
    normal = build_global_explainability(
        ex, bg, handle.drivers, config=cfg, artefact_dir=tmp_path / "n", log_path=dq
    )
    reordered = build_global_explainability(
        ex, bg, list(reversed(handle.drivers)), config=cfg, artefact_dir=tmp_path / "r", log_path=dq
    )
    assert normal.per_driver_shap == reordered.per_driver_shap
    assert normal.per_driver_radar == reordered.per_driver_radar


def test_describe_style_drops_zero_contribution_domain(tmp_path: Path) -> None:
    """A domain that contributed exactly zero must NOT be named a dominant trait (§6.2 FM3)."""
    cfg = _cfg(tmp_path)
    radar = {"throttle": 10.0, "braking": 0.0, "speed": 0.0, "gear": 0.0, "engine": 0.0,
             "cornering": 0.0}
    style = describe_style(radar, config=cfg)
    assert style.generic is False
    assert style.dominant_domains == ["throttle"]  # not ["throttle", "braking"]
    assert "braking" not in style.sentence


def test_stability_all_nan_pair_is_none(tmp_path: Path, read_dq: Any) -> None:
    """A season whose SHAP is entirely NaN makes its pair uncomparable → missing data, not zero
    drift; with no comparable pair the score is None, not a fabricated 0.0."""
    cfg = _cfg(tmp_path)
    nan_season = {fid: [float("nan")] * 10 for fid in FEATURE_IDS[:5]}
    r = season_stability({"2022": _season(0.0, seed=1), "2023": nan_season}, config=cfg)
    assert r.stability_score is None
    assert r.status is Status.DEGRADED and r.logged is True
    dq = read_dq(cfg.paths.data_quality_log)
    assert any(row["error_code"] == ErrorCode.SHAP_UNAVAILABLE for row in dq)


def test_stability_mixed_valid_and_nan_pairs(tmp_path: Path) -> None:
    """With one all-NaN season among three, the uncomparable pairs are dropped and the score is the
    mean of the valid pair(s) only — never dragged toward 0 by a fabricated NaN→0.0."""
    cfg = _cfg(tmp_path)
    nan_season = {fid: [float("nan")] * 10 for fid in FEATURE_IDS[:5]}
    per = {"2022": _season(0.0, seed=1), "2023": _season(0.0, seed=1), "2024": nan_season}
    r = season_stability(per, config=cfg)
    assert r.stability_score is not None
    assert set(r.pairwise_js) == {"2022|2023"}  # the two NaN pairs excluded


def test_explainer_handle_driver_mismatch_degrades(
    make_handle: Callable[..., Any], clean_vector: Callable[[], dict[str, float]], tmp_path: Path
) -> None:
    """A stale explainer whose driver order disagrees with the handle must degrade to
    SHAP_UNAVAILABLE, never silently attribute SHAP to the wrong driver."""
    cfg = _cfg(tmp_path)
    handle = make_handle([0.7, 0.3], drivers=["VER", "HAM"])
    ex = _stub_explainer(  # reversed order vs the handle
        drivers=["HAM", "VER"], k=0, shap_vec=np.ones(len(FEATURE_IDS)), base_val=0.0
    )
    r = explain_prediction(
        clean_vector(), circuit="Bahrain", predicted_class="VER", explainer=ex, handle=handle,
        config=cfg,
    )
    assert r.error_code is ErrorCode.SHAP_UNAVAILABLE and r.shap_available is False
