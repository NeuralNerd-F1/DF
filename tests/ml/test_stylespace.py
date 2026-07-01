"""F08 — driver style space tests: one per acceptance criterion AND per failure mode (§6/§7).

Fast tests (pure helpers, config validator, StubHandle boundaries) run without fitting a UMAP; the
real-fit tests share a module-scoped ``built``/``handle`` so UMAP + numba JIT is paid once.
"""

from __future__ import annotations

import inspect
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from app.contracts.config import ConfigError, load_config
from app.contracts.enums import ConfidenceState, Status
from app.contracts.errors import ErrorCode
from app.ml import stylespace as ss
from app.pipeline.features import FEATURE_IDS
from tests.ml.conftest import REPO_CONFIG, SS_DRIVERS, build_matrix


def _codes(read_dq: Any, cfg: Any) -> list[str]:
    return [r["error_code"] for r in read_dq(cfg.paths.data_quality_log)]


# ----------------------------------------------------------------------------------
# Pure helpers (handcrafted arrays — no UMAP)
# ----------------------------------------------------------------------------------


def test_aggregate_driver_56dim_ddof0_single_lap_no_nan() -> None:
    one = pd.DataFrame([{f: float(i) for i, f in enumerate(FEATURE_IDS)}])
    agg = ss.aggregate_driver(one)
    assert agg.shape == (56,)
    assert np.all(np.isfinite(agg))
    assert np.allclose(agg[28:], 0.0)  # single lap -> population std half all-zero (no NaN)
    assert np.allclose(agg[:28], np.arange(28.0))


def test_aggregate_all_null_feature_is_zero_not_nan() -> None:
    rows = pd.DataFrame([{f: 1.0 for f in FEATURE_IDS}, {f: 3.0 for f in FEATURE_IDS}])
    rows["F05"] = np.nan  # a fully-NULL feature
    agg = ss.aggregate_driver(rows)
    idx = FEATURE_IDS.index("F05")
    assert agg[idx] == 0.0 and agg[28 + idx] == 0.0  # explicit 0.0, never NaN
    assert np.all(np.isfinite(agg))


def test_nearest_neighbours_cosine_order_and_zero_norm() -> None:
    known = {"A": np.array([1.0, 0.0]), "B": np.array([0.9, 0.1]), "C": np.array([0.0, 1.0])}
    nn = ss.nearest_neighbours(np.array([1.0, 0.0]), known, 2)
    assert [n.driver for n in nn] == ["A", "B"]
    assert nn[0].cosine_similarity == pytest.approx(1.0)
    assert nn[0].distance == pytest.approx(0.0)
    # Zero-norm query -> similarity 0.0, no NaN / div-by-zero.
    zn = ss.nearest_neighbours(np.array([0.0, 0.0]), known, 3)
    assert all(n.cosine_similarity == 0.0 for n in zn)


def test_ood_check_threshold_both_sides() -> None:
    pca_lo, pca_hi = ss_stub_pca([2.9, 0, 0, 0, 0]), ss_stub_pca([3.1, 0, 0, 0, 0])
    pc_std = np.ones(5)
    ood_lo, max_lo = ss.ood_check(np.zeros(56), pca_lo, pc_std, 3.0, 1e-8)
    ood_hi, max_hi = ss.ood_check(np.zeros(56), pca_hi, pc_std, 3.0, 1e-8)
    assert ood_lo is False and max_lo == pytest.approx(2.9)
    assert ood_hi is True and max_hi == pytest.approx(3.1)


def test_risk_flags_out_vs_in_range() -> None:
    agg = np.zeros(56)
    agg[FEATURE_IDS.index("F03")] = 10.0  # 10 SD out (fstd=1)
    flags = ss._risk_flags(agg, np.zeros(56), np.ones(56), 3.0, 1e-8)
    assert flags == ["F03"]  # only the out-of-range feature flagged, not the in-range ones


def ss_stub_pca(scores: list[float]) -> Any:
    from tests.ml.conftest import StubPCA

    return StubPCA(scores)


# ----------------------------------------------------------------------------------
# Config validator (§7 — CONFIG_001 on any bad stylespace value)
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("umap_n_components_2d", 0),
        ("umap_n_components_5d", 2),  # <= 2d
        ("pca_n_components", 4),  # < 5d
        ("umap_n_neighbors", 1),
        ("umap_min_dist", 1.0),  # not in (0,1)
        ("umap_metric", ""),
        ("ood_sd_threshold", 0.0),
        ("nearest_neighbours_k", 0),
        ("stability_n_runs", 1),
        ("stability_variance_threshold", 0.0),
        ("mlflow_experiment", ""),
        ("mlflow_tracking_uri", ""),
    ],
)
def test_config_validator_rejects_bad_stylespace(tmp_path: Path, key: str, value: Any) -> None:
    import yaml

    raw = yaml.safe_load(Path(REPO_CONFIG).read_text(encoding="utf-8"))
    raw["stylespace"][key] = value
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p)


def test_config_validator_rejects_equal_filenames(tmp_path: Path) -> None:
    import yaml

    raw = yaml.safe_load(Path(REPO_CONFIG).read_text(encoding="utf-8"))
    raw["stylespace"]["pca_model_filename"] = raw["stylespace"]["umap_model_filename"]
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p)


# ----------------------------------------------------------------------------------
# Build (real fit, module-scoped) — 5D/2D/PCA, per-season, artefacts, reproducibility
# ----------------------------------------------------------------------------------


def test_build_fits_5d_2d_pca_and_coord_lengths(built: tuple[Any, Any, Any]) -> None:
    res, cfg, _matrix = built
    assert res.error_code is None  # DEGRADED (stability) is a finding, never an error code
    assert res.n_drivers == len(SS_DRIVERS)
    assert res.n_features == 56
    assert res.effective_n_neighbors == min(cfg.stylespace.umap_n_neighbors, len(SS_DRIVERS) - 1)
    for d in SS_DRIVERS:
        assert len(res.coords_5d[d]) == cfg.stylespace.umap_n_components_5d == 5
        assert len(res.coords_2d[d]) == cfg.stylespace.umap_n_components_2d == 2
    # exactly two fitted UMAP models persisted (not one-per-season) + a PCA fallback.
    from joblib import load as joblib_load

    umap_bundle = joblib_load(res.umap_artefact_path)
    assert set(umap_bundle) >= {"umap_5d", "umap_2d"}
    assert Path(res.pca_artefact_path).is_file()
    assert (
        Path(res.umap_artefact_path + ".sha256").read_text(encoding="utf-8").strip()
        == res.umap_sha256
    )


def test_per_season_coords_emitted_via_transform(built: tuple[Any, Any, Any]) -> None:
    res, _cfg, matrix = built
    train = matrix[matrix["split"] == "train"]
    expected = {
        (str(d), str(s)) for d, s in train[["driver", "season"]].drop_duplicates().to_numpy()
    }
    got = {(d, s) for d, seasons in res.season_coords_5d.items() for s in seasons}
    assert got == expected  # exactly one coord per (driver, season) present in the matrix
    for d, seasons in res.season_coords_5d.items():
        for s, coord in seasons.items():
            assert len(coord) == 5
            assert len(res.season_coords_2d[d][s]) == 2
    # per-season coords are a forward pass: the serving/helper bodies contain no `.fit(`.
    assert ".fit(" not in inspect.getsource(ss._project_aggregate)


def test_build_reproducible_pca_identical_umap_within_tol(tmp_path: Path, ss_matrix: Any) -> None:
    from app.pipeline.normalise import fit_normaliser

    def _one(sub: str) -> tuple[bytes, dict[str, list[float]]]:
        cfg = load_config(REPO_CONFIG)
        d = tmp_path / sub
        cfg.paths.artefacts_dir = str(d / "artefacts")
        cfg.paths.data_quality_log = str(d / "dq.jsonl")
        cfg.stylespace.mlflow_tracking_uri = f"sqlite:///{(d / 'm.db').as_posix()}"
        cfg.stylespace.stability_n_runs = 2
        (d / "artefacts").mkdir(parents=True)
        m = build_matrix(drivers=SS_DRIVERS, laps_per_combo=12, separation=4.0)
        fit_normaliser(m, config=cfg)
        r = ss.build_style_space(m, config=cfg)
        return Path(r.pca_artefact_path).read_bytes(), r.coords_5d

    pca_a, coords_a = _one("a")
    pca_b, coords_b = _one("b")
    assert pca_a == pca_b  # PCA fallback is deterministic -> byte-identical
    for d in SS_DRIVERS:  # UMAP not byte-identical, but coords equal within tolerance (fixed seed)
        assert np.allclose(coords_a[d], coords_b[d], atol=1e-4)


def test_assert_no_leakage_raises_on_held_out(stylespace_config: Any) -> None:
    # The tripwire fires on an unfiltered frame carrying a held-out circuit (mirrors F04/F06a).
    frame = build_matrix(drivers=SS_DRIVERS)  # includes Belgium/Italy rows
    with pytest.raises(ss.StyleSpaceBuildError) as exc:
        ss._assert_no_leakage(frame, stylespace_config)
    assert exc.value.error_code is ErrorCode.TRAIN_LEAKAGE


def test_build_leakage_logs_and_writes_no_artefact(
    stylespace_config: Any, read_dq: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Simulate a broken exclusion filter: force the tripwire; build logs + writes no artefact.
    from app.pipeline.normalise import fit_normaliser

    cfg = stylespace_config
    matrix = build_matrix(drivers=SS_DRIVERS)
    fit_normaliser(matrix, config=cfg)

    def _boom(_train: Any, _config: Any) -> None:
        raise ss.StyleSpaceBuildError("held-out leaked", error_code=ErrorCode.TRAIN_LEAKAGE)

    monkeypatch.setattr(ss, "_assert_no_leakage", _boom)
    with pytest.raises(ss.StyleSpaceBuildError) as exc:
        ss.build_style_space(matrix, config=cfg)
    assert exc.value.error_code is ErrorCode.TRAIN_LEAKAGE
    assert not (Path(cfg.paths.artefacts_dir) / cfg.stylespace.umap_model_filename).is_file()
    assert "TRAIN_LEAKAGE" in _codes(read_dq, cfg)


def test_build_too_few_drivers_raises(stylespace_config: Any, read_dq: Any) -> None:
    from app.pipeline.normalise import fit_normaliser

    cfg = stylespace_config
    matrix = build_matrix(drivers=["VER", "HAM", "LEC"])  # 3 < 5+1
    fit_normaliser(matrix, config=cfg)
    with pytest.raises(ss.StyleSpaceBuildError) as exc:
        ss.build_style_space(matrix, config=cfg)
    assert exc.value.error_code is ErrorCode.MODEL_TRAINING_FAILED
    assert "MODEL_TRAINING_FAILED" in _codes(read_dq, cfg)


def test_build_missing_columns_raises(stylespace_config: Any, read_dq: Any) -> None:
    with pytest.raises(ss.StyleSpaceBuildError) as exc:
        ss.build_style_space(pd.DataFrame({"driver": ["VER"]}), config=stylespace_config)
    assert exc.value.error_code is ErrorCode.MODEL_TRAINING_FAILED
    assert "MODEL_TRAINING_FAILED" in _codes(read_dq, stylespace_config)


def test_build_empty_pool_raises(stylespace_config: Any) -> None:
    from app.pipeline.normalise import fit_normaliser

    cfg = stylespace_config
    matrix = build_matrix(drivers=SS_DRIVERS)
    fit_normaliser(matrix, config=cfg)
    all_test = matrix.copy()
    all_test["split"] = "test"  # nothing left after the train quarantine
    with pytest.raises(ss.StyleSpaceBuildError) as exc:
        ss.build_style_space(all_test, config=cfg)
    assert exc.value.error_code is ErrorCode.MODEL_TRAINING_FAILED


def test_build_fit_exception_is_model_training_failed(
    stylespace_config: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.pipeline.normalise import fit_normaliser

    cfg = stylespace_config
    matrix = build_matrix(drivers=SS_DRIVERS)
    fit_normaliser(matrix, config=cfg)

    def _boom(**_kwargs: Any) -> Any:
        raise RuntimeError("umap fit boom")

    monkeypatch.setattr(ss.umap, "UMAP", _boom)  # any fit fault -> coded build failure
    with pytest.raises(ss.StyleSpaceBuildError) as exc:
        ss.build_style_space(matrix, config=cfg)
    assert exc.value.error_code is ErrorCode.MODEL_TRAINING_FAILED


def test_build_computes_proximity_when_teams_given(built: tuple[Any, Any, Any]) -> None:
    res, _cfg, _m = built  # the `built` fixture passes teams -> proximity is computed
    assert res.proximity is not None
    assert res.proximity.mean_intra_team is not None


# ----------------------------------------------------------------------------------
# Load — fallback chain + fail-loud
# ----------------------------------------------------------------------------------


def test_load_umap_hash_mismatch_falls_back_to_pca(
    built: tuple[Any, Any, Any], tmp_path: Path, read_dq: Any
) -> None:
    _res, cfg, _m = built
    art = tmp_path / "art"
    art.mkdir()
    src = Path(cfg.paths.artefacts_dir)
    for name in (
        cfg.stylespace.umap_model_filename,
        cfg.stylespace.umap_model_filename + ".sha256",
        cfg.stylespace.pca_model_filename,
        cfg.normalise.artefact_filename,
    ):
        shutil.copy(src / name, art / name)
    (art / (cfg.stylespace.umap_model_filename + ".sha256")).write_text(
        "deadbeef", encoding="utf-8"
    )
    dq = tmp_path / "dq.jsonl"
    handle = ss.load_style_space(config=cfg, artefact_dir=art, log_path=dq)
    assert handle.engine == "pca" and handle.ready
    assert "UMAP_FALLBACK" in [r["error_code"] for r in read_dq(dq)]


def test_load_no_engine_raises_model_not_found(
    stylespace_config: Any, tmp_path: Path, read_dq: Any
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    dq = tmp_path / "dq.jsonl"
    with pytest.raises(ss.StyleSpaceLoadError) as exc:
        ss.load_style_space(config=stylespace_config, artefact_dir=empty, log_path=dq)
    assert exc.value.error_code is ErrorCode.MODEL_NOT_FOUND
    assert "MODEL_NOT_FOUND" in [r["error_code"] for r in read_dq(dq)]


def test_load_missing_normalisation_raises_model_not_found(
    built: tuple[Any, Any, Any], tmp_path: Path
) -> None:
    _res, cfg, _m = built
    art = tmp_path / "art"
    art.mkdir()
    src = Path(cfg.paths.artefacts_dir)
    for name in (
        cfg.stylespace.umap_model_filename,
        cfg.stylespace.umap_model_filename + ".sha256",
        cfg.stylespace.pca_model_filename,
    ):
        shutil.copy(src / name, art / name)  # PCA present, normalisation.json absent
    with pytest.raises(ss.StyleSpaceLoadError) as exc:
        ss.load_style_space(config=cfg, artefact_dir=art, log_path=tmp_path / "dq.jsonl")
    assert exc.value.error_code is ErrorCode.MODEL_NOT_FOUND


def test_load_corrupt_pca_raises_model_not_found(stylespace_config: Any, read_dq: Any) -> None:
    from app.pipeline.normalise import fit_normaliser

    cfg = stylespace_config
    fit_normaliser(build_matrix(drivers=SS_DRIVERS), config=cfg)  # writes normalisation.json
    art = Path(cfg.paths.artefacts_dir)
    (art / cfg.stylespace.pca_model_filename).write_bytes(b"not a joblib pickle")  # corrupt PCA
    with pytest.raises(ss.StyleSpaceLoadError) as exc:
        ss.load_style_space(config=cfg)
    assert exc.value.error_code is ErrorCode.MODEL_NOT_FOUND
    assert "MODEL_NOT_FOUND" in _codes(read_dq, cfg)


def test_health_none_and_ready(handle: Any) -> None:
    assert ss.health(None).error_code is ErrorCode.HEALTH_FAIL
    assert ss.health(None).reason is ErrorCode.MODEL_NOT_FOUND
    ok = ss.health(handle)
    assert ok.status is Status.OK and ok.ready and ok.engine == "umap"


# ----------------------------------------------------------------------------------
# Project (real handle happy path + get_style_space)
# ----------------------------------------------------------------------------------


def test_project_happy_high_ok_neighbours(
    handle: Any, built: tuple[Any, Any, Any], unknown_laps: Any
) -> None:
    _res, cfg, _m = built
    res = ss.project_unknown(
        unknown_laps(30, circuit="Bahrain", offset=0.0),
        circuit="Bahrain",
        handle=handle,
        config=cfg,
    )
    assert res.status is Status.OK
    assert res.confidence_state is ConfidenceState.HIGH
    assert res.projection_engine == "umap"
    assert res.coords_5d is not None and len(res.coords_5d) == 5
    assert res.coords_2d is not None and len(res.coords_2d) == 2
    assert len(res.nearest_neighbours) == cfg.stylespace.nearest_neighbours_k
    sims = [n.cosine_similarity for n in res.nearest_neighbours]
    assert sims == sorted(sims, reverse=True)


def test_get_style_space_driver_and_season(handle: Any) -> None:
    r = ss.get_style_space(handle)
    assert set(r.drivers) == set(SS_DRIVERS)
    for d in SS_DRIVERS:
        assert len(r.coords_5d[d]) == 5 and len(r.coords_2d[d]) == 2
        assert "2023" in r.season_coords_5d[d]  # per-season served for GET /stylespace


# ----------------------------------------------------------------------------------
# Project boundaries + fallbacks (StubHandle — deterministic, no UMAP fit)
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize(("n", "sparse"), [(29, True), (30, False)])
def test_sparse_gate_29_vs_30(
    make_stub_handle: Any,
    stylespace_config: Any,
    unknown_laps: Any,
    read_dq: Any,
    n: int,
    sparse: bool,
) -> None:
    h = make_stub_handle()  # pca scores 0 -> not ood; umap engine returns a coord
    res = ss.project_unknown(
        unknown_laps(n, circuit="Bahrain"), circuit="Bahrain", handle=h, config=stylespace_config
    )
    assert res.sparse is sparse
    if sparse:
        assert res.confidence_state is ConfidenceState.LOW_CONFIDENCE
        assert res.status is Status.DEGRADED
        assert ErrorCode.SPARSE_PROJECTION in res.flags
        assert res.coords_5d is not None  # projection still computed
        assert "SPARSE_PROJECTION" in _codes(read_dq, stylespace_config)
    else:
        assert res.confidence_state is ConfidenceState.HIGH and res.status is Status.OK


@pytest.mark.parametrize(("score", "ood"), [(2.9, False), (3.1, True)])
def test_ood_gate_2p9_vs_3p1(
    make_stub_handle: Any,
    stylespace_config: Any,
    unknown_laps: Any,
    read_dq: Any,
    score: float,
    ood: bool,
) -> None:
    h = make_stub_handle(pca_scores=[score, 0, 0, 0, 0], pc_std=[1, 1, 1, 1, 1])
    res = ss.project_unknown(
        unknown_laps(30, circuit="Bahrain"), circuit="Bahrain", handle=h, config=stylespace_config
    )
    assert res.ood is ood
    assert res.coords_5d is not None  # projection computed either way
    if ood:
        assert res.error_code is ErrorCode.OUT_OF_DISTRIBUTION
        assert res.confidence_state is ConfidenceState.OUT_OF_DISTRIBUTION
        assert res.status is Status.DEGRADED
        assert "OUT_OF_DISTRIBUTION" in _codes(read_dq, stylespace_config)
    else:
        assert res.error_code is None and res.confidence_state is ConfidenceState.HIGH


def test_transform_raises_falls_back_to_raw_nn(
    make_stub_handle: Any, stylespace_config: Any, unknown_laps: Any, read_dq: Any
) -> None:
    h = make_stub_handle(umap_raises=True)
    res = ss.project_unknown(
        unknown_laps(30, circuit="Bahrain"), circuit="Bahrain", handle=h, config=stylespace_config
    )
    assert res.error_code is ErrorCode.PROJECTION_FALLBACK
    assert res.projection_engine == "raw_nn"
    assert res.coords_5d is None
    assert res.status is Status.DEGRADED
    assert res.nearest_neighbours  # raw-56D cosine neighbours returned
    assert res.confidence_state is ConfidenceState.LOW_CONFIDENCE
    assert "PROJECTION_FALLBACK" in _codes(read_dq, stylespace_config)


def test_transform_raises_with_ood_precedence(
    make_stub_handle: Any, stylespace_config: Any, unknown_laps: Any
) -> None:
    h = make_stub_handle(umap_raises=True, pca_scores=[9.0, 0, 0, 0, 0], pc_std=[1, 1, 1, 1, 1])
    res = ss.project_unknown(
        unknown_laps(30, circuit="Bahrain"), circuit="Bahrain", handle=h, config=stylespace_config
    )
    assert res.error_code is ErrorCode.PROJECTION_FALLBACK  # operative code
    assert ErrorCode.OUT_OF_DISTRIBUTION in res.flags  # OOD retained on flags


def test_umap_fallback_engine_yields_pca_coords(
    make_stub_handle: Any, stylespace_config: Any, unknown_laps: Any
) -> None:
    # pc_std large so the coord scores are not themselves OOD (isolate the engine-fallback path).
    h = make_stub_handle(engine="pca", pca_scores=[1, 2, 3, 4, 5], pc_std=[10, 10, 10, 10, 10])
    res = ss.project_unknown(
        unknown_laps(30, circuit="Bahrain"), circuit="Bahrain", handle=h, config=stylespace_config
    )
    assert res.projection_engine == "pca"
    assert ErrorCode.UMAP_FALLBACK in res.flags
    assert res.status is Status.DEGRADED
    assert res.confidence_state is ConfidenceState.HIGH  # engine fallback alone stays HIGH
    assert res.coords_2d == [1.0, 2.0]  # PCA first-2 comps


def test_zero_laps_projection_fallback(make_stub_handle: Any, stylespace_config: Any) -> None:
    h = make_stub_handle()
    res = ss.project_unknown(
        pd.DataFrame(columns=[*FEATURE_IDS, "circuit"]),
        circuit="Bahrain",
        handle=h,
        config=stylespace_config,
    )
    assert res.error_code is ErrorCode.PROJECTION_FALLBACK
    assert res.coords_5d is None
    assert res.confidence_state is ConfidenceState.LOW_CONFIDENCE


def test_nonfinite_aggregate_guard(
    make_stub_handle: Any,
    stylespace_config: Any,
    unknown_laps: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = make_stub_handle()
    monkeypatch.setattr(ss, "aggregate_driver", lambda _rows: np.full(56, np.nan))
    res = ss.project_unknown(
        unknown_laps(30, circuit="Bahrain"), circuit="Bahrain", handle=h, config=stylespace_config
    )
    assert res.error_code is ErrorCode.PROJECTION_FALLBACK  # never a NaN coord / silent OOD pass
    assert res.coords_5d is None


def test_normalisation_fallback_unknown_circuit(
    make_stub_handle: Any, stylespace_config: Any, unknown_laps: Any
) -> None:
    h = make_stub_handle()
    res = ss.project_unknown(
        unknown_laps(30, circuit="Atlantis"), circuit="Atlantis", handle=h, config=stylespace_config
    )
    assert ErrorCode.NORMALISATION_FALLBACK in res.flags
    assert res.status is Status.DEGRADED
    assert res.confidence_state is ConfidenceState.HIGH  # a norm fallback alone stays HIGH


def test_confidence_state_always_populated(
    make_stub_handle: Any, stylespace_config: Any, unknown_laps: Any
) -> None:
    scenarios = [
        make_stub_handle(),  # happy
        make_stub_handle(pca_scores=[9, 0, 0, 0, 0]),  # ood
        make_stub_handle(engine="pca"),  # umap fallback engine
        make_stub_handle(umap_raises=True),  # raw-nn fallback
    ]
    for h in scenarios:
        res = ss.project_unknown(
            unknown_laps(30, circuit="Bahrain"),
            circuit="Bahrain",
            handle=h,
            config=stylespace_config,
        )
        assert res.confidence_state is not None


def test_dq_log_unwritable_surfaces_on_flags(
    make_stub_handle: Any,
    stylespace_config: Any,
    unknown_laps: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.contracts.logging import DataQualityLogError

    def _boom(**_kwargs: Any) -> None:
        raise DataQualityLogError("sink down")

    monkeypatch.setattr(ss, "log_data_quality", _boom)  # audit sink is down
    h = make_stub_handle()
    res = ss.project_unknown(
        unknown_laps(5, circuit="Bahrain"), circuit="Bahrain", handle=h, config=stylespace_config
    )
    assert ErrorCode.DATA_QUALITY_LOG_UNWRITABLE in res.flags  # surfaced, never crashes


def test_project_none_laps_never_raises(make_stub_handle: Any, stylespace_config: Any) -> None:
    # A non-sized `laps` must not make the never-raises function raise (backstop must not re-deref).
    h = make_stub_handle()
    res = ss.project_unknown(None, circuit="Bahrain", handle=h, config=stylespace_config)  # type: ignore[arg-type]
    assert res.error_code is ErrorCode.PROJECTION_FALLBACK
    assert res.coords_5d is None


def test_all_null_laps_degrade_not_confident(make_stub_handle: Any, stylespace_config: Any) -> None:
    # 30 entirely-NULL laps carry no signal: must degrade, never a confident origin coordinate.
    h = make_stub_handle()
    laps = pd.DataFrame({**{f: [np.nan] * 30 for f in FEATURE_IDS}, "circuit": ["Bahrain"] * 30})
    res = ss.project_unknown(laps, circuit="Bahrain", handle=h, config=stylespace_config)
    assert res.error_code is ErrorCode.PROJECTION_FALLBACK
    assert res.confidence_state is not ConfidenceState.HIGH
    assert res.n_laps_used == 0 and res.laps_rejected == 30


def test_project_never_raises_on_broken_handle(stylespace_config: Any, unknown_laps: Any) -> None:
    class Boom:
        ready = True
        engine = "umap"
        min_std = 1e-8
        normalisation_params = None  # will explode inside -> backstop catches it

    res = ss.project_unknown(
        unknown_laps(5, circuit="Bahrain"),
        circuit="Bahrain",
        handle=Boom(),
        config=stylespace_config,
    )
    assert res.error_code is ErrorCode.PROJECTION_FALLBACK  # never-raises backstop


# ----------------------------------------------------------------------------------
# Dissertation functions (§5.2/§11.2) — never raise, never block
# ----------------------------------------------------------------------------------


def test_stability_report_structure(built: tuple[Any, Any, Any]) -> None:
    _res, cfg, matrix = built
    rep = ss.stability_report(matrix, config=cfg)
    assert rep.n_runs == cfg.stylespace.stability_n_runs
    assert len(rep.per_run_disparity) == cfg.stylespace.stability_n_runs - 1
    assert rep.procrustes_variance is not None and 0.0 <= rep.procrustes_variance <= 1.0
    assert isinstance(rep.passes, bool)


def test_stability_backstop_degrades_never_raises(
    built: tuple[Any, Any, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _res, cfg, matrix = built

    def _boom(_train: Any, _config: Any) -> None:
        raise ss.StyleSpaceBuildError("leak", error_code=ErrorCode.TRAIN_LEAKAGE)

    monkeypatch.setattr(ss, "_assert_no_leakage", _boom)  # force the tripwire
    rep = ss.stability_report(matrix, config=cfg)  # backstop catches it -> DEGRADED, never raises
    assert rep.status is Status.DEGRADED and rep.procrustes_variance is None


def test_teammate_proximity_intra_inter_and_none(stylespace_config: Any) -> None:
    # intra < inter -> passes: put two drivers close, teammates.
    coords = {"A": np.array([0.0, 0.0]), "B": np.array([0.1, 0.0]), "C": np.array([9.0, 9.0])}
    teams = {"A": "T1", "B": "T1", "C": "T2"}
    r = ss.teammate_proximity(coords, teams, config=stylespace_config)
    assert r.mean_intra_team is not None and r.mean_inter_team is not None
    assert r.passes is True
    # single-driver team only -> no comparable pair -> None (never fabricated 0.0), never raises.
    solo = ss.teammate_proximity({"A": np.array([0.0])}, {"A": "T1"}, config=stylespace_config)
    assert solo.mean_intra_team is None and solo.mean_inter_team is None and solo.passes is False


# ----------------------------------------------------------------------------------
# Static / literal scans (§7)
# ----------------------------------------------------------------------------------

_SOURCE = Path(ss.__file__).read_text(encoding="utf-8")
_SERVING = (
    ss.load_style_space,
    ss.project_unknown,
    ss.get_style_space,
    ss.health,
    ss.teammate_proximity,
    ss._normalise_unknown_laps,
    ss._raw_nn,
    ss.aggregate_driver,
    ss.nearest_neighbours,
    ss.ood_check,
    ss._project_aggregate,
    ss._risk_flags,
    ss._projection_fallback,
)


def test_no_fit_in_serving_bodies() -> None:
    for fn in _SERVING:
        src = inspect.getsource(fn)
        assert ".fit(" not in src, f"{fn.__name__} must not fit"
        assert ".fit_transform(" not in src, f"{fn.__name__} must not fit_transform"


def test_no_normalisation_refit_anywhere() -> None:
    assert "fit_normaliser" not in _SOURCE  # F08 reuses F04 params, never refits normalisation


def test_no_magic_threshold_literals() -> None:
    # The two F08-specific float thresholds must never be hardcoded (read from config instead).
    body = "\n".join(
        line
        for line in _SOURCE.splitlines()
        if "_MIN_STD" not in line and not line.strip().startswith("#")
    )
    for literal in ("3.0", "0.05"):
        assert literal not in body, f"threshold {literal!r} hardcoded"
    # The integer thresholds (5/2/7/30) are read from config, never bare literals — assert the
    # config accessors are present (a bare-int scan would false-positive on slices / 2*len).
    for key in (
        "umap_n_components_5d",
        "umap_n_components_2d",
        "umap_n_neighbors",
        "ood_sd_threshold",
        "min_clean_laps",
    ):
        assert key in _SOURCE, f"config key {key!r} must drive the module (no magic number)"


def test_two_fit_sites_only() -> None:
    # UMAP/PCA .fit lives only in build_style_space + stability_report.
    for fn in (ss.build_style_space, ss.stability_report):
        assert ".fit" in inspect.getsource(fn)
