"""F09 corner-type decomposition + sub-fingerprints + ablation tests (§7 + §6).

Every §6 failure row has a test; ``decompose_lap`` / ``run_ablation`` never-raise is proven by
fault injection; the corner segmentation is pinned against F03a's ``_corner_indices`` (parity);
leakage + determinism are proven behaviourally. Uses a synthetic X/Y lap generator with a CONSTANT
arc-length step so curvature κ = 1/R on each arc (radius R exactly classifiable).
"""

from __future__ import annotations

import inspect
import json
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from app.contracts.config import AppConfig, load_config
from app.contracts.enums import Status
from app.contracts.errors import ErrorCode
from app.ml import corners as cor
from app.ml.corners import (
    CornerLoadError,
    CornerType,
    SubFingerprintBuildError,
    build_sub_fingerprints,
    corner_count_columns,
    corner_sub_columns,
    decompose_lap,
    decomposition_row,
    health,
    load_sub_fingerprints,
    run_ablation,
)
from app.pipeline.features import _corner_indices, _extract_channels
from tests.ml.conftest import build_matrix

warnings.filterwarnings("ignore")

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config.yaml"


def _make_config(tmp_dir: Path) -> AppConfig:
    cfg = load_config(REPO_CONFIG)
    cfg.paths.data_quality_log = str(tmp_dir / "logs" / "dq.jsonl")
    cfg.paths.artefacts_dir = str(tmp_dir / "artefacts")
    assert cfg.corners is not None
    cfg.corners.mlflow_tracking_uri = f"sqlite:///{(tmp_dir / 'mlflow.db').as_posix()}"
    return cfg


def make_lap(radii: list[float], *, straight_len: int = 40, turn_rad: float = 1.2,
             v_straight: float = 300.0, ds: float = 5.0, dt_s: float = 0.1) -> pd.DataFrame:
    """straight → arc(R) → straight → … at a CONSTANT step ``ds`` so κ = 1/R on each arc."""
    xs: list[float] = []
    ys: list[float] = []
    sp: list[float] = []
    heading = 0.0
    px = py = 0.0

    def add(v: float) -> None:
        xs.append(px)
        ys.append(py)
        sp.append(v)

    def straight(n: int) -> None:
        nonlocal px, py
        for _ in range(n):
            add(v_straight)
            px += ds * math.cos(heading)
            py += ds * math.sin(heading)

    straight(straight_len)
    for r in radii:
        v_apex = float(np.clip(math.sqrt(r) * 12.0, 60.0, v_straight - 40.0))
        n_arc = max(6, round(turn_rad * r / ds))
        dtheta = turn_rad / n_arc
        for j in range(n_arc):
            frac = j / (n_arc - 1)
            v = v_straight - (v_straight - v_apex) * (1.0 - abs(2.0 * frac - 1.0))
            add(v)
            px += ds * math.cos(heading)
            py += ds * math.sin(heading)
            heading += dtheta
        straight(straight_len)
    speed = np.array(sp, dtype=float)
    n = speed.size
    return pd.DataFrame({
        "X": xs, "Y": ys, "Speed": speed,
        "Throttle": np.clip(speed / v_straight * 100.0, 0, 100),
        "Brake": np.where(speed < v_straight * 0.9, 3.0, 0.0),
        "Time": np.arange(n) * dt_s,
    })


@pytest.fixture
def corners_config(tmp_path: Path) -> AppConfig:
    return _make_config(tmp_path)


@pytest.fixture
def read_dq() -> Any:
    def _read(path: str | Path) -> list[dict]:
        p = Path(path)
        if not p.is_file():
            return []
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line]

    return _read


def _codes(read_dq: Any, cfg: Any) -> set[str]:
    return {r["error_code"] for r in read_dq(cfg.paths.data_quality_log)}


# ==================================================================================
# Per-lap decomposition (§7 decompose criteria, §6 CORNER_001)
# ==================================================================================


def test_classifies_three_corner_types(corners_config: AppConfig) -> None:
    res = decompose_lap(make_lap([50.0, 150.0, 400.0]), lap_number=1, config=corners_config)
    assert res.status is Status.OK and res.error_code is None
    assert res.xy_available is True
    assert res.sub_vectors["SLOW"].n_corners == 1  # radius 50 < 100
    assert res.sub_vectors["MEDIUM"].n_corners == 1  # 100 ≤ 150 < 200
    assert res.sub_vectors["HIGH_SPEED"].n_corners == 1  # 400 ≥ 200
    assert set(res.sub_vectors) == {ct.value for ct in CornerType}  # all 3 keys always present


def test_single_type_lap_degraded_all_keys(corners_config: AppConfig) -> None:
    res = decompose_lap(make_lap([50.0, 60.0]), lap_number=1, config=corners_config)
    assert res.status is Status.DEGRADED  # medium/high have 0 corners
    assert res.sub_vectors["SLOW"].n_corners == 2
    assert res.sub_vectors["MEDIUM"].n_corners == 0
    assert res.sub_vectors["MEDIUM"].features["apex_speed_mean"] is None  # NULL sub-vector
    assert set(res.sub_vectors) == {ct.value for ct in CornerType}


def test_sub_features_finite_for_present_type(corners_config: AppConfig) -> None:
    res = decompose_lap(make_lap([150.0]), lap_number=1, config=corners_config)
    med = res.sub_vectors["MEDIUM"]
    assert med.n_corners == 1
    for f in ("apex_speed_mean", "lateral_g_mean", "throttle_mean", "brake_mean"):
        assert med.features[f] is not None and math.isfinite(med.features[f])


def test_corner_count_parity_with_indices(corners_config: AppConfig) -> None:
    """F09 classifies exactly F03a's ``_corner_indices`` apices (no re-segmentation)."""
    lap = make_lap([50.0, 150.0, 400.0])
    res = decompose_lap(lap, lap_number=1, config=corners_config)
    ch = _extract_channels(lap)
    n_apices = len(_corner_indices(ch.speed, corners_config.features))
    assert res.n_corners_total == n_apices == 3


def test_xy_unavailable_corner_001(corners_config: AppConfig, read_dq: Any) -> None:
    lap = make_lap([150.0]).drop(columns=["X", "Y"])  # no position trace
    res = decompose_lap(lap, lap_number=7, config=corners_config)
    assert res.status is Status.DEGRADED and res.error_code is ErrorCode.CORNER_001
    assert res.xy_available is False
    assert all(sv.n_corners == 0 for sv in res.sub_vectors.values())
    assert "CORNER_001" in _codes(read_dq, corners_config)


def test_xy_sparse_corner_001(corners_config: AppConfig) -> None:
    lap = make_lap([150.0])
    lap.loc[lap.index[2:], "X"] = np.nan  # near-total X dropout → below min_xy_finite_fraction
    res = decompose_lap(lap, lap_number=1, config=corners_config)
    assert res.error_code is ErrorCode.CORNER_001


def test_decompose_never_raises(corners_config: AppConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("curvature blew up")

    monkeypatch.setattr(cor, "_signed_lateral_g", _boom)
    res = decompose_lap(make_lap([150.0]), lap_number=1, config=corners_config)  # must not raise
    assert res.status is Status.DEGRADED and res.error_code is ErrorCode.CORNER_001


def test_subfeature_uncomputable_feature_001(corners_config: AppConfig, read_dq: Any) -> None:
    # A real MEDIUM corner but Throttle is all-NaN → throttle_mean NULL → FEATURE_001 flag,
    # that key None, the lap still continues DEGRADED (never a raised exception, §6 row 3).
    lap = make_lap([150.0])
    lap["Throttle"] = np.nan
    res = decompose_lap(lap, lap_number=3, config=corners_config)
    assert res.status is Status.DEGRADED
    med = res.sub_vectors["MEDIUM"]
    assert med.n_corners == 1 and med.features["throttle_mean"] is None  # NULL, not fabricated
    assert med.features["apex_speed_mean"] is not None  # other sub-features still computed
    assert "FEATURE_001" in _codes(read_dq, corners_config)


def test_dq_log_unwritable_flagged_never_raises(
    corners_config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The DQ sink raising must NOT propagate — the result carries DATA_QUALITY_LOG_UNWRITABLE
    # and decompose_lap still returns its typed CORNER_001 result (§6 last row, non-negotiable #1).
    def _boom(*_a: Any, **_k: Any) -> None:
        raise cor.DataQualityLogError("sink down")

    monkeypatch.setattr(cor, "log_data_quality", _boom)
    lap = make_lap([150.0]).drop(columns=["X", "Y"])  # forces a CORNER_001 DQ write attempt
    res = decompose_lap(lap, lap_number=1, config=corners_config)  # must not raise
    assert res.status is Status.DEGRADED and res.error_code is ErrorCode.CORNER_001
    assert ErrorCode.DATA_QUALITY_LOG_UNWRITABLE in res.flags
    assert res.logged is False  # nothing reached the sink


def test_decomposition_row_flattens(corners_config: AppConfig) -> None:
    res = decompose_lap(make_lap([50.0, 150.0, 400.0]), lap_number=1, config=corners_config)
    row = decomposition_row(res)
    assert set(row) == set(corner_sub_columns()) | set(corner_count_columns())  # 15 + 3
    assert row["SLOW_n_corners"] == 1.0
    assert row["MEDIUM_apex_speed_mean"] is not None


# ==================================================================================
# Sub-fingerprints (§7 sub-fingerprint criteria, §4.2)
# ==================================================================================


def _subfp_frame(driver_radii: dict[str, list[float]], *, circuits: list[str], laps: int = 6,
                 split: str = "train") -> pd.DataFrame:
    cfg = load_config(REPO_CONFIG)
    rows: list[dict[str, Any]] = []
    for driver, radii in driver_radii.items():
        for circuit in circuits:
            for lap_i in range(laps):
                res = decompose_lap(make_lap(radii), lap_number=lap_i + 1, config=cfg)
                row = decomposition_row(res)
                row.update(driver=driver, circuit=circuit, split=split)
                rows.append(row)
    return pd.DataFrame(rows)


def test_build_status_and_shape(corners_config: AppConfig, tmp_path: Path) -> None:
    frame = _subfp_frame(
        {"VER": [50.0, 150.0, 400.0], "HAM": [50.0, 150.0, 400.0]},
        circuits=["Bahrain", "Monaco"], laps=4,
    )
    fit = build_sub_fingerprints(frame, config=corners_config,
                                 artefact_dir=Path(corners_config.paths.artefacts_dir))
    assert fit.status is Status.OK
    assert fit.n_drivers == 2
    assert fit.n_entries == 6  # 2 drivers × 3 corner types (all above min_corners)
    assert (Path(corners_config.paths.artefacts_dir) / "corner_subfingerprints.json").is_file()


def test_build_deterministic_byte_identical(tmp_path: Path) -> None:
    frame = _subfp_frame({"VER": [50.0, 150.0, 400.0]}, circuits=["Bahrain", "Monaco"], laps=4)

    def _one(sub: str) -> bytes:
        cfg = _make_config(tmp_path / sub)
        build_sub_fingerprints(frame, config=cfg, artefact_dir=Path(cfg.paths.artefacts_dir))
        return (Path(cfg.paths.artefacts_dir) / "corner_subfingerprints.json").read_bytes()

    assert _one("a") == _one("b")


def test_build_thin_cell_data_insufficient(corners_config: AppConfig, read_dq: Any) -> None:
    # VER only takes SLOW corners → medium/high cells have 0 corners → DATA_INSUFFICIENT.
    frame = _subfp_frame({"VER": [50.0], "HAM": [50.0, 150.0, 400.0]},
                         circuits=["Bahrain", "Monaco"], laps=4)
    fit = build_sub_fingerprints(frame, config=corners_config,
                                 artefact_dir=Path(corners_config.paths.artefacts_dir))
    assert "DATA_INSUFFICIENT" in _codes(read_dq, corners_config)
    # VER contributes only a SLOW cell; HAM contributes all three → 4 cells.
    assert fit.n_entries == 4
    # H-1: a dropped thin cell must surface on the TYPED result, not only in the DQ log.
    assert fit.status is Status.DEGRADED
    assert ErrorCode.DATA_INSUFFICIENT in fit.flags


def test_build_missing_columns_training_failed(corners_config: AppConfig, read_dq: Any) -> None:
    frame = _subfp_frame({"VER": [50.0, 150.0, 400.0]}, circuits=["Bahrain", "Monaco"], laps=4)
    frame = frame.drop(columns=["SLOW_apex_speed_mean"])  # a required sub-feature column absent
    with pytest.raises(SubFingerprintBuildError) as exc:
        build_sub_fingerprints(frame, config=corners_config,
                               artefact_dir=Path(corners_config.paths.artefacts_dir))
    assert exc.value.error_code is ErrorCode.MODEL_TRAINING_FAILED
    assert "MODEL_TRAINING_FAILED" in _codes(read_dq, corners_config)


def test_build_all_cells_thin_no_usable(corners_config: AppConfig) -> None:
    # Every cell below min_corners_per_type → the REASON_NO_CELLS branch (distinct from empty pool).
    frame = _subfp_frame({"VER": [50.0], "HAM": [50.0]}, circuits=["Bahrain", "Monaco"], laps=1)
    with pytest.raises(SubFingerprintBuildError) as exc:
        build_sub_fingerprints(frame, config=corners_config,
                               artefact_dir=Path(corners_config.paths.artefacts_dir))
    assert exc.value.error_code is ErrorCode.MODEL_TRAINING_FAILED
    assert not (Path(corners_config.paths.artefacts_dir) / "corner_subfingerprints.json").is_file()


def test_build_write_fault_training_failed(
    corners_config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _subfp_frame({"VER": [50.0, 150.0, 400.0]}, circuits=["Bahrain", "Monaco"], laps=4)
    real_write = Path.write_text

    def _no_write(self: Path, *a: Any, **k: Any) -> int:
        if self.name == "corner_subfingerprints.json":
            raise OSError("disk full")
        return real_write(self, *a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "write_text", _no_write)
    with pytest.raises(SubFingerprintBuildError) as exc:
        build_sub_fingerprints(frame, config=corners_config,
                               artefact_dir=Path(corners_config.paths.artefacts_dir))
    assert exc.value.error_code is ErrorCode.MODEL_TRAINING_FAILED


def test_build_leakage_direct(corners_config: AppConfig) -> None:
    frame = _subfp_frame({"VER": [50.0, 150.0, 400.0]}, circuits=["Belgium"], laps=4)
    with pytest.raises(SubFingerprintBuildError) as exc:
        cor._assert_no_leakage(frame, corners_config)
    assert exc.value.error_code is ErrorCode.TRAIN_LEAKAGE


def test_build_leakage_no_artefact(
    corners_config: AppConfig, read_dq: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    art = Path(corners_config.paths.artefacts_dir)
    frame = _subfp_frame({"VER": [50.0, 150.0, 400.0]}, circuits=["Bahrain", "Monaco"], laps=4)

    def _boom(_t: Any, _c: Any) -> None:
        raise SubFingerprintBuildError("leak", error_code=ErrorCode.TRAIN_LEAKAGE)

    monkeypatch.setattr(cor, "_assert_no_leakage", _boom)
    with pytest.raises(SubFingerprintBuildError) as exc:
        build_sub_fingerprints(frame, config=corners_config, artefact_dir=art)
    assert exc.value.error_code is ErrorCode.TRAIN_LEAKAGE
    assert not (art / "corner_subfingerprints.json").is_file()
    assert "TRAIN_LEAKAGE" in _codes(read_dq, corners_config)


def test_build_empty_post_quarantine(corners_config: AppConfig) -> None:
    frame = _subfp_frame({"VER": [50.0, 150.0, 400.0]}, circuits=["Bahrain"], laps=4, split="val")
    with pytest.raises(SubFingerprintBuildError) as exc:
        build_sub_fingerprints(frame, config=corners_config,
                               artefact_dir=Path(corners_config.paths.artefacts_dir))
    assert exc.value.error_code is ErrorCode.MODEL_TRAINING_FAILED


def test_build_mlflow_fault_degrades(
    corners_config: AppConfig, read_dq: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    art = Path(corners_config.paths.artefacts_dir)
    frame = _subfp_frame({"VER": [50.0, 150.0, 400.0]}, circuits=["Bahrain", "Monaco"], laps=4)

    def _boom(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("mlflow down")

    monkeypatch.setattr(cor.mlflow, "set_tracking_uri", _boom)
    fit = build_sub_fingerprints(frame, config=corners_config, artefact_dir=art)
    assert fit.status is Status.DEGRADED
    assert fit.mlflow_run_id is None
    assert (art / "corner_subfingerprints.json").is_file()
    assert "MODEL_TRAINING_FAILED" in _codes(read_dq, corners_config)


# ==================================================================================
# Load + health (§7 load criterion)
# ==================================================================================


@pytest.fixture
def built(tmp_path: Path) -> tuple[Path, AppConfig]:
    cfg = _make_config(tmp_path / "built")
    frame = _subfp_frame({"VER": [50.0, 150.0, 400.0], "HAM": [50.0, 150.0, 400.0]},
                         circuits=["Bahrain", "Monaco"], laps=4)
    build_sub_fingerprints(frame, config=cfg, artefact_dir=Path(cfg.paths.artefacts_dir))
    return Path(cfg.paths.artefacts_dir), cfg


def test_load_and_health(built: tuple[Path, AppConfig]) -> None:
    art, cfg = built
    handle = load_sub_fingerprints(config=cfg, artefact_dir=art)
    assert handle.ready and handle.model_version
    assert (("VER", "SLOW")) in handle.cells
    assert health(None).error_code is ErrorCode.HEALTH_FAIL
    assert health(handle).status is Status.OK


def test_load_missing_refuses(corners_config: AppConfig) -> None:
    with pytest.raises(CornerLoadError) as exc:
        load_sub_fingerprints(config=corners_config,
                              artefact_dir=Path(corners_config.paths.artefacts_dir))
    assert exc.value.error_code is ErrorCode.MODEL_NOT_FOUND


def test_load_corrupt_refuses(corners_config: AppConfig) -> None:
    art = Path(corners_config.paths.artefacts_dir)
    art.mkdir(parents=True, exist_ok=True)
    (art / "corner_subfingerprints.json").write_text("{ not json", encoding="utf-8")
    with pytest.raises(CornerLoadError):
        load_sub_fingerprints(config=corners_config, artefact_dir=art)


def test_load_schema_mismatch_refuses(corners_config: AppConfig) -> None:
    art = Path(corners_config.paths.artefacts_dir)
    art.mkdir(parents=True, exist_ok=True)
    # Parses as JSON but is missing the required `cells` key → Pydantic ValidationError branch.
    (art / "corner_subfingerprints.json").write_text(
        json.dumps({"schema_version": "1.0", "sub_features": ["x"], "content_hash": "abc"}),
        encoding="utf-8",
    )
    with pytest.raises(CornerLoadError) as exc:
        load_sub_fingerprints(config=corners_config, artefact_dir=art)
    assert exc.value.error_code is ErrorCode.MODEL_NOT_FOUND


def test_load_empty_cells_refuses(corners_config: AppConfig) -> None:
    art = Path(corners_config.paths.artefacts_dir)
    art.mkdir(parents=True, exist_ok=True)
    # A structurally valid artefact with a correct hash but NO cells → the empty-table branch
    # (which sits AFTER the hash check, so the hash must genuinely match).
    empty = cor._SubFingerprintArtefact(
        schema_version=cor.SCHEMA_VERSION, sub_features=list(cor.CORNER_SUB_FEATURES),
        cells=[], content_hash="",
    )
    empty.content_hash = cor._hash_artefact(empty)
    (art / "corner_subfingerprints.json").write_text(
        json.dumps(empty.model_dump()), encoding="utf-8"
    )
    with pytest.raises(CornerLoadError):
        load_sub_fingerprints(config=corners_config, artefact_dir=art)


def test_load_hash_tampered_refuses(built: tuple[Path, AppConfig], tmp_path: Path) -> None:
    art, cfg = built
    data = json.loads((art / "corner_subfingerprints.json").read_text(encoding="utf-8"))
    data["content_hash"] = "deadbeef"
    tampered = tmp_path / "tamper"
    tampered.mkdir()
    (tampered / "corner_subfingerprints.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(CornerLoadError):
        load_sub_fingerprints(config=cfg, artefact_dir=tampered)


# ==================================================================================
# Ablation (§7-ablation, §4.4-c)
# ==================================================================================


def _ablation_matrix(*, base_sep: float, corner_signal: float, seed: int = 0) -> pd.DataFrame:
    """build_matrix base (28) + 15 corner columns; ``corner_signal`` scales the per-driver offset in
    the corner columns (large → drivers separable by corners → ablation IMPROVES; 0 → noise)."""
    m = build_matrix(separation=base_sep, seed=seed)
    rng = np.random.default_rng(seed + 1)
    drivers = sorted(m["driver"].unique())
    offset = {d: i for i, d in enumerate(drivers)}
    for col in corner_sub_columns():
        base = m["driver"].map(offset).to_numpy(dtype=float) * corner_signal
        m[col] = base + rng.normal(0.0, 1.0, len(m))
    return m


def test_ablation_improves(corners_config: AppConfig) -> None:
    # Weak base separation + strong corner signal → expanded macro-F1 > base by > margin.
    matrix = _ablation_matrix(base_sep=0.3, corner_signal=6.0)
    report = run_ablation(matrix, config=corners_config)
    assert report.status is Status.OK
    assert report.delta is not None and report.delta > corners_config.corners.ablation_margin
    assert report.finding == "IMPROVES"
    assert report.n_base_features == 28 and report.n_expanded_features == 43
    assert report.per_driver_delta


def test_ablation_noise_not_improves(corners_config: AppConfig) -> None:
    # Strong base separation + pure-noise corner columns → corners do not help (NEUTRAL/DEGRADES).
    matrix = _ablation_matrix(base_sep=6.0, corner_signal=0.0)
    report = run_ablation(matrix, config=corners_config)
    assert report.finding in ("NEUTRAL", "DEGRADES")


def test_ablation_reproducible(corners_config: AppConfig) -> None:
    matrix = _ablation_matrix(base_sep=0.3, corner_signal=6.0)
    a = run_ablation(matrix, config=corners_config)
    b = run_ablation(matrix, config=corners_config)
    assert a.base_macro_f1 == pytest.approx(b.base_macro_f1, rel=1e-9, abs=1e-9)
    assert a.expanded_macro_f1 == pytest.approx(b.expanded_macro_f1, rel=1e-9, abs=1e-9)


def test_ablation_native_nan_expanded(corners_config: AppConfig) -> None:
    # PRODUCTION shape: a lap with no corners of a type leaves those columns NaN (a real
    # decomposition_row). XGBoost consumes NaN natively — must still score, no imputation (div #21).
    matrix = _ablation_matrix(base_sep=0.3, corner_signal=6.0)
    hs_cols = [c for c in corner_sub_columns() if c.startswith("HIGH_SPEED_")]
    matrix.loc[matrix.index[::2], hs_cols] = np.nan  # half the laps have no fast corners
    assert matrix[hs_cols].isna().to_numpy().any()  # the NaN path is genuinely exercised
    report = run_ablation(matrix, config=corners_config)
    assert report.status is Status.OK and report.delta is not None
    assert report.finding in ("IMPROVES", "NEUTRAL", "DEGRADES")


def test_ablation_quarantines_held_out(corners_config: AppConfig) -> None:
    # A held-out circuit row marked split="train" is excluded by the quarantine mask
    # (~isin(held_out)) BEFORE the CV — the ablation runs clean, never leaks Belgium into a fit.
    matrix = _ablation_matrix(base_sep=0.3, corner_signal=6.0)
    belgium = matrix.iloc[:8].copy()
    belgium["circuit"] = "Belgium"
    belgium["split"] = "train"
    poisoned = pd.concat([matrix, belgium], ignore_index=True)
    report = run_ablation(poisoned, config=corners_config)
    assert report.status is Status.OK  # Belgium filtered, no crash, no leak
    assert report.finding in ("IMPROVES", "NEUTRAL", "DEGRADES")


def test_ablation_mlflow_fault_degrades(
    corners_config: AppConfig, read_dq: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("mlflow down")

    monkeypatch.setattr(cor.mlflow, "set_tracking_uri", _boom)
    report = run_ablation(_ablation_matrix(base_sep=0.3, corner_signal=6.0), config=corners_config)
    assert report.status is Status.DEGRADED  # the report still lands; only logging degraded
    assert report.mlflow_run_id is None and report.delta is not None
    assert "MODEL_TRAINING_FAILED" in _codes(read_dq, corners_config)


def test_ablation_malformed_invalid(corners_config: AppConfig) -> None:
    report = run_ablation(pd.DataFrame(), config=corners_config)
    assert report.status is Status.FAILED and report.error_code is ErrorCode.INVALID_INPUT


def test_ablation_never_raises(corners_config: AppConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("cv blew up")

    monkeypatch.setattr(cor, "_cv_f1", _boom)
    report = run_ablation(_ablation_matrix(base_sep=1.0, corner_signal=1.0), config=corners_config)
    assert report.status is Status.FAILED and report.error_code is ErrorCode.INVALID_INPUT


# ==================================================================================
# Static no-fit scan (§7 leakage criterion)
# ==================================================================================


_SOURCE = Path(cor.__file__).read_text(encoding="utf-8")
_SERVING = (cor.decompose_lap, cor.load_sub_fingerprints, cor.health, cor.decomposition_row,
            cor._apex_radius, cor._corner_windows, cor._classify)


def test_no_fit_in_serving_bodies() -> None:
    for fn in _SERVING:
        src = inspect.getsource(fn)
        assert ".fit(" not in src
        assert ".fit_transform(" not in src


def test_no_magic_threshold_literals() -> None:
    body = "\n".join(ln for ln in _SOURCE.splitlines() if not ln.strip().startswith("#"))
    for literal in ("100.0", "200.0", "0.01"):  # radius bands / ablation margin come from config
        assert literal not in body, f"threshold {literal!r} hardcoded"
    for key in ("radius_slow_max", "radius_medium_max", "ablation_margin", "min_corners_per_type"):
        assert key in _SOURCE, f"config key {key!r} must drive the module"
