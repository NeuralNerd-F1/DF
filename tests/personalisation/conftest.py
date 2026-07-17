"""Fixtures for the F11 personalisation-engine tests.

Mirrors ``tests/ml/conftest.py``: a config pinned under ``tmp_path`` (own sqlite MLflow store), a
synthetic wide matrix (8 drivers × 6 circuits across the 4 circuit types + a ``lap_time_s`` column,
held-out Belgium/Italy `split="test"`), and a module-scoped ``trained`` fixture that builds a real
F08 ``StyleSpaceHandle`` and then trains + loads the F11 bundle ONCE. A ``stub_handle`` factory
builds a `PersonalisationHandle` with fixed-prediction stub regressors for exact constraint/
confidence assertions without depending on the trained model's outputs.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from app.contracts.config import AppConfig, load_config
from app.ml.stylespace import build_style_space, load_style_space
from app.personalisation.engine import (
    PersonalisationHandle,
    load_personalisation,
    train_personalisation,
)
from app.pipeline.features import FEATURE_IDS
from app.pipeline.normalise import fit_normaliser

warnings.filterwarnings("ignore")

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config.yaml"

DRIVERS = ["VER", "HAM", "LEC", "NOR", "PIA", "RUS", "ALO", "SAI"]  # ≥ umap n5 + 1
CIRCUITS = [
    ("Bahrain", "high_speed"),
    ("Spain", "high_speed"),
    ("Monaco", "street"),
    ("Singapore", "street"),
    ("Suzuka", "technical"),
    ("Silverstone", "mixed"),
]
HELD_OUT = [("Belgium", "high_speed"), ("Italy", "technical")]
LAPS_PER_CELL = 34  # > thresholds.min_clean_laps (30)


def _make_config(tmp_dir: Path) -> AppConfig:
    cfg = load_config(REPO_CONFIG)
    cfg.paths.data_quality_log = str(tmp_dir / "logs" / "dq.jsonl")
    cfg.paths.artefacts_dir = str(tmp_dir / "artefacts")
    assert cfg.personalisation is not None and cfg.stylespace is not None
    cfg.personalisation.mlflow_tracking_uri = f"sqlite:///{(tmp_dir / 'mlflow.db').as_posix()}"
    cfg.stylespace.mlflow_tracking_uri = f"sqlite:///{(tmp_dir / 'ss_mlflow.db').as_posix()}"
    cfg.stylespace.stability_n_runs = 2
    return cfg


def _driver_base(driver: str) -> np.ndarray:
    idx = DRIVERS.index(driver) if driver in DRIVERS else 99
    rng = np.random.default_rng(2000 + idx)
    return 50.0 + 8.0 * idx + rng.normal(0.0, 4.0, len(FEATURE_IDS))


def build_matrix(
    *, jitter: float = 0.6, seed: int = 0, laps_per_cell: int = LAPS_PER_CELL
) -> pd.DataFrame:
    """8 drivers × 6 (circuit, circuit_type) × ``laps_per_cell`` train laps + 2 held-out."""
    rows: list[dict[str, Any]] = []
    ct_offset = {ct: i for i, ct in enumerate(["street", "high_speed", "technical", "mixed"])}

    def add(driver: str, circuit: str, ct: str, split: str, n: int, s: int) -> None:
        base = _driver_base(driver)
        rng = np.random.default_rng(seed + s)
        for lap in range(n):
            vec = base + ct_offset[ct] * 0.8 + rng.normal(0.0, jitter, len(FEATURE_IDS))
            row: dict[str, Any] = {fid: float(vec[j]) for j, fid in enumerate(FEATURE_IDS)}
            row.update(
                driver=driver, circuit=circuit, circuit_type=ct, season=2023, split=split,
                lap_number=lap + 1, lap_time_s=float(80.0 + rng.normal(0.0, 1.5)),
            )
            rows.append(row)

    for di, driver in enumerate(DRIVERS):
        for ci, (circuit, ct) in enumerate(CIRCUITS):
            add(driver, circuit, ct, "train", laps_per_cell, di * 100 + ci * 7 + 1)
        for hi, (circuit, ct) in enumerate(HELD_OUT):
            add(driver, circuit, ct, "test", 8, di * 100 + 90 + hi)
    return pd.DataFrame(rows)


@pytest.fixture
def personalisation_config(tmp_path: Path) -> AppConfig:
    return _make_config(tmp_path)


@pytest.fixture
def read_dq() -> Any:
    def _read(path: str | Path) -> list[dict]:
        p = Path(path)
        if not p.is_file():
            return []
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line]

    return _read


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Build the F08 style handle + train/load the F11 bundle ONCE; shared across happy tests."""
    tmp_dir = tmp_path_factory.mktemp("f11_trained")
    cfg = _make_config(tmp_dir)
    art_dir = Path(cfg.paths.artefacts_dir)
    matrix = build_matrix()
    fit_normaliser(matrix, config=cfg)
    build_style_space(matrix, config=cfg, artefact_dir=art_dir)
    style_handle = load_style_space(config=cfg, artefact_dir=art_dir)
    fit = train_personalisation(matrix, style_handle=style_handle, config=cfg, artefact_dir=art_dir)
    handle = load_personalisation(config=cfg, artefact_dir=art_dir)
    return {
        "handle": handle, "style_handle": style_handle, "config": cfg, "art_dir": art_dir,
        "matrix": matrix, "fit": fit,
    }


class _StubRegressor:
    """A regressor whose ``predict`` returns a fixed value regardless of input (for exact tests).
    ``boom=True`` makes ``predict`` raise (drives the never-raises backstop)."""

    def __init__(self, value: float, *, boom: bool = False) -> None:
        self.value = value
        self.boom = boom

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.boom:
            raise RuntimeError("injected regressor fault")
        return np.full(len(x), self.value, dtype=float)


class _BoomExplainer:
    """A SHAP explainer whose call raises — drives the ``SHAP_UNAVAILABLE`` degrade path."""

    def __call__(self, x: np.ndarray) -> Any:
        raise RuntimeError("injected SHAP fault")


def make_stub_handle(
    cfg: AppConfig, *, predictions: dict[str, float], reliability: dict[str, float] | None = None,
    boom: bool = False, boom_shap: bool = False,
) -> PersonalisationHandle:
    """A `PersonalisationHandle` with fixed-prediction stub regressors + chosen reliability. Lets a
    test pin exact clip/Monaco/WIDE/consistency behaviour. ``boom`` makes every ``predict`` raise;
    ``boom_shap`` attaches a raising explainer (else explainers are None → SHAP skipped)."""
    assert cfg.personalisation is not None
    pcfg = cfg.personalisation
    default_r = reliability or {}
    explainer: Any = _BoomExplainer() if boom_shap else None
    return PersonalisationHandle(
        regressors={name: _StubRegressor(predictions[name], boom=boom) for name in predictions},
        reliability={name: default_r.get(name, 0.9) for name in predictions},
        explainers={name: explainer for name in predictions},
        fia_bounds=dict(pcfg.fia_bounds),
        circuit_types=tuple(cfg.circuit_types),
        input_dim=5 + max(len(cfg.circuit_types) - 1, 0),
        model_version="sha256:stub",
    )


def full_predictions(value: float = 5.0, **overrides: float) -> dict[str, float]:
    """Fixed predictions for all 10 parameters (default ``value``), with per-param overrides."""
    from app.personalisation.engine import _PARAM_NAMES

    preds = {name: value for name in _PARAM_NAMES}
    preds.update(overrides)
    return preds
