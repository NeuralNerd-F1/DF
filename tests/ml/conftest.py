"""Fixtures for the F06a classifier-training tests.

Mirrors ``tests/pipeline/conftest.py``: a config fixture pinned under ``tmp_path`` (plus
tmp-contained sqlite stores for Optuna + MLflow so runs are isolated and fast), a
``training_matrix`` builder for the wide 28-feature matrix, and a ``read_dq`` helper. A
module-scoped ``trained`` fixture runs ``train_classifier`` ONCE on a standard separable matrix so
the many "inspect the happy result" assertions do not each pay for a full training run.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest
from app.contracts.config import AppConfig, load_config
from app.ml.classifier import train_classifier
from app.pipeline.features import FEATURE_IDS

if TYPE_CHECKING:
    from collections.abc import Callable

# MLflow + sklearn + xgboost emit many deprecation / convergence warnings on tiny synthetic data;
# silence them at import so the test output stays readable (they are not assertions).
warnings.filterwarnings("ignore")

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config.yaml"

# Small-but-valid knobs so a full train_classifier run is seconds, not minutes.
SMALL_TRIALS = 3
SMALL_FOLDS = 3
TRAIN_CIRCUITS = ["Bahrain", "Spain", "Japan", "USA", "Brazil", "Canada"]
DRIVERS = ["VER", "HAM", "LEC", "NOR"]


def _make_config(tmp_dir: Path) -> AppConfig:
    cfg = load_config(REPO_CONFIG)
    cfg.paths.data_quality_log = str(tmp_dir / "logs" / "dq.jsonl")
    cfg.paths.artefacts_dir = str(tmp_dir / "artefacts")
    assert cfg.classifier is not None
    cfg.classifier.optuna_trials = SMALL_TRIALS
    cfg.classifier.cv_folds = SMALL_FOLDS
    cfg.classifier.optuna_storage = f"sqlite:///{(tmp_dir / 'optuna.db').as_posix()}"
    cfg.classifier.mlflow_tracking_uri = f"sqlite:///{(tmp_dir / 'mlflow.db').as_posix()}"
    return cfg


def build_matrix(
    *,
    drivers: list[str] = DRIVERS,
    train_circuits: list[str] = TRAIN_CIRCUITS,
    held_out: list[str] | None = None,
    laps_per_combo: int = 12,
    held_out_laps: int = 8,
    separation: float = 5.0,
    seed: int = 0,
    held_out_split: str = "test",
) -> pd.DataFrame:
    """Build a wide training matrix (F01..F28 + driver/circuit/season/split).

    Each driver gets a distinct per-feature mean offset (``separation`` apart) so the classes are
    linearly separable when ``separation`` is large and overlap when it is small. ``held_out``
    defaults to the repo's quarantined circuits (Belgium, Italy).
    """
    held_out = held_out if held_out is not None else ["Belgium", "Italy"]
    rng = np.random.default_rng(seed)
    offset = {d: i * separation for i, d in enumerate(drivers)}
    rows: list[dict[str, Any]] = []

    def add(circuit: str, split: str, n: int) -> None:
        for d in drivers:
            for _ in range(n):
                vec = {f: float(rng.normal(offset[d], 1.0)) for f in FEATURE_IDS}
                rows.append(
                    {"driver": d, "circuit": circuit, "season": 2023, "split": split, **vec}
                )

    for c in train_circuits:
        add(c, "train", laps_per_combo)
    for c in held_out:
        add(c, held_out_split, held_out_laps)
    return pd.DataFrame(rows)


@pytest.fixture
def classifier_config(tmp_path: Path) -> AppConfig:
    """Function-scoped config pinned under ``tmp_path`` (fresh stores per test)."""
    return _make_config(tmp_path)


@pytest.fixture
def make_config() -> Callable[[Path], AppConfig]:
    """Factory: build a tmp-pinned config under an arbitrary directory (for multi-run tests)."""
    return _make_config


@pytest.fixture
def training_matrix() -> Callable[..., pd.DataFrame]:
    """Expose the matrix builder so tests can vary drivers/circuits/separability/held-out."""
    return build_matrix


@pytest.fixture
def read_dq() -> Callable[[str | Path], list[dict]]:
    def _read(path: str | Path) -> list[dict]:
        p = Path(path)
        if not p.is_file():
            return []
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line]

    return _read


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> tuple[Any, AppConfig]:
    """Run train_classifier ONCE on a standard separable matrix; shared across happy-path tests."""
    tmp_dir = tmp_path_factory.mktemp("f06a_trained")
    cfg = _make_config(tmp_dir)
    matrix = build_matrix()
    result = train_classifier(matrix, config=cfg)
    return result, cfg
