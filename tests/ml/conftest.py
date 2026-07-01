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


# ----------------------------------------------------------------------------------
# F06b inference harness
# ----------------------------------------------------------------------------------


class StubModel:
    """A minimal sklearn-shaped classifier for pinning F06b gate boundaries deterministically.

    ``predict_proba`` ignores X and returns the caller-supplied row (so a test sets the exact
    max-prob / entropy / argmax). ``raises=True`` makes it throw, to exercise the INFERENCE_ERROR
    backstop. ``classes_`` are integers [0..n-1] like the real F06a champion; a spy counter records
    whether the model was called (for the NULL-gate "model not invoked" assertion)."""

    def __init__(
        self, proba_row: list[float], *, n_features: int = 28, raises: bool = False
    ) -> None:
        self.proba_row = np.asarray(proba_row, dtype=float)
        self.classes_ = np.arange(len(proba_row))
        self.n_features_in_ = n_features
        self.raises = raises
        self.calls = 0

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        self.calls += 1
        if self.raises:
            raise RuntimeError("stub predict_proba boom")
        return np.tile(self.proba_row, (len(x), 1))


@pytest.fixture
def clean_vector() -> Callable[[], dict[str, float]]:
    """A full 28-key feature vector (all present, no NULLs)."""
    return lambda: {fid: 1.0 + i for i, fid in enumerate(FEATURE_IDS)}


@pytest.fixture
def norm_params(trained: tuple[Any, AppConfig]) -> Any:
    """The real NormalisationParams from the F06a `trained` run (so apply_normalisation works)."""
    from app.pipeline.normalise import load_normalisation_params

    _result, cfg = trained
    assert cfg.normalise is not None
    path = Path(cfg.paths.artefacts_dir) / cfg.normalise.artefact_filename
    return load_normalisation_params(path)


@pytest.fixture
def make_handle(norm_params: Any) -> Callable[..., Any]:
    """Factory: wrap a StubModel + the real params into a ModelHandle for gate-boundary tests."""
    from app.ml.inference import ModelHandle

    def _make(
        proba_row: list[float], *, drivers: list[str] | None = None, raises: bool = False
    ) -> Any:
        model = StubModel(proba_row, raises=raises)
        names = drivers or [f"D{i}" for i in range(len(proba_row))]
        return ModelHandle(
            model=model, normalisation_params=norm_params, drivers=names, model_version="test:v1"
        )

    return _make


@pytest.fixture
def bundle_dir(trained: tuple[Any, AppConfig], tmp_path: Path) -> Path:
    """A function-scoped COPY of the trained champion bundle (pkl + normalisation.json + labels),
    so load_model failure tests can mutate/delete files without corrupting the module-scoped run."""
    import shutil

    _result, cfg = trained
    src = Path(cfg.paths.artefacts_dir)
    dst = tmp_path / "bundle"
    dst.mkdir()
    assert cfg.classifier is not None and cfg.normalise is not None
    for name in (
        cfg.classifier.model_artefact_filename,
        cfg.normalise.artefact_filename,
        cfg.classifier.labels_filename,
    ):
        shutil.copy(src / name, dst / name)
    return dst
