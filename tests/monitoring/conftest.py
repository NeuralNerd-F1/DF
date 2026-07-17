"""Fixtures for the F10 degradation-detector tests.

Mirrors ``tests/ml/conftest.py``: a config pinned under ``tmp_path`` (with a tmp sqlite MLflow
store so runs are isolated), a synthetic **stint** generator (drift ramp / safety-car laps /
interpolation / NaN poisoning / known-vs-UNKNOWN tyre age), a **training matrix** builder
(2 drivers × 2 circuit types × ≥ ``min_calibration_stints`` stints), a module-scoped
``calibrated`` fixture that runs F04 ``fit_normaliser`` + ``calibrate_stint_variation`` +
``load_detector`` ONCE, and a ``read_dq`` helper.

The synthetic laps are built so the detector's STANDARDISED-space geometry is predictable: each
driver has a large, distinct base vector (so the standardised driver offset dominates per-lap
jitter → a no-drift lap sits near the baseline, cosine distance ≈ 0), and an injected drift adds
a consistent direction from lap ``drift_from`` (→ a monotonically rising deviation that crosses τ).
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
from app.monitoring.detector import calibrate_stint_variation, load_detector
from app.pipeline.features import FEATURE_IDS
from app.pipeline.normalise import fit_normaliser

if TYPE_CHECKING:
    from collections.abc import Callable

warnings.filterwarnings("ignore")

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config.yaml"

DRIVERS = ["VER", "HAM"]
# (circuit, circuit_type) — two circuit types, each a single non-held-out circuit with plenty of
# training laps (>= thresholds.min_clean_laps=30) so F04 fits per-circuit params (no fallback).
CIRCUITS = [("Bahrain", "high_speed"), ("Monaco", "street")]
STINTS_PER_CELL = 4  # >= min_calibration_stints (3) → (driver, circuit_type) cells are usable
STINT_LEN = 12  # > features.min_stint_laps (8)
GREEN = "1"


def _make_config(tmp_dir: Path) -> AppConfig:
    cfg = load_config(REPO_CONFIG)
    cfg.paths.data_quality_log = str(tmp_dir / "logs" / "dq.jsonl")
    cfg.paths.artefacts_dir = str(tmp_dir / "artefacts")
    assert cfg.monitoring is not None
    cfg.monitoring.mlflow_tracking_uri = f"sqlite:///{(tmp_dir / 'mlflow.db').as_posix()}"
    return cfg


def _driver_base(driver: str) -> np.ndarray:
    """A large, distinct per-driver 28-vector (deterministic). Large magnitude + wide per-driver
    spread → the standardised driver offset dominates jitter, so a no-drift lap sits near the
    within-stint baseline."""
    idx = DRIVERS.index(driver) if driver in DRIVERS else abs(hash(driver)) % 7
    rng = np.random.default_rng(1000 + idx)
    return 60.0 + 12.0 * idx + rng.normal(0.0, 6.0, len(FEATURE_IDS))


def make_stint(
    *,
    driver: str = "VER",
    circuit: str = "Bahrain",
    circuit_type: str | None = "high_speed",
    season: int = 2023,
    session: str = "R1",
    stint: int = 1,
    split: str = "train",
    n_laps: int = STINT_LEN,
    start_lap: int = 1,
    drift_from: int | None = None,
    drift_mag: float = 0.0,
    drift_feats: tuple[int, ...] | None = None,
    sc_laps: frozenset[int] = frozenset(),
    interp_laps: frozenset[int] = frozenset(),
    nan_laps: frozenset[int] = frozenset(),
    tyre_age_known: bool = True,
    constant_tyre_age: bool = False,
    compound: str = "Soft",
    pit_lap: int | None = None,
    jitter: float = 0.4,
    seed: int = 0,
) -> pd.DataFrame:
    """Build one stint frame. ``sc_laps`` / ``interp_laps`` / ``nan_laps`` are **lap numbers**
    (absolute, honoring ``start_lap``). ``drift_from`` (a lap number) injects a rising offset of
    magnitude ``drift_mag`` on every feature from that lap onward."""
    base = _driver_base(driver)
    rng = np.random.default_rng(seed + 7 * len(driver) + hash(session) % 97)
    drift_dir = np.zeros(len(FEATURE_IDS), dtype=float)
    if drift_feats is None:
        drift_dir[:] = 1.0  # uniform drift across every feature
    else:
        for j in drift_feats:
            drift_dir[j] = 1.0
    rows: list[dict[str, Any]] = []
    for i in range(n_laps):
        lap = start_lap + i
        age_idx = 0 if constant_tyre_age else i
        vec = base + 0.1 * age_idx * np.ones(len(FEATURE_IDS)) + rng.normal(0.0, jitter,
                                                                            len(FEATURE_IDS))
        if drift_from is not None and lap >= drift_from and drift_mag:
            ramp = (lap - drift_from + 1)
            vec = vec + drift_mag * ramp * drift_dir
        row: dict[str, Any] = {fid: float(vec[j]) for j, fid in enumerate(FEATURE_IDS)}
        if lap in nan_laps:
            row["F01"] = float("nan")  # poison a cosine-core feature → NaN distance
        row.update(
            driver=driver, circuit=circuit, circuit_type=circuit_type, season=season,
            session=session, stint=stint, lap_number=lap, split=split,
            track_status="4" if lap in sc_laps else GREEN,
            is_clean=lap not in sc_laps,
            has_interpolation=lap in interp_laps,
            tyre_age=(
                None if not tyre_age_known
                else float(1 if constant_tyre_age else age_idx + 1)
            ),
            compound=compound,
        )
        if pit_lap is not None:
            row["pit_lap"] = pit_lap
        rows.append(row)
    return pd.DataFrame(rows)


def build_training_matrix(*, jitter: float = 0.4) -> pd.DataFrame:
    """2 drivers × 2 (circuit, circuit_type) × ``STINTS_PER_CELL`` stints, all ``split="train"``."""
    frames: list[pd.DataFrame] = []
    for driver in DRIVERS:
        for circuit, ct in CIRCUITS:
            for s in range(STINTS_PER_CELL):
                frames.append(
                    make_stint(
                        driver=driver, circuit=circuit, circuit_type=ct,
                        session=f"{circuit}_{s}", stint=1, n_laps=STINT_LEN, jitter=jitter,
                        seed=s * 13 + 1,
                    )
                )
    return pd.concat(frames, ignore_index=True)


def build_eval_matrix() -> pd.DataFrame:
    """Evaluation matrix (adds ``compound`` + ``pit_lap``, includes one ``"UNKNOWN"`` stint)."""
    frames: list[pd.DataFrame] = []
    sess = 0
    for driver in DRIVERS:
        for circuit, ct in CIRCUITS:
            for s in range(3):
                compound = "UNKNOWN" if (driver == "VER" and circuit == "Bahrain" and s == 0) \
                    else ["Soft", "Medium", "Hard"][s % 3]
                frames.append(
                    make_stint(
                        driver=driver, circuit=circuit, circuit_type=ct, session=f"E{sess}",
                        split="val", n_laps=STINT_LEN, drift_from=7, drift_mag=6.0,
                        compound=compound, pit_lap=STINT_LEN, seed=sess * 5 + 3,
                    )
                )
                sess += 1
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def monitoring_config(tmp_path: Path) -> AppConfig:
    return _make_config(tmp_path)


@pytest.fixture
def make_config() -> Callable[[Path], AppConfig]:
    return _make_config


@pytest.fixture
def read_dq() -> Callable[[str | Path], list[dict]]:
    def _read(path: str | Path) -> list[dict]:
        p = Path(path)
        if not p.is_file():
            return []
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line]

    return _read


@pytest.fixture(scope="module")
def calibrated(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Run F04 fit + F10 calibration + load ONCE; shared across the happy-path serving tests."""
    tmp_dir = tmp_path_factory.mktemp("f10_calibrated")
    cfg = _make_config(tmp_dir)
    art_dir = Path(cfg.paths.artefacts_dir)
    matrix = build_training_matrix()
    fit_normaliser(matrix, config=cfg)  # writes config.paths.artefacts_dir/normalisation.json
    fit = calibrate_stint_variation(matrix, config=cfg, artefact_dir=art_dir)
    handle = load_detector(config=cfg, artefact_dir=art_dir)
    return {"handle": handle, "config": cfg, "art_dir": art_dir, "matrix": matrix, "fit": fit}
