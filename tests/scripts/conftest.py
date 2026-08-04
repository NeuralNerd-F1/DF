"""Shared fixtures for the Stage A/B/C script tests.

The scripts live outside ``app/`` but are imported as a namespace package (``pythonpath = ["."]``
in pyproject), so ``import scripts.build_training_matrix`` works from the suite unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from app.contracts.config import AppConfig, load_config


@pytest.fixture
def scripts_config(repo_config_path: Path, tmp_path: Path) -> AppConfig:
    """Real repo config (shipped thresholds, real held-out list) with the DQ log in tmp."""
    cfg = load_config(repo_config_path)
    cfg.paths.data_quality_log = str(tmp_path / "logs" / "dq.jsonl")
    return cfg


def read_dq(path: str | Path) -> list[dict]:
    """Parse the JSONL data_quality_log back into rows (empty if never written)."""
    p = Path(path)
    if not p.is_file():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def record(
    *,
    driver: str,
    circuit: str,
    circuit_type: str | None = "high_speed",
    season: int = 2024,
    lap: int = 1,
    raw: dict[str, float | None] | None = None,
    forced_null: list[str] | None = None,
) -> dict[str, Any]:
    """One :func:`extract_session`-shaped record, the input contract of the median/assembly
    stages. Mirrors the real record layout: public context keys plus the ``_``-prefixed
    internals that :func:`assemble_rows` consumes and strips."""
    return {
        "season": season,
        "circuit": circuit,
        "driver": driver,
        "lap": lap,
        "circuit_type": circuit_type,
        "session": "R",
        "lap_number": lap,
        "stint": 1,
        "tyre_age": 5,
        "compound": "MEDIUM",
        "track_status": "1",
        "lap_time_s": 92.5,
        "is_clean": True,
        "has_interpolation": False,
        "_raw": dict(raw or {}),
        "_forced_null": forced_null,
        "_flags": None,
    }


def telemetry_lap(lap: int, *, n: int = 20, hz_in: float = 10.0, **cols: Any) -> pd.DataFrame:
    """One lap of Stage-B-shaped telemetry on a ``hz_in`` grid (pre-resample).

    Channel values ramp with the sample index so interpolation and zero-order hold produce
    distinguishable results; per-lap context is constant within the lap, as in the real corpus.
    """
    t0 = float(lap) * 100.0
    frame = pd.DataFrame(
        {
            "time": [t0 + i / hz_in for i in range(n)],
            "Speed": [200.0 + i for i in range(n)],
            "Throttle": [50.0 + i for i in range(n)],
            "Brake": [0.0] * n,
            "RPM": [10000.0 + i for i in range(n)],
            "X": [float(i) for i in range(n)],
            "Y": [0.0] * n,
            "Gear": [float(6 + (i % 2)) for i in range(n)],
            "DRS": [float(i % 2) for i in range(n)],
            "lap": [lap] * n,
            "lap_time": [92.5] * n,
            "track_status": ["1"] * n,
            "stint": [1] * n,
            "tyre_age": [5] * n,
            "compound": ["MEDIUM"] * n,
        }
    )
    for name, value in cols.items():
        frame[name] = value
    return frame
