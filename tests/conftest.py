"""Shared fixtures for the contract test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# A complete, valid config used by the loader's happy-path and as a base to corrupt
# for failure tests. Mirrors the foundational blocks in the repo's config.yaml.
VALID_CONFIG_YAML = """\
seeds:
  numpy: 42
  xgboost: 42
  optuna: 42
paths:
  artefacts_dir: artefacts
  cache_dir: cache
  data_quality_log: logs/data_quality_log.jsonl
held_out_circuits: [Belgium, Italy]
thresholds:
  confidence_gate: 0.65
  entropy_ambiguous_bits: 2.9
  null_ceiling: 6
  min_clean_laps: 30
  sample_rate_floor_hz: 20
  laptime_bound_pct: 115
"""


@pytest.fixture
def valid_config_file(tmp_path: Path) -> Path:
    """Write a valid config.yaml to a temp dir and return its path."""
    p = tmp_path / "config.yaml"
    p.write_text(VALID_CONFIG_YAML, encoding="utf-8")
    return p


@pytest.fixture
def repo_config_path() -> Path:
    """Path to the real repo config.yaml (asserts the shipped values are valid)."""
    return REPO_ROOT / "config.yaml"
