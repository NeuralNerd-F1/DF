"""The config loader reads typed values and fails loudly on bad config."""

from __future__ import annotations

from pathlib import Path

import pytest
from app.contracts.config import ConfigError, load_config
from app.contracts.errors import ErrorCode


def test_repo_config_has_shipped_foundation_values(repo_config_path: Path) -> None:
    """The real config.yaml shipped with F00 carries the constitution's constants."""
    cfg = load_config(repo_config_path)
    assert cfg.seeds.numpy == 42
    assert cfg.seeds.xgboost == 42
    assert cfg.seeds.optuna == 42
    assert cfg.held_out_circuits == ["Belgium", "Italy"]
    assert cfg.thresholds.confidence_gate == 0.65
    assert cfg.thresholds.entropy_ambiguous_bits == 2.9
    assert cfg.thresholds.null_ceiling == 6
    assert cfg.thresholds.min_clean_laps == 30
    assert cfg.thresholds.sample_rate_floor_hz == 20
    assert cfg.thresholds.laptime_bound_pct == 115


def test_loads_valid_fixture(valid_config_file: Path) -> None:
    cfg = load_config(valid_config_file)
    assert cfg.paths.data_quality_log.endswith("data_quality_log.jsonl")


def test_missing_file_raises_config_001(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as exc:
        load_config(tmp_path / "does_not_exist.yaml")
    assert exc.value.error_code is ErrorCode.CONFIG_001


def test_unreadable_file_raises_config_001(
    valid_config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An existing-but-unreadable config surfaces CONFIG_001, not a bare OSError."""

    def _boom(*_args: object, **_kwargs: object) -> str:
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "read_text", _boom)
    with pytest.raises(ConfigError) as exc:
        load_config(valid_config_file)
    assert exc.value.error_code is ErrorCode.CONFIG_001


def test_missing_required_key_raises(tmp_path: Path) -> None:
    p = tmp_path / "config.yaml"
    p.write_text("paths: {artefacts_dir: a, cache_dir: c, data_quality_log: l}\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p)


def test_unknown_key_in_strict_block_raises(valid_config_file: Path, tmp_path: Path) -> None:
    """Foundational blocks forbid extra keys — a stray key is a typo, not config."""
    bad = valid_config_file.read_text(encoding="utf-8").replace(
        "  optuna: 42\n", "  optuna: 42\n  numby: 42\n"
    )
    p = tmp_path / "bad.yaml"
    p.write_text(bad, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p)


def test_feature_owned_section_is_tolerated(valid_config_file: Path, tmp_path: Path) -> None:
    """A downstream feature appending its own block must not break the loader."""
    extended = valid_config_file.read_text(encoding="utf-8") + (
        "ingest:\n  cache_dir: cache\n  cache_max_age_days: 7\n"
    )
    p = tmp_path / "extended.yaml"
    p.write_text(extended, encoding="utf-8")
    cfg = load_config(p)
    assert cfg.seeds.numpy == 42  # foundational block still intact


def test_non_mapping_top_level_raises(tmp_path: Path) -> None:
    p = tmp_path / "config.yaml"
    p.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p)


def test_invalid_yaml_raises(tmp_path: Path) -> None:
    p = tmp_path / "config.yaml"
    p.write_text("seeds: [unbalanced\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p)
