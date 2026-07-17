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
    assert cfg.ingest is None  # minimal foundation config has no ingest: block


def test_repo_config_parses_ingest_block(repo_config_path: Path) -> None:
    """F01's ingest: block validates into the typed IngestConfig sub-model."""
    cfg = load_config(repo_config_path)
    assert cfg.ingest is not None
    assert cfg.ingest.cache_max_age_days == 7
    assert cfg.ingest.breakers.fastf1.open_after == 3
    assert cfg.ingest.breakers.fastf1.backoff_s == [5, 15, 30]
    assert cfg.ingest.breakers.ergast.retries == 2


def test_malformed_ingest_block_raises(valid_config_file: Path, tmp_path: Path) -> None:
    """A typed feature block is strictly validated — a missing breaker key raises."""
    extended = valid_config_file.read_text(encoding="utf-8") + (
        "ingest:\n  cache_max_age_days: 7\n  breakers:\n"
        "    fastf1: { timeout_s: 30, retries: 3, backoff_s: [5], open_after: 3, window_s: 60 }\n"
    )  # openf1 + ergast breakers missing -> IngestBreakers validation fails
    p = tmp_path / "bad_ingest.yaml"
    p.write_text(extended, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p)


def test_repo_config_parses_features_block(repo_config_path: Path) -> None:
    """The F03a+F03b features block loads, covers exactly the 26 range-bearing IDs, and the
    F27/F28 unconstrained slopes carry no range entry."""
    cfg = load_config(repo_config_path)
    assert cfg.features is not None
    assert len(cfg.features.valid_ranges) == 26
    assert "F27" not in cfg.features.valid_ranges and "F28" not in cfg.features.valid_ranges
    assert cfg.features.min_stint_laps == 8


def test_repo_config_parses_shap_block(repo_config_path: Path) -> None:
    """F07's shap: block validates into the typed ShapConfig sub-model (§6)."""
    cfg = load_config(repo_config_path)
    assert cfg.shap is not None
    assert cfg.shap.consistency_tolerance == 0.01
    assert cfg.shap.nl_dominance_margin == 0.10
    assert cfg.shap.nl_top_domains == 2
    assert cfg.shap.stability_min_seasons == 3
    assert cfg.shap.beeswarm_filename == "shap_beeswarm.json"


def test_zero_lateral_g_peak_top_n_rejected(repo_config_path: Path, tmp_path: Path) -> None:
    """lateral_g_peak_top_n=0 would make the top-N slice return ALL peaks -> rejected at load."""
    text = repo_config_path.read_text(encoding="utf-8").replace(
        "lateral_g_peak_top_n: 5", "lateral_g_peak_top_n: 0"
    )
    p = tmp_path / "bad_topn.yaml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p)


def test_bad_min_xy_finite_fraction_rejected(repo_config_path: Path, tmp_path: Path) -> None:
    text = repo_config_path.read_text(encoding="utf-8").replace(
        "min_xy_finite_fraction: 0.9", "min_xy_finite_fraction: 1.5"
    )
    p = tmp_path / "bad_xy_fraction.yaml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p)


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


def test_unknown_top_level_section_is_rejected(valid_config_file: Path, tmp_path: Path) -> None:
    """Standing rule (F01): AppConfig forbids extras, so a typo'd/untyped block raises.

    Replaces the old extra="allow" tolerance — feature sections must be typed fields on
    AppConfig, never silently-passed-through unknown keys.
    """
    extended = valid_config_file.read_text(encoding="utf-8") + (
        "wibble:\n  not_a_real_section: true\n"
    )
    p = tmp_path / "extended.yaml"
    p.write_text(extended, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p)


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


# --- F10 monitoring: block (MonitoringConfig) -----------------------------------------


def test_repo_config_parses_monitoring_block(repo_config_path: Path) -> None:
    """F10's monitoring: block validates into the typed MonitoringConfig sub-model (§5)."""
    cfg = load_config(repo_config_path)
    assert cfg.monitoring is not None
    assert cfg.monitoring.baseline_laps == 3
    assert cfg.monitoring.baseline_window_extended == 5
    assert cfg.monitoring.rolling_window == 3
    assert cfg.monitoring.alert_threshold_sd == 2.0
    assert cfg.monitoring.alert_confirm_laps == 2
    assert cfg.monitoring.max_consecutive_nan == 3
    assert cfg.monitoring.slope_weight == 0.25
    assert cfg.monitoring.confidence_medium_sd == 2.5
    assert cfg.monitoring.confidence_high_sd == 3.0
    assert cfg.monitoring.min_calibration_stints == 3
    assert cfg.monitoring.attribution_top_k == 3
    assert cfg.monitoring.sensitivity_grid_sd == [1.0, 1.5, 2.0, 2.5]
    assert cfg.monitoring.variation_artefact_filename == "stint_variation.json"
    assert cfg.monitoring.mlflow_experiment == "apex-degradation"


def test_monitoring_band_ordering_rejected(repo_config_path: Path, tmp_path: Path) -> None:
    """confidence_medium_sd must sit strictly above alert_threshold_sd (bands must be coherent)."""
    text = repo_config_path.read_text(encoding="utf-8").replace(
        "confidence_medium_sd: 2.5", "confidence_medium_sd: 1.5"
    )
    p = tmp_path / "bad_bands.yaml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p)


def test_monitoring_baseline_window_below_baseline_rejected(
    repo_config_path: Path, tmp_path: Path
) -> None:
    text = repo_config_path.read_text(encoding="utf-8").replace(
        "baseline_window_extended: 5", "baseline_window_extended: 2"
    )
    p = tmp_path / "bad_window.yaml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p)


def test_monitoring_empty_sensitivity_grid_rejected(
    repo_config_path: Path, tmp_path: Path
) -> None:
    text = repo_config_path.read_text(encoding="utf-8").replace(
        "sensitivity_grid_sd: [1.0, 1.5, 2.0, 2.5]", "sensitivity_grid_sd: []"
    )
    p = tmp_path / "bad_grid.yaml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p)


def test_monitoring_requires_sibling_blocks(repo_config_path: Path, tmp_path: Path) -> None:
    """A monitoring: block without features/validate/normalise (whose keys F10 reads) is a
    build-breaking CONFIG_001 — the detector would otherwise crash at first use."""
    import yaml

    raw = yaml.safe_load(repo_config_path.read_text(encoding="utf-8"))
    del raw["features"]  # remove a required sibling block
    p = tmp_path / "no_sibling.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p)
