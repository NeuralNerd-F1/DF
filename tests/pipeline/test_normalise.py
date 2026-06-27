"""F04 — circuit-aware normalisation tests.

Covers the spec's acceptance criteria and every failure mode, with the **no-leakage contract
as the primary subject**: the held-out leakage guard (direct + byte-identical-artefact proof),
exclusion reporting, deterministic serialisation, the one-writer/no-refit source guards, the
round-trip, and the three coded fallbacks (unknown circuit, zero-variance, NULL input) plus the
fail-loud load. Boundaries (30-vs-29 lap floor, std at/below min_std) and config-driven thresholds
are pinned both sides.
"""

from __future__ import annotations

import inspect
import json
import math
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from app.contracts.config import (
    NORMALISED_FEATURE_IDS,
    AppConfig,
    ConfigError,
    NormaliseConfig,
    load_config,
)
from app.contracts.enums import Status
from app.contracts.errors import ErrorCode
from app.pipeline import normalise as norm
from app.pipeline.features import FEATURE_IDS
from app.pipeline.normalise import (
    NormalisationError,
    apply_normalisation,
    fit_normaliser,
    load_normalisation_params,
)
from pydantic import ValidationError

from tests.pipeline.conftest import feature_frame, read_dq

# Default ramp knobs (mirror conftest.feature_frame): values base, base+step, ...
BASE = 100.0
STEP = 1.0


def _expected_mean(n: int, base: float = BASE, step: float = STEP) -> float:
    return base + step * (n - 1) / 2


def _expected_std(n: int, step: float = STEP) -> float:
    """Population std (ddof=0) of an arithmetic ramp of n values."""
    return step * math.sqrt((n * n - 1) / 12)


def _full_vector(value: float = 100.0) -> dict[str, float | None]:
    """A 28-key vector: the six normalised features at ``value``, the other 22 at unique
    sentinels (so a passthrough test can assert each is bit-identical)."""
    vec: dict[str, float | None] = {}
    for i, fid in enumerate(FEATURE_IDS):
        vec[fid] = value if fid in NORMALISED_FEATURE_IDS else float(1000 + i)
    return vec


def _art(cfg: AppConfig, name: str = "normalisation.json") -> Path:
    return Path(cfg.paths.artefacts_dir) / name


# ----------------------------------------------------------------------------------
# Leakage guard — the primary contract
# ----------------------------------------------------------------------------------


def test_assert_no_leakage_raises_on_held_out_present(normalise_config: AppConfig) -> None:
    df = feature_frame({"Belgium": 30, "Bahrain": 30})
    with pytest.raises(NormalisationError, match="held-out"):
        norm._assert_no_leakage(df, normalise_config)


def test_held_out_rows_have_zero_effect_on_artefact(normalise_config: AppConfig) -> None:
    """Adding pathological Belgium/Italy train rows yields a BYTE-IDENTICAL artefact — proof
    they never reach aggregation (excluded before any reduction)."""
    clean = feature_frame({"Bahrain": 35, "Monaco": 35})
    poison = feature_frame({"Belgium": 50, "Italy": 50}, base=9_999.0, step=100.0)
    with_poison = pd.concat([clean, poison], ignore_index=True)

    p_clean = _art(normalise_config, "clean.json")
    p_poison = _art(normalise_config, "poison.json")
    fit_normaliser(clean, config=normalise_config, artefact_path=p_clean)
    fit_normaliser(with_poison, config=normalise_config, artefact_path=p_poison)

    assert p_clean.read_bytes() == p_poison.read_bytes()


# ----------------------------------------------------------------------------------
# Exclusion reporting + determinism
# ----------------------------------------------------------------------------------


def test_excluded_circuits_equals_config_held_out(normalise_config: AppConfig) -> None:
    res = fit_normaliser(feature_frame({"Bahrain": 35}), config=normalise_config)
    assert res.excluded_circuits == sorted(normalise_config.held_out_circuits)
    params = load_normalisation_params(res.artefact_path)
    assert params.excluded_circuits == sorted(normalise_config.held_out_circuits)


def test_fit_is_byte_deterministic(normalise_config: AppConfig) -> None:
    df = feature_frame({"Bahrain": 35, "Monaco": 40})
    p1 = _art(normalise_config, "a.json")
    p2 = _art(normalise_config, "b.json")
    r1 = fit_normaliser(df, config=normalise_config, artefact_path=p1)
    r2 = fit_normaliser(df, config=normalise_config, artefact_path=p2)
    assert p1.read_bytes() == p2.read_bytes()
    assert r1.content_hash == r2.content_hash


def test_artefact_has_no_timestamp(normalise_config: AppConfig) -> None:
    res = fit_normaliser(feature_frame({"Bahrain": 35}), config=normalise_config)
    raw = json.loads(Path(res.artefact_path).read_text())
    assert "created_at" not in raw and "timestamp" not in raw


# ----------------------------------------------------------------------------------
# Round-trip + scope
# ----------------------------------------------------------------------------------


def test_clean_fit_is_ok_with_no_error_code(normalise_config: AppConfig) -> None:
    """A fit with every circuit above the floor and no zero-variance fallback is OK (no code)."""
    res = fit_normaliser(feature_frame({"Bahrain": 35, "Monaco": 40}), config=normalise_config)
    assert res.status is Status.OK
    assert res.error_code is None
    assert not res.skipped_circuits


def test_per_circuit_params_match_hand_computed(normalise_config: AppConfig) -> None:
    n = 35
    res = fit_normaliser(feature_frame({"Bahrain": n}), config=normalise_config)
    params = load_normalisation_params(res.artefact_path)
    stats = params.per_circuit["Bahrain"]["F09"]
    assert stats.mean == pytest.approx(_expected_mean(n))
    assert stats.std == pytest.approx(_expected_std(n))


def test_round_trip_reproduces_zscore(normalise_config: AppConfig) -> None:
    n = 35
    res = fit_normaliser(feature_frame({"Bahrain": n}), config=normalise_config)
    params = load_normalisation_params(res.artefact_path)
    out = apply_normalisation(
        _full_vector(value=130.0), circuit="Bahrain", params=params, config=normalise_config
    )
    expected = (130.0 - _expected_mean(n)) / _expected_std(n)
    assert out.normalised_vector["F09"] == pytest.approx(expected)
    assert out.status is Status.OK and not out.used_global


def test_only_the_six_features_change(normalise_config: AppConfig) -> None:
    res = fit_normaliser(feature_frame({"Bahrain": 35}), config=normalise_config)
    params = load_normalisation_params(res.artefact_path)
    vec = _full_vector(value=130.0)
    out = apply_normalisation(vec, circuit="Bahrain", params=params, config=normalise_config)
    assert set(out.normalised_vector) == set(FEATURE_IDS)  # full 28-key vector
    for fid in FEATURE_IDS:
        if fid in NORMALISED_FEATURE_IDS:
            assert out.normalised_vector[fid] != vec[fid]  # transformed
        else:
            assert out.normalised_vector[fid] == vec[fid]  # bit-identical passthrough


# ----------------------------------------------------------------------------------
# No refit downstream (mtime/bytes unchanged + static source guards)
# ----------------------------------------------------------------------------------


def test_apply_never_rewrites_the_artefact(normalise_config: AppConfig) -> None:
    res = fit_normaliser(feature_frame({"Bahrain": 35, "Monaco": 35}), config=normalise_config)
    path = Path(res.artefact_path)
    before_hash = sha256(path.read_bytes()).hexdigest()
    before_mtime = path.stat().st_mtime_ns
    params = load_normalisation_params(path)
    for circ in ("Bahrain", "Monaco", "Suzuka"):
        apply_normalisation(_full_vector(), circuit=circ, params=params, config=normalise_config)
    assert sha256(path.read_bytes()).hexdigest() == before_hash
    assert path.stat().st_mtime_ns == before_mtime


_READER_SRC = inspect.getsource(load_normalisation_params) + inspect.getsource(apply_normalisation)
_WRITER_SRC = inspect.getsource(fit_normaliser)
_FUNC_SRC = _READER_SRC + _WRITER_SRC


def test_readers_contain_no_fit_or_aggregation() -> None:
    for token in (".fit(", "groupby", ".mean(", ".std("):
        assert token not in _READER_SRC, f"{token!r} found in a read-only function"


def test_only_the_writer_aggregates() -> None:
    assert "groupby" in _WRITER_SRC  # fit is the one place aggregation happens
    assert "groupby" not in _READER_SRC


def test_no_hardcoded_config_values_in_functions() -> None:
    """The six IDs, the artefact filename, and the min_std value come from config — never a
    literal in the function bodies (module docstring is exempt)."""
    for token in (*NORMALISED_FEATURE_IDS, "normalisation.json", "1e-8", "1.0e-8", "1e-08"):
        assert token not in _FUNC_SRC, f"hardcoded config value {token!r} in function body"


# ----------------------------------------------------------------------------------
# Coded fallbacks (every NORMALISATION_FALLBACK path)
# ----------------------------------------------------------------------------------


def test_unknown_circuit_falls_back_to_global(normalise_config: AppConfig) -> None:
    res = fit_normaliser(feature_frame({"Bahrain": 35, "Monaco": 35}), config=normalise_config)
    params = load_normalisation_params(res.artefact_path)
    out = apply_normalisation(
        _full_vector(), circuit="Suzuka", params=params, config=normalise_config
    )
    assert out.used_global is True
    assert out.status is Status.DEGRADED
    assert out.error_code is ErrorCode.NORMALISATION_FALLBACK
    assert out.logged is True
    assert set(out.normalised_vector) == set(FEATURE_IDS)
    rows = read_dq(normalise_config.paths.data_quality_log)
    apply_rows = [r for r in rows if norm.REASON_UNKNOWN_CIRCUIT in r["error_detail"]]
    assert len(apply_rows) == 1 and apply_rows[0]["error_code"] == "NORMALISATION_FALLBACK"


def test_sub_min_circuit_is_skipped_and_uses_global(normalise_config: AppConfig) -> None:
    res = fit_normaliser(
        feature_frame({"Big": 30, "Small": 29}), config=normalise_config
    )
    assert res.skipped_circuits == ["Small"]
    assert res.status is Status.DEGRADED
    assert res.error_code is ErrorCode.NORMALISATION_FALLBACK  # degraded fit names the condition
    assert res.logged is True
    params = load_normalisation_params(res.artefact_path)
    assert "Big" in params.per_circuit and "Small" not in params.per_circuit
    out = apply_normalisation(
        _full_vector(), circuit="Small", params=params, config=normalise_config
    )
    assert out.used_global is True  # skipped circuit → global at apply
    rows = read_dq(normalise_config.paths.data_quality_log)
    assert any(norm.REASON_SUB_MIN_CIRCUIT in r["error_detail"] for r in rows)


def test_zero_variance_feature_centres_without_nan(normalise_config: AppConfig) -> None:
    """A feature constant at one circuit (std=0) falls back to the global std — no NaN/Inf, one
    logged ZERO_VARIANCE row for that circuit×feature."""
    const = feature_frame({"Flat": 35}, constant_feature="F09")  # F09 zero variance at Flat
    vary = feature_frame({"Hills": 35})  # F09 varies → global F09 std > 0
    df = pd.concat([const, vary], ignore_index=True)
    res = fit_normaliser(df, config=normalise_config)
    assert res.status is Status.DEGRADED  # zero-variance fallback names the condition
    assert res.error_code is ErrorCode.NORMALISATION_FALLBACK
    assert res.logged is True
    params = load_normalisation_params(res.artefact_path)
    stats = params.per_circuit["Flat"]["F09"]
    assert math.isfinite(stats.std) and stats.std > 0  # took global std, not 0
    out = apply_normalisation(
        _full_vector(120.0), circuit="Flat", params=params, config=normalise_config
    )
    assert math.isfinite(out.normalised_vector["F09"])
    rows = read_dq(normalise_config.paths.data_quality_log)
    zv = [
        r
        for r in rows
        if norm.REASON_ZERO_VARIANCE in r["error_detail"] and "F09" in r["error_detail"]
    ]
    assert len(zv) == 1 and zv[0]["entity_type"] == "circuit"


def test_single_constant_circuit_centres_to_one(normalise_config: AppConfig) -> None:
    """When the global std is also zero, the feature is centred only (std=1.0) — still finite."""
    df = feature_frame({"Flat": 35}, constant_feature="F11")
    res = fit_normaliser(df, config=normalise_config)
    params = load_normalisation_params(res.artefact_path)
    assert params.global_["F11"].std == 1.0
    out = apply_normalisation(
        _full_vector(105.0), circuit="Flat", params=params, config=normalise_config
    )
    assert math.isfinite(out.normalised_vector["F11"])


def test_null_input_feature_passthrough(normalise_config: AppConfig) -> None:
    res = fit_normaliser(feature_frame({"Bahrain": 35}), config=normalise_config)
    params = load_normalisation_params(res.artefact_path)
    vec = _full_vector()
    vec["F13"] = None
    out = apply_normalisation(vec, circuit="Bahrain", params=params, config=normalise_config)
    assert "F13" in out.fallback_features
    assert out.normalised_vector["F13"] is None
    assert out.error_code is ErrorCode.NORMALISATION_FALLBACK
    assert out.logged is True
    rows = read_dq(normalise_config.paths.data_quality_log)
    assert any(norm.REASON_NULL_INPUT in r["error_detail"] for r in rows)


def test_apply_never_raises_on_non_numeric(normalise_config: AppConfig) -> None:
    """A garbage (non-float) value among the six is caught per-feature, never propagated."""
    res = fit_normaliser(feature_frame({"Bahrain": 35}), config=normalise_config)
    params = load_normalisation_params(res.artefact_path)
    vec: dict[str, float | None] = _full_vector()
    vec["F09"] = "not-a-number"  # type: ignore[assignment]
    out = apply_normalisation(vec, circuit="Bahrain", params=params, config=normalise_config)
    assert "F09" in out.fallback_features
    assert out.error_code is ErrorCode.NORMALISATION_FALLBACK
    assert out.logged is True
    # The UNEXPECTED catch must still write the audit row — assert it, so a deleted ledger
    # call can't silently drop the trail while the type contract still appears to hold.
    rows = read_dq(normalise_config.paths.data_quality_log)
    unexpected = [
        r
        for r in rows
        if norm.REASON_UNEXPECTED in r["error_detail"] and "F09" in r["error_detail"]
    ]
    assert len(unexpected) == 1 and unexpected[0]["error_code"] == "NORMALISATION_FALLBACK"


# ----------------------------------------------------------------------------------
# Fail-loud load
# ----------------------------------------------------------------------------------


def test_load_missing_artefact_raises(normalise_config: AppConfig) -> None:
    with pytest.raises(NormalisationError, match="not found"):
        load_normalisation_params(_art(normalise_config, "absent.json"))


def test_load_corrupt_json_raises(normalise_config: AppConfig, tmp_path: Path) -> None:
    p = tmp_path / "corrupt.json"
    p.write_text("{ this is not valid json", encoding="utf-8")
    with pytest.raises(NormalisationError, match="corrupt|unreadable"):
        load_normalisation_params(p)


def test_load_schema_mismatch_raises(normalise_config: AppConfig, tmp_path: Path) -> None:
    p = tmp_path / "wrong_schema.json"
    p.write_text(json.dumps({"foo": "bar"}), encoding="utf-8")
    with pytest.raises(NormalisationError, match="schema mismatch"):
        load_normalisation_params(p)


def test_load_tampered_hash_raises(normalise_config: AppConfig) -> None:
    res = fit_normaliser(feature_frame({"Bahrain": 35}), config=normalise_config)
    path = Path(res.artefact_path)
    raw = json.loads(path.read_text())
    raw["per_circuit"]["Bahrain"]["F09"]["mean"] += 1.0  # tamper, leave hash stale
    path.write_text(json.dumps(raw, sort_keys=True, indent=2), encoding="utf-8")
    with pytest.raises(NormalisationError, match="content_hash mismatch"):
        load_normalisation_params(path)


def test_load_wrong_schema_version_raises(normalise_config: AppConfig) -> None:
    res = fit_normaliser(feature_frame({"Bahrain": 35}), config=normalise_config)
    path = Path(res.artefact_path)
    raw = json.loads(path.read_text())
    raw["schema_version"] = "0.0"
    # Re-hash the tampered body so we isolate the schema_version check (not the hash check).
    body = {k: v for k, v in raw.items() if k != "content_hash"}
    raw["content_hash"] = sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path.write_text(json.dumps(raw, sort_keys=True, indent=2), encoding="utf-8")
    with pytest.raises(NormalisationError, match="schema_version"):
        load_normalisation_params(path)


# ----------------------------------------------------------------------------------
# Fail-loud fit (degenerate inputs never write a bad artefact)
# ----------------------------------------------------------------------------------


def test_fit_missing_columns_raises(normalise_config: AppConfig) -> None:
    df = feature_frame({"Bahrain": 35}).drop(columns=["F09"])
    with pytest.raises(NormalisationError, match="missing required columns"):
        fit_normaliser(df, config=normalise_config)


def test_fit_no_train_rows_raises(normalise_config: AppConfig) -> None:
    df = feature_frame({"Bahrain": 35}, split="val")  # nothing in the train split
    with pytest.raises(NormalisationError, match="no training rows"):
        fit_normaliser(df, config=normalise_config)


def test_fit_config_normalise_absent_raises(normalise_config: AppConfig) -> None:
    """A config without the normalise: block fails loud (typed), never silently no-ops."""
    normalise_config.normalise = None
    with pytest.raises(NormalisationError, match="block absent"):
        fit_normaliser(feature_frame({"Bahrain": 35}), config=normalise_config)


def test_assert_no_leakage_config_normalise_absent_raises(normalise_config: AppConfig) -> None:
    """Called directly without the normalise: block, the guard raises NormalisationError (not a
    bare AttributeError) — non-negotiable #1 holds even on the direct-call path."""
    normalise_config.normalise = None
    with pytest.raises(NormalisationError, match="block absent"):
        norm._assert_no_leakage(feature_frame({"Bahrain": 5}), normalise_config)


# ----------------------------------------------------------------------------------
# Boundaries + config-driven thresholds (pinned both sides)
# ----------------------------------------------------------------------------------


def test_min_clean_laps_boundary_30_vs_29(normalise_config: AppConfig) -> None:
    res = fit_normaliser(feature_frame({"Exactly30": 30, "Only29": 29}), config=normalise_config)
    assert "Exactly30" not in res.skipped_circuits
    assert res.skipped_circuits == ["Only29"]


def test_min_clean_laps_is_config_driven(normalise_config: AppConfig) -> None:
    """Raising the floor to 40 skips a 35-lap circuit that was fitted at 30 (no hardcoded 30)."""
    normalise_config.thresholds.min_clean_laps = 40
    res = fit_normaliser(feature_frame({"Mid": 35}), config=normalise_config)
    assert res.skipped_circuits == ["Mid"]


def test_min_std_at_boundary_is_kept(normalise_config: AppConfig) -> None:
    """std exactly == min_std is KEPT (inclusive `>=`); the measured ramp std is used as min_std."""
    n = 35
    measured = _expected_std(n)
    normalise_config.normalise.min_std = measured  # type: ignore[union-attr]
    res = fit_normaliser(feature_frame({"Bahrain": n}), config=normalise_config)
    rows = read_dq(normalise_config.paths.data_quality_log)
    assert not any(norm.REASON_ZERO_VARIANCE in r["error_detail"] for r in rows)
    params = load_normalisation_params(res.artefact_path)
    assert params.per_circuit["Bahrain"]["F09"].std == pytest.approx(measured)


def test_min_std_just_above_triggers_fallback(normalise_config: AppConfig) -> None:
    """min_std just above the ramp std forces every normalised feature to zero-variance fallback
    (proves min_std is config-driven, not a hardcoded 1e-8)."""
    n = 35
    normalise_config.normalise.min_std = _expected_std(n) * 2  # type: ignore[union-attr]
    fit_normaliser(feature_frame({"Bahrain": n}), config=normalise_config)
    rows = read_dq(normalise_config.paths.data_quality_log)
    zv = {
        fid
        for fid in NORMALISED_FEATURE_IDS
        for r in rows
        if norm.REASON_ZERO_VARIANCE in r["error_detail"] and fid in r["error_detail"]
    }
    assert zv == set(NORMALISED_FEATURE_IDS)


def test_artefact_filename_is_config_driven(normalise_config: AppConfig) -> None:
    normalise_config.normalise.artefact_filename = "custom_norm.json"  # type: ignore[union-attr]
    res = fit_normaliser(feature_frame({"Bahrain": 35}), config=normalise_config)
    assert Path(res.artefact_path).name == "custom_norm.json"
    assert Path(res.artefact_path).is_file()


def test_normalised_features_come_from_config(normalise_config: AppConfig) -> None:
    res = fit_normaliser(feature_frame({"Bahrain": 35}), config=normalise_config)
    assert res.normalised_features == list(normalise_config.normalise.features)
    assert tuple(res.normalised_features) == NORMALISED_FEATURE_IDS


# ----------------------------------------------------------------------------------
# Config-load guards for the chokepoint (leakage scope + quarantine list)
# ----------------------------------------------------------------------------------


def test_normalise_features_must_be_in_canonical_order() -> None:
    """Reordering the six IDs is rejected at config-load: the artefact stores and hashes the
    feature order, so a reorder would silently change content_hash and break byte-determinism."""
    reordered = list(reversed(NORMALISED_FEATURE_IDS))
    with pytest.raises(ValidationError, match="in this order"):
        NormaliseConfig(
            features=reordered,
            grouping_key="circuit",
            artefact_filename="normalisation.json",
            ddof=0,
            min_std=1e-8,
        )


def test_empty_held_out_circuits_is_rejected(repo_config_path: Path, tmp_path: Path) -> None:
    """An empty held_out_circuits would silently disable the train-mask exclusion AND the
    _assert_no_leakage tripwire — so config-load rejects it (CONFIG_001), never a silent hole."""
    raw = yaml.safe_load(Path(repo_config_path).read_text(encoding="utf-8"))
    raw["held_out_circuits"] = []
    p = tmp_path / "empty_held_out.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="held_out_circuits"):
        load_config(p)


def _normalise_cfg_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "features": list(NORMALISED_FEATURE_IDS),
        "grouping_key": "circuit",
        "artefact_filename": "normalisation.json",
        "ddof": 0,
        "min_std": 1e-8,
    }
    base.update(overrides)
    return base


def test_grouping_key_must_be_circuit() -> None:
    with pytest.raises(ValidationError, match="grouping_key must be 'circuit'"):
        NormaliseConfig(**_normalise_cfg_kwargs(grouping_key="driver"))  # type: ignore[arg-type]


def test_artefact_filename_must_be_non_empty() -> None:
    with pytest.raises(ValidationError, match="artefact_filename"):
        NormaliseConfig(**_normalise_cfg_kwargs(artefact_filename=""))  # type: ignore[arg-type]


def test_ddof_must_be_non_negative() -> None:
    with pytest.raises(ValidationError, match="ddof"):
        NormaliseConfig(**_normalise_cfg_kwargs(ddof=-1))  # type: ignore[arg-type]


def test_min_std_must_be_positive() -> None:
    with pytest.raises(ValidationError, match="min_std"):
        NormaliseConfig(**_normalise_cfg_kwargs(min_std=0.0))  # type: ignore[arg-type]


# ----------------------------------------------------------------------------------
# NaN / non-finite safety (fit stores no NaN; apply ships no NaN; null circuit fails loud)
# ----------------------------------------------------------------------------------


def test_all_null_feature_at_circuit_falls_back_to_global_mean(normalise_config: AppConfig) -> None:
    """A feature entirely NULL at one circuit must NOT store mean=NaN: it falls back to the global
    mean (finite, since the feature varies elsewhere) and logs NON_FINITE_MEAN. The artefact stays
    valid JSON (no bare NaN token)."""
    df = feature_frame({"Bahrain": 35, "Monaco": 35})
    df.loc[df["circuit"] == "Bahrain", "F09"] = np.nan
    res = fit_normaliser(df, config=normalise_config)
    params = load_normalisation_params(res.artefact_path)
    stats = params.per_circuit["Bahrain"]["F09"]
    assert math.isfinite(stats.mean) and math.isfinite(stats.std)
    assert "NaN" not in Path(res.artefact_path).read_text()
    rows = read_dq(normalise_config.paths.data_quality_log)
    assert any(
        norm.REASON_NON_FINITE_MEAN in r["error_detail"] and "F09" in r["error_detail"]
        for r in rows
    )


def test_globally_all_null_feature_centres_to_zero(normalise_config: AppConfig) -> None:
    """When a feature is all-NULL across the whole train split, the global mean cannot be computed;
    it is centred to 0.0 (std 1.0) so the artefact is finite + valid JSON, never NaN."""
    df = feature_frame({"Bahrain": 35})
    df["F09"] = np.nan
    res = fit_normaliser(df, config=normalise_config)
    params = load_normalisation_params(res.artefact_path)
    assert params.global_["F09"].mean == 0.0
    assert params.global_["F09"].std == 1.0
    assert math.isfinite(params.per_circuit["Bahrain"]["F09"].mean)
    assert "NaN" not in Path(res.artefact_path).read_text()


def test_nan_input_feature_is_treated_as_missing(normalise_config: AppConfig) -> None:
    """A NaN among the six at apply is NOT normalised to NaN: it is nulled, listed in
    fallback_features, coded, and logged (NON_FINITE_INPUT) — never shipped to a model."""
    res = fit_normaliser(feature_frame({"Bahrain": 35}), config=normalise_config)
    params = load_normalisation_params(res.artefact_path)
    vec = _full_vector()
    vec["F11"] = float("nan")
    out = apply_normalisation(vec, circuit="Bahrain", params=params, config=normalise_config)
    assert out.normalised_vector["F11"] is None
    assert "F11" in out.fallback_features
    assert out.error_code is ErrorCode.NORMALISATION_FALLBACK
    assert out.logged is True
    rows = read_dq(normalise_config.paths.data_quality_log)
    assert any(
        norm.REASON_NON_FINITE_INPUT in r["error_detail"] and "F11" in r["error_detail"]
        for r in rows
    )


def test_null_circuit_key_in_fit_raises(normalise_config: AppConfig) -> None:
    """A training row with a null circuit key would be dropped from the per-circuit fit yet skew
    global params — fit_normaliser fails loud rather than fit on inconsistent populations."""
    df = feature_frame({"Bahrain": 35})
    df.loc[df.index[:3], "circuit"] = np.nan
    with pytest.raises(NormalisationError, match="null 'circuit' key"):
        fit_normaliser(df, config=normalise_config)
