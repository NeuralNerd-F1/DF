"""``app/api/profiles.py`` — the profiles artefact loader + the key-signal ranking.

The loader's failure paths matter most: a missing/corrupt artefact must raise the coded error AND
write a ``data_quality_log`` row (non-negotiable #1) — startup then degrades the field to ``None``
and the endpoint serialises honest nulls. ``top_signals`` is the one place attribution magnitude
and grid rank meet; its tests pin that neither is ever inferred from the other.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from app.api.profiles import (
    PROFILES_FILENAME,
    DriverProfile,
    DriverProfilesUnavailableError,
    load_driver_profiles,
    top_signals,
)
from app.contracts.config import AppConfig, load_config

from tests.scripts.conftest import read_dq


@pytest.fixture
def cfg(repo_config_path: Path, tmp_path: Path) -> AppConfig:
    config = load_config(repo_config_path)
    config.paths.data_quality_log = str(tmp_path / "dq.jsonl")
    return config


def write_artefact(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / PROFILES_FILENAME
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


VALID = {
    "content_hash": "abc",
    "n_laps": 100,
    "held_out_circuits": ["Belgium", "Italy"],
    "source_matrix": "m.parquet",
    "drivers": {
        "VER": {
            "laps_analysed": 60,
            "seasons": [2022, 2023],
            "n_circuits": 5,
            "lap_time_consistency_s": 1.1,
            "identification_f1": 0.53,
            "feature_percentile": {"F01": 0.5, "F02": 1.0},
        },
        "UNK": {
            "laps_analysed": 40,
            "seasons": [2024],
            "n_circuits": 3,
            "lap_time_consistency_s": None,  # too few laps per session — honest absence
            "identification_f1": None,  # never scored by the champion summary
            "feature_percentile": {},
        },
    },
}


# --- load_driver_profiles ---------------------------------------------------------------------


def test_load_parses_fields_including_honest_nulls(cfg: AppConfig, tmp_path: Path) -> None:
    write_artefact(tmp_path, VALID)
    profiles = load_driver_profiles(config=cfg, artefact_dir=tmp_path)
    ver = profiles.get("VER")
    assert ver is not None
    assert ver.laps_analysed == 60
    assert ver.seasons == (2022, 2023)
    assert ver.identification_f1 == 0.53
    unk = profiles.get("UNK")
    assert unk is not None
    assert unk.lap_time_consistency_s is None  # None survives, never coerced to 0.0
    assert unk.identification_f1 is None
    assert profiles.held_out_circuits == ("Belgium", "Italy")


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        (None, "missing"),
        ("{not json", "unreadable"),
        ({"drivers": "nope"}, "no drivers object"),
        ({"drivers": {}}, "empty drivers"),
        ({"drivers": {"VER": {"seasons": []}}}, "driver entry missing laps_analysed"),
    ],
)
def test_load_failure_paths_raise_coded_and_log(
    cfg: AppConfig, tmp_path: Path, payload: object, reason: str
) -> None:
    """Every failure shape → the typed error + a data_quality_log row. Never a silent None."""
    if payload == "{not json":
        (tmp_path / PROFILES_FILENAME).write_text("{not json", encoding="utf-8")
    elif payload is not None:
        write_artefact(tmp_path, payload)
    with pytest.raises(DriverProfilesUnavailableError):
        load_driver_profiles(config=cfg, artefact_dir=tmp_path)
    rows = read_dq(cfg.paths.data_quality_log)
    assert rows, f"no DQ row written for: {reason}"
    assert rows[-1]["error_code"] == "DATA_SOURCE_UNAVAILABLE"
    assert rows[-1]["entity_type"] == "driver_profiles"


# --- top_signals -------------------------------------------------------------------------------

PROFILE = DriverProfile(
    laps_analysed=60,
    seasons=(2022,),
    n_circuits=5,
    lap_time_consistency_s=1.1,
    identification_f1=0.5,
    feature_percentile={"F01": 0.9, "F02": 0.1},
)


def test_top_signals_ranks_by_magnitude_and_joins_percentile() -> None:
    shap = {"F01": 0.2, "F02": 0.9, "F03": 0.5}
    out = top_signals(shap, PROFILE, limit=2)
    assert [s.feature for s in out] == ["F02", "F03"]
    assert out[0].percentile == 0.1  # joined from the profile, NOT derived from attribution
    assert out[1].percentile is None  # F03 has no rank — stays None, never guessed


def test_top_signals_tie_breaks_on_feature_id() -> None:
    out = top_signals({"F09": 0.5, "F03": 0.5, "F01": 0.5}, None)
    assert [s.feature for s in out] == ["F01", "F03", "F09"]  # deterministic order


def test_top_signals_empty_or_absent_attribution_returns_empty() -> None:
    """No stored SHAP → []: the card says 'no attribution', it never falls back to raw values."""
    assert top_signals(None, PROFILE) == []
    assert top_signals({}, PROFILE) == []


def test_top_signals_ignores_non_numeric_values() -> None:
    out = top_signals({"F01": 0.5, "F02": "bad"}, None)  # type: ignore[dict-item]
    assert [s.feature for s in out] == ["F01"]


def test_load_parses_shap_signed_mean(cfg: AppConfig, tmp_path: Path) -> None:
    payload = {
        "drivers": {
            "VER": {
                "laps_analysed": 10,
                "seasons": [2024],
                "n_circuits": 1,
                "lap_time_consistency_s": None,
                "identification_f1": None,
                "feature_percentile": {},
                "shap_signed_mean": {"F01": -0.25, "F16": 0.455},
            }
        }
    }
    write_artefact(tmp_path, payload)
    profiles = load_driver_profiles(config=cfg, artefact_dir=tmp_path)
    ver = profiles.get("VER")
    assert ver is not None
    assert ver.shap_signed_mean == {"F01": -0.25, "F16": 0.455}  # sign survives the round trip


def test_load_pre_signed_artefact_defaults_to_empty(cfg: AppConfig, tmp_path: Path) -> None:
    """An artefact built before the signed fill parses with {} — old artefacts stay loadable."""
    write_artefact(tmp_path, VALID)  # VALID has no shap_signed_mean key
    ver = load_driver_profiles(config=cfg, artefact_dir=tmp_path).get("VER")
    assert ver is not None
    assert ver.shap_signed_mean == {}


def test_load_parses_segments(cfg: AppConfig, tmp_path: Path) -> None:
    payload = {
        "drivers": VALID["drivers"],
        "segments": [
            {
                "id": "Belgium-2022-VER-34",
                "label": "Belgium 2022 · Race · Lap 34",
                "circuit": "Belgium",
                "season": 2022,
                "session": "Race",
                "lap_number": 34,
                "true_driver": "VER",
                "features": {"F01": 62.1, "F02": None},
            }
        ],
    }
    write_artefact(tmp_path, payload)
    profiles = load_driver_profiles(config=cfg, artefact_dir=tmp_path)
    assert len(profiles.segments) == 1
    seg = profiles.segments[0]
    assert seg.true_driver == "VER"
    assert seg.features["F01"] == 62.1
    assert seg.features["F02"] is None  # a lap's genuine NULL survives the round trip


def test_load_segmentless_artefact_defaults_to_empty(cfg: AppConfig, tmp_path: Path) -> None:
    write_artefact(tmp_path, VALID)  # no segments key
    assert load_driver_profiles(config=cfg, artefact_dir=tmp_path).segments == ()


def test_load_malformed_segments_raise_coded(cfg: AppConfig, tmp_path: Path) -> None:
    write_artefact(tmp_path, {"drivers": VALID["drivers"], "segments": [{"id": "x"}]})
    with pytest.raises(DriverProfilesUnavailableError):
        load_driver_profiles(config=cfg, artefact_dir=tmp_path)
    assert read_dq(cfg.paths.data_quality_log)[-1]["error_code"] == "DATA_SOURCE_UNAVAILABLE"


def test_load_parses_tyre_wear_passthrough(cfg: AppConfig, tmp_path: Path) -> None:
    payload = {
        "drivers": {
            "VER": {
                "laps_analysed": 10,
                "seasons": [2024],
                "n_circuits": 1,
                "lap_time_consistency_s": None,
                "identification_f1": None,
                "feature_percentile": {},
                "tyre_wear": {"MEDIUM": {"n_laps": 100, "deg_slope_s_per_lap": 0.06}},
            }
        }
    }
    write_artefact(tmp_path, payload)
    ver = load_driver_profiles(config=cfg, artefact_dir=tmp_path).get("VER")
    assert ver is not None
    assert ver.tyre_wear["MEDIUM"]["deg_slope_s_per_lap"] == 0.06  # raw mapping passthrough


def test_load_non_dict_tyre_wear_raises_coded(cfg: AppConfig, tmp_path: Path) -> None:
    payload = {"drivers": {"VER": {"laps_analysed": 1, "tyre_wear": ["nope"]}}}
    write_artefact(tmp_path, payload)
    with pytest.raises(DriverProfilesUnavailableError):
        load_driver_profiles(config=cfg, artefact_dir=tmp_path)
