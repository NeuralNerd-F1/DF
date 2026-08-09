"""GET /drivers, /drivers/{id}/fingerprint, /setup/{circuit}, /stability."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app.api.deps import get_recommender

from tests.api.conftest import fake_setup, seed_fingerprint


async def test_list_drivers_empty_is_200(api: SimpleNamespace) -> None:
    r = await api.client.get("/drivers")
    assert r.status_code == 200
    assert r.json() == []


async def test_list_drivers_with_data(api: SimpleNamespace) -> None:
    await seed_fingerprint(api.session_factory, code="VER")
    r = await api.client.get("/drivers")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1
    assert body[0]["code"] == "VER"
    assert body[0]["radar_values"]["throttle"] == 0.5


async def test_list_drivers_surfaces_team(api: SimpleNamespace) -> None:
    """F16 seam — the team-colour bar needs the driver's team on the summary; a driver with no
    team (unknown driver) serialises ``team: null``, never a fabricated value (§4.1 / div #47)."""
    await seed_fingerprint(api.session_factory, code="VER", team="Red Bull")
    await seed_fingerprint(api.session_factory, code="UNK", team=None)
    body = {d["code"]: d for d in (await api.client.get("/drivers")).json()}
    assert body["VER"]["team"] == "Red Bull"
    assert body["UNK"]["team"] is None


async def test_driver_fingerprint_three_seasons_is_200(api: SimpleNamespace) -> None:
    for season in (2022, 2023, 2024):
        await seed_fingerprint(api.session_factory, code="VER", season=season)
    r = await api.client.get("/drivers/VER/fingerprint")
    assert r.status_code == 200  # 3 seasons -> not partial
    body = r.json()
    assert len(body["vector_5d"]) == 5
    assert body["stability"]["seasons_available"] == 3


async def test_driver_fingerprint_partial_seasons_is_206(api: SimpleNamespace) -> None:
    await seed_fingerprint(api.session_factory, code="VER", season=2023)
    r = await api.client.get("/drivers/VER/fingerprint")
    assert r.status_code == 206  # < 3 seasons -> partial


async def test_driver_fingerprint_not_found_is_404(api: SimpleNamespace) -> None:
    r = await api.client.get("/drivers/NOPE/fingerprint")
    assert r.status_code == 404
    assert r.json()["error_code"] == "NOT_FOUND"


async def test_driver_stability_partial_is_206(api: SimpleNamespace) -> None:
    await seed_fingerprint(api.session_factory, code="VER", season=2023)
    r = await api.client.get("/drivers/VER/stability")
    assert r.status_code == 206
    assert r.json()["partial"] is True


async def test_driver_setup_ok(api: SimpleNamespace) -> None:
    api.app.dependency_overrides[get_recommender] = lambda: (
        lambda driver, circuit, ct: fake_setup(driver, circuit)
    )
    r = await api.client.get("/drivers/VER/setup/Monaco")
    assert r.status_code == 200
    body = r.json()
    assert body["driver"] == "VER"
    assert body["parameters"][0]["name"] == "front_wing_angle"
    assert body["parameters"][0]["rating"] == "HIGH"
    assert "approximated" in body["approximation_disclaimer"]
    # System input #2: the grid reference + heuristic basis ride the wire for the tendency line.
    assert body["parameters"][0]["grid_mean"] == 4.2
    assert body["parameters"][0]["basis_features"] == ["F19", "F14"]


async def test_driver_setup_unknown_driver_is_404(api: SimpleNamespace) -> None:
    # "ZZZ" is not in the fake style_handle.drivers (["VER","HAM"]) -> 404
    api.app.dependency_overrides[get_recommender] = lambda: (
        lambda driver, circuit, ct: fake_setup(driver, circuit)
    )
    r = await api.client.get("/drivers/ZZZ/setup/Monaco")
    assert r.status_code == 404
    assert r.json()["error_code"] == "NOT_FOUND"


async def test_driver_setup_wide_confidence_in_body(api: SimpleNamespace) -> None:
    """§8.5 — a low-reliability parameter is served with a WIDE rating + a value_range, at 200."""
    from app.contracts.errors import ErrorCode
    from app.personalisation.engine import (
        SetupConfidence,
        SetupOutputKind,
        SetupParameter,
        SetupRecommendation,
    )

    wide = SetupParameter(
        name="front_wing_angle",
        kind=SetupOutputKind.DEGREES,
        value=None,  # WIDE returns a range, not a point
        value_range=(2.0, 8.0),
        confidence=0.1,
        rating=SetupConfidence.WIDE,
        unit="deg",
        fia_min=0.0,
        fia_max=12.0,
        constraint_flags=[ErrorCode.WIDE_CONFIDENCE],
    )
    rec = SetupRecommendation(
        driver="VER",
        circuit="Monaco",
        circuit_type=None,
        parameters=[wide],
        approximation_disclaimer="Setup targets are approximated from telemetry, not measured.",
        model_version="test-v1",
    )
    api.app.dependency_overrides[get_recommender] = lambda: (lambda d, c, ct: rec)
    r = await api.client.get("/drivers/VER/setup/Monaco")
    assert r.status_code == 200
    p = r.json()["parameters"][0]
    assert p["rating"] == "WIDE"
    assert p["value"] is None
    assert p["value_range"] == [2.0, 8.0]
    assert "WIDE_CONFIDENCE" in p["constraint_flags"]


def _full_radar(throttle: float) -> dict[str, float]:
    return {
        "throttle": throttle, "braking": 0.3, "speed": 0.7,
        "gear": 0.4, "engine": 0.6, "cornering": 0.5,
    }


async def test_radar_endpoints_agree_on_latest_season(api: SimpleNamespace) -> None:
    """§11/§4.0(i) — /drivers, /drivers/{id}/fingerprint and /compare must all report the SAME
    (latest-season) radar for a driver; no oldest-vs-latest divergence between endpoints."""
    await seed_fingerprint(api.session_factory, code="VER", season=2022, radar=_full_radar(0.2))
    await seed_fingerprint(api.session_factory, code="VER", season=2024, radar=_full_radar(0.8))
    await seed_fingerprint(api.session_factory, code="HAM", season=2023, radar=_full_radar(0.5))

    listed = {d["code"]: d["radar_values"] for d in (await api.client.get("/drivers")).json()}
    fp = (await api.client.get("/drivers/VER/fingerprint")).json()["shap_radar"]
    cmp_ = (await api.client.get("/compare/VER/HAM")).json()["driver_1_radar"]

    assert listed["VER"]["throttle"] == 0.8  # latest season (2024), not 0.2 (2022)
    assert fp["throttle"] == 0.8
    assert cmp_["throttle"] == 0.8


async def test_driver_stability_three_seasons_is_200(api: SimpleNamespace) -> None:
    for season in (2022, 2023, 2024):
        await seed_fingerprint(api.session_factory, code="VER", season=season)
    r = await api.client.get("/drivers/VER/stability")
    assert r.status_code == 200  # 3 seasons -> not partial
    assert r.json()["partial"] is False
    assert r.json()["seasons_available"] == 3


async def test_driver_stability_unknown_is_404(api: SimpleNamespace) -> None:
    r = await api.client.get("/drivers/NOPE/stability")
    assert r.status_code == 404
    assert r.json()["error_code"] == "NOT_FOUND"


async def test_driver_setup_returns_all_parameters(api: SimpleNamespace) -> None:
    """§8.5 — the endpoint returns every parameter F11 provides (the 10-param sheet is F11's
    contract; F13 maps 1:1 and must not drop any)."""
    from app.personalisation.engine import (
        SetupConfidence,
        SetupOutputKind,
        SetupParameter,
        SetupRecommendation,
    )

    params = [
        SetupParameter(
            name=f"param_{i}",
            kind=SetupOutputKind.DEGREES,
            value=float(i),
            confidence=0.8,
            rating=SetupConfidence.HIGH,
            unit="deg",
            fia_min=0.0,
            fia_max=12.0,
        )
        for i in range(10)
    ]
    rec = SetupRecommendation(
        driver="VER",
        circuit="Monaco",
        circuit_type=None,
        parameters=params,
        approximation_disclaimer="Setup targets are approximated from telemetry, not measured.",
        model_version="test-v1",
    )
    api.app.dependency_overrides[get_recommender] = lambda: (lambda d, c, ct: rec)
    r = await api.client.get("/drivers/VER/setup/Monaco")
    assert r.status_code == 200
    assert len(r.json()["parameters"]) == 10
    assert [p["name"] for p in r.json()["parameters"]] == [f"param_{i}" for i in range(10)]


# --- /drivers enrichment from the profiles artefact (Driver Overview) --------------------------


async def test_list_drivers_without_profiles_serialises_honest_nulls(
    api: SimpleNamespace,
) -> None:
    """A checkout with no profiles artefact still serves the row — profile null, and the
    signature ranked from the stored shap_global with percentile null (no rank source)."""
    await seed_fingerprint(api.session_factory, code="VER", shap_global={"F01": 0.2, "F02": 0.9})
    body = (await api.client.get("/drivers")).json()
    assert body[0]["profile"] is None
    assert [s["feature"] for s in body[0]["signature"]] == ["F02", "F01"]
    assert all(s["percentile"] is None for s in body[0]["signature"])


async def test_list_drivers_enriched_from_profiles(api: SimpleNamespace) -> None:
    """With the artefact loaded, the summary carries the dataset facts and the signature joins
    the grid percentile; a driver ABSENT from the artefact keeps profile null — never borrowed."""
    from app.api.profiles import DriverProfile, DriverProfiles

    await seed_fingerprint(api.session_factory, code="VER", shap_global={"F01": 0.2, "F02": 0.9})
    await seed_fingerprint(api.session_factory, code="UNK", shap_global={"F03": 0.5})
    api.state.driver_profiles = DriverProfiles(
        content_hash="abc",
        n_laps=100,
        held_out_circuits=("Belgium", "Italy"),
        source_matrix="m.parquet",
        by_code={
            "VER": DriverProfile(
                laps_analysed=60,
                seasons=(2022, 2023),
                n_circuits=5,
                lap_time_consistency_s=1.14,
                identification_f1=0.53,
                feature_percentile={"F02": 1.0},
            )
        },
    )
    body = {d["code"]: d for d in (await api.client.get("/drivers")).json()}
    ver = body["VER"]
    assert ver["profile"] == {
        "laps_analysed": 60,
        "seasons": [2022, 2023],
        "n_circuits": 5,
        "lap_time_consistency_s": 1.14,
        "identification_f1": 0.53,
    }
    assert ver["signature"][0] == {"feature": "F02", "attribution": 0.9, "percentile": 1.0}
    assert ver["signature"][1]["percentile"] is None  # F01 has no rank in the artefact
    assert body["UNK"]["profile"] is None  # not in the artefact -> null, not a neighbour's facts


async def test_fingerprint_carries_signed_shap_from_profiles(api: SimpleNamespace) -> None:
    """/fingerprint serves the signed per-feature attribution from the profiles artefact; sign is
    preserved, and a checkout without the artefact serialises {} (never derived from magnitudes)."""
    from app.api.profiles import DriverProfile, DriverProfiles

    await seed_fingerprint(api.session_factory, code="VER")
    body = (await api.client.get("/drivers/VER/fingerprint")).json()
    assert body["shap_signed"] == {}  # no artefact loaded -> empty, not invented

    api.state.driver_profiles = DriverProfiles(
        content_hash="abc",
        n_laps=10,
        held_out_circuits=(),
        source_matrix="m.parquet",
        by_code={
            "VER": DriverProfile(
                laps_analysed=10,
                seasons=(2024,),
                n_circuits=1,
                lap_time_consistency_s=None,
                identification_f1=None,
                feature_percentile={},
                shap_signed_mean={"F16": 0.455, "F08": -0.001},
            )
        },
    )
    body = (await api.client.get("/drivers/VER/fingerprint")).json()
    assert body["shap_signed"] == {"F16": 0.455, "F08": -0.001}


# --- GET /tyres (Tyre Wear screen) -------------------------------------------------------------


def _profiles_with_tyres(tyre_wear: dict) -> object:
    from app.api.profiles import DriverProfile, DriverProfiles

    return DriverProfiles(
        content_hash="abc",
        n_laps=10,
        held_out_circuits=(),
        source_matrix="m.parquet",
        by_code={
            "VER": DriverProfile(
                laps_analysed=10,
                seasons=(2024,),
                n_circuits=1,
                lap_time_consistency_s=None,
                identification_f1=None,
                feature_percentile={},
                tyre_wear=tyre_wear,
            )
        },
    )


async def test_tyres_empty_without_profiles(api: SimpleNamespace) -> None:
    r = await api.client.get("/tyres")
    assert r.status_code == 200
    assert r.json() == {"drivers": {}}


async def test_tyres_serves_validated_blocks(api: SimpleNamespace) -> None:
    api.state.driver_profiles = _profiles_with_tyres(
        {
            "MEDIUM": {
                "n_laps": 1657,
                "n_stints": 83,
                "median_stint_len": 19,
                "max_stint_len": 32,
                "deg_slope_s_per_lap": 0.062,
                "total_delta_s": 1.12,
                "fuel_slope_s_per_lap": -0.062,
                "cliff_age": None,
                "curve": [{"age": 2, "delta_s": 0.0, "q25": -0.2, "q75": 0.2, "n": 62}],
                "f27_median": 0.036,
                "f28_median": None,
            }
        }
    )
    body = (await api.client.get("/tyres")).json()
    stats = body["drivers"]["VER"]["MEDIUM"]
    assert stats["deg_slope_s_per_lap"] == 0.062
    assert stats["cliff_age"] is None  # honest no-cliff survives serialisation
    assert stats["curve"][0]["n"] == 62


async def test_tyres_skips_malformed_block_with_dq_row(api: SimpleNamespace) -> None:
    """A corrupt per-compound block is dropped AND logged — served curves stay trustworthy."""
    import json as _json

    api.state.driver_profiles = _profiles_with_tyres({"MEDIUM": {"n_laps": "not-a-number"}})
    body = (await api.client.get("/tyres")).json()
    assert body == {"drivers": {}}
    log = Path(api.config.paths.data_quality_log)
    rows = [_json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert any(r["entity_type"] == "tyre_wear" and r["entity_id"] == "VER/MEDIUM" for r in rows)
