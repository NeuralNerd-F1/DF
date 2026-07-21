"""GET /drivers, /drivers/{id}/fingerprint, /setup/{circuit}, /stability."""

from __future__ import annotations

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
