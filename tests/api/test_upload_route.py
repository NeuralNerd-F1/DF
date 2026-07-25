"""``POST /ingest/upload`` route tests (F05) — the multipart surface over ``process_upload``.

Drives the assembled app via the shared ``api`` fixture (``create_app`` + fake ``AppState`` +
``ASGITransport``). Medians are injected by setting ``state.medians`` (the ``get_medians`` provider
reads it) — mirroring how the pipeline tests inject a synthetic table. Asserts the HTTP status/code
mapping: fatal §3.4 edge cases → 400 with the pre-seeded code, non-fatal → 200 with ``warnings``.
"""

from __future__ import annotations

from types import SimpleNamespace

from app.pipeline.features import FEATURE_IDS, FeatureMedians

from tests.pipeline.test_upload import good_csv


def _inject_medians(api: SimpleNamespace) -> None:
    api.state.medians = FeatureMedians(global_medians=dict.fromkeys(FEATURE_IDS, 0.5))


async def _post(api: SimpleNamespace, raw: bytes, **data: str):
    return await api.client.post(
        "/ingest/upload", files={"file": ("t.csv", raw, "text/csv")}, data=data
    )


async def test_upload_route_happy_path_200(api: SimpleNamespace) -> None:
    _inject_medians(api)
    r = await _post(api, good_csv(n_laps=3), driver_label="VER")
    assert r.status_code == 200
    body = r.json()
    expected = {"upload_id", "laps_accepted", "laps_rejected", "cleaning_report", "warnings"}
    assert set(body) == expected
    assert body["upload_id"].startswith("up_")
    assert body["laps_accepted"] > 0
    assert body["cleaning_report"]["laps_total"] == 3


async def test_upload_route_no_longer_deferred(api: SimpleNamespace) -> None:
    """The F13 stub returned 503 for any call — a valid upload now succeeds (200), not 503."""
    _inject_medians(api)
    r = await _post(api, good_csv(n_laps=2))
    assert r.status_code != 503
    assert r.status_code == 200


async def test_upload_route_missing_column_400_upload_001(api: SimpleNamespace) -> None:
    _inject_medians(api)
    raw = b"LapNumber,Time,Speed,Throttle,Brake,Gear\n1,0.0,250,80,0,6\n1,0.05,250,80,0,6\n"
    r = await _post(api, raw)
    assert r.status_code == 400
    assert r.json()["error_code"] == "UPLOAD_001"


async def test_upload_route_empty_file_400_upload_004(api: SimpleNamespace) -> None:
    _inject_medians(api)
    r = await _post(api, b"LapNumber,Time,Speed,Throttle,Brake,RPM,Gear\n")
    assert r.status_code == 400
    assert r.json()["error_code"] == "UPLOAD_004"


async def test_upload_route_corrupt_over_threshold_400_upload_003(api: SimpleNamespace) -> None:
    _inject_medians(api)
    import io

    import pandas as pd

    df = pd.read_csv(io.BytesIO(good_csv(n_laps=2)))
    df["Speed"] = df["Speed"].astype(object)
    df.iloc[: int(len(df) * 0.2), df.columns.get_loc("Speed")] = "bad"
    r = await _post(api, df.to_csv(index=False).encode())
    assert r.status_code == 400
    assert r.json()["error_code"] == "UPLOAD_003"


async def test_upload_route_mph_warning_200(api: SimpleNamespace) -> None:
    _inject_medians(api)
    r = await _post(api, good_csv(n_laps=2, speed_scale=2.0))
    assert r.status_code == 200
    assert any("UPLOAD_002" in w for w in r.json()["warnings"])


async def test_upload_route_unparseable_400_invalid_input(api: SimpleNamespace) -> None:
    _inject_medians(api)
    r = await _post(api, b"\x00\x01\x02\xff not a csv \x00")
    assert r.status_code == 400
    assert r.json()["error_code"] == "INVALID_INPUT"


async def test_upload_route_unknown_driver_flag(api: SimpleNamespace) -> None:
    _inject_medians(api)
    r = await _post(api, good_csv(n_laps=2), unknown_driver="true")
    assert r.status_code == 200
    assert r.json()["cleaning_report"]["unknown_driver"] is True


async def test_upload_route_without_medians_still_returns_report(api: SimpleNamespace) -> None:
    """No medians injected (fresh-checkout posture) → 200 cleaning report, no crash."""
    api.state.medians = None
    r = await _post(api, good_csv(n_laps=3))
    assert r.status_code == 200
    body = r.json()
    assert body["laps_accepted"] == 3  # validation-clean laps
    assert any("training medians" in w for w in body["warnings"])


def test_openapi_upload_has_multipart_request_body(api: SimpleNamespace) -> None:
    """The live route publishes a multipart request body (the F17 client regenerates against it)."""
    op = api.app.openapi()["paths"]["/ingest/upload"]["post"]
    assert "requestBody" in op
    assert "multipart/form-data" in op["requestBody"]["content"]
