"""POST /predict — states map to 200, gates to 400/503, never a 5xx on model output (§8.5)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from app.api.deps import get_explainer_fn, get_predictor
from app.contracts.enums import ConfidenceState, Status
from app.contracts.errors import ErrorCode

from tests.api.conftest import fake_explanation, fake_prediction

_BODY = {"feature_vector": {"F01": 1.0, "F02": 0.5}, "circuit": "Monaco"}


def _predictor(api: SimpleNamespace, pred: object) -> None:
    api.app.dependency_overrides[get_predictor] = lambda: (lambda fv, circuit: pred)


async def test_predict_ok(api: SimpleNamespace) -> None:
    _predictor(api, fake_prediction(state=ConfidenceState.HIGH))
    r = await api.client.post("/predict", json=_BODY)
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_driver"] == "VER"
    assert body["confidence_state"] == "HIGH"
    assert len(body["top_3"]) == 2
    # No breakdown on the fake → served as an explicit null, never absent (cache-compatible field).
    assert body["confidence_breakdown"] is None


async def test_predict_serves_the_confidence_breakdown(api: SimpleNamespace) -> None:
    """System input #6: the propagated-confidence composition rides the response verbatim."""
    from app.ml.inference import ConfidenceBreakdown

    bd = ConfidenceBreakdown(
        feature_quality=0.5, distribution=0.83, model_confidence=0.91, overall=0.5 * 0.83 * 0.91
    )
    _predictor(api, fake_prediction(state=ConfidenceState.HIGH, breakdown=bd))
    r = await api.client.post("/predict", json=_BODY)
    assert r.status_code == 200
    served = r.json()["confidence_breakdown"]
    assert served["feature_quality"] == pytest.approx(0.5)
    assert served["distribution"] == pytest.approx(0.83)
    assert served["model_confidence"] == pytest.approx(0.91)
    assert served["overall"] == pytest.approx(0.5 * 0.83 * 0.91)


async def test_predict_low_confidence_is_200(api: SimpleNamespace) -> None:
    _predictor(api, fake_prediction(state=ConfidenceState.LOW_CONFIDENCE, confidence=0.3))
    r = await api.client.post("/predict", json=_BODY)
    assert r.status_code == 200
    assert r.json()["confidence_state"] == "LOW_CONFIDENCE"


async def test_predict_ambiguous_is_200(api: SimpleNamespace) -> None:
    _predictor(api, fake_prediction(state=ConfidenceState.AMBIGUOUS))
    r = await api.client.post("/predict", json=_BODY)
    assert r.status_code == 200
    assert r.json()["confidence_state"] == "AMBIGUOUS"


async def test_predict_too_many_nulls_is_400(api: SimpleNamespace) -> None:
    _predictor(
        api,
        fake_prediction(
            status=Status.FAILED, error_code=ErrorCode.INVALID_INPUT, top=None, state=None
        ),
    )
    r = await api.client.post("/predict", json=_BODY)
    assert r.status_code == 400
    assert r.json()["error_code"] == "INVALID_INPUT"


async def test_predict_malformed_body_is_422(api: SimpleNamespace) -> None:
    r = await api.client.post("/predict", json={"circuit": "Monaco"})  # feature_vector missing
    assert r.status_code == 422


async def test_predict_inference_fault_is_503_never_500(api: SimpleNamespace) -> None:
    _predictor(
        api,
        fake_prediction(
            status=Status.FAILED, error_code=ErrorCode.INFERENCE_ERROR, top=None, state=None
        ),
    )
    r = await api.client.post("/predict", json=_BODY)
    assert r.status_code == 503  # never 500 on model output (§8.5)
    # distinct code preserved — a genuine inference fault is NOT collapsed into HEALTH_FAIL
    assert r.json()["error_code"] == "INFERENCE_ERROR"


async def test_predict_shap_available(api: SimpleNamespace) -> None:
    _predictor(api, fake_prediction())
    api.app.dependency_overrides[get_explainer_fn] = lambda: (
        lambda fv, circuit, cls: fake_explanation(shap_available=True)
    )
    r = await api.client.post("/predict", json=_BODY)
    assert r.status_code == 200
    assert r.json()["shap_available"] is True
    assert r.json()["shap_values"] == {"F01": 0.2}


async def test_predict_shap_unavailable_is_200(api: SimpleNamespace) -> None:
    _predictor(api, fake_prediction())
    api.app.dependency_overrides[get_explainer_fn] = lambda: (
        lambda fv, circuit, cls: fake_explanation(shap_available=False)
    )
    r = await api.client.post("/predict", json=_BODY)
    assert r.status_code == 200
    assert r.json()["shap_available"] is False


async def test_predict_model_not_loaded_is_503(api: SimpleNamespace) -> None:
    api.state.model_handle = None  # real get_predictor (not overridden) -> 503
    r = await api.client.post("/predict", json=_BODY)
    assert r.status_code == 503
    assert r.json()["error_code"] == "HEALTH_FAIL"


async def test_predict_too_many_feature_keys_is_400(api: SimpleNamespace) -> None:
    """§11 DoS guard — an oversized feature_vector (> api.max_feature_keys) is rejected at 400
    before the model is touched."""
    body = {"feature_vector": {f"F{i}": 1.0 for i in range(200)}, "circuit": "Monaco"}
    r = await api.client.post("/predict", json=body)
    assert r.status_code == 400
    assert r.json()["error_code"] == "INVALID_INPUT"


# --- GET /segments + class_probabilities (Prediction screen) ----------------------------------


async def test_segments_empty_without_profiles(api: SimpleNamespace) -> None:
    """No profiles artefact → 200 [] — a missing demo source is not a service failure."""
    r = await api.client.get("/segments")
    assert r.status_code == 200
    assert r.json() == []


async def test_segments_serve_the_curated_laps(api: SimpleNamespace) -> None:
    from app.api.profiles import DriverProfiles, SampleSegment

    api.state.driver_profiles = DriverProfiles(
        content_hash="abc",
        n_laps=1,
        held_out_circuits=("Belgium",),
        source_matrix="m.parquet",
        by_code={},
        segments=(
            SampleSegment(
                id="Belgium-2022-VER-34",
                label="Belgium 2022 · Race · Lap 34",
                circuit="Belgium",
                season=2022,
                session="Race",
                lap_number=34,
                true_driver="VER",
                features={"F01": 62.1, "F02": None},
            ),
        ),
    )
    body = (await api.client.get("/segments")).json()
    assert len(body) == 1
    assert body[0]["true_driver"] == "VER"
    assert body[0]["features"] == {"F01": 62.1, "F02": None}


async def test_predict_serialises_the_full_class_distribution(api: SimpleNamespace) -> None:
    """PredictResponse carries every class the model scored, not just the top-3."""
    _predictor(api, fake_prediction(confidence=0.91))
    r = await api.client.post("/predict", json=_BODY)
    assert r.status_code == 200
    probs = r.json()["class_probabilities"]
    assert set(probs) == {"VER", "HAM"}  # the result's full distribution, every class present
    assert probs["VER"] == 0.91
    assert abs(sum(probs.values()) - 1.0) < 1e-9
