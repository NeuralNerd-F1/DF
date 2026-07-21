"""POST /predict — states map to 200, gates to 400/503, never a 5xx on model output (§8.5)."""

from __future__ import annotations

from types import SimpleNamespace

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
