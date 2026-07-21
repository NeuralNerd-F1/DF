"""POST /predict — F06b classifier + F07 SHAP. Never a 5xx on model output (§8.5/§11).

A valid vector always yields a 200 carrying the confidence_state (HIGH/MEDIUM/LOW_CONFIDENCE/
AMBIGUOUS — the gate returns a state, never a forced answer). Only two non-200s: 400 on the >6-NULL
input gate, and 503 if the classifier is not loaded / an inference fault occurs (never 500).
Stateless compute — no DB, so it stays 200 even when the DB is down.
"""

from __future__ import annotations

from contextlib import suppress
from typing import Any

from fastapi import APIRouter, Depends

from app.api import metrics as metrics_mod
from app.api.deps import (
    AppState,
    ExplainFn,
    PredictFn,
    get_apex_state,
    get_cache,
    get_explainer_fn,
    get_predictor,
)
from app.api.errors import ApiError, invalid_input
from app.api.schemas import PredictRequest, PredictResponse, TopK
from app.cache.redis import RedisCache
from app.contracts.enums import Status
from app.contracts.errors import ErrorCode
from app.contracts.logging import log_data_quality

router = APIRouter(tags=["predict"])

# §11 DoS guard fallback; primary source is api.max_feature_keys (create_app requires api).
_MAX_FEATURE_KEYS_FALLBACK = 128
_DEFAULT_TTL_S = 3600


@router.post("/predict", response_model=PredictResponse)
async def post_predict(
    req: PredictRequest,
    predictor: PredictFn = Depends(get_predictor),
    explainer_fn: ExplainFn | None = Depends(get_explainer_fn),
    state: AppState = Depends(get_apex_state),
    cache: RedisCache | None = Depends(get_cache),
) -> PredictResponse:
    api = state.config.api
    max_keys = api.max_feature_keys if api is not None else _MAX_FEATURE_KEYS_FALLBACK
    if len(req.feature_vector) > max_keys:  # §11 DoS guard — reject before the model touches it
        with suppress(Exception):
            log_data_quality(
                error_code=ErrorCode.INVALID_INPUT,
                entity_type="predict",
                entity_id=req.circuit,
                error_detail=f"{len(req.feature_vector)} feature keys > {max_keys} cap",
                fallback_used=False,
                log_path=state.config.paths.data_quality_log,
            )
        raise invalid_input(f"feature_vector has too many keys (max {max_keys})")

    def _compute() -> dict[str, Any]:
        """Compute the prediction (+ best-effort SHAP) and return the response as a JSON dict. Any
        input-gate / inference fault raises here (never cached)."""
        result = predictor(req.feature_vector, req.circuit)
        if result.status is Status.FAILED:
            if result.error_code is ErrorCode.INVALID_INPUT:
                raise invalid_input("input rejected: more than the allowed NULL features")
            # A genuine internal inference fault — 503 with the DISTINCT INFERENCE_ERROR code (not
            # HEALTH_FAIL, which get_predictor reserves for "classifier never loaded"); never a 500
            # on model output (§8.5).
            raise ApiError(
                status_code=503,
                error_code=result.error_code or ErrorCode.INFERENCE_ERROR,
                detail="prediction could not be computed",
            )
        shap_values = result.shap_values
        shap_available = result.shap_available
        if result.top_prediction is not None and explainer_fn is not None:
            expl = explainer_fn(req.feature_vector, req.circuit, result.top_prediction)
            shap_values = expl.shap_values
            shap_available = expl.shap_available
        return PredictResponse(
            predicted_driver=result.top_prediction,
            confidence=result.confidence,
            confidence_state=result.confidence_state,
            top_3=[TopK(driver=d, probability=p) for d, p in result.top_3],
            shap_values=shap_values,
            shap_available=shap_available,
            model_version=result.model_version,
        ).model_dump(mode="json")

    # Model-output cache (§8.2). Keyed by a hash of the feature vector + circuit (deterministic →
    # same input, same output). The cache is transparent: a miss/down Redis still computes → 200,
    # so a Redis outage never fails a prediction (RQ5, §11.4).
    cc = state.config.cache
    ttl = cc.model_output_ttl_s if cc is not None else _DEFAULT_TTL_S
    if cache is not None:
        ident = RedisCache.hash_ident({"fv": req.feature_vector, "circuit": req.circuit})

        async def _acompute() -> dict[str, Any]:
            return _compute()

        payload, cres = await cache.get_or_compute(
            namespace="predict", ident=ident, ttl_s=ttl, compute=_acompute
        )
        metrics_mod.record_cache(cres)
    else:
        payload = _compute()

    resp = PredictResponse(**payload)
    metrics_mod.observe_confidence(resp.confidence)
    return resp
