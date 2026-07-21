"""Prometheus instrumentation (§8.5 /metrics · §9 Prometheus + Grafana monitoring).

A module-level custom registry (created once at import — no duplicate-timeseries errors across app
instances) tracking request count, request latency, and the model-confidence distribution. Exposed
at ``/metrics``, gated internal-only (403 to external callers).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

if TYPE_CHECKING:
    from fastapi import FastAPI
    from starlette.requests import Request
    from starlette.responses import Response

    from app.contracts.config import AppConfig

CONTENT_TYPE = CONTENT_TYPE_LATEST

REGISTRY = CollectorRegistry()
REQUEST_COUNT = Counter(
    "apex_requests_total", "API request count", ["method", "path", "status"], registry=REGISTRY
)
REQUEST_LATENCY = Histogram(
    "apex_request_latency_seconds", "API request latency (s)", ["path"], registry=REGISTRY
)
PREDICTION_CONFIDENCE = Histogram(
    "apex_prediction_confidence", "Classifier confidence distribution", registry=REGISTRY
)

_LOOPBACK = {"127.0.0.1", "::1", "localhost", "testclient"}


async def _dispatch(request: Request, call_next: RequestResponseEndpoint) -> Response:
    start = time.perf_counter()
    response = await call_next(request)
    # Label with the matched ROUTE TEMPLATE, never the raw path: a raw path (path params, or an
    # unmatched 404 URL) mints unbounded label cardinality in the process-lifetime registry — an
    # unauthenticated memory-exhaustion DoS. Unmatched requests bucket under a single label.
    route = request.scope.get("route")
    path = getattr(route, "path_format", None) or "__unmatched__"
    REQUEST_LATENCY.labels(path=path).observe(time.perf_counter() - start)
    REQUEST_COUNT.labels(method=request.method, path=path, status=str(response.status_code)).inc()
    return response


def instrument(app: FastAPI) -> None:
    """Add the request-timing middleware."""
    app.add_middleware(BaseHTTPMiddleware, dispatch=_dispatch)


def observe_confidence(value: float) -> None:
    """Record one prediction's confidence (called from the /predict route)."""
    PREDICTION_CONFIDENCE.observe(value)


def render() -> bytes:
    """The Prometheus text exposition of the registry."""
    return generate_latest(REGISTRY)


def is_internal(request: Request, config: AppConfig) -> bool:
    """True if the caller may read /metrics: loopback, or the internal-only gate is off (§8.5).

    A client-supplied ``X-Forwarded-For`` is only honoured when the DIRECT peer is itself loopback
    (a co-located reverse proxy). A direct external peer is rejected outright — otherwise any caller
    could spoof ``X-Forwarded-For: 127.0.0.1`` and defeat the gate.
    """
    api = config.api
    if api is None or not api.metrics_internal_only:
        return True
    peer = request.client.host if request.client else ""
    if peer not in _LOOPBACK:
        return False  # direct external peer — never trust its X-Forwarded-For header
    xff = request.headers.get("x-forwarded-for")  # peer is a local proxy: trust forwarded origin
    if xff:
        return xff.split(",")[0].strip() in _LOOPBACK
    return True
