"""HTTP error mapping — typed ``ErrorCode`` → status + ``ErrorResponse`` (never a raw stack).

Every non-2xx body is an ``ErrorResponse`` carrying an ``ErrorCode`` enum member (house rule: error
strings are never hard-coded). The global ``Exception`` handler is the "no unhandled exceptions"
backstop (§11): any uncaught fault on any route → typed **500 `API_ERROR`**, logged best-effort to
``data_quality_log``, never a stack trace out of the process.
"""

from __future__ import annotations

from contextlib import suppress

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.schemas import ErrorResponse
from app.contracts.errors import ErrorCode
from app.contracts.logging import log_data_quality


class ApiError(Exception):
    """A typed HTTP failure: carries the status, the ``ErrorCode``, and a user-facing detail."""

    def __init__(
        self,
        *,
        status_code: int,
        error_code: ErrorCode,
        detail: str,
        flags: list[ErrorCode] | None = None,
    ) -> None:
        self.status_code = status_code
        self.error_code = error_code
        self.detail = detail
        self.flags = flags or []
        super().__init__(detail)


def not_found(detail: str) -> ApiError:
    """404 — a requested driver / combination does not exist."""
    return ApiError(status_code=404, error_code=ErrorCode.NOT_FOUND, detail=detail)


def forbidden(detail: str) -> ApiError:
    """403 — e.g. ``/metrics`` called by an external client."""
    return ApiError(status_code=403, error_code=ErrorCode.FORBIDDEN, detail=detail)


def service_unavailable(detail: str) -> ApiError:
    """503 — a critical component (model handle / DB) is not ready, or an endpoint is deferred."""
    return ApiError(status_code=503, error_code=ErrorCode.HEALTH_FAIL, detail=detail)


def invalid_input(detail: str) -> ApiError:
    """400 — a request that parsed but violates a business gate (>6 NULL features, <5 laps)."""
    return ApiError(status_code=400, error_code=ErrorCode.INVALID_INPUT, detail=detail)


def _json(
    status_code: int, error_code: ErrorCode, detail: str, flags: list[ErrorCode] | None = None
) -> JSONResponse:
    body = ErrorResponse(error_code=error_code, detail=detail, flags=flags or [])
    return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"))


def _log_fault(request: Request, code: ErrorCode, exc: Exception) -> None:
    """Best-effort ``data_quality_log`` write for a typed fault (§6 'Logged? yes' rows). The
    log-write itself is suppressed so a broken sink can never mask the primary fault or crash the
    handler (mirrors ``deps._try_load`` / F12's ``_safe_rollback``)."""
    cfg = getattr(request.app.state, "apex_config", None)
    if cfg is None:  # pragma: no cover — create_app always sets apex_config
        return
    with suppress(Exception):
        log_data_quality(
            error_code=code,
            entity_type="api",
            entity_id=str(request.url.path),
            error_detail=f"{type(exc).__name__}: {exc}",
            fallback_used=False,
            log_path=cfg.paths.data_quality_log,
        )


def register_exception_handlers(app: FastAPI) -> None:
    """Wire the typed exception → HTTP mapping onto the app."""
    # Imported here (not at module top) to keep app.api.errors import-light of the DB stack.
    from app.db.results import DatabaseUnavailableError, DbIntegrityError

    @app.exception_handler(ApiError)
    async def _handle_api_error(_request: Request, exc: ApiError) -> JSONResponse:
        # NOT_FOUND/FORBIDDEN/HEALTH_FAIL are §6 "Logged? no"; INVALID_INPUT is logged at its source
        # (F06b's NULL gate / the /fingerprint <5-laps gate) — so this handler does not double-log.
        return _json(exc.status_code, exc.error_code, exc.detail, exc.flags)

    @app.exception_handler(DatabaseUnavailableError)
    async def _handle_db_unavailable(
        request: Request, exc: DatabaseUnavailableError
    ) -> JSONResponse:
        # §11: DB down → 503 DB_UNAVAILABLE (session_scope already rolled back — no partial write).
        _log_fault(request, ErrorCode.DB_UNAVAILABLE, exc)  # §6 "Logged? yes"
        return _json(503, ErrorCode.DB_UNAVAILABLE, "Database temporarily unavailable.")

    @app.exception_handler(DbIntegrityError)
    async def _handle_db_integrity(request: Request, exc: DbIntegrityError) -> JSONResponse:
        _log_fault(request, ErrorCode.DB_INTEGRITY, exc)  # §6 "Logged? yes -> data_quality_log"
        return _json(409, ErrorCode.DB_INTEGRITY, "Record rejected: integrity constraint.")

    @app.exception_handler(Exception)
    async def _handle_unhandled(request: Request, exc: Exception) -> JSONResponse:
        # The "no unhandled exceptions" backstop (§11): log best-effort, return a typed 500.
        _log_fault(request, ErrorCode.API_ERROR, exc)
        return _json(500, ErrorCode.API_ERROR, "Internal error.")
