"""``POST /ingest/upload`` — the F05 manual CSV upload route (§3.4 / §8.5).

Replaces F13's 503 stub with the real, synchronous pipeline: read the multipart file, run
:func:`app.pipeline.upload.process_upload` off the event loop (pandas is CPU-bound), and return the
cleaning report as an :class:`UploadResponse`. A fatal edge case (UPLOAD_001/003/004 or a DoS/parse
INVALID_INPUT) → **400** with the pre-seeded code; non-fatal issues (UPLOAD_002 / RESAMPLED /
single-lap / SPARSE_PROJECTION) → **200** with ``warnings`` (§8.5). The route is stateless — writes
nothing to the DB (mirrors ``/predict`` #30); the assembled vectors are returned for a downstream
``/predict`` / ``/fingerprint/unknown`` call.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.api.deps import AppState, get_apex_state, get_medians
from app.api.errors import ApiError
from app.api.schemas import UploadResponse
from app.contracts.enums import Status
from app.contracts.errors import ErrorCode

router = APIRouter(tags=["upload"])

# Fatal §3.4 edge cases + the DoS/parse guard map to a 400 (client-fixable). Anything else fatal
# (e.g. a missing config block) is a server-state problem → 503.
_CLIENT_400 = frozenset(
    {ErrorCode.UPLOAD_001, ErrorCode.UPLOAD_003, ErrorCode.UPLOAD_004, ErrorCode.INVALID_INPUT}
)


@router.post(
    "/ingest/upload",
    response_model=UploadResponse,
    include_in_schema=False,  # dissertation branch: functional, outside the presented surface
)
async def ingest_upload(
    file: UploadFile = File(...),
    driver_label: str | None = Form(default=None),
    circuit: str | None = Form(default=None),
    unknown_driver: bool = Form(default=False),
    state: AppState = Depends(get_apex_state),
    medians: Any = Depends(get_medians),
) -> UploadResponse:
    # Bound the in-memory read to the configured cap + 1 byte so an oversize upload is rejected
    # without buffering the whole body (DoS). process_upload re-checks `len(raw) > max_file_bytes`
    # and returns INVALID_INPUT — reached here because the capped read yields cap + 1 bytes.
    ucfg = state.config.upload
    raw = await file.read(ucfg.max_file_bytes + 1) if ucfg is not None else await file.read()

    # Local import keeps the API module import-light of pandas/numpy (mirrors deps.load_state).
    from app.pipeline.upload import process_upload

    result = await run_in_threadpool(
        process_upload,
        raw,
        filename=file.filename or "upload.csv",
        driver_label=None if unknown_driver else driver_label,
        circuit=circuit,
        config=state.config,
        medians=medians,
    )

    if result.status is Status.FAILED and result.error_code is not None:
        code = result.error_code
        status = 400 if code in _CLIENT_400 else 503
        raise ApiError(
            status_code=status,
            error_code=code,
            detail=f"Upload rejected ({code.value}).",
        )

    return UploadResponse(
        upload_id=result.upload_id,
        laps_accepted=result.laps_accepted,
        laps_rejected=result.laps_rejected,
        cleaning_report=result.cleaning_report.model_dump(mode="json"),
        warnings=result.warnings,
    )
