"""The ``data_quality_log`` writer — the spine of "no component fails silently".

Every failure path in APEX writes one row here. F00 ships a structured append-only
JSONL writer; the row schema matches the master-spec §8.4 ``data_quality_log`` table
columns exactly so the F12 PostgreSQL swap is a drop-in with no caller changes.

The logger itself must never fail silently: if the sink is unwritable it raises
``DataQualityLogError`` (``DATA_QUALITY_LOG_UNWRITABLE``) rather than swallowing the
error and losing the audit trail.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from app.contracts.errors import ErrorCode

# Column order matches master-spec §8.4 data_quality_log (id is generated per row).
ROW_COLUMNS = (
    "id",
    "created_at",
    "entity_type",
    "entity_id",
    "error_code",
    "error_detail",
    "fallback_used",
)


class DataQualityLogError(Exception):
    """Raised when the ``data_quality_log`` sink cannot be written.

    Carries ``ErrorCode.DATA_QUALITY_LOG_UNWRITABLE``. The failure logger raising
    (rather than swallowing) is itself an instance of non-negotiable #1.
    """

    error_code = ErrorCode.DATA_QUALITY_LOG_UNWRITABLE

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def log_data_quality(
    *,
    error_code: ErrorCode,
    entity_type: str,
    entity_id: str,
    error_detail: str,
    fallback_used: bool,
    log_path: str | Path,
    created_at: datetime | None = None,
) -> dict[str, object]:
    """Append one row to the ``data_quality_log`` sink and return it.

    Args:
        error_code: the ``ErrorCode`` being logged.
        entity_type: what the row is about (e.g. ``"lap"``, ``"session"``, ``"upload"``).
        entity_id: identifier of that entity (e.g. lap number, session id).
        error_detail: human-readable detail (channel name, dropout %, null list, ...).
        fallback_used: whether a documented fallback was taken.
        log_path: JSONL sink path (read from ``config.yaml`` ``paths.data_quality_log``).
        created_at: timestamp override; defaults to ``now(UTC)``.

    Raises:
        DataQualityLogError: if the sink directory/file cannot be written.
    """
    row: dict[str, object] = {
        "id": str(uuid.uuid4()),
        "created_at": (created_at or datetime.now(UTC)).isoformat(),
        "entity_type": entity_type,
        "entity_id": entity_id,
        "error_code": error_code.value,
        "error_detail": error_detail,
        "fallback_used": fallback_used,
    }
    p = Path(log_path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
    except OSError as exc:
        raise DataQualityLogError(f"cannot write data_quality_log at {p}: {exc}") from exc
    return row
