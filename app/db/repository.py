"""Invariant-enforcing write helpers + the ``data_quality_log`` DB sink.

Thin by design — repository functions add to a caller-supplied session and ``flush`` (surfacing a
constraint violation immediately as ``DbIntegrityError``); the surrounding ``session_scope`` owns
the commit/rollback and the breaker. Connection-level faults are deliberately NOT caught here — they
propagate to ``session_scope`` which rolls back, trips the breaker, and maps them to
``DatabaseUnavailableError``. This module owns exactly two things: the §8.4 application-layer
invariants and the JSONL→Postgres ``data_quality_log`` swap.

``data_quality_log`` is **append-only** ("Never deleted", §8.4): this module exposes no delete.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.contracts.enums import Status
from app.contracts.errors import ErrorCode
from app.db import models
from app.db.results import DbIntegrityError, PersistResult


async def _flush_one(session: AsyncSession, obj: Any, entity_type: str) -> PersistResult:
    """Add + flush one ORM object, translating a constraint violation to ``DbIntegrityError``.

    A flush (not commit) so the violation surfaces here while the enclosing ``session_scope`` still
    owns the transaction boundary. Connection errors are not caught — they belong to the scope.
    """
    session.add(obj)
    try:
        await session.flush()
    except IntegrityError as exc:
        raise DbIntegrityError(
            f"{entity_type} write violates a constraint: {exc.orig}", cause=exc
        ) from exc
    entity_id = getattr(obj, "id", None)
    return PersistResult(
        status=Status.OK,
        entity_type=entity_type,
        entity_id=str(entity_id) if entity_id is not None else None,
        written=True,
    )


async def persist(session: AsyncSession, obj: Any) -> PersistResult:
    """Generic write: add + flush any ORM row, returning a typed ``PersistResult``.

    Success → ``OK`` + ``written=True`` + the row's id. A constraint violation raises
    ``DbIntegrityError`` (``DB_INTEGRITY``); a connection fault propagates to ``session_scope``.
    """
    return await _flush_one(session, obj, obj.__tablename__)


async def insert_lap_features(
    session: AsyncSession,
    *,
    lap_id: Any,
    feature_vector: dict[str, float | None],
    null_count: int | None = None,
    imputed_features: list[str] | None = None,
    corner_type: models.CornerType | None = None,
) -> PersistResult:
    """Insert one ``lap_features`` row, enforcing the §8.4 application-layer invariant.

    ``null_count`` must equal the number of NULL values in ``feature_vector``. It is computed here
    when omitted; when a caller supplies one that disagrees, the write is rejected with
    ``DbIntegrityError`` (``DB_INTEGRITY``) **before** any DB round-trip — the invariant the master
    assigns to "the application layer".
    """
    actual_nulls = sum(1 for v in feature_vector.values() if v is None)
    if null_count is None:
        null_count = actual_nulls
    elif null_count != actual_nulls:
        raise DbIntegrityError(
            f"lap_features.null_count ({null_count}) != NULLs in feature_vector ({actual_nulls})"
        )
    row = models.LapFeatures(
        lap_id=lap_id,
        feature_vector=feature_vector,
        null_count=null_count,
        imputed_features=imputed_features if imputed_features is not None else [],
        corner_type=corner_type,
    )
    return await _flush_one(session, row, "lap_features")


async def write_quality_log_row(
    session: AsyncSession,
    *,
    error_code: ErrorCode,
    entity_type: str,
    entity_id: str | None,
    error_detail: str | None,
    fallback_used: bool,
    created_at: datetime | None = None,
) -> PersistResult:
    """Append one row to the ``data_quality_log`` table — the Postgres side of the JSONL swap.

    Mirrors ``app/contracts/logging.py::log_data_quality`` exactly (same 7 columns, ``error_code``
    stored as its ``.value``), so the F00 audit trail lands in the DB with no caller changes. The
    ``created_at`` default is the DB's ``now()``; pass an explicit value only for determinism.
    Append-only — there is no delete counterpart (§8.4 "Never deleted").
    """
    row = models.DataQualityLog(
        entity_type=entity_type,
        entity_id=entity_id,
        error_code=error_code.value,
        error_detail=error_detail,
        fallback_used=fallback_used,
    )
    if created_at is not None:
        row.created_at = created_at
    return await _flush_one(session, row, "data_quality_log")
