"""The data_quality_log writer logs every failure and never fails silently."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from app.contracts.errors import ErrorCode
from app.contracts.logging import (
    ROW_COLUMNS,
    DataQualityLogError,
    log_data_quality,
)


def test_writes_row_with_spec_8_4_schema(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "dq.jsonl"
    row = log_data_quality(
        error_code=ErrorCode.LAP_003,
        entity_type="lap",
        entity_id="2024-Bahrain-VER-L12",
        error_detail="Speed channel 14% consecutive NaN",
        fallback_used=False,
        log_path=log_path,
    )
    # Returned row matches the §8.4 column set exactly.
    assert set(row.keys()) == set(ROW_COLUMNS)
    assert row["error_code"] == "LAP_003"
    assert row["entity_type"] == "lap"
    assert row["fallback_used"] is False

    # And it was actually persisted as one parseable JSONL line.
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["error_code"] == "LAP_003"


def test_appends_rather_than_overwrites(tmp_path: Path) -> None:
    log_path = tmp_path / "dq.jsonl"
    for compound_lap in ("L1", "L2"):
        log_data_quality(
            error_code=ErrorCode.COMPOUND_UNKNOWN,
            entity_type="lap",
            entity_id=compound_lap,
            error_detail="OpenF1 and Ergast both unavailable",
            fallback_used=True,
            log_path=log_path,
        )
    assert len(log_path.read_text(encoding="utf-8").splitlines()) == 2


def test_created_at_override_is_used(tmp_path: Path) -> None:
    ts = datetime(2026, 6, 25, 12, 0, tzinfo=UTC)
    row = log_data_quality(
        error_code=ErrorCode.DATA_001,
        entity_type="session",
        entity_id="2024-Italy-R",
        error_detail="FastF1 session unavailable",
        fallback_used=False,
        log_path=tmp_path / "dq.jsonl",
        created_at=ts,
    )
    assert row["created_at"] == ts.isoformat()


def test_unwritable_sink_raises_not_silent(tmp_path: Path) -> None:
    """If the log sink cannot be written, the logger raises — it never swallows."""
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file, not a directory", encoding="utf-8")
    # parent path traverses through a file -> mkdir/open must fail.
    bad_path = blocker / "nested" / "dq.jsonl"
    with pytest.raises(DataQualityLogError) as exc:
        log_data_quality(
            error_code=ErrorCode.LAP_001,
            entity_type="lap",
            entity_id="L1",
            error_detail="safety car lap",
            fallback_used=False,
            log_path=bad_path,
        )
    assert exc.value.error_code is ErrorCode.DATA_QUALITY_LOG_UNWRITABLE
