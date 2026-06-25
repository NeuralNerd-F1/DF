"""BaseResult is the typed, strict base every stage returns."""

from __future__ import annotations

import pytest
from app.contracts.enums import Status
from app.contracts.errors import ErrorCode
from app.contracts.results import BaseResult
from pydantic import ValidationError


def test_defaults_are_clean_success() -> None:
    r = BaseResult()
    assert r.status is Status.OK
    assert r.error_code is None
    assert r.flags == []
    assert r.logged is False
    assert r.ok is True


def test_failed_status_is_not_ok() -> None:
    r = BaseResult(status=Status.FAILED, error_code=ErrorCode.DATA_001)
    assert r.ok is False
    assert r.error_code is ErrorCode.DATA_001


def test_extra_fields_are_forbidden() -> None:
    """A typo'd field is a hard error, never a silently-dropped attribute."""
    with pytest.raises(ValidationError):
        BaseResult(staus=Status.OK)  # type: ignore[call-arg]


def test_flags_accept_error_codes() -> None:
    r = BaseResult(status=Status.DEGRADED, flags=[ErrorCode.STALE_DATA, ErrorCode.DERIVED_AGE])
    assert ErrorCode.STALE_DATA in r.flags


def test_subclass_inherits_base_fields() -> None:
    class IngestResultStub(BaseResult):
        rows: int

    r = IngestResultStub(rows=42, status=Status.OK)
    assert r.rows == 42
    assert r.ok is True
    # subclass also enforces extra=forbid inherited from the base config
    with pytest.raises(ValidationError):
        IngestResultStub(rows=1, bogus=True)  # type: ignore[call-arg]


def test_json_serialises_enums_to_strings() -> None:
    r = BaseResult(status=Status.FAILED, error_code=ErrorCode.LAP_003)
    dumped = r.model_dump(mode="json")
    assert dumped["status"] == "FAILED"
    assert dumped["error_code"] == "LAP_003"
