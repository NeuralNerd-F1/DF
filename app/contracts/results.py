"""Typed result objects — the foundation base every pipeline stage returns.

The constitution forbids bare tuples / dicts crossing module boundaries: each stage
returns a Pydantic result object carrying an explicit ``status`` and, on failure, an
``error_code``. F00 ships only the *base*; concrete per-feature results
(``IngestResult``, ``ValidationResult``, ``FeatureResult``, ``PredictionResult``, ...)
subclass it in their own features.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.contracts.enums import Status
from app.contracts.errors import ErrorCode


class BaseResult(BaseModel):
    """Base class for every typed result object in APEX.

    ``extra="forbid"`` makes a typo'd field a build-breaking validation error rather
    than a silently-ignored attribute — the constitution's "no silent failure" rule
    applied to the contract layer itself.
    """

    model_config = ConfigDict(extra="forbid")

    status: Status = Status.OK
    error_code: ErrorCode | None = None
    flags: list[ErrorCode] = Field(default_factory=list)
    logged: bool = False

    @property
    def ok(self) -> bool:
        """True only when the stage succeeded cleanly (no fallback, no failure)."""
        return self.status is Status.OK
