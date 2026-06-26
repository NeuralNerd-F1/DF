"""Shared enumerations used across every APEX module.

These are foundation contracts (F00): the small, cross-cutting value sets that
result objects and downstream features reference. Feature-specific enums (e.g. the
degradation alert band) live with their owning feature, not here.
"""

from __future__ import annotations

from enum import StrEnum


class Status(StrEnum):
    """High-level outcome of a pipeline stage or model call.

    Carried by every ``BaseResult``. ``OK`` = success; ``DEGRADED`` = produced a
    result via a documented fallback (e.g. served stale cache); ``FAILED`` = no
    usable result, an ``error_code`` explains why. A ``str`` enum so it serialises
    to its name in JSON / the database.
    """

    OK = "OK"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"


class LapStatus(StrEnum):
    """Admit/reject outcome for a single lap in the F02 filter chain (§3.3).

    A deliberately binary axis — a lap either enters the feature matrix or it does
    not. This is distinct from ``Status`` (OK/DEGRADED/FAILED), which describes a
    whole pipeline stage; a per-lap result carries ``LapStatus`` plus the documented
    ``error_code`` (``OK`` when clean) and any accept-time ``flags``
    (``LAP_INTERP``, ``LAP_PARTIAL``).
    """

    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"


class ConfidenceState(StrEnum):
    """Explicit confidence / uncertainty state for a classifier prediction.

    Mirrors the ``predictions.confidence_state`` column (master spec §8.4) and adds
    ``OUT_OF_DISTRIBUTION`` — the style-space state (§5.2) that the project
    constitution lists as a mandatory uncertainty state (non-negotiable #4). The
    classifier never returns a forced label below the confidence gate; it returns
    one of these states instead.
    """

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    AMBIGUOUS = "AMBIGUOUS"
    OUT_OF_DISTRIBUTION = "OUT_OF_DISTRIBUTION"
