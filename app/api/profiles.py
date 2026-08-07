"""Read side of ``artefacts/driver_profiles.json`` — the per-driver dataset + evaluation facts.

``scripts/build_driver_profiles.py`` writes the artefact from one training run; this module loads it
at startup and derives the "key signals" the Driver Overview card shows. It is deliberately a
read-only artefact reader, not a computation: nothing here re-derives a statistic, so a number on
the screen can always be traced back to the generating script and the matrix it names.

**Absence is a state, not an error.** A fresh checkout has no profiles artefact. The loader raises a
coded, logged failure that ``_try_load`` turns into ``profiles = None``, and every field the screen
sources from it becomes an explicit ``null`` the frontend renders as a dash. The alternative — a
zero lap count or a 0.0 identification score — would be indistinguishable from a real measurement.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.contracts.config import AppConfig
from app.contracts.errors import ErrorCode
from app.contracts.logging import log_data_quality

# The artefact's filename is fixed rather than configurable: it is written by one script and read by
# one loader, and a configurable path would let the two drift apart silently.
PROFILES_FILENAME = "driver_profiles.json"

# How many features the card calls out per driver. Three fits the reference layout and stays within
# the range where per-feature attribution is meaningfully separated from the tail.
DEFAULT_SIGNAL_COUNT = 3


class DriverProfilesUnavailableError(RuntimeError):
    """The profiles artefact is missing, unreadable, or not shaped like a profiles artefact."""

    error_code = ErrorCode.DATA_SOURCE_UNAVAILABLE


@dataclass(frozen=True, slots=True)
class DriverProfile:
    """One driver's facts. Every optional field is ``None`` when the artefact could not measure it
    (too few laps for a within-session SD, or a driver the champion's summary never scored).

    ``shap_signed_mean`` is the mean **signed** SHAP per feature over the driver's sampled laps —
    positive pushes the champion toward this driver, negative away. It comes from the beeswarm
    artefact via the build script; an artefact built without one carries ``{}`` and the profile
    screen omits the signed panel rather than inventing directions.
    """

    laps_analysed: int
    seasons: tuple[int, ...]
    n_circuits: int
    lap_time_consistency_s: float | None
    identification_f1: float | None
    feature_percentile: Mapping[str, float]
    shap_signed_mean: Mapping[str, float] = field(default_factory=dict)
    # Per-compound stint-degradation stats (curve, slope, cliff, F27/F28 medians). Carried as the
    # artefact's raw mapping — the /tyres route validates the shape into its typed schema, so the
    # loader stays a reader, not a re-modeller. {} = matrix had no usable stints for this driver.
    tyre_wear: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DriverSignal:
    """A feature the model leans on for this driver, plus where the driver sits on it.

    ``percentile`` is the driver's rank against the rest of the grid on that feature (0 = lowest
    mean, 1 = highest), and is ``None`` when the artefact carries no rank for it — the card then
    names the feature without claiming a direction.
    """

    feature: str
    attribution: float
    percentile: float | None


@dataclass(frozen=True, slots=True)
class SampleSegment:
    """One real held-out lap for the Prediction screen's picker. ``true_driver`` is the matrix's
    ground truth — the screen grades the classifier against it after predicting. Features are the
    RAW per-lap values (None = the lap's genuine NULL); /predict normalises them exactly as the
    evaluation pipeline does."""

    id: str
    label: str
    circuit: str
    season: int
    session: str
    lap_number: int
    true_driver: str
    features: Mapping[str, float | None]


@dataclass(frozen=True, slots=True)
class DriverProfiles:
    content_hash: str
    n_laps: int
    held_out_circuits: tuple[str, ...]
    source_matrix: str
    by_code: Mapping[str, DriverProfile]
    segments: tuple[SampleSegment, ...] = ()

    def get(self, code: str) -> DriverProfile | None:
        return self.by_code.get(code)


def _profile_from_json(raw: Any) -> DriverProfile:
    if not isinstance(raw, dict):
        raise DriverProfilesUnavailableError(f"driver entry is {type(raw).__name__}, not an object")
    percentile = raw.get("feature_percentile") or {}
    if not isinstance(percentile, dict):
        raise DriverProfilesUnavailableError("feature_percentile is not an object")
    signed = raw.get("shap_signed_mean") or {}
    if not isinstance(signed, dict):
        raise DriverProfilesUnavailableError("shap_signed_mean is not an object")
    wear = raw.get("tyre_wear") or {}
    if not isinstance(wear, dict):
        raise DriverProfilesUnavailableError("tyre_wear is not an object")
    return DriverProfile(
        laps_analysed=int(raw["laps_analysed"]),
        seasons=tuple(int(s) for s in raw.get("seasons") or ()),
        n_circuits=int(raw.get("n_circuits", 0)),
        lap_time_consistency_s=_opt_float(raw.get("lap_time_consistency_s")),
        identification_f1=_opt_float(raw.get("identification_f1")),
        feature_percentile={str(k): float(v) for k, v in percentile.items()},
        shap_signed_mean={str(k): float(v) for k, v in signed.items()},
        tyre_wear=wear,
    )


def _opt_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _segment_from_json(raw: Any) -> SampleSegment:
    if not isinstance(raw, dict):
        raise DriverProfilesUnavailableError(f"segment is {type(raw).__name__}, not an object")
    features = raw.get("features") or {}
    if not isinstance(features, dict):
        raise DriverProfilesUnavailableError("segment features is not an object")
    return SampleSegment(
        id=str(raw["id"]),
        label=str(raw["label"]),
        circuit=str(raw["circuit"]),
        season=int(raw["season"]),
        session=str(raw.get("session", "")),
        lap_number=int(raw.get("lap_number", 0)),
        true_driver=str(raw["true_driver"]),
        features={str(k): _opt_float(v) for k, v in features.items()},
    )


def load_driver_profiles(
    *,
    config: AppConfig,
    artefact_dir: Path | None = None,
    log_path: str | None = None,
) -> DriverProfiles:
    """Load the artefact, or raise ``DriverProfilesUnavailableError`` (coded + logged).

    Raising rather than returning ``None`` keeps this consistent with every other startup loader:
    ``_try_load`` writes the component row and degrades the field to ``None``, so the outage is
    recorded in exactly one place instead of each caller inventing its own fallback.
    """
    directory = artefact_dir if artefact_dir is not None else Path(config.paths.artefacts_dir)
    path = directory / PROFILES_FILENAME
    dq_log = log_path if log_path is not None else config.paths.data_quality_log

    def _fail(detail: str) -> DriverProfilesUnavailableError:
        log_data_quality(
            error_code=ErrorCode.DATA_SOURCE_UNAVAILABLE,
            entity_type="driver_profiles",
            entity_id=str(path),
            error_detail=detail,
            fallback_used=False,
            log_path=dq_log,
        )
        return DriverProfilesUnavailableError(detail)

    if not path.exists():
        raise _fail(f"driver profiles artefact not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _fail(f"driver profiles artefact unreadable: {type(exc).__name__}: {exc}") from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("drivers"), dict):
        raise _fail("driver profiles artefact has no 'drivers' object")

    try:
        by_code = {
            str(code): _profile_from_json(raw) for code, raw in payload["drivers"].items()
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise _fail(f"driver profiles artefact malformed: {type(exc).__name__}: {exc}") from exc

    if not by_code:
        raise _fail("driver profiles artefact carries no drivers")

    raw_segments = payload.get("segments") or []
    if not isinstance(raw_segments, list):
        raise _fail("segments is not a list")
    try:
        segments = tuple(_segment_from_json(raw) for raw in raw_segments)
    except (KeyError, TypeError, ValueError) as exc:
        raise _fail(f"segments malformed: {type(exc).__name__}: {exc}") from exc

    return DriverProfiles(
        content_hash=str(payload.get("content_hash", "")),
        n_laps=int(payload.get("n_laps", 0)),
        held_out_circuits=tuple(str(c) for c in payload.get("held_out_circuits") or ()),
        source_matrix=str(payload.get("source_matrix", "")),
        by_code=by_code,
        segments=segments,
    )


def top_signals(
    shap_global: Mapping[str, float] | None,
    profile: DriverProfile | None,
    *,
    limit: int = DEFAULT_SIGNAL_COUNT,
) -> list[DriverSignal]:
    """The ``limit`` features the champion attributes most to this driver, richest first.

    Ranking is by mean |SHAP| (what the artefact stores), which is a *magnitude* — it says the model
    watches this channel for this driver, not which way. The direction comes from ``percentile``,
    which is measured from the driver's own feature means, so the two halves of a signal come from
    different evidence and neither is inferred from the other.

    Returns ``[]`` when there is no attribution to rank — a driver whose fingerprint row predates
    the SHAP fill. The card then shows an explicit "no attribution available" line rather than
    falling back to whichever features happen to be numerically largest in the raw vector.
    """
    if not shap_global:
        return []
    ranked = sorted(
        ((str(f), float(v)) for f, v in shap_global.items() if isinstance(v, (int, float))),
        key=lambda kv: (-kv[1], kv[0]),  # magnitude desc, then feature ID for a stable tie-break
    )
    percentiles = profile.feature_percentile if profile is not None else {}
    return [
        DriverSignal(feature=f, attribution=v, percentile=percentiles.get(f))
        for f, v in ranked[: max(0, limit)]
    ]
