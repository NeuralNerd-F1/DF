"""F05 — manual CSV upload pipeline (§3.4 edge-case handling).

A user drops a telemetry CSV; this module runs the master's eight §3.4 pre-processing edge cases
(wrong columns, mph units, mixed sampling, corrupt rows, single-lap, unknown-driver, duplicate laps,
empty file) and then feeds the cleaned frame through the **existing** F02 filter chain
(:func:`validate_session`) and F03 feature extraction (:func:`compute_scalar_features` /
:func:`compute_geometry_features` / :func:`assemble_feature_vector`) — unchanged. It returns a typed
:class:`UploadResult` carrying the cleaning report; nothing is silently dropped, imputed, or faked.

Design (see ``specs/F05-manual-upload.md``):

* **Synchronous + stateless** — the caller receives the cleaning report directly; F05 writes nothing
  to the DB. The assembled ``clean_vectors`` are handed back for a downstream ``/predict`` /
  ``/fingerprint/unknown`` call.
* **No new error codes** — ``UPLOAD_001``–``UPLOAD_004`` + ``RESAMPLED`` and the inherited
  ``LAP_*``/``FEATURE_001``/``IMPUTED``/``INVALID_INPUT`` codes are all pre-seeded.
* **Train-only imputation (constitution #2)** — the ``FeatureMedians`` table is *injected*; this
  module never derives a median/mean from the uploaded frame (a whole-module leakage scan gates it).
* **Documented context defaults** — an uploaded CSV lacks session context (track status, pit events,
  tyre age), so :func:`_to_session_frame` supplies explicit defaults (all-green ``track_status``,
  per-lap ``lap_time`` from the time column, no pit events, tyre age UNKNOWN → F27/F28 NULL). These
  are surfaced in the report, never assumed away.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from app.contracts.config import AppConfig
from app.contracts.enums import Status
from app.contracts.errors import ErrorCode
from app.contracts.logging import log_data_quality
from app.contracts.results import BaseResult
from app.pipeline.features import (
    FeatureMedians,
    FeatureResult,
    assemble_feature_vector,
    compute_geometry_features,
    compute_scalar_features,
)
from app.pipeline.validate import validate_session

# Optional CSV columns the pipeline consumes when present (never required by UPLOAD_001, which is
# the master's five channels). ``Distance`` (metres) feeds F15 (gear changes / km); ``X``/``Y`` feed
# the F03b cornering features (F19–F21, F25). Absent → those features NULL, honestly.
_DISTANCE_COLUMN = "Distance"
_POSITION_COLUMNS = ("X", "Y")

# Placeholder identity for the (season, circuit, driver) triple ``validate_session`` needs for its
# entity ids — an upload is not a real FastF1 session. ``season=0`` is a sentinel, never persisted.
_UPLOAD_SEASON = 0
_UPLOAD_CIRCUIT = "UPLOAD"
_UPLOAD_DRIVER = "UNKNOWN"


# ----------------------------------------------------------------------------------
# Result objects
# ----------------------------------------------------------------------------------


class LapRejection(BaseModel):
    """One lap that did not enter ``clean_vectors``, with the code that rejected it."""

    model_config = ConfigDict(extra="forbid")

    lap_number: int
    error_code: ErrorCode
    reason: str


class CleaningReport(BaseModel):
    """The §3.4 cleaning summary — the payload the master's ``/ingest/upload`` returns.

    Every count is derived from the real F02/F03 outcome; the ``context_defaults`` field records the
    synthesised session context (§4.0-c) so the honesty is visible, never hidden.
    """

    model_config = ConfigDict(extra="forbid")

    total_rows: int = 0
    corrupt_rows_dropped: int = 0
    corrupt_row_fraction: float = 0.0
    duplicate_laps_removed: int = 0
    resampled: bool = False
    resample_target_hz: int | None = None
    missing_columns: list[str] = Field(default_factory=list)
    laps_total: int = 0
    laps_clean: int = 0
    laps_rejected: int = 0
    single_lap: bool = False
    unknown_driver: bool = False
    combination_excluded: bool = False  # F02 COMBO_001/DATA_INSUFFICIENT (< 30 clean laps) surfaced
    per_lap_rejections: list[LapRejection] = Field(default_factory=list)
    features_null_by_id: dict[str, int] = Field(default_factory=dict)
    context_defaults: list[str] = Field(default_factory=list)


class UploadResult(BaseResult):
    """Outcome of :func:`process_upload`.

    ``status`` is ``FAILED`` with ``error_code`` set on a fatal fault (UPLOAD_001/003/004 or
    INVALID_INPUT — empty ``clean_vectors``); ``OK`` otherwise, **even with warnings** (UPLOAD_002 /
    RESAMPLED / single-lap / SPARSE_PROJECTION are non-fatal). ``clean_vectors`` holds the assembled
    28-key vectors for accepted laps (empty when no ``FeatureMedians`` were injected — the cleaning
    report is still fully populated).
    """

    upload_id: str = ""
    laps_accepted: int = 0
    laps_rejected: int = 0
    cleaning_report: CleaningReport = Field(default_factory=CleaningReport)
    warnings: list[str] = Field(default_factory=list)
    clean_vectors: list[FeatureResult] = Field(default_factory=list)
    unknown_driver: bool = False


# ----------------------------------------------------------------------------------
# data_quality_log ledger
# ----------------------------------------------------------------------------------


class _Ledger:
    """One ``data_quality_log`` row per upload edge case / rejection / tag (never silent, #1)."""

    def __init__(self, log_path: Path, upload_id: str) -> None:
        self.log_path = log_path
        self.upload_id = upload_id
        self.rows = 0

    def record(
        self, *, code: ErrorCode, detail: str, fallback_used: bool, lap_number: int | None = None
    ) -> None:
        entity_id = self.upload_id if lap_number is None else f"{self.upload_id}/lap{lap_number}"
        log_data_quality(
            error_code=code,
            entity_type="upload" if lap_number is None else "lap",
            entity_id=entity_id,
            error_detail=detail,
            fallback_used=fallback_used,
            log_path=self.log_path,
        )
        self.rows += 1


# ----------------------------------------------------------------------------------
# §3.4 pre-processing helpers
# ----------------------------------------------------------------------------------


def _numeric_columns(df: pd.DataFrame, cfg: AppConfig) -> list[str]:
    """The columns coerced to numeric for the corrupt-row check: the required channels + time +
    lap-number + any optional numeric columns present."""
    ucfg = cfg.upload
    assert ucfg is not None  # guarded by the caller
    required = cfg.lap_validation.required_channels if cfg.lap_validation is not None else []
    candidates = [
        *required,
        ucfg.time_column,
        ucfg.lap_number_column,
        _DISTANCE_COLUMN,
        *_POSITION_COLUMNS,
    ]
    seen: dict[str, None] = {}
    for col in candidates:
        if col in df.columns and col not in seen:
            seen[col] = None
    return list(seen)


def _corrupt_row_mask(df: pd.DataFrame, numeric_cols: list[str]) -> pd.Series:
    """Boolean mask of rows with a *non-numeric* value in any numeric column (§3.4). A blank cell
    (already NaN on read) is not corrupt — only a value that fails numeric coercion while being
    present. Vectorised across columns."""
    mask = pd.Series(False, index=df.index)
    for col in numeric_cols:
        raw = df[col]
        coerced = pd.to_numeric(raw, errors="coerce")
        mask = mask | (coerced.isna() & raw.notna())
    return mask


def _dedupe_laps(df: pd.DataFrame, lap_col: str) -> tuple[pd.DataFrame, int]:
    """De-duplicate by keeping the first occurrence of each lap number (§3.4 duplicate laps).

    A duplicated lap number is one whose value reappears **after** a different lap number has
    intervened (an F1 telemetry file has all samples of a lap contiguous). We keep the first
    contiguous block per lap and drop any later block for a lap already seen. Returns the frame and
    the count of dropped *laps*.
    """
    laps = pd.to_numeric(df[lap_col], errors="coerce")
    # Block id increments whenever the lap number changes between consecutive rows.
    block = (laps != laps.shift()).cumsum()
    first_block_per_lap = laps.groupby(block).first()
    # For each block, is its lap the first block that lap appears in?
    keep_block = ~first_block_per_lap.duplicated(keep="first")
    keep_blocks = set(keep_block[keep_block].index)
    dropped_laps = int((~keep_block).sum())
    kept = df[block.isin(keep_blocks)].reset_index(drop=True)
    return kept, dropped_laps


def _inter_sample_inconsistent(df: pd.DataFrame, time_col: str, lap_col: str, frac: float) -> bool:
    """§3.4 mixed-sampling detection: pool the per-lap inter-sample Δt and flag when its std exceeds
    ``frac`` × its mean. Returns False when Δt cannot be measured (handled elsewhere)."""
    if time_col not in df.columns:
        return False
    deltas: list[np.ndarray] = []
    for _, lap in df.groupby(pd.to_numeric(df[lap_col], errors="coerce")):
        t = pd.to_numeric(lap[time_col], errors="coerce").to_numpy(dtype=float)
        d = np.diff(t)
        d = d[np.isfinite(d) & (d > 0)]
        if d.size:
            deltas.append(d)
    if not deltas:
        return False
    pooled = np.concatenate(deltas)
    mean = float(pooled.mean())
    if mean <= 0:
        return False
    return float(pooled.std()) > frac * mean


def _resample_lap_to_hz(
    lap: pd.DataFrame, time_col: str, hz: int, numeric_cols: list[str]
) -> pd.DataFrame:
    """Linear-interpolate one lap's numeric channels onto a uniform ``hz`` grid (§3.4)."""
    t = pd.to_numeric(lap[time_col], errors="coerce").to_numpy(dtype=float)
    order = np.argsort(t, kind="stable")
    t = t[order]
    finite = np.isfinite(t)
    t = t[finite]
    if t.size < 2:
        return lap  # too few samples to resample — leave as-is
    # Drop duplicate timestamps (np.interp requires strictly increasing x).
    keep = np.concatenate(([True], np.diff(t) > 0))
    t = t[keep]
    if t.size < 2:
        return lap
    new_t = np.arange(t[0], t[-1], 1.0 / hz, dtype=float)
    if new_t.size == 0:
        return lap
    out: dict[str, np.ndarray] = {time_col: new_t}
    for col in numeric_cols:
        if col == time_col:
            continue
        vals = pd.to_numeric(lap[col], errors="coerce").to_numpy(dtype=float)[order][finite][keep]
        out[col] = np.interp(new_t, t, vals)
    # Carry non-numeric columns (e.g. the lap key if it slipped in) as their first value.
    frame = pd.DataFrame(out)
    return frame


def _resample_if_inconsistent(
    df: pd.DataFrame, cfg: AppConfig, numeric_cols: list[str]
) -> tuple[pd.DataFrame, bool]:
    """Resample every lap to ``resample_target_hz`` when the pooled inter-sample Δt varies."""
    ucfg = cfg.upload
    assert ucfg is not None
    lap_col = ucfg.lap_number_column
    if not _inter_sample_inconsistent(
        df, ucfg.time_column, lap_col, ucfg.sampling_delta_std_fraction
    ):
        return df, False
    resampled_laps: list[pd.DataFrame] = []
    for lap_no, lap in df.groupby(pd.to_numeric(df[lap_col], errors="coerce")):
        new_lap = _resample_lap_to_hz(lap, ucfg.time_column, ucfg.resample_target_hz, numeric_cols)
        new_lap = new_lap.copy()
        new_lap[lap_col] = lap_no
        resampled_laps.append(new_lap)
    if not resampled_laps:
        return df, False
    return pd.concat(resampled_laps, ignore_index=True), True


def _to_session_frame(df: pd.DataFrame, cfg: AppConfig) -> tuple[pd.DataFrame, list[str]]:
    """Adapt the cleaned CSV to the F01→F02 wide-frame shape ``validate_session`` expects.

    Synthesises the session context an uploaded CSV lacks, with **documented** defaults (returned as
    the second element so the caller can surface them): green ``track_status`` (LAP_001), per-lap
    ``lap_time`` from the time column (LAP_004 reads it), no pit events (LAP_002), and tyre age
    UNKNOWN → F27/F28 NULL (LAP_PARTIAL). X/Y and Distance pass through only when present.
    """
    ucfg = cfg.upload
    assert ucfg is not None
    vcfg = cfg.lap_validation
    green = vcfg.green_track_status_code if vcfg is not None else "1"
    required = vcfg.required_channels if vcfg is not None else []

    out = pd.DataFrame(index=df.index)
    out["lap"] = pd.to_numeric(df[ucfg.lap_number_column], errors="coerce").astype("Int64")
    for ch in required:
        out[ch] = pd.to_numeric(df[ch], errors="coerce")
    for opt in (*_POSITION_COLUMNS, _DISTANCE_COLUMN):
        if opt in df.columns:
            out[opt] = pd.to_numeric(df[opt], errors="coerce")
    time_present = ucfg.time_column in df.columns
    if time_present:
        out[ucfg.time_column] = pd.to_numeric(df[ucfg.time_column], errors="coerce")

    defaults: list[str] = []
    out["track_status"] = green
    defaults.append("track_status=green")
    out["pit_in"] = False
    out["pit_out"] = False
    defaults.append("pit_in/out=False")
    # Per-lap lap_time = span of the time column (LAP_004 proxy). Absent time column → NA, which the
    # validator surfaces honestly as LAP_004 rejections rather than a fabricated rate.
    if time_present:
        span = out.groupby("lap")[ucfg.time_column].transform(lambda t: t.max() - t.min())
        out["lap_time"] = span
        defaults.append("lap_time=from-time-column")
    else:
        out["lap_time"] = pd.NA
    out["stint"] = 1
    out["tyre_age"] = pd.NA
    defaults.append("tyre_age=UNKNOWN(F27/F28 NULL)")
    return out, defaults


def _lap_distance_m(frame: pd.DataFrame, lap_number: int) -> float | None:
    """Lap distance for F15 (gear changes / km): the span of an optional ``Distance`` column, else
    ``None`` (F15 NULLs — an honest limitation of a telemetry CSV with no odometer channel)."""
    if _DISTANCE_COLUMN not in frame.columns:
        return None
    col = pd.to_numeric(frame[_DISTANCE_COLUMN], errors="coerce")
    d = col[frame["lap"] == lap_number].to_numpy(dtype=float)
    d = d[np.isfinite(d)]
    if d.size < 2:
        return None
    span = float(d.max() - d.min())
    return span if span > 0 else None


# ----------------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------------


def _fail(
    upload_id: str, code: ErrorCode, report: CleaningReport, warnings: list[str], *, unknown: bool
) -> UploadResult:
    return UploadResult(
        status=Status.FAILED,
        error_code=code,
        logged=True,
        upload_id=upload_id,
        laps_accepted=0,
        laps_rejected=report.laps_total,
        cleaning_report=report,
        warnings=warnings,
        unknown_driver=unknown,
    )


def process_upload(
    raw: bytes,
    *,
    filename: str,
    driver_label: str | None,
    circuit: str | None,
    config: AppConfig,
    medians: FeatureMedians | None,
    log_path: str | Path | None = None,
) -> UploadResult:
    """Run one manual CSV upload through the §3.4 pipeline. Never raises for a data fault (#1)."""
    ucfg = config.upload
    upload_id = "up_" + hashlib.sha256(raw).hexdigest()[:16]
    path = Path(log_path) if log_path is not None else Path(config.paths.data_quality_log)
    unknown_driver = driver_label is None
    report = CleaningReport(unknown_driver=unknown_driver)
    warnings: list[str] = []

    if ucfg is None:
        # Config block absent — typed failure, no bare raise (#1).
        _Ledger(path, upload_id).record(
            code=ErrorCode.CONFIG_001, detail="config.upload block absent", fallback_used=False
        )
        return _fail(upload_id, ErrorCode.CONFIG_001, report, warnings, unknown=unknown_driver)

    ledger = _Ledger(path, upload_id)

    # 1. DoS guard — reject oversize before parsing (no silent truncation).
    if len(raw) > ucfg.max_file_bytes:
        ledger.record(
            code=ErrorCode.INVALID_INPUT,
            detail=f"file {len(raw)} bytes > max_file_bytes {ucfg.max_file_bytes}",
            fallback_used=False,
        )
        return _fail(upload_id, ErrorCode.INVALID_INPUT, report, warnings, unknown=unknown_driver)

    # 2. Parse. A malformed / non-CSV / bad-encoding file is INVALID_INPUT, not a crash.
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:  # noqa: BLE001 — any parse fault -> typed result, never propagate (#1)
        ledger.record(
            code=ErrorCode.INVALID_INPUT, detail=f"unparseable CSV: {exc!r}", fallback_used=False
        )
        return _fail(upload_id, ErrorCode.INVALID_INPUT, report, warnings, unknown=unknown_driver)

    report.total_rows = int(len(df))
    if len(df) > ucfg.max_rows:
        ledger.record(
            code=ErrorCode.INVALID_INPUT,
            detail=f"{len(df)} rows > max_rows {ucfg.max_rows}",
            fallback_used=False,
        )
        return _fail(upload_id, ErrorCode.INVALID_INPUT, report, warnings, unknown=unknown_driver)

    # 3. UPLOAD_004 — empty file (zero data rows after header). Immediate, before any other check.
    if df.empty:
        ledger.record(code=ErrorCode.UPLOAD_004, detail="zero data rows", fallback_used=False)
        return _fail(upload_id, ErrorCode.UPLOAD_004, report, warnings, unknown=unknown_driver)

    # 4. UPLOAD_001 — required columns (exact, case-sensitive match). Do not proceed.
    required = config.lap_validation.required_channels if config.lap_validation is not None else []
    missing = [c for c in required if c not in df.columns]
    if ucfg.lap_number_column not in df.columns:
        missing = [*missing, ucfg.lap_number_column]
    if missing:
        report.missing_columns = missing
        ledger.record(
            code=ErrorCode.UPLOAD_001, detail=f"missing columns: {missing}", fallback_used=False
        )
        return _fail(upload_id, ErrorCode.UPLOAD_001, report, warnings, unknown=unknown_driver)

    # 5. UPLOAD_002 — units warning (non-fatal; continue). Strict '>' per the master threshold.
    speed_median = float(pd.to_numeric(df["Speed"], errors="coerce").median())
    if np.isfinite(speed_median) and speed_median > ucfg.mph_speed_median_kmh:
        msg = "UPLOAD_002: Speed values appear to be in mph. Convert before uploading?"
        warnings.append(msg)
        ledger.record(code=ErrorCode.UPLOAD_002, detail=msg, fallback_used=True)

    # 6. UPLOAD_003 — corrupt rows. > threshold -> reject; at/below -> drop + log + continue.
    numeric_cols = _numeric_columns(df, config)
    corrupt_mask = _corrupt_row_mask(df, numeric_cols)
    corrupt = int(corrupt_mask.sum())
    frac = corrupt / len(df) if len(df) else 0.0
    cap = ucfg.max_corrupt_row_fraction
    report.corrupt_row_fraction = frac
    if frac > cap:
        report.corrupt_rows_dropped = 0
        ledger.record(
            code=ErrorCode.UPLOAD_003,
            detail=f"corrupt {corrupt}/{len(df)} ({frac:.1%}) > {cap:.0%}",
            fallback_used=False,
        )
        return _fail(upload_id, ErrorCode.UPLOAD_003, report, warnings, unknown=unknown_driver)
    if corrupt:
        df = cast("pd.DataFrame", df.loc[~corrupt_mask]).reset_index(drop=True)
        report.corrupt_rows_dropped = corrupt
        warnings.append(f"{corrupt} corrupt row(s) dropped ({frac:.1%} <= {cap:.0%}).")
        ledger.record(
            code=ErrorCode.UPLOAD_003,
            detail=f"dropped {corrupt} corrupt rows ({frac:.1%} <= threshold)",
            fallback_used=True,
        )

    # 7. Duplicate laps — keep first occurrence, log the count. Logged under ORPHAN_SAMPLES
    # (rows dropped before validation) — NOT F02's per-lap LAP_PARTIAL, so queries don't conflate.
    df, dup_laps = _dedupe_laps(df, ucfg.lap_number_column)
    report.duplicate_laps_removed = dup_laps
    if dup_laps:
        warnings.append(f"{dup_laps} duplicate lap(s) removed (kept first occurrence).")
        ledger.record(
            code=ErrorCode.ORPHAN_SAMPLES,
            detail=f"de-duplicated {dup_laps} repeated lap number(s), kept first",
            fallback_used=True,
        )

    # 8. Mixed sampling -> resample to target Hz (RESAMPLED flag on every lap).
    df, resampled = _resample_if_inconsistent(df, config, numeric_cols)
    report.resampled = resampled
    if resampled:
        report.resample_target_hz = ucfg.resample_target_hz
        warnings.append(f"Laps resampled to {ucfg.resample_target_hz} Hz (inconsistent sampling).")
        ledger.record(
            code=ErrorCode.RESAMPLED,
            detail=f"resampled to {ucfg.resample_target_hz} Hz (inconsistent Δt)",
            fallback_used=True,
        )

    # 9. Adapt to the F01 wide frame, then run the real F02 filter chain.
    frame, defaults = _to_session_frame(df, config)
    report.context_defaults = defaults
    lap_ids = [int(x) for x in pd.unique(frame["lap"].dropna())]
    report.laps_total = len(lap_ids)
    report.single_lap = report.laps_total == 1

    vr = validate_session(
        frame,
        season=_UPLOAD_SEASON,
        circuit=circuit or _UPLOAD_CIRCUIT,
        driver=driver_label or _UPLOAD_DRIVER,
        config=config,
        log_path=path,
    )
    # Surface F02's combination-exclusion (COMBO_001 / DATA_INSUFFICIENT, < 30 clean laps) in the
    # report rather than discarding vr.combination_excluded (F02 already logged it).
    report.combination_excluded = vr.combination_excluded

    rejections: list[LapRejection] = []
    for lap in vr.rejected_laps:
        rejections.append(
            LapRejection(
                lap_number=lap.lap_number,
                error_code=lap.error_code,
                reason=f"validation: {lap.error_code.value}",
            )
        )

    # 10. Feature extraction for the clean laps (only when a train-only median table is available).
    flags: list[ErrorCode] = [ErrorCode.RESAMPLED] if resampled else []
    clean_vectors: list[FeatureResult] = []
    null_counts: dict[str, int] = {}

    if medians is None and vr.clean_laps:
        warnings.append(
            "Feature vectors unavailable — no training medians loaded; cleaning report only."
        )
        ledger.record(
            code=ErrorCode.MODEL_NOT_FOUND,
            detail="FeatureMedians not injected; skipping feature assembly",
            fallback_used=True,
        )

    session_max_rpm = _finite_max(frame.get("RPM"))
    driver_id = driver_label or _UPLOAD_DRIVER
    for lap in vr.clean_laps:
        n = lap.lap_number
        if medians is None:
            continue
        try:
            lap_df = cast("pd.DataFrame", frame.loc[frame["lap"] == n])
            lap_time_s = _finite_scalar(lap_df.get("lap_time"))
            scalar = compute_scalar_features(
                lap_df,
                lap_number=n,
                session_max_rpm=session_max_rpm,
                lap_distance_m=_lap_distance_m(frame, n),
                config=config,
                lap_time_s=lap_time_s,
                log_path=path,
                entity_id=f"{upload_id}/lap{n}",
            )
            geo = compute_geometry_features(
                lap_df, lap_number=n, config=config, log_path=path, entity_id=f"{upload_id}/lap{n}"
            )
            values: dict[str, float | None] = {**scalar.features, **geo.features}
            fr = assemble_feature_vector(
                values,
                lap_number=n,
                driver=driver_id,
                circuit_type=None,
                medians=medians,
                config=config,
                forced_null=list(lap.null_features),
                flags=flags,
                log_path=path,
                entity_id=f"{upload_id}/lap{n}",
            )
        except Exception as exc:  # noqa: BLE001 — a single lap's fault is a rejection, not a crash (#1)
            ledger.record(
                code=ErrorCode.FEATURE_001,
                detail=f"unexpected feature error: {exc!r}",
                fallback_used=False,
                lap_number=n,
            )
            rejections.append(
                LapRejection(lap_number=n, error_code=ErrorCode.FEATURE_001, reason=f"{exc!r}")
            )
            continue
        if fr.status is Status.FAILED:
            rejections.append(
                LapRejection(
                    lap_number=n,
                    error_code=fr.error_code or ErrorCode.FEATURE_001,
                    reason=f"assembly: {(fr.error_code or ErrorCode.FEATURE_001).value}",
                )
            )
            continue
        clean_vectors.append(fr)
        for fid in fr.imputed_features:
            null_counts[fid] = null_counts.get(fid, 0) + 1

    report.per_lap_rejections = rejections
    report.features_null_by_id = null_counts

    if medians is None:
        laps_accepted = len(vr.clean_laps)
    else:
        laps_accepted = len(clean_vectors)
    report.laps_clean = laps_accepted
    report.laps_rejected = report.laps_total - laps_accepted

    # 11. Non-fatal warnings — single-lap partial + sparse projection.
    if report.single_lap:
        warnings.append("Tyre degradation features require multiple laps (F27/F28 NULL).")
    min_clean = config.thresholds.min_clean_laps
    if 0 < laps_accepted < min_clean:
        warnings.append(
            f"SPARSE_PROJECTION: {laps_accepted} clean laps < {min_clean} — projection is sparse."
        )
        ledger.record(
            code=ErrorCode.SPARSE_PROJECTION,
            detail=f"{laps_accepted} clean laps < {min_clean}",
            fallback_used=True,
        )
    # No-silent-failure surface (#1): laps existed but none survived. The detail is in
    # per_lap_rejections; this makes the empty result visible in the warnings surface too.
    if report.laps_total > 0 and laps_accepted == 0:
        warnings.append(f"All {report.laps_total} lap(s) rejected — no usable output produced.")

    return UploadResult(
        status=Status.OK,
        logged=True,
        upload_id=upload_id,
        laps_accepted=laps_accepted,
        laps_rejected=report.laps_rejected,
        cleaning_report=report,
        warnings=warnings,
        clean_vectors=clean_vectors,
        unknown_driver=unknown_driver,
    )


def _finite_max(series: pd.Series | None) -> float | None:
    if series is None:
        return None
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(vals.max()) if vals.size else None


def _finite_scalar(series: pd.Series | None) -> float | None:
    if series is None:
        return None
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(vals[0]) if vals.size else None
