"""F05 — manual CSV upload pipeline tests (§3.4 edge cases + the F02/F03 chain).

One test per acceptance criterion (spec §7) AND one per failure mode (spec §6), plus the
boundary cases (6-vs-7 NULL, corrupt fraction at 0.05, Speed median at exactly 350, content-hash
stability) and the constitution-#2 leakage scan. Telemetry is built from the same engineered
"corner block" pattern the F03a happy-path uses, so every scalar feature computes and the only NULLs
are the ones an upload structurally loses (X/Y cornering, the F15 distance feature, the F27/F28 tyre
slopes) — which is exactly what the edge cases probe.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from app.contracts.config import AppConfig, load_config
from app.contracts.enums import Status
from app.contracts.errors import ErrorCode
from app.pipeline.features import FEATURE_IDS, FeatureMedians
from app.pipeline.upload import process_upload

from tests.pipeline.conftest import read_dq

# The engineered per-lap "corner" blocks (approach / braking / apex / exit) that make all 22 scalar
# features compute in range — the F03a canonical good lap, tiled into a multi-lap session here.
_SPEED_BLOCK = [280.0] * 6 + [260.0, 220.0, 180.0, 140.0] + [120.0, 120.0] + [
    140.0, 160.0, 180.0, 200.0, 220.0, 240.0, 260.0, 280.0
]
_THROTTLE_BLOCK = [100.0] * 6 + [0.0] * 4 + [0.0] * 2 + [
    12.5, 25.0, 37.5, 50.0, 62.5, 75.0, 87.5, 100.0,
]
_BRAKE_BLOCK = [0.0] * 6 + [80.0] * 4 + [0.0] * 2 + [0.0] * 8
_RPM_BLOCK = [12000.0] * 6 + [10000.0, 9000.0, 8000.0, 7000.0] + [7000.0, 7000.0] + [
    7500.0, 8000.0, 8500.0, 9000.0, 9500.0, 10000.0, 10500.0, 11000.0
]
_GEAR_BLOCK = [7.0] * 6 + [6.0, 5.0, 4.0, 3.0] + [3.0, 3.0] + [
    3.0, 4.0, 5.0, 6.0, 7.0, 7.0, 7.0, 7.0,
]
_BLOCK_LEN = len(_SPEED_BLOCK)


@pytest.fixture
def upload_config(repo_config_path: Path, tmp_path: Path) -> AppConfig:
    """Real repo config (shipped upload/threshold values) with the DQ log redirected to tmp."""
    cfg = load_config(repo_config_path)
    cfg.paths.data_quality_log = str(tmp_path / "logs" / "dq.jsonl")
    return cfg


@pytest.fixture
def medians() -> FeatureMedians:
    """Injected train-only imputation table (synthetic — the harness recipe)."""
    return FeatureMedians(global_medians=dict.fromkeys(FEATURE_IDS, 0.5))


def _good_frame(
    lap_ids: list[int],
    *,
    with_xy: bool,
    with_distance: bool,
    dt: float,
    speed_scale: float,
    n_blocks: int,
) -> pd.DataFrame:
    """A multi-lap telemetry DataFrame with a continuous time base."""
    rows: list[dict[str, object]] = []
    t = 0.0
    for lap in lap_ids:
        speed = np.tile(_SPEED_BLOCK, n_blocks) * speed_scale
        throttle = np.tile(_THROTTLE_BLOCK, n_blocks)
        brake = np.tile(_BRAKE_BLOCK, n_blocks)
        rpm = np.tile(_RPM_BLOCK, n_blocks)
        gear = np.tile(_GEAR_BLOCK, n_blocks)
        n = speed.size
        theta = np.linspace(0.0, 2.0 * np.pi, n)
        dist_step = 5500.0 / (n - 1)  # ~5.5 km lap (keeps the distance-scaled features in range)
        for s in range(n):
            t += dt
            row: dict[str, object] = {
                "LapNumber": lap,
                "Time": round(t, 5),
                "Speed": float(speed[s]),
                "Throttle": float(throttle[s]),
                "Brake": float(brake[s]),
                "RPM": float(rpm[s]),
                "Gear": float(gear[s]),
            }
            if with_xy:
                row["X"] = float(np.cos(theta[s]) * 500.0)
                row["Y"] = float(np.sin(theta[s]) * 500.0)
            if with_distance:
                row["Distance"] = float(s * dist_step)
            rows.append(row)
    return pd.DataFrame(rows)


def good_csv(
    n_laps: int = 3,
    *,
    with_xy: bool = True,
    with_distance: bool = False,
    dt: float = 0.05,
    speed_scale: float = 1.0,
    n_blocks: int = 6,
    lap_ids: list[int] | None = None,
) -> bytes:
    ids = lap_ids if lap_ids is not None else list(range(1, n_laps + 1))
    df = _good_frame(
        ids,
        with_xy=with_xy,
        with_distance=with_distance,
        dt=dt,
        speed_scale=speed_scale,
        n_blocks=n_blocks,
    )
    return df.to_csv(index=False).encode()


def _run(raw: bytes, cfg: AppConfig, medians: FeatureMedians | None, *, driver: str | None = "VER"):
    return process_upload(
        raw,
        filename="t.csv",
        driver_label=driver,
        circuit="Silverstone",
        config=cfg,
        medians=medians,
    )


# --- Happy path ---------------------------------------------------------------------


def test_wellformed_csv_accepts_laps(
    upload_config: AppConfig, medians: FeatureMedians
) -> None:
    """A well-formed multi-lap CSV with X/Y yields laps_accepted > 0 (not all rejected)."""
    res = _run(good_csv(n_laps=3), upload_config, medians)
    assert res.status is Status.OK
    assert res.error_code is None
    assert res.laps_accepted > 0
    assert res.cleaning_report.laps_total == 3
    assert res.upload_id.startswith("up_")


def test_accepted_lap_imputes_from_injected_medians(
    upload_config: AppConfig, medians: FeatureMedians
) -> None:
    """A <=6-NULL vector is IMPUTED (F27/F28 NULL for an upload) from the injected medians."""
    res = _run(good_csv(n_laps=3), upload_config, medians)
    assert res.clean_vectors, "expected at least one assembled vector"
    fr = res.clean_vectors[0]
    assert "F27" in fr.imputed_features and "F28" in fr.imputed_features
    # Every 28 key is present and non-NULL after imputation.
    assert set(fr.feature_vector) == set(FEATURE_IDS)
    assert all(v is not None for v in fr.feature_vector.values())


def test_context_defaults_are_surfaced(
    upload_config: AppConfig, medians: FeatureMedians
) -> None:
    """The synthesised session context is reported, never hidden (§4.0-c)."""
    res = _run(good_csv(n_laps=3), upload_config, medians)
    joined = " ".join(res.cleaning_report.context_defaults)
    assert "track_status=green" in joined
    assert "tyre_age=UNKNOWN" in joined


# --- UPLOAD_001 missing columns -----------------------------------------------------


def test_missing_required_column_upload_001(
    upload_config: AppConfig, medians: FeatureMedians
) -> None:
    df = pd.read_csv(io.BytesIO(good_csv(n_laps=2)))
    raw = df.drop(columns=["RPM"]).to_csv(index=False).encode()
    res = _run(raw, upload_config, medians)
    assert res.status is Status.FAILED
    assert res.error_code is ErrorCode.UPLOAD_001
    assert "RPM" in res.cleaning_report.missing_columns
    assert res.laps_accepted == 0
    assert res.cleaning_report.laps_total == 0  # did not proceed
    rows = read_dq(upload_config.paths.data_quality_log)
    assert any(r["error_code"] == "UPLOAD_001" for r in rows)


# --- UPLOAD_002 units warning -------------------------------------------------------


def test_mph_units_warning_upload_002(
    upload_config: AppConfig, medians: FeatureMedians
) -> None:
    res = _run(good_csv(n_laps=2, speed_scale=2.0), upload_config, medians)  # median well over 350
    assert res.status is Status.OK  # non-fatal — processing continues
    assert any("UPLOAD_002" in w for w in res.warnings)


def test_speed_median_at_threshold_no_warning(
    upload_config: AppConfig, medians: FeatureMedians
) -> None:
    """Strict '>' — a Speed median exactly at the threshold does not warn."""
    thresh = upload_config.upload.mph_speed_median_kmh  # type: ignore[union-attr]
    df = _good_frame(
        [1, 2], with_xy=True, with_distance=False, dt=0.05, speed_scale=1.0, n_blocks=6
    )
    # Shift Speed so its median is exactly the threshold.
    df["Speed"] = df["Speed"] - float(df["Speed"].median()) + thresh
    res = _run(df.to_csv(index=False).encode(), upload_config, medians)
    assert not any("UPLOAD_002" in w for w in res.warnings)


# --- RESAMPLED ----------------------------------------------------------------------


def test_inconsistent_sampling_resampled(
    upload_config: AppConfig, medians: FeatureMedians
) -> None:
    df = _good_frame(
        [1, 2, 3], with_xy=True, with_distance=False, dt=0.05, speed_scale=1.0, n_blocks=6
    )
    # Jitter the time base so per-sample Δt std exceeds 20% of the mean.
    rng = np.random.default_rng(0)
    df["Time"] = np.cumsum(0.02 * (1.0 + rng.uniform(-0.6, 0.6, len(df))))
    res = _run(df.to_csv(index=False).encode(), upload_config, medians)
    assert res.cleaning_report.resampled is True
    assert res.cleaning_report.resample_target_hz == upload_config.upload.resample_target_hz  # type: ignore[union-attr]
    assert any("resampled" in w.lower() for w in res.warnings)
    rows = read_dq(upload_config.paths.data_quality_log)
    assert any(r["error_code"] == "RESAMPLED" for r in rows)


def test_consistent_sampling_not_resampled(
    upload_config: AppConfig, medians: FeatureMedians
) -> None:
    res = _run(good_csv(n_laps=2), upload_config, medians)  # uniform dt
    assert res.cleaning_report.resampled is False


# --- UPLOAD_003 corrupt rows --------------------------------------------------------


def _corrupt(raw: bytes, fraction: float) -> bytes:
    df = pd.read_csv(io.BytesIO(raw))
    df["Speed"] = df["Speed"].astype(object)
    k = max(1, int(len(df) * fraction))
    df.iloc[:k, df.columns.get_loc("Speed")] = "not_a_number"
    return df.to_csv(index=False).encode()


def test_corrupt_rows_over_threshold_upload_003(
    upload_config: AppConfig, medians: FeatureMedians
) -> None:
    res = _run(_corrupt(good_csv(n_laps=2), 0.10), upload_config, medians)  # 10% > 5%
    assert res.status is Status.FAILED
    assert res.error_code is ErrorCode.UPLOAD_003
    assert res.laps_accepted == 0


def test_corrupt_rows_under_threshold_dropped_and_continue(
    upload_config: AppConfig, medians: FeatureMedians
) -> None:
    res = _run(_corrupt(good_csv(n_laps=3), 0.02), upload_config, medians)  # 2% <= 5%
    assert res.status is Status.OK
    assert res.cleaning_report.corrupt_rows_dropped > 0


def test_corrupt_fraction_exactly_at_threshold_continues(
    upload_config: AppConfig, medians: FeatureMedians
) -> None:
    """Boundary: frac == max_corrupt_row_fraction is NOT '> threshold' -> drop + continue."""
    frac = upload_config.upload.max_corrupt_row_fraction  # type: ignore[union-attr]
    res = _run(_corrupt(good_csv(n_laps=4), frac), upload_config, medians)
    assert res.status is Status.OK  # exactly at the bound continues (strict '>')


# --- Single-lap ---------------------------------------------------------------------


def test_single_lap_partial_fingerprint(
    upload_config: AppConfig, medians: FeatureMedians
) -> None:
    res = _run(good_csv(n_laps=1), upload_config, medians)
    assert res.cleaning_report.single_lap is True
    assert res.laps_accepted == 1
    fr = res.clean_vectors[0]
    assert "F27" in fr.imputed_features and "F28" in fr.imputed_features  # tyre slopes NULL'd
    assert any("multiple laps" in w for w in res.warnings)


# --- Unknown driver + sparse --------------------------------------------------------


def test_unknown_driver_mode(
    upload_config: AppConfig, medians: FeatureMedians
) -> None:
    res = _run(good_csv(n_laps=2), upload_config, medians, driver=None)
    assert res.unknown_driver is True
    assert res.cleaning_report.unknown_driver is True


def test_sparse_projection_warning_under_min_laps(
    upload_config: AppConfig, medians: FeatureMedians
) -> None:
    res = _run(good_csv(n_laps=3), upload_config, medians)  # 3 < 30
    assert any("SPARSE_PROJECTION" in w for w in res.warnings)


# --- Duplicate laps -----------------------------------------------------------------


def test_duplicate_lap_numbers_deduped(
    upload_config: AppConfig, medians: FeatureMedians
) -> None:
    # Lap 2 reappears after lap 3 -> one duplicate lap block, keep the first.
    res = _run(good_csv(lap_ids=[1, 2, 3, 2]), upload_config, medians)
    assert res.cleaning_report.duplicate_laps_removed == 1
    assert res.cleaning_report.laps_total == 3


# --- UPLOAD_004 empty ---------------------------------------------------------------


def test_empty_file_upload_004(
    upload_config: AppConfig, medians: FeatureMedians
) -> None:
    raw = b"LapNumber,Time,Speed,Throttle,Brake,RPM,Gear\n"  # header only
    res = _run(raw, upload_config, medians)
    assert res.status is Status.FAILED
    assert res.error_code is ErrorCode.UPLOAD_004
    assert res.laps_accepted == 0


# --- INVALID_INPUT (DoS / non-CSV) --------------------------------------------------


def test_oversize_file_rejected_before_parse(
    upload_config: AppConfig, medians: FeatureMedians
) -> None:
    cfg = upload_config
    cfg.upload.max_file_bytes = 100  # type: ignore[union-attr]
    res = _run(good_csv(n_laps=2), cfg, medians)
    assert res.status is Status.FAILED
    assert res.error_code is ErrorCode.INVALID_INPUT
    assert res.cleaning_report.total_rows == 0  # no partial read


def test_too_many_rows_rejected(
    upload_config: AppConfig, medians: FeatureMedians
) -> None:
    cfg = upload_config
    cfg.upload.max_rows = 10  # type: ignore[union-attr]
    res = _run(good_csv(n_laps=2), cfg, medians)
    assert res.status is Status.FAILED
    assert res.error_code is ErrorCode.INVALID_INPUT


def test_unparseable_binary_blob_invalid_input(
    upload_config: AppConfig, medians: FeatureMedians
) -> None:
    res = _run(b"\x00\x01\x02\xff\xfe not a csv \x00", upload_config, medians)
    assert res.status is Status.FAILED
    assert res.error_code is ErrorCode.INVALID_INPUT


# --- NULL budget: IMPUTED (<=6) vs LAP_NULL_EXCESS (>6) -----------------------------


def test_no_xy_with_distance_is_six_null_accepted(
    upload_config: AppConfig, medians: FeatureMedians
) -> None:
    """No X/Y (F19-21,F25 NULL) + Distance (F15 OK) + F27/F28 = exactly 6 NULL -> IMPUTED."""
    res = _run(good_csv(n_laps=2, with_xy=False, with_distance=True), upload_config, medians)
    assert res.laps_accepted > 0
    fr = res.clean_vectors[0]
    assert {"F19", "F20", "F21", "F25", "F27", "F28"} <= set(fr.imputed_features)


def test_no_xy_no_distance_is_seven_null_rejected(
    upload_config: AppConfig, medians: FeatureMedians
) -> None:
    """No X/Y + no Distance (F15 NULL) + F27/F28 = 7 NULL > ceiling -> LAP_NULL_EXCESS rejection."""
    res = _run(good_csv(n_laps=2, with_xy=False, with_distance=False), upload_config, medians)
    assert res.laps_accepted == 0
    codes = {r.error_code for r in res.cleaning_report.per_lap_rejections}
    assert ErrorCode.LAP_NULL_EXCESS in codes
    # No-silent-failure: an all-rejected upload still surfaces a warning (not a silent empty OK).
    assert res.status is Status.OK
    assert any("rejected" in w.lower() for w in res.warnings)


# --- Medians absent: cleaning report still returned ---------------------------------


def test_no_medians_returns_cleaning_report_without_vectors(upload_config: AppConfig) -> None:
    res = _run(good_csv(n_laps=3), upload_config, None)
    assert res.status is Status.OK
    assert res.clean_vectors == []
    assert res.laps_accepted == 3  # validation-clean laps still counted
    assert any("training medians" in w for w in res.warnings)


# --- Leakage guard (constitution #2) ------------------------------------------------


def test_upload_module_has_no_fit_or_median_table_construction() -> None:
    """The upload module must never call .fit() nor construct its own FeatureMedians — imputation
    uses the INJECTED train-only table (constitution #2). (The one .median() call is the §3.4
    UPLOAD_002 units heuristic, not imputation — asserted absent from the assembly path.)"""
    src = Path("app/pipeline/upload.py").read_text(encoding="utf-8")
    assert ".fit(" not in src
    assert "FeatureMedians(" not in src  # never builds a median table from the upload


# --- Reproducibility ----------------------------------------------------------------


def test_upload_id_is_content_hash_stable(
    upload_config: AppConfig, medians: FeatureMedians
) -> None:
    raw = good_csv(n_laps=2)
    a = _run(raw, upload_config, medians)
    b = _run(raw, upload_config, medians)
    assert a.upload_id == b.upload_id
    assert _run(good_csv(n_laps=3), upload_config, medians).upload_id != a.upload_id


# --- Never-raises -------------------------------------------------------------------


def test_config_block_absent_typed_failure(
    upload_config: AppConfig, medians: FeatureMedians
) -> None:
    upload_config.upload = None
    res = _run(good_csv(n_laps=2), upload_config, medians)
    assert res.status is Status.FAILED
    assert res.error_code is ErrorCode.CONFIG_001


def test_feature_exception_becomes_rejection_not_crash(
    upload_config: AppConfig, medians: FeatureMedians, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unexpected fault in a per-lap feature call is a coded rejection, never a raise (#1)."""
    import app.pipeline.upload as upload_mod

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("synthetic feature fault")

    monkeypatch.setattr(upload_mod, "compute_scalar_features", _boom)
    res = _run(good_csv(n_laps=2), upload_config, medians)
    assert res.status is Status.OK  # the upload itself never raises
    assert res.laps_accepted == 0
    codes = {r.error_code for r in res.cleaning_report.per_lap_rejections}
    assert ErrorCode.FEATURE_001 in codes
    rows = read_dq(upload_config.paths.data_quality_log)
    assert any(r["error_code"] == "FEATURE_001" for r in rows)


def test_combination_excluded_surfaced(upload_config: AppConfig, medians: FeatureMedians) -> None:
    """F02's < 30-clean-laps exclusion is surfaced in the report, not discarded (§6)."""
    res = _run(good_csv(n_laps=3), upload_config, medians)  # 3 < 30
    assert res.cleaning_report.combination_excluded is True


# --- Finding-6 warnings: corrupt-drop + dedup now surface in warnings ---------------


def test_corrupt_drop_emits_warning(upload_config: AppConfig, medians: FeatureMedians) -> None:
    res = _run(_corrupt(good_csv(n_laps=3), 0.02), upload_config, medians)
    assert any("corrupt" in w.lower() for w in res.warnings)


def test_dedup_emits_warning(upload_config: AppConfig, medians: FeatureMedians) -> None:
    res = _run(good_csv(lap_ids=[1, 2, 3, 2]), upload_config, medians)
    assert any("duplicate" in w.lower() for w in res.warnings)


# --- AC §7: every code writes a data_quality_log row -------------------------------


def _dq_codes(cfg: AppConfig) -> set[str]:
    return {r["error_code"] for r in read_dq(cfg.paths.data_quality_log)}


def test_dq_log_upload_002(upload_config: AppConfig, medians: FeatureMedians) -> None:
    _run(good_csv(n_laps=2, speed_scale=2.0), upload_config, medians)
    assert "UPLOAD_002" in _dq_codes(upload_config)


def test_dq_log_upload_003_over_threshold(
    upload_config: AppConfig, medians: FeatureMedians
) -> None:
    _run(_corrupt(good_csv(n_laps=2), 0.10), upload_config, medians)
    assert "UPLOAD_003" in _dq_codes(upload_config)


def test_dq_log_upload_004(upload_config: AppConfig, medians: FeatureMedians) -> None:
    _run(b"LapNumber,Time,Speed,Throttle,Brake,RPM,Gear\n", upload_config, medians)
    assert "UPLOAD_004" in _dq_codes(upload_config)


def test_dq_log_dedup_orphan_samples(upload_config: AppConfig, medians: FeatureMedians) -> None:
    _run(good_csv(lap_ids=[1, 2, 3, 2]), upload_config, medians)
    assert "ORPHAN_SAMPLES" in _dq_codes(upload_config)


def test_dq_log_sparse_projection(upload_config: AppConfig, medians: FeatureMedians) -> None:
    _run(good_csv(n_laps=3), upload_config, medians)
    assert "SPARSE_PROJECTION" in _dq_codes(upload_config)


def test_dq_log_medians_absent_model_not_found(upload_config: AppConfig) -> None:
    _run(good_csv(n_laps=2), upload_config, None)
    assert "MODEL_NOT_FOUND" in _dq_codes(upload_config)


def test_dq_log_invalid_input(upload_config: AppConfig, medians: FeatureMedians) -> None:
    cfg = upload_config
    cfg.upload.max_file_bytes = 50  # type: ignore[union-attr]  # oversize -> deterministic INVALID_INPUT
    _run(good_csv(n_laps=2), cfg, medians)
    assert "INVALID_INPUT" in _dq_codes(cfg)


def test_dq_log_lap_null_excess(upload_config: AppConfig, medians: FeatureMedians) -> None:
    _run(good_csv(n_laps=2, with_xy=False, with_distance=False), upload_config, medians)  # 7 NULL
    assert "LAP_NULL_EXCESS" in _dq_codes(upload_config)
