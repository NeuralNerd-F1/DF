"""Stage B — training-matrix build tests.

The load-bearing case is :func:`derive_medians`: it is where non-negotiable #2 is enforced for
the real-data build. The single-pass rewrite turned a structural guard (held-out sessions were
never extracted) into a filter expression over already-extracted records, so the guard now
depends on a predicate rather than on control flow — and a silent break there would poison the
imputation table with held-out data without failing anything downstream. Those tests come first.

Also covers the §4.5 NULL gate in assembly, the two fail-loud paths in corpus discovery
(unreadable session / absent season), resample channel-kind handling, and the DQ helper's own
failure path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from app.contracts.config import AppConfig
from app.contracts.errors import ErrorCode
from app.pipeline.features import FEATURE_IDS, FeatureMedians
from tests.scripts.conftest import read_dq, record, telemetry_lap

from scripts import build_training_matrix as btm

HELD_OUT = {"Belgium", "Italy"}


# --- derive_medians: the non-negotiable #2 guard --------------------------------------------


def test_derive_medians_excludes_held_out_rows() -> None:
    """A held-out lap must not move the median by even one imputation cell.

    Train rows sit at F01=10.0; the held-out rows are pushed to 1000.0. If the filter ever
    breaks, the global median jumps — the assertion is on the exact train-only value, not on a
    tolerance, because there is no acceptable amount of held-out leakage.
    """
    train = [record(driver="VER", circuit="Bahrain", raw={"F01": 10.0}, lap=i) for i in range(20)]
    leak = [record(driver="VER", circuit="Belgium", raw={"F01": 1000.0}, lap=i) for i in range(20)]

    medians = btm.derive_medians(train + leak, HELD_OUT)

    assert medians.global_medians["F01"] == 10.0
    assert btm.derive_medians(train, HELD_OUT).global_medians == medians.global_medians


def test_derive_medians_creates_no_cell_for_a_held_out_circuit_type() -> None:
    """Held-out laps must not create a driver×circuit-type cell either — the cell is the
    PREFERRED imputation source (``median_for`` consults it before the global table), so a
    leaked cell would be used ahead of the legitimate global value."""
    records = [
        record(driver="LEC", circuit="Bahrain", circuit_type="high_speed", raw={"F01": 1.0}),
        record(driver="LEC", circuit="Italy", circuit_type="street", raw={"F01": 500.0}),
    ]

    medians = btm.derive_medians(records, HELD_OUT)

    assert ("LEC", "high_speed") in medians.driver_circuit_medians
    assert ("LEC", "street") not in medians.driver_circuit_medians
    assert medians.median_for("F01", "LEC", "street") == 1.0  # falls back to the train global


def test_derive_medians_held_out_set_comes_from_config(scripts_config: AppConfig) -> None:
    """The circuit names the build filters on must match ``config.held_out_circuits`` exactly.

    This is the seam the mapping in fetch_fastf1 has to hit: if the corpus ever labelled Spa as
    "Spa" rather than "Belgium", ``not in held_out`` would silently pass it into the fit.
    """
    held_out = set(scripts_config.held_out_circuits)
    assert held_out == HELD_OUT

    records = [
        record(driver="VER", circuit=c, raw={"F01": 5.0})
        for c in ("Bahrain", "Monaco", *held_out)
    ]
    medians = btm.derive_medians(records, held_out)

    assert set(medians.driver_circuit_medians) == {("VER", "high_speed")}


def test_derive_medians_builds_global_and_per_cell_tables() -> None:
    """Happy path: the global table medians every feature seen in train, and each
    driver×circuit-type combination gets its own cell."""
    records = [
        record(driver="VER", circuit="Bahrain", circuit_type="high_speed", raw={"F01": v})
        for v in (1.0, 3.0, 5.0)
    ] + [
        record(driver="HAM", circuit="Monaco", circuit_type="street", raw={"F01": v})
        for v in (10.0, 20.0, 30.0)
    ]

    medians = btm.derive_medians(records, HELD_OUT)

    assert medians.global_medians["F01"] == 7.5  # median of 1,3,5,10,20,30
    assert medians.driver_circuit_medians[("VER", "high_speed")]["F01"] == 3.0
    assert medians.driver_circuit_medians[("HAM", "street")]["F01"] == 20.0


def test_derive_medians_omits_a_feature_with_no_train_values() -> None:
    """An always-NULL feature gets NO median rather than a fabricated one, so assembly fails
    the lap loudly (FEATURE_001) instead of imputing from nothing."""
    records = [record(driver="VER", circuit="Bahrain", raw={"F01": 1.0, "F02": None})] * 3

    medians = btm.derive_medians(records, HELD_OUT)

    assert "F01" in medians.global_medians
    assert "F02" not in medians.global_medians
    assert medians.median_for("F02", "VER", "high_speed") is None


def test_derive_medians_ignores_rows_with_no_circuit_type() -> None:
    """An unmapped circuit carries ``circuit_type=None``; it still feeds the global table but
    cannot key a cell (a None-typed cell would be looked up by nothing)."""
    records = [
        record(driver="VER", circuit="Bahrain", circuit_type=None, raw={"F01": 2.0}),
        record(driver="VER", circuit="Monaco", circuit_type="street", raw={"F01": 4.0}),
    ]

    medians = btm.derive_medians(records, HELD_OUT)

    assert medians.global_medians["F01"] == 3.0
    assert set(medians.driver_circuit_medians) == {("VER", "street")}


# --- assemble_rows: the §4.5 NULL gate ------------------------------------------------------


def _full_raw(value: float = 1.0) -> dict[str, float | None]:
    return {fid: value for fid in FEATURE_IDS}


def _medians(value: float = 1.0) -> FeatureMedians:
    return FeatureMedians(global_medians={fid: value for fid in FEATURE_IDS})


def test_assemble_rows_keeps_context_and_strips_internals(
    scripts_config: AppConfig, tmp_path: Path
) -> None:
    """An accepted lap yields the 28 features PLUS the F10/F11 context, and none of the
    ``_``-prefixed internals leak into the matrix."""
    rows, stats = btm.assemble_rows(
        [record(driver="VER", circuit="Bahrain", raw=_full_raw())],
        _medians(), scripts_config, tmp_path / "dq.jsonl",
    )

    assert stats["vectors_built"] == 1
    row = rows[0]
    assert all(fid in row for fid in FEATURE_IDS)
    for ctx in ("session", "stint", "tyre_age", "compound", "track_status", "lap_time_s",
                "is_clean", "has_interpolation", "lap_number"):
        assert ctx in row, ctx
    assert not [k for k in row if k.startswith("_")]


def test_assemble_rows_rejects_over_the_null_ceiling(
    scripts_config: AppConfig, tmp_path: Path
) -> None:
    """Over ``thresholds.null_ceiling`` NULLs -> the lap is rejected, tallied under its error
    code, and kept out of the matrix (never silently imputed past the gate)."""
    ceiling = scripts_config.thresholds.null_ceiling
    raw = _full_raw()
    over = [fid for fid in FEATURE_IDS[: ceiling + 1]]

    rows, stats = btm.assemble_rows(
        [record(driver="VER", circuit="Bahrain", raw=raw, forced_null=over)],
        _medians(), scripts_config, tmp_path / "dq.jsonl",
    )

    assert rows == []
    assert stats[f"vec_{ErrorCode.LAP_NULL_EXCESS.value}"] == 1
    assert stats["vectors_built"] == 0


def test_assemble_rows_accepts_at_the_null_ceiling(
    scripts_config: AppConfig, tmp_path: Path
) -> None:
    """The boundary itself accepts and imputes — ceiling NULLs is the last accepted lap."""
    ceiling = scripts_config.thresholds.null_ceiling
    at = [fid for fid in FEATURE_IDS[:ceiling]]

    rows, stats = btm.assemble_rows(
        [record(driver="VER", circuit="Bahrain", raw=_full_raw(), forced_null=at)],
        _medians(value=7.0), scripts_config, tmp_path / "dq.jsonl",
    )

    assert stats["vectors_built"] == 1
    assert rows[0][at[0]] == 7.0  # imputed from the injected train median


def test_assemble_rows_fails_a_lap_with_no_train_median(
    scripts_config: AppConfig, tmp_path: Path
) -> None:
    """A NULL that cannot be imputed fails the lap rather than surviving as a NULL — the
    downstream half of the "feature with no train median" warning."""
    medians = FeatureMedians(global_medians={fid: 1.0 for fid in FEATURE_IDS[1:]})

    rows, stats = btm.assemble_rows(
        [record(driver="VER", circuit="Bahrain", raw=_full_raw(),
                forced_null=[FEATURE_IDS[0]])],
        medians, scripts_config, tmp_path / "dq.jsonl",
    )

    assert rows == []
    assert sum(v for k, v in stats.items() if k.startswith("vec_")) == 1


def test_assemble_rows_assembles_held_out_laps(
    scripts_config: AppConfig, tmp_path: Path
) -> None:
    """Held-out laps ARE assembled — they are the test split. Only the FIT (the median table)
    excludes them; excluding them here would leave nothing to evaluate on."""
    rows, _ = btm.assemble_rows(
        [record(driver="VER", circuit="Belgium", raw=_full_raw()),
         record(driver="VER", circuit="Italy", raw=_full_raw())],
        _medians(), scripts_config, tmp_path / "dq.jsonl",
    )

    assert {r["circuit"] for r in rows} == {"Belgium", "Italy"}


# --- discover / stable_core -----------------------------------------------------------------


def _corpus(root: Path, layout: dict[int, dict[str, list[str]]]) -> Path:
    """A Stage-A-shaped corpus: season/circuit/DRIVER.parquet holding readable telemetry."""
    session = pd.concat([telemetry_lap(i) for i in (1, 2, 3)], ignore_index=True)
    for season, circuits in layout.items():
        for circuit, drivers in circuits.items():
            d = root / str(season) / circuit
            d.mkdir(parents=True, exist_ok=True)
            for drv in drivers:
                session.to_parquet(d / f"{drv}.parquet", index=False)
    return root


def test_discover_returns_units_and_no_missing(tmp_path: Path) -> None:
    root = _corpus(tmp_path / "c", {2024: {"Bahrain": ["VER", "HAM"]}})

    units, missing = btm.discover(root, [2024])

    assert missing == []
    assert [(s, c, d) for s, c, d, _ in units] == [
        (2024, "Bahrain", "HAM"), (2024, "Bahrain", "VER"),
    ]


def test_discover_reports_an_absent_season(tmp_path: Path) -> None:
    """A requested season with no corpus is RETURNED, not skipped in silence — otherwise the
    build produces a matrix quietly missing a whole season."""
    root = _corpus(tmp_path / "c", {2024: {"Bahrain": ["VER"]}})

    units, missing = btm.discover(root, [2023, 2024, 2025])

    assert missing == [2023, 2025]
    assert len(units) == 1


def test_stable_core_keeps_only_drivers_present_every_season(tmp_path: Path) -> None:
    root = _corpus(
        tmp_path / "c",
        {2023: {"Bahrain": ["VER", "HAM", "RIC"]}, 2024: {"Bahrain": ["VER", "HAM"]}},
    )
    units, _ = btm.discover(root, [2023, 2024])

    assert btm.stable_core(units, [2023, 2024]) == {"VER", "HAM"}


# --- resample: channel kind is load-bearing -------------------------------------------------


def test_resample_interpolates_continuous_and_holds_discrete() -> None:
    """Continuous channels interpolate; Gear/DRS carry the previous observed value.

    Linearly interpolating a category code invents a transition at essentially every sample,
    which blows F15 (gear changes per km) past its range and NULLs it for every lap — the
    documented reason the two channel kinds are treated separately.
    """
    lap = telemetry_lap(1, n=4, hz_in=10.0)
    lap["Gear"] = [6.0, 6.0, 7.0, 7.0]
    lap["Speed"] = [100.0, 110.0, 120.0, 130.0]

    out = btm.resample_lap(lap, hz=50)

    assert set(np.unique(out["Gear"])) <= {6.0, 7.0}          # only observed codes
    assert len(np.unique(out["Speed"])) > len(lap)            # genuinely interpolated
    assert out["Speed"].between(100.0, 130.0).all()


def test_resample_carries_per_lap_constants() -> None:
    out = btm.resample_lap(telemetry_lap(3, n=6), hz=50)

    assert (out["lap"] == 3).all()
    assert (out["compound"] == "MEDIUM").all()
    assert (out["lap_time"] == 92.5).all()


def test_resample_returns_a_degenerate_lap_unchanged() -> None:
    """Under two usable timestamps there is no grid to build; the lap is returned as-is rather
    than being dropped or fabricated."""
    lap = telemetry_lap(1, n=1)

    assert btm.resample_lap(lap, hz=50).equals(lap)


def test_resample_session_covers_every_lap() -> None:
    frame = pd.concat([telemetry_lap(i, n=8) for i in (1, 2, 3)], ignore_index=True)

    out = btm.resample_session(frame, hz=50)

    assert sorted(out["lap"].unique()) == [1, 2, 3]


# --- record_dq: the logger's own failure path ------------------------------------------------


def test_record_dq_writes_a_typed_row(tmp_path: Path) -> None:
    log = tmp_path / "dq.jsonl"

    btm.record_dq(
        log, code=ErrorCode.CACHE_CORRUPT, entity_type="session", entity_id="2024/Bahrain/VER",
        detail=f"{btm.REASON_SESSION_UNREADABLE}: boom",
    )

    row = read_dq(log)[0]
    assert row["error_code"] == ErrorCode.CACHE_CORRUPT.value
    assert row["entity_type"] == "session"
    assert row["entity_id"] == "2024/Bahrain/VER"
    assert btm.REASON_SESSION_UNREADABLE in row["error_detail"]


def test_record_dq_surfaces_an_unwritable_sink_without_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A build must not die because the audit sink is unwritable — but the sink failing must
    not be silent either (non-negotiable #1 applies to the logger itself)."""
    def _boom(**_kwargs: Any) -> None:
        raise btm.DataQualityLogError("sink is read-only")

    monkeypatch.setattr(btm, "log_data_quality", _boom)

    btm.record_dq(
        tmp_path / "dq.jsonl", code=ErrorCode.CACHE_CORRUPT, entity_type="session",
        entity_id="2024/Bahrain/VER", detail="lost this",
    )

    err = capsys.readouterr().err
    assert "UNWRITABLE" in err
    assert "lost this" in err  # the row that could not be written is echoed, not discarded


# --- main(): the two fail-loud corpus paths --------------------------------------------------


def _run_main(monkeypatch: pytest.MonkeyPatch, **argv: Any) -> None:
    args = ["build_training_matrix.py"]
    for k, v in argv.items():
        args += [f"--{k.replace('_', '-')}", *(v if isinstance(v, list) else [v])]
    monkeypatch.setattr("sys.argv", args)
    btm.main()


def test_main_logs_and_survives_an_unreadable_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One corrupt parquet must not void the build: it is tallied, logged CACHE_CORRUPT, and
    the loop continues. Before this guard a single truncated file raised ArrowInvalid and
    discarded every session already extracted, none of it yet on disk.
    """
    corpus = _corpus(tmp_path / "c", {2024: {"Bahrain": ["VER"]}})
    (corpus / "2024" / "Bahrain" / "CORRUPT.parquet").write_bytes(b"not a parquet file")
    out = tmp_path / "out" / "m.parquet"

    with pytest.raises(SystemExit) as exc:
        _run_main(monkeypatch, seasons="2024", data_dir=str(corpus), out=str(out))

    # No usable laps in this toy corpus, so the build still refuses to write an empty matrix —
    # but it got there by HANDLING the corrupt file, not by dying on it.
    assert exc.value.code == 3
    rows = read_dq(out.parent / "build_dq.jsonl")
    corrupt = [r for r in rows if r["error_code"] == ErrorCode.CACHE_CORRUPT.value]
    assert len(corrupt) == 1
    assert btm.REASON_SESSION_UNREADABLE in corrupt[0]["error_detail"]
    assert "2024/Bahrain/CORRUPT" == corrupt[0]["entity_id"]


def test_main_logs_an_absent_season(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    corpus = _corpus(tmp_path / "c", {2024: {"Bahrain": ["VER"]}})
    out = tmp_path / "out" / "m.parquet"

    with pytest.raises(SystemExit):
        _run_main(monkeypatch, seasons=["2024", "2025"], data_dir=str(corpus), out=str(out))

    rows = read_dq(out.parent / "build_dq.jsonl")
    absent = [r for r in rows if r["error_code"] == ErrorCode.DATA_SOURCE_UNAVAILABLE.value]
    assert [r["entity_id"] for r in absent] == ["2025"]
    assert btm.REASON_SEASON_ABSENT in absent[0]["error_detail"]


def test_main_refuses_an_empty_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit) as exc:
        _run_main(
            monkeypatch, seasons="2024", data_dir=str(tmp_path / "nothing"),
            out=str(tmp_path / "out" / "m.parquet"),
        )

    assert exc.value.code == 2


def test_main_refuses_to_assume_a_resample_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, repo_config_path: Path
) -> None:
    """No ``upload`` block -> exit 2. Defaulting the rate would put an unstated Hz into every
    feature in the matrix (a magic number AND a silent fallback)."""
    loaded = btm.load_config(repo_config_path)
    loaded.upload = None
    monkeypatch.setattr(btm, "load_config", lambda _p: loaded)

    with pytest.raises(SystemExit) as exc:
        _run_main(
            monkeypatch, seasons="2024", data_dir=str(tmp_path),
            out=str(tmp_path / "out" / "m.parquet"),
        )

    assert exc.value.code == 2


# --- circuit-type mapping --------------------------------------------------------------------


def test_every_mapped_circuit_type_is_a_config_member(scripts_config: AppConfig) -> None:
    """The mapping this module owns must speak the config's vocabulary — an invented type would
    silently mis-key F04 grouping, F10's lookup and F11's setup mapping."""
    assert set(btm._CIRCUIT_TYPE.values()) <= set(scripts_config.circuit_types)


def test_held_out_circuits_are_mapped(scripts_config: AppConfig) -> None:
    """Belgium and Italy must be present in the mapping under exactly the config's spelling,
    since that same string is what the median filter tests for."""
    for circuit in scripts_config.held_out_circuits:
        assert circuit in btm._CIRCUIT_TYPE
