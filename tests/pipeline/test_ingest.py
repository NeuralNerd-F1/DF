"""F01 ingestion — one test per acceptance criterion AND one per failure mode.

The acceptance criteria (spec §7) and the failure-mode table (spec §6) are the
definition of done; each is exercised below, including its data_quality_log row.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import pytest
from app.contracts.config import AppConfig
from app.contracts.enums import Status
from app.contracts.errors import ErrorCode
from app.pipeline.ingest import (
    CircuitBreaker,
    CompoundSource,
    IngestSourceError,
    SessionRequest,
    SourceUnavailableError,
    TyreAgeSource,
    build_breakers,
    cache_path_for,
    call_with_breaker,
    ingest_session,
)

from tests.pipeline.conftest import (
    DAY_S,
    FIXED_NOW_EPOCH,
    FakeClock,
    Recorder,
    ergast_frame,
    no_sleep,
    openf1_frame,
    raises_session_unavailable,
    raises_source_down,
    returns,
    returns_empty,
    telemetry_frame,
)

# --- helpers -----------------------------------------------------------------------


def run(
    request: SessionRequest,
    cfg: AppConfig,
    *,
    fastf1: Recorder,
    openf1: Recorder,
    ergast: Recorder,
    now_epoch: float = FIXED_NOW_EPOCH,
):
    """Drive ingest_session with injected clocks/sleep so nothing touches wall time."""
    clock = FakeClock()
    return ingest_session(
        request,
        cfg,
        fastf1_fetch=fastf1,
        openf1_fetch=openf1,
        ergast_fetch=ergast,
        clock=clock,
        sleep=no_sleep,
        now_epoch=lambda: now_epoch,
    )


def read_dq(cfg: AppConfig) -> list[dict]:
    p = Path(cfg.paths.data_quality_log)
    if not p.is_file():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]


def write_cache(cfg: AppConfig, request: SessionRequest, age_days: float) -> Path:
    """Write a small valid Parquet cache for `request` and back-date its mtime."""
    cpath = cache_path_for(request, Path(cfg.paths.cache_dir))
    cpath.parent.mkdir(parents=True, exist_ok=True)
    df = telemetry_frame(laps=(1, 2))
    df["compound"] = "SOFT"
    df["tyre_age"] = 0
    df.to_parquet(cpath, index=False)
    import os

    mtime = FIXED_NOW_EPOCH - age_days * DAY_S
    os.utime(cpath, (mtime, mtime))
    return cpath


# --- Happy path (spec §8) ----------------------------------------------------------


def test_happy_path_openf1_compound_ergast_age(request_obj, ingest_config):
    res = run(
        request_obj,
        ingest_config,
        fastf1=returns(telemetry_frame()),
        openf1=returns(openf1_frame()),
        ergast=returns(ergast_frame()),
    )
    assert res.status is Status.OK
    assert res.ok
    assert res.compound_source is CompoundSource.OPENF1
    assert res.tyre_age_source is TyreAgeSource.ERGAST
    assert res.rows == 6
    assert res.error_code is None
    assert res.flags == []
    assert read_dq(ingest_config) == []  # a clean run logs nothing


# --- AC1 + freshness (§7, §8) ------------------------------------------------------


def test_fresh_cache_is_used_without_any_network_call(request_obj, ingest_config):
    write_cache(ingest_config, request_obj, age_days=6)  # < 7 days -> fresh
    fastf1 = returns(telemetry_frame())
    openf1 = returns(openf1_frame())
    ergast = returns(ergast_frame())
    res = run(request_obj, ingest_config, fastf1=fastf1, openf1=openf1, ergast=ergast)
    assert res.status is Status.OK
    assert res.source_used == "cache"
    assert fastf1.calls == 0 and openf1.calls == 0 and ergast.calls == 0  # no network


def test_stale_cache_is_refetched(request_obj, ingest_config):
    write_cache(ingest_config, request_obj, age_days=8)  # > 7 days -> stale
    fastf1 = returns(telemetry_frame())
    res = run(
        request_obj, ingest_config,
        fastf1=fastf1, openf1=returns(openf1_frame()), ergast=returns(ergast_frame()),
    )
    assert fastf1.calls == 1  # cache too old -> network consulted
    assert res.source_used == "fastf1"


# --- AC2 / failure row 1: DATA_001 -------------------------------------------------


def test_data_001_when_fastf1_session_unavailable(request_obj, ingest_config):
    res = run(
        request_obj, ingest_config,
        fastf1=raises_session_unavailable(),
        openf1=returns(openf1_frame()), ergast=returns(ergast_frame()),
    )
    assert res.status is Status.FAILED
    assert res.error_code is ErrorCode.DATA_001
    assert res.rows == 0
    assert res.logged is True
    rows = read_dq(ingest_config)
    assert [r["error_code"] for r in rows] == ["DATA_001"]
    assert rows[0]["fallback_used"] is False  # never substituted


def test_data_001_when_fastf1_returns_empty(request_obj, ingest_config):
    res = run(
        request_obj, ingest_config,
        fastf1=returns_empty(),
        openf1=returns(openf1_frame()), ergast=returns(ergast_frame()),
    )
    assert res.error_code is ErrorCode.DATA_001


def test_data_001_does_not_substitute_even_when_cache_exists(request_obj, ingest_config):
    write_cache(ingest_config, request_obj, age_days=8)  # stale cache present
    res = run(
        request_obj, ingest_config,
        fastf1=raises_session_unavailable(),
        openf1=returns(openf1_frame()), ergast=returns(ergast_frame()),
    )
    assert res.error_code is ErrorCode.DATA_001  # NOT served from cache
    assert res.source_used == "fastf1"


# --- failure row 2: STALE_DATA -----------------------------------------------------


def test_stale_data_served_when_fastf1_down_and_cache_present(request_obj, ingest_config):
    write_cache(ingest_config, request_obj, age_days=8)
    res = run(
        request_obj, ingest_config,
        fastf1=raises_source_down(),
        openf1=returns(openf1_frame()), ergast=returns(ergast_frame()),
    )
    assert res.status is Status.DEGRADED
    assert ErrorCode.STALE_DATA in res.flags
    assert res.source_used == "cache"
    assert res.rows > 0
    rows = read_dq(ingest_config)
    assert rows[0]["error_code"] == "STALE_DATA"
    assert rows[0]["fallback_used"] is True


# --- failure row 3: DATA_SOURCE_UNAVAILABLE ----------------------------------------


def test_data_source_unavailable_when_fastf1_down_and_no_cache(request_obj, ingest_config):
    res = run(
        request_obj, ingest_config,
        fastf1=raises_source_down(),
        openf1=returns(openf1_frame()), ergast=returns(ergast_frame()),
    )
    assert res.status is Status.FAILED
    assert res.error_code is ErrorCode.DATA_SOURCE_UNAVAILABLE
    assert res.logged is True
    assert read_dq(ingest_config)[0]["error_code"] == "DATA_SOURCE_UNAVAILABLE"


# --- AC3 / failure row 4: OpenF1 timeout -> Ergast compound fallback ---------------


def test_openf1_timeout_falls_back_to_ergast_for_compound(request_obj, ingest_config):
    res = run(
        request_obj, ingest_config,
        fastf1=returns(telemetry_frame()),
        openf1=raises_source_down(),  # OpenF1 timeout / down
        ergast=returns(ergast_frame(with_compound=True)),
    )
    assert res.compound_source is CompoundSource.ERGAST
    assert ErrorCode.COMPOUND_FALLBACK in res.flags
    assert res.status is Status.DEGRADED
    codes = [r["error_code"] for r in read_dq(ingest_config)]
    assert "COMPOUND_FALLBACK" in codes


# --- AC4 / failure row 5: both down -> COMPOUND_UNKNOWN, core intact ---------------


def test_compound_unknown_when_openf1_and_ergast_both_down(request_obj, ingest_config):
    res = run(
        request_obj, ingest_config,
        fastf1=returns(telemetry_frame()),
        openf1=raises_source_down(),
        ergast=raises_source_down(),
    )
    assert res.compound_source is CompoundSource.UNKNOWN
    assert ErrorCode.COMPOUND_UNKNOWN in res.flags
    # Core telemetry is untouched — the fingerprint path is unaffected.
    assert res.rows == 6
    assert res.cache_path is not None and res.cache_path.is_file()
    codes = [r["error_code"] for r in read_dq(ingest_config)]
    assert "COMPOUND_UNKNOWN" in codes


# --- AC5 / failure row 6: Ergast down -> DERIVED_AGE -------------------------------


def test_derived_age_when_ergast_down(request_obj, ingest_config):
    res = run(
        request_obj, ingest_config,
        fastf1=returns(telemetry_frame()),
        openf1=returns(openf1_frame()),  # pit flags available for derivation
        ergast=raises_source_down(),
    )
    assert res.tyre_age_source is TyreAgeSource.DERIVED
    assert res.compound_source is CompoundSource.OPENF1  # compound still fine from OpenF1
    assert ErrorCode.DERIVED_AGE in res.flags
    cached = pd.read_parquet(res.cache_path)
    assert "tyre_age" in cached.columns  # age was derived and persisted
    codes = [r["error_code"] for r in read_dq(ingest_config)]
    assert "DERIVED_AGE" in codes


# --- AC6 + breaker test: opens after configured count, one failure per call --------


def test_breaker_opens_after_configured_count_one_failure_per_exhausted_call(ingest_config):
    clock = FakeClock()
    bcfg = ingest_config.ingest.breakers.fastf1  # open_after=3, retries=3, window_s=60
    breaker = CircuitBreaker("fastf1", bcfg, clock=clock, sleep=no_sleep)
    calls = {"n": 0}

    def always_fail():
        calls["n"] += 1
        raise IngestSourceError("down")

    # Three fully-exhausted invocations within the window -> exactly three failures.
    for i in range(bcfg.open_after):
        with pytest.raises(SourceUnavailableError) as exc:
            call_with_breaker(breaker, always_fail)
        assert exc.value.short_circuited is False
        assert breaker.failure_count() == i + 1  # one failure per exhausted call
        clock.advance(1.0)  # still well inside the 60 s window

    assert breaker.is_open()
    # 4th call short-circuits: no source call, returns a fallback in well under 500 ms.
    before = calls["n"]
    t0 = time.perf_counter()
    with pytest.raises(SourceUnavailableError) as exc:
        call_with_breaker(breaker, always_fail)
    assert exc.value.short_circuited is True
    assert calls["n"] == before  # the source was never called
    assert (time.perf_counter() - t0) < 0.5


def test_breaker_open_short_circuits_through_ingest_session(request_obj, ingest_config):
    """AC6 at the integration level: a pre-opened breaker yields a typed result fast,
    without ever calling the source."""
    write_cache(ingest_config, request_obj, age_days=8)  # stale cache available to serve
    clock = FakeClock()
    breakers = build_breakers(ingest_config, clock=clock, sleep=no_sleep)

    def fail():
        raise IngestSourceError("down")

    for _ in range(ingest_config.ingest.breakers.fastf1.open_after):
        with pytest.raises(SourceUnavailableError):
            call_with_breaker(breakers["fastf1"], fail)
        clock.advance(1.0)
    assert breakers["fastf1"].is_open()

    fastf1 = returns(telemetry_frame())
    t0 = time.perf_counter()
    res = ingest_session(
        request_obj, ingest_config,
        fastf1_fetch=fastf1,
        openf1_fetch=returns(openf1_frame()),
        ergast_fetch=returns(ergast_frame()),
        clock=clock, sleep=no_sleep, now_epoch=lambda: FIXED_NOW_EPOCH,
        breakers=breakers,
    )
    assert (time.perf_counter() - t0) < 0.5  # no hang
    assert fastf1.calls == 0  # breaker open -> source never called
    assert res.status is Status.DEGRADED and ErrorCode.STALE_DATA in res.flags


def test_breaker_recovers_when_failures_age_out_of_window(ingest_config):
    clock = FakeClock()
    bcfg = ingest_config.ingest.breakers.ergast  # open_after=2, retries=2, window_s=60
    breaker = CircuitBreaker("ergast", bcfg, clock=clock, sleep=no_sleep)

    def fail():
        raise IngestSourceError("down")

    for _ in range(bcfg.open_after):
        with pytest.raises(SourceUnavailableError):
            call_with_breaker(breaker, fail)
    assert breaker.is_open()
    clock.advance(bcfg.window_s + 1)  # let both failures age out
    assert not breaker.is_open()


# --- AC7: every failure path writes exactly one data_quality row -------------------


@pytest.mark.parametrize(
    ("fastf1", "openf1", "ergast", "expected_code"),
    [
        (raises_session_unavailable, returns, returns, "DATA_001"),
        (raises_source_down, returns, returns, "DATA_SOURCE_UNAVAILABLE"),
    ],
)
def test_failed_paths_log_one_row(
    request_obj, ingest_config, fastf1, openf1, ergast, expected_code
):
    res = run(
        request_obj, ingest_config,
        fastf1=fastf1(),
        openf1=openf1(openf1_frame()) if openf1 is returns else openf1(),
        ergast=ergast(ergast_frame()) if ergast is returns else ergast(),
    )
    rows = read_dq(ingest_config)
    assert len(rows) == 1
    assert rows[0]["error_code"] == expected_code
    assert res.logged is True


def test_stale_path_logs_one_row(request_obj, ingest_config):
    write_cache(ingest_config, request_obj, age_days=8)
    run(
        request_obj, ingest_config,
        fastf1=raises_source_down(),
        openf1=returns(openf1_frame()), ergast=returns(ergast_frame()),
    )
    rows = read_dq(ingest_config)
    assert len(rows) == 1 and rows[0]["error_code"] == "STALE_DATA"


def test_compound_fallback_logs_one_row(request_obj, ingest_config):
    run(
        request_obj, ingest_config,
        fastf1=returns(telemetry_frame()),
        openf1=raises_source_down(),
        ergast=returns(ergast_frame(with_compound=True)),  # ergast still serves tyre age
    )
    rows = read_dq(ingest_config)
    assert len(rows) == 1 and rows[0]["error_code"] == "COMPOUND_FALLBACK"


def test_derived_age_logs_one_row(request_obj, ingest_config):
    run(
        request_obj, ingest_config,
        fastf1=returns(telemetry_frame()),
        openf1=returns(openf1_frame()),  # compound + pit flags fine
        ergast=raises_source_down(),
    )
    rows = read_dq(ingest_config)
    assert len(rows) == 1 and rows[0]["error_code"] == "DERIVED_AGE"


def test_both_down_logs_compound_unknown_and_derived_age(request_obj, ingest_config):
    """Combined OpenF1+Ergast outage writes exactly two rows: the compound + age faults."""
    run(
        request_obj, ingest_config,
        fastf1=returns(telemetry_frame()),
        openf1=raises_source_down(),
        ergast=raises_source_down(),
    )
    codes = sorted(r["error_code"] for r in read_dq(ingest_config))
    assert codes == ["COMPOUND_UNKNOWN", "DERIVED_AGE"]


# --- AC8: all timeouts/retries/backoffs come from config ---------------------------


def test_breaker_policy_is_read_from_config(ingest_config):
    breakers = build_breakers(ingest_config)
    cfg = ingest_config.ingest.breakers
    assert breakers["fastf1"].config.timeout_s == cfg.fastf1.timeout_s == 30
    assert breakers["fastf1"].config.retries == cfg.fastf1.retries == 3
    assert breakers["fastf1"].config.backoff_s == cfg.fastf1.backoff_s == [5, 15, 30]
    assert breakers["openf1"].config.open_after == cfg.openf1.open_after == 3
    assert breakers["ergast"].config.retries == cfg.ergast.retries == 2


# --- reinforcement (c): fresh fetch writes Parquet to the canonical path -----------


def test_fresh_fetch_writes_parquet_to_canonical_cache_path(request_obj, ingest_config):
    res = run(
        request_obj, ingest_config,
        fastf1=returns(telemetry_frame()),
        openf1=returns(openf1_frame()), ergast=returns(ergast_frame()),
    )
    expected = Path(ingest_config.paths.cache_dir) / "2024" / "Bahrain" / "VER.parquet"
    assert res.cache_path == expected
    assert expected.is_file()
    cached = pd.read_parquet(expected)
    # The F01 -> F02 column contract is present in the persisted frame.
    for col in ("Speed", "Throttle", "Brake", "RPM", "Gear", "DRS", "X", "Y",
                "lap", "compound", "pit_out", "lap_time", "stint", "tyre_age"):
        assert col in cached.columns, col


# --- Cache integrity: no-silent-failure on corrupt / unwritable Parquet ------------


def _write_corrupt_cache(cfg: AppConfig, request: SessionRequest, age_days: float) -> Path:
    cpath = cache_path_for(request, Path(cfg.paths.cache_dir))
    cpath.parent.mkdir(parents=True, exist_ok=True)
    cpath.write_bytes(b"this is not a parquet file")
    import os

    mtime = FIXED_NOW_EPOCH - age_days * DAY_S
    os.utime(cpath, (mtime, mtime))
    return cpath


def test_corrupt_fresh_cache_is_logged_and_refetched(request_obj, ingest_config):
    _write_corrupt_cache(ingest_config, request_obj, age_days=6)  # fresh but corrupt
    fastf1 = returns(telemetry_frame())
    res = run(
        request_obj, ingest_config,
        fastf1=fastf1, openf1=returns(openf1_frame()), ergast=returns(ergast_frame()),
    )
    assert fastf1.calls == 1  # corrupt cache treated as a miss -> refetch, no hang
    assert ErrorCode.CACHE_CORRUPT in res.flags
    assert res.rows == 6
    assert "CACHE_CORRUPT" in [r["error_code"] for r in read_dq(ingest_config)]


def test_corrupt_stale_cache_with_fastf1_down_is_data_source_unavailable(
    request_obj, ingest_config
):
    _write_corrupt_cache(ingest_config, request_obj, age_days=8)  # stale AND corrupt
    res = run(
        request_obj, ingest_config,
        fastf1=raises_source_down(),
        openf1=returns(openf1_frame()), ergast=returns(ergast_frame()),
    )
    assert res.status is Status.FAILED
    assert res.error_code is ErrorCode.DATA_SOURCE_UNAVAILABLE  # corrupt cache not served
    assert read_dq(ingest_config)[0]["error_code"] == "DATA_SOURCE_UNAVAILABLE"


def test_cache_write_failure_is_degraded_not_fatal(request_obj, ingest_config, tmp_path):
    # Point the cache dir under an existing *file* so the mkdir/write must fail.
    blocker = tmp_path / "blocker_file"
    blocker.write_text("i am a file, not a directory", encoding="utf-8")
    ingest_config.paths.cache_dir = str(blocker / "cache")
    res = run(
        request_obj, ingest_config,
        fastf1=returns(telemetry_frame()),
        openf1=returns(openf1_frame()), ergast=returns(ergast_frame()),
    )
    assert res.status is Status.DEGRADED  # cache is an optimisation, not a hard dep
    assert res.cache_path is None
    assert res.rows == 6  # the freshly-fetched telemetry is still returned
    assert ErrorCode.CACHE_CORRUPT in res.flags
    assert "CACHE_CORRUPT" in [r["error_code"] for r in read_dq(ingest_config)]
