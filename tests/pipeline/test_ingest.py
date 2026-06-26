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
    cache_meta_path,
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
    track_status_frame,
)

# --- helpers -----------------------------------------------------------------------


def run(
    request: SessionRequest,
    cfg: AppConfig,
    *,
    fastf1: Recorder,
    openf1: Recorder,
    ergast: Recorder,
    fastf1_laps: Recorder | None = None,
    now_epoch: float = FIXED_NOW_EPOCH,
):
    """Drive ingest_session with injected clocks/sleep so nothing touches wall time.

    Track status is available (all-green) by default so tests that don't care about it
    don't emit a TRACK_STATUS_UNAVAILABLE row; pass ``fastf1_laps`` to override.
    """
    clock = FakeClock()
    if fastf1_laps is None:
        fastf1_laps = returns(track_status_frame())
    return ingest_session(
        request,
        cfg,
        fastf1_fetch=fastf1,
        openf1_fetch=openf1,
        ergast_fetch=ergast,
        fastf1_laps_fetch=fastf1_laps,
        clock=clock,
        sleep=no_sleep,
        now_epoch=lambda: now_epoch,
    )


def read_dq(cfg: AppConfig) -> list[dict]:
    p = Path(cfg.paths.data_quality_log)
    if not p.is_file():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]


def write_cache(
    cfg: AppConfig,
    request: SessionRequest,
    age_days: float,
    *,
    compound_source: str = "OPENF1",
    tyre_age_source: str = "ERGAST",
    flags: list[str] | None = None,
) -> Path:
    """Write a small valid Parquet cache + provenance sidecar and back-date the mtime."""
    cpath = cache_path_for(request, Path(cfg.paths.cache_dir))
    cpath.parent.mkdir(parents=True, exist_ok=True)
    df = telemetry_frame(laps=(1, 2))
    df["compound"] = "SOFT"
    df["tyre_age"] = 0
    df.to_parquet(cpath, index=False)
    meta = {
        "compound_source": compound_source,
        "tyre_age_source": tyre_age_source,
        "flags": flags or [],
    }
    cache_meta_path(cpath).write_text(json.dumps(meta), encoding="utf-8")
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
                "lap", "track_status", "compound", "pit_out", "lap_time", "stint", "tyre_age"):
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


# --- Breaker opens under realistic backoff timing (review finding #1) ---------------


def test_fastf1_breaker_opens_under_realistic_backoff_timing(ingest_config):
    """The breaker must open during a sustained outage even though FastF1's own backoff
    (5+15+30=50 s/call) exceeds window_s=60 s. Counting consecutive failures (not a
    wall-clock window) is what makes this work."""
    clock = FakeClock()

    def advancing_sleep(seconds: float) -> None:
        clock.advance(seconds)  # backoff consumes time, as it would in production

    bcfg = ingest_config.ingest.breakers.fastf1
    breaker = CircuitBreaker("fastf1", bcfg, clock=clock, sleep=advancing_sleep)

    def fail():
        raise IngestSourceError("down")

    for _ in range(bcfg.open_after):
        with pytest.raises(SourceUnavailableError):
            call_with_breaker(breaker, fail)
    assert breaker.is_open()  # would never open under a naive sliding-window count


def test_breaker_success_resets_the_failure_streak(ingest_config):
    clock = FakeClock()
    bcfg = ingest_config.ingest.breakers.ergast  # open_after=2
    breaker = CircuitBreaker("ergast", bcfg, clock=clock, sleep=no_sleep)
    state = {"fail": True}

    def flaky():
        if state["fail"]:
            raise IngestSourceError("down")
        return pd.DataFrame({"lap": [1]})

    with pytest.raises(SourceUnavailableError):
        call_with_breaker(breaker, flaky)
    assert breaker.failure_count() == 1
    state["fail"] = False
    call_with_breaker(breaker, flaky)  # a success resets the streak
    assert breaker.failure_count() == 0


# --- Cache re-read restores provenance, never launders (review finding #2) ----------


def test_degraded_session_is_not_laundered_on_cache_reread(request_obj, ingest_config):
    """A session first ingested via a fallback must replay as DEGRADED with the same
    flags and provenance — not silently downgraded to OK on the cached read."""
    import os

    first = run(
        request_obj, ingest_config,
        fastf1=returns(telemetry_frame()),
        openf1=returns(openf1_frame()), ergast=raises_source_down(),  # -> DERIVED_AGE
    )
    assert first.status is Status.DEGRADED and ErrorCode.DERIVED_AGE in first.flags
    os.utime(first.cache_path, (FIXED_NOW_EPOCH, FIXED_NOW_EPOCH))  # make it look fresh

    fastf1 = returns(telemetry_frame())
    openf1 = returns(openf1_frame())
    ergast = returns(ergast_frame())
    replay = run(request_obj, ingest_config, fastf1=fastf1, openf1=openf1, ergast=ergast)
    assert fastf1.calls == 0 and openf1.calls == 0 and ergast.calls == 0  # served from cache
    assert replay.source_used == "cache"
    assert replay.status is Status.DEGRADED  # NOT laundered to OK
    assert ErrorCode.DERIVED_AGE in replay.flags
    assert replay.tyre_age_source is TyreAgeSource.DERIVED


def test_cache_reread_preserves_ergast_compound_provenance(request_obj, ingest_config):
    import os

    first = run(
        request_obj, ingest_config,
        fastf1=returns(telemetry_frame()),
        openf1=raises_source_down(), ergast=returns(ergast_frame(with_compound=True)),
    )
    assert first.compound_source is CompoundSource.ERGAST
    os.utime(first.cache_path, (FIXED_NOW_EPOCH, FIXED_NOW_EPOCH))
    replay = run(
        request_obj, ingest_config,
        fastf1=returns(telemetry_frame()),
        openf1=returns(openf1_frame()), ergast=returns(ergast_frame()),
    )
    assert replay.source_used == "cache"
    assert replay.compound_source is CompoundSource.ERGAST  # not re-inferred as OPENF1
    assert ErrorCode.COMPOUND_FALLBACK in replay.flags


def test_cache_without_provenance_sidecar_is_treated_as_corrupt(request_obj, ingest_config):
    """A Parquet present but missing its provenance sidecar must not be served with
    guessed provenance — it is treated as corrupt and refetched."""
    import os

    cpath = cache_path_for(request_obj, Path(ingest_config.paths.cache_dir))
    cpath.parent.mkdir(parents=True, exist_ok=True)
    telemetry_frame(laps=(1, 2)).to_parquet(cpath, index=False)  # no .meta.json sidecar
    os.utime(cpath, (FIXED_NOW_EPOCH, FIXED_NOW_EPOCH))
    fastf1 = returns(telemetry_frame())
    res = run(
        request_obj, ingest_config,
        fastf1=fastf1, openf1=returns(openf1_frame()), ergast=returns(ergast_frame()),
    )
    assert fastf1.calls == 1  # incomplete cache -> refetch
    assert ErrorCode.CACHE_CORRUPT in res.flags


# --- Contract columns are never dropped on a fallback (review findings #3, #4) ------


@pytest.mark.parametrize(
    ("openf1_factory", "ergast_factory"),
    [
        (lambda: returns(openf1_frame()), lambda: returns(ergast_frame())),
        (raises_source_down, lambda: returns(ergast_frame(with_compound=True))),
        (lambda: returns(openf1_frame()), raises_source_down),
        (raises_source_down, raises_source_down),
    ],
)
def test_persisted_cache_always_has_full_contract_columns(
    request_obj, ingest_config, openf1_factory, ergast_factory
):
    res = run(
        request_obj, ingest_config,
        fastf1=returns(telemetry_frame()),
        openf1=openf1_factory(), ergast=ergast_factory(),
    )
    cached = pd.read_parquet(res.cache_path)
    for col in ("Speed", "Throttle", "Brake", "RPM", "Gear", "DRS", "X", "Y", "lap",
                "track_status", "compound", "pit_in", "pit_out", "lap_time", "stint",
                "tyre_age"):
        assert col in cached.columns, col


# --- Malformed adapter frames produce typed outcomes, never crashes (#5, #6, #8) ----


def test_fastf1_frame_without_lap_column_is_data_001(request_obj, ingest_config):
    no_lap = returns(pd.DataFrame({"Speed": [250.0, 251.0]}))  # missing the 'lap' key
    res = run(
        request_obj, ingest_config,
        fastf1=no_lap, openf1=returns(openf1_frame()), ergast=returns(ergast_frame()),
    )
    assert res.status is Status.FAILED and res.error_code is ErrorCode.DATA_001


def test_context_frame_without_lap_does_not_crash(request_obj, ingest_config):
    openf1_no_lap = returns(pd.DataFrame({"compound": ["SOFT"], "pit_out": [True]}))  # no 'lap'
    res = run(
        request_obj, ingest_config,
        fastf1=returns(telemetry_frame()), openf1=openf1_no_lap, ergast=returns(ergast_frame()),
    )
    assert res.rows == 6  # no KeyError; compound unmergeable -> UNKNOWN fallback
    assert res.compound_source is CompoundSource.UNKNOWN


def test_openf1_without_compound_column_falls_back_to_ergast(request_obj, ingest_config):
    openf1_no_compound = returns(
        pd.DataFrame({"lap": [1, 2, 3], "pit_in": [False] * 3, "pit_out": [True, False, False]})
    )
    res = run(
        request_obj, ingest_config,
        fastf1=returns(telemetry_frame()),
        openf1=openf1_no_compound, ergast=returns(ergast_frame(with_compound=True)),
    )
    assert res.compound_source is CompoundSource.ERGAST  # not falsely reported OPENF1
    assert ErrorCode.COMPOUND_FALLBACK in res.flags
    cached = pd.read_parquet(res.cache_path)
    assert "compound" in cached.columns
    assert "pit_out" in cached.columns  # pit flags from OpenF1 are still kept


def test_duplicate_lap_in_context_does_not_multiply_rows(request_obj, ingest_config):
    dup = returns(
        pd.DataFrame(
            {
                "lap": [1, 1, 2, 2, 3, 3],
                "compound": ["SOFT"] * 6,
                "pit_in": [False] * 6,
                "pit_out": [True, True, False, False, False, False],
            }
        )
    )
    res = run(
        request_obj, ingest_config,
        fastf1=returns(telemetry_frame()), openf1=dup, ergast=returns(ergast_frame()),
    )
    assert res.rows == 6  # 3 laps x 2 samples; the duplicate-lap join must not explode


# --- Missing ingest: config block returns a typed result, not a bare raise (#7) -----


def test_config_without_ingest_block_returns_typed_config_001(
    request_obj, valid_config_file, tmp_path
):
    from app.contracts.config import load_config

    cfg = load_config(valid_config_file)  # VALID_CONFIG_YAML has no ingest: block
    assert cfg.ingest is None
    cfg.paths.data_quality_log = str(tmp_path / "dq.jsonl")
    cfg.paths.cache_dir = str(tmp_path / "cache")
    res = ingest_session(
        request_obj, cfg,
        fastf1_fetch=returns(telemetry_frame()),
        openf1_fetch=returns(openf1_frame()),
        ergast_fetch=returns(ergast_frame()),
        now_epoch=lambda: FIXED_NOW_EPOCH,
    )
    assert res.status is Status.FAILED
    assert res.error_code is ErrorCode.CONFIG_001
    assert res.logged is True


# --- Raw FastF1 track status persisted to the cache (feature/F01-track-status) ------


def test_track_status_column_present_and_carries_raw_sc_vsc_status(request_obj, ingest_config):
    """The raw per-lap track_status is persisted; a known SC/VSC lap keeps its non-green
    code (no green-flag interpretation in F01)."""
    laps = (1, 2, 3, 4)
    status = {2: "4", 3: "6"}  # lap 2 = Safety Car, lap 3 = Virtual Safety Car
    res = run(
        request_obj, ingest_config,
        fastf1=returns(telemetry_frame(laps=laps)),
        openf1=returns(openf1_frame(laps=laps)),
        ergast=returns(ergast_frame(laps=laps)),
        fastf1_laps=returns(track_status_frame(laps=laps, status=status)),
    )
    cached = pd.read_parquet(res.cache_path)
    assert "track_status" in cached.columns
    # Raw codes are stored verbatim, broadcast across each lap's samples.
    assert set(cached.loc[cached["lap"] == 2, "track_status"]) == {"4"}  # SC
    assert set(cached.loc[cached["lap"] == 3, "track_status"]) == {"6"}  # VSC
    assert set(cached.loc[cached["lap"] == 1, "track_status"]) == {"1"}  # green
    assert res.status is Status.OK  # track status available -> no degradation


def test_track_status_unavailable_nulls_column_and_logs_without_failing(
    request_obj, ingest_config
):
    res = run(
        request_obj, ingest_config,
        fastf1=returns(telemetry_frame()),
        openf1=returns(openf1_frame()),
        ergast=returns(ergast_frame()),
        fastf1_laps=raises_source_down(),  # Laps.TrackStatus unavailable
    )
    assert res.status is Status.DEGRADED  # session continues, not failed
    assert ErrorCode.TRACK_STATUS_UNAVAILABLE in res.flags
    cached = pd.read_parquet(res.cache_path)
    assert "track_status" in cached.columns  # column present...
    assert cached["track_status"].isna().all()  # ...but null for every lap
    assert "TRACK_STATUS_UNAVAILABLE" in [r["error_code"] for r in read_dq(ingest_config)]


def test_track_status_breaker_open_does_not_escape_untyped(request_obj, ingest_config):
    """A laps adapter that wraps call_with_breaker raises SourceUnavailableError when its
    breaker is open; ingest_session must absorb it (TRACK_STATUS_UNAVAILABLE), not crash."""

    def laps_breaker_open(_req):
        raise SourceUnavailableError("fastf1_laps", short_circuited=True)

    res = run(
        request_obj, ingest_config,
        fastf1=returns(telemetry_frame()),
        openf1=returns(openf1_frame()), ergast=returns(ergast_frame()),
        fastf1_laps=laps_breaker_open,
    )
    assert res.status is Status.DEGRADED  # no untyped exception escaped
    assert ErrorCode.TRACK_STATUS_UNAVAILABLE in res.flags


def test_track_status_session_unavailable_is_recorded(request_obj, ingest_config):
    res = run(
        request_obj, ingest_config,
        fastf1=returns(telemetry_frame()),
        openf1=returns(openf1_frame()), ergast=returns(ergast_frame()),
        fastf1_laps=raises_session_unavailable(),
    )
    assert ErrorCode.TRACK_STATUS_UNAVAILABLE in res.flags


def test_track_status_omitted_fetcher_records_unavailable(request_obj, ingest_config):
    """No laps fetcher supplied -> track status recorded unavailable, not silently OK."""
    res = ingest_session(
        request_obj, ingest_config,
        fastf1_fetch=returns(telemetry_frame()),
        openf1_fetch=returns(openf1_frame()),
        ergast_fetch=returns(ergast_frame()),
        fastf1_laps_fetch=None,
        clock=FakeClock(), sleep=no_sleep, now_epoch=lambda: FIXED_NOW_EPOCH,
    )
    assert res.status is Status.DEGRADED
    assert ErrorCode.TRACK_STATUS_UNAVAILABLE in res.flags
