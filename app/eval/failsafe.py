"""The RQ5 failsafe evaluation harness — the 7 §11.4 tests, executed with real fault injection.

Each ``rq5_*`` assembles the real system (or drives the F01 pipeline for Test 2), injects the
fault, measures the behaviour, and returns a :class:`FailsafeTestResult` — **never raising past
itself** (a harness-internal fault → ``passed=False`` + a finding, §4.0-h). ``run_failsafe_suite``
runs all 7, writes the content-hashed **RQ5 results table** (`failsafe_report.json`), and returns
the report. F15 is §11.4 only; F20 calls ``run_failsafe_suite`` for that pillar (divergence #39).

The two model-*rate* assertions (Test 1's 90% gate rate, Test 6's 5-SD detection) need committed
artefacts; F15 asserts the *mechanism* offline (a sub-gate prediction surfaces LOW_CONFIDENCE; a
flagged projection surfaces OUT_OF_DISTRIBUTION) and marks the rate as artefact-gated → F20 (#41).
"""

from __future__ import annotations

import hashlib
import json
import statistics
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from app.contracts.config import AppConfig


# --- typed results -----------------------------------------------------------------------------
class FailsafeTestResult(BaseModel):
    """One §11.4 test outcome. ``expected`` is the master's verbatim criterion; ``measured`` the
    observed numbers; ``findings`` any honest caveat (§11 'documented as findings')."""

    test_id: int
    name: str
    passed: bool
    skipped: bool = False
    skip_reason: str | None = None
    measured: dict[str, Any] = Field(default_factory=dict)
    expected: str = ""
    findings: list[str] = Field(default_factory=list)


class FailsafeReport(BaseModel):
    """The RQ5 evaluation results table (§13 Week-14 deliverable). Timestamp-free; ``content_hash``
    covers the reproducible OUTCOME (pass/fail per test), not the run-specific ``measured`` timing —
    so same inputs → same outcomes → same hash (reproducibility #3)."""

    results: list[FailsafeTestResult] = Field(default_factory=list)
    total: int = 0
    passed: int = 0
    skipped: int = 0
    content_hash: str = ""

    def render_table(self) -> str:
        """The markdown RQ5 results table the dissertation embeds — incl. the Findings column that
        carries the honesty caveats (mechanism-vs-rate, in-process-vs-production; §4.3)."""
        lines = [
            "| # | Test | Result | Measured | Expected | Findings |",
            "|---|------|--------|----------|----------|----------|",
        ]
        for r in self.results:
            result = "SKIP" if r.skipped else ("PASS" if r.passed else "FAIL")
            measured = "; ".join(f"{k}={v}" for k, v in r.measured.items())
            findings = "; ".join(r.findings) or "—"
            lines.append(
                f"| {r.test_id} | {r.name} | {result} | {measured} | {r.expected} | {findings} |"
            )
        lines.append(f"\n**{self.passed}/{self.total} passed** ({self.skipped} skipped).")
        return "\n".join(lines)


# --- helpers -----------------------------------------------------------------------------------
def _harness_config(config: AppConfig) -> AppConfig:
    """A deep copy of the config with the DQ log AND the telemetry cache dir redirected to a fresh
    temp dir, so a harness run never pollutes the real ``data_quality_log`` or ``cache/`` (a shipped
    harness must not clobber a user's runtime dirs). Each test reads only its own rows."""
    cfg = config.model_copy(deep=True)
    tmp = Path(tempfile.mkdtemp(prefix="apex_failsafe_"))
    cfg.paths.data_quality_log = str(tmp / "dq.jsonl")
    cfg.paths.cache_dir = str(tmp / "cache")  # Test 2's fixture Parquet lands here, not real cache/
    return cfg


def _read_dq(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line]


def _errored(test_id: int, name: str, expected: str, exc: BaseException) -> FailsafeTestResult:
    """A harness-internal fault → a FAILED result with a finding (never a raise; §4.0-h)."""
    return FailsafeTestResult(
        test_id=test_id, name=name, passed=False, expected=expected,
        findings=[f"harness error: {type(exc).__name__}: {exc}"],
    )


def _telemetry_frame() -> Any:
    """A minimal FastF1-shaped cache frame for Test 2 (8 channels + lap + compound + tyre_age)."""
    import pandas as pd

    rows = [
        {"Speed": 250 + s, "Throttle": 80, "Brake": 0, "RPM": 11000, "Gear": 6, "DRS": 0,
         "X": float(lap * 10 + s), "Y": float(lap), "lap": lap, "compound": "SOFT", "tyre_age": 0}
        for lap in (1, 2)
        for s in range(2)
    ]
    return pd.DataFrame(rows)


# --- the 7 §11.4 tests -------------------------------------------------------------------------
async def rq5_confidence_gate(config: AppConfig) -> FailsafeTestResult:
    """Test 1 — a sub-gate prediction on ambiguous input surfaces 200 LOW_CONFIDENCE (mechanism)."""
    from app.contracts.enums import ConfidenceState
    from app.eval.harness import assemble, fake_prediction

    expected = "LOW_CONFIDENCE for 90% of ambiguous inputs rather than a wrong confident prediction"
    try:
        cfg = _harness_config(config)
        pred = fake_prediction(top="VER", confidence=0.40, state=ConfidenceState.LOW_CONFIDENCE)
        async with assemble(cfg, predictor=lambda fv, c: pred) as h:
            r = await h.client.post(
                "/predict", json={"feature_vector": {"F01": 0.5}, "circuit": "Bahrain"}
            )
        body = r.json()
        passed = r.status_code == 200 and body.get("confidence_state") == "LOW_CONFIDENCE"
        return FailsafeTestResult(
            test_id=1, name="Confidence gate effectiveness", passed=passed, expected=expected,
            measured={"status": r.status_code, "confidence_state": body.get("confidence_state")},
            findings=[
                "route passthrough of confidence_state (the real 0.65 gate is unit-tested in "
                "tests/ml); the 90% rate on real ambiguous input needs the champion → F20",
            ],
        )
    except Exception as exc:  # never raise past the harness (§4.0-h)
        return _errored(1, "Confidence gate effectiveness", expected, exc)


async def rq5_circuit_breaker(config: AppConfig) -> FailsafeTestResult:
    """Test 2 — FastF1 breaker opens after 3 failures; cached data served with STALE_DATA <500ms."""
    import os
    import time as _time

    from app.contracts.enums import Status
    from app.contracts.errors import ErrorCode
    from app.pipeline.ingest import (
        IngestSourceError,
        SourceUnavailableError,
        build_breakers,
        cache_meta_path,
        cache_path_for,
        call_with_breaker,
        ingest_session,
    )

    expected = (
        "No uncaught exception. System returns cached data with STALE_DATA flag within 500ms."
    )
    try:
        cfg = _harness_config(config)
        assert cfg.failsafe is not None
        from app.pipeline.ingest import SessionRequest

        req = SessionRequest(season=2024, circuit="Bahrain", driver="VER")
        cpath = cache_path_for(req, Path(cfg.paths.cache_dir))
        cpath.parent.mkdir(parents=True, exist_ok=True)
        _telemetry_frame().to_parquet(cpath, index=False)
        cache_meta_path(cpath).write_text(
            json.dumps({"compound_source": "OPENF1", "tyre_age_source": "ERGAST", "flags": []}),
            encoding="utf-8",
        )
        os.utime(cpath, (1000.0, 1000.0))  # far in the past → stale vs the future now_epoch below

        def down(_r: SessionRequest) -> Any:
            raise IngestSourceError("FastF1 down")

        def empty(_r: SessionRequest) -> Any:
            import pandas as pd

            return pd.DataFrame()

        breakers = build_breakers(cfg, sleep=lambda _s: None)
        for _ in range(3):  # 3 consecutive failures → the circuit opens (Test 2 method)
            try:
                call_with_breaker(breakers["fastf1"], lambda: down(req))
            except SourceUnavailableError:
                pass
        opened = breakers["fastf1"].is_open()

        t0 = _time.perf_counter()
        res = ingest_session(
            req, cfg, fastf1_fetch=down, openf1_fetch=empty, ergast_fetch=empty,
            sleep=lambda _s: None, now_epoch=lambda: 2_000_000_000.0, breakers=breakers,
        )
        elapsed_ms = (_time.perf_counter() - t0) * 1000
        passed = (
            opened
            and ErrorCode.STALE_DATA in res.flags
            and res.status is Status.DEGRADED
            and elapsed_ms < cfg.failsafe.breaker_fallback_ms
        )
        return FailsafeTestResult(
            test_id=2, name="Circuit breaker activation", passed=passed, expected=expected,
            measured={
                "breaker_opened": opened, "stale_data": ErrorCode.STALE_DATA in res.flags,
                "status": res.status.value, "elapsed_ms": round(elapsed_ms, 2),
            },
        )
    except Exception as exc:
        return _errored(2, "Circuit breaker activation", expected, exc)


async def rq5_null_tolerance(config: AppConfig) -> FailsafeTestResult:
    """Test 3 — 6 NULLs accepted+IMPUTED, 7 rejected LAP_NULL_EXCESS (no silent imputation)."""
    from app.contracts.enums import Status
    from app.contracts.errors import ErrorCode
    from app.pipeline.features import FEATURE_IDS, FeatureMedians, assemble_feature_vector

    expected = (
        "6 NULLs accepted with imputation. 7+ NULLs rejected with LAP_NULL_EXCESS. "
        "No silent imputation."
    )
    try:
        cfg = _harness_config(config)  # redirected DQ log — never pollute the real one
        assert cfg.failsafe is not None
        medians = FeatureMedians(global_medians=dict.fromkeys(FEATURE_IDS, 0.5))
        ceiling = cfg.thresholds.null_ceiling
        outcomes: dict[int, Any] = {}
        for n in cfg.failsafe.null_boundary:
            values: dict[str, float | None] = dict.fromkeys(FEATURE_IDS, 1.0)
            for fid in list(FEATURE_IDS)[:n]:
                values[fid] = None
            outcomes[n] = assemble_feature_vector(
                values, lap_number=1, driver="VER", circuit_type="high_speed",
                medians=medians, config=cfg,
            )
        at6 = outcomes[ceiling]
        at7 = outcomes[ceiling + 1]
        # Accepted-with-imputation is status DEGRADED (imputation IS a degradation), not OK.
        accept6 = at6.status is not Status.FAILED and ErrorCode.IMPUTED in at6.flags
        reject7 = (
            at7.status is Status.FAILED
            and at7.error_code is ErrorCode.LAP_NULL_EXCESS
            and ErrorCode.IMPUTED not in at7.flags  # no silent imputation
        )
        return FailsafeTestResult(
            test_id=3, name="NULL feature tolerance", passed=accept6 and reject7, expected=expected,
            measured={str(n): outcomes[n].status.value for n in outcomes},
        )
    except Exception as exc:
        return _errored(3, "NULL feature tolerance", expected, exc)


async def rq5_cache(config: AppConfig) -> FailsafeTestResult:
    """Test 4 — Redis down: 100 predictions all 200, degradation <200ms, CACHE_UNAVAILABLE seen."""
    from app.cache.redis import RedisCache
    from app.contracts.errors import ErrorCode
    from app.eval.harness import assemble, fake_prediction, measure_post

    expected = (
        "Zero failed predictions. Response time increases by < 200ms average. "
        "Log shows CACHE_UNAVAILABLE throughout."
    )
    try:
        cfg = _harness_config(config)
        assert cfg.failsafe is not None and cfg.cache is not None and cfg.api is not None
        pred = fake_prediction()
        body = {"feature_vector": {"F01": 0.5, "F02": 0.3}, "circuit": "Bahrain"}
        n = cfg.failsafe.n_predictions
        async with assemble(cfg, cache=None, predictor=lambda fv, c: pred) as h:
            base_times, _ = await measure_post(h.client, "/predict", body, n)
        down = RedisCache(
            _DownClient(), config=cfg.cache, api_version=cfg.api.version,
            log_path=cfg.paths.data_quality_log,
        )
        async with assemble(cfg, cache=down, predictor=lambda fv, c: pred) as h:
            down_times, down_status = await measure_post(h.client, "/predict", body, n)
        zero_failed = all(s == 200 for s in down_status)
        degradation_ms = (statistics.mean(down_times) - statistics.mean(base_times)) * 1000
        logged = any(
            r.get("error_code") == ErrorCode.CACHE_UNAVAILABLE.value
            for r in _read_dq(cfg.paths.data_quality_log)
        )
        passed = zero_failed and logged and degradation_ms < cfg.failsafe.cache_degradation_ms
        return FailsafeTestResult(
            test_id=4, name="Cache unavailability resilience", passed=passed, expected=expected,
            measured={
                "failed": sum(1 for s in down_status if s != 200),
                "avg_degradation_ms": round(degradation_ms, 2), "cache_unavailable_logged": logged,
            },
            findings=["in-process latency (fakes) — mechanism; production degradation is F20"],
        )
    except Exception as exc:
        return _errored(4, "Cache unavailability resilience", expected, exc)


async def rq5_database(config: AppConfig) -> FailsafeTestResult:
    """Test 5 — DB breaker open: DB-backed read → 503 DB_UNAVAILABLE, no partial write, no crash."""
    from sqlalchemy import func, select

    from app.db import models
    from app.db.results import DatabaseUnavailableError
    from app.db.session import session_scope
    from app.eval.harness import assemble

    expected = (
        "API returns 503 with DB_UNAVAILABLE reason. No partial writes. No unhandled exceptions."
    )
    try:
        cfg = _harness_config(config)
        async with assemble(cfg) as h:
            for _ in range(h.db_breaker.config.open_after):  # open the Supabase-policy breaker
                h.db_breaker.record_failure()
            r = await h.client.get("/drivers")
            http_503 = r.status_code == 503 and r.json().get("error_code") == "DB_UNAVAILABLE"
            # No partial write: with the breaker open, session_scope fast-fails BEFORE yielding a
            # session, so the write is never even attempted → rows stay 0. (The mid-transaction
            # rollback path itself is exercised by F12's tests, not re-proven here.)
            try:
                async with session_scope(h.session_factory, h.db_breaker) as s:
                    s.add(models.Circuit(name="X", type=models.CircuitType.HIGH_SPEED))
            except DatabaseUnavailableError:
                pass
            h.db_breaker.record_success()  # reopen for the verification read
            async with h.session_factory() as s:
                rows = (
                    await s.execute(select(func.count()).select_from(models.Circuit))
                ).scalar_one()
        passed = http_503 and rows == 0
        return FailsafeTestResult(
            test_id=5, name="Database unavailability", passed=passed, expected=expected,
            measured={"status": r.status_code, "error_code": r.json().get("error_code"),
                      "rows_after_failed_write": rows},
        )
    except Exception as exc:
        return _errored(5, "Database unavailability", expected, exc)


async def rq5_umap_ood(config: AppConfig) -> FailsafeTestResult:
    """Test 6 — a flagged projection surfaces OUT_OF_DISTRIBUTION with coords present, no crash."""
    from app.eval.harness import assemble, fake_projection

    expected = (
        "Warning returned. Projection computed but flagged. UI shows dashed marker. No crash."
    )
    try:
        cfg = _harness_config(config)
        proj = fake_projection(ood=True)
        n_laps = (cfg.api.min_projection_laps if cfg.api else 5) + 1
        laps = [{"F01": 0.5} for _ in range(n_laps)]
        async with assemble(cfg, projector=lambda lp, c, ct: proj) as h:
            r = await h.client.post(
                "/fingerprint/unknown",
                json={"laps": laps, "circuit": "Bahrain", "circuit_type": "high_speed"},
            )
        body = r.json()
        ood_flagged = "OUT_OF_DISTRIBUTION" in body.get("warnings", [])
        coords_present = body.get("fingerprint_vector") is not None
        passed = r.status_code == 200 and ood_flagged and coords_present
        return FailsafeTestResult(
            test_id=6, name="UMAP out-of-distribution", passed=passed, expected=expected,
            measured={"status": r.status_code, "ood_flagged": ood_flagged,
                      "coords_present": coords_present},
            findings=["mechanism only; 5-SD detection needs the umap artefact (→ F20, @artefacts)"],
        )
    except Exception as exc:
        return _errored(6, "UMAP out-of-distribution", expected, exc)


async def rq5_concurrent(config: AppConfig) -> FailsafeTestResult:
    """Test 7 — 50 concurrent /predict all complete 200, identical bodies, p95 ≤ latency_slo_ms."""
    import asyncio
    import time as _time

    from app.eval.harness import assemble, fake_prediction, p95

    expected = (
        "All 50 requests complete. Results consistent (same input = same output). "
        "p95 latency < 500ms."
    )
    try:
        cfg = _harness_config(config)
        assert cfg.failsafe is not None
        pred = fake_prediction()
        body = {"feature_vector": {"F01": 0.5}, "circuit": "Bahrain"}
        n = cfg.failsafe.n_concurrent
        slo_ms = cfg.api.latency_slo_ms if cfg.api else 500.0
        async with assemble(cfg, predictor=lambda fv, c: pred) as h:

            async def one() -> tuple[float, int, dict[str, Any]]:
                t0 = _time.perf_counter()
                resp = await h.client.post("/predict", json=body)
                return _time.perf_counter() - t0, resp.status_code, resp.json()

            outcomes = await asyncio.gather(*(one() for _ in range(n)))
        times = [t for t, _, _ in outcomes]
        statuses = [s for _, s, _ in outcomes]
        bodies = [b for _, _, b in outcomes]
        all_200 = all(s == 200 for s in statuses)
        consistent = all(b == bodies[0] for b in bodies)
        p95_ms = p95(times) * 1000
        # Completion + consistency are the real in-process guarantees (§4.0-c). In-process p95
        # measures event-loop SERIALISATION, not production latency, so it is REPORTED against the
        # SLO, not gated on — real p95 under a multi-worker deploy is F20 (divergence #40).
        passed = all_200 and consistent
        within_slo = p95_ms < slo_ms
        return FailsafeTestResult(
            test_id=7, name="Concurrent request handling", passed=passed, expected=expected,
            measured={"completed": len(outcomes), "consistent": consistent,
                      "p95_ms": round(p95_ms, 2), "in_process_within_slo": within_slo},
            findings=[
                f"in-process p95={round(p95_ms, 2)}ms vs SLO {slo_ms}ms — measures event-loop "
                "serialisation not production latency; real p95 (multi-worker) is F20 (§4.0-c)",
            ],
        )
    except Exception as exc:
        return _errored(7, "Concurrent request handling", expected, exc)


class _DownClient:
    """A Redis client that fails every op (Test 4 — Redis unreachable)."""

    async def get(self, key: str) -> Any:
        raise ConnectionError("redis down")

    async def set(self, key: str, value: str, *, ex: int | None = None) -> Any:
        raise ConnectionError("redis down")

    async def delete(self, *keys: str) -> Any:
        raise ConnectionError("redis down")

    async def ping(self) -> Any:
        raise ConnectionError("redis down")

    async def aclose(self) -> None:
        return None


# --- suite + report ----------------------------------------------------------------------------
def _canonical(report: FailsafeReport) -> str:
    """Canonical body for the content hash: the reproducible **outcome** (pass/fail/skip + criterion
    per test) — NOT the run-specific ``measured`` timing (elapsed/p95 vary per run). So the hash
    certifies "same inputs → same outcomes", the meaningful reproducibility for an evidence run."""
    outcomes = [
        {"test_id": r.test_id, "name": r.name, "passed": r.passed, "skipped": r.skipped,
         "expected": r.expected}
        for r in report.results
    ]
    body = {
        "results": outcomes, "total": report.total, "passed": report.passed,
        "skipped": report.skipped,
    }
    return json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)


async def run_failsafe_suite(
    config: AppConfig, *, artefact_dir: str | Path | None = None
) -> FailsafeReport:
    """Run all 7 §11.4 tests, write the content-hashed RQ5 results table, and return the report.

    Never raises past a single test — each returns a typed result AND the suite wraps every call in
    its own try/except (defense-in-depth), so one test's fault (even in its pre-``try`` setup) never
    aborts the suite (§4.0-h). F20 calls this for the §11.4 pillar."""
    tests: list[tuple[Any, int, str]] = [
        (rq5_confidence_gate, 1, "Confidence gate effectiveness"),
        (rq5_circuit_breaker, 2, "Circuit breaker activation"),
        (rq5_null_tolerance, 3, "NULL feature tolerance"),
        (rq5_cache, 4, "Cache unavailability resilience"),
        (rq5_database, 5, "Database unavailability"),
        (rq5_umap_ood, 6, "UMAP out-of-distribution"),
        (rq5_concurrent, 7, "Concurrent request handling"),
    ]
    results: list[FailsafeTestResult] = []
    for fn, tid, name in tests:
        try:
            results.append(await fn(config))
        except Exception as exc:  # a test must never abort the suite (§4.0-h)
            results.append(_errored(tid, name, "", exc))
    report = FailsafeReport(
        results=results, total=len(results),
        passed=sum(1 for r in results if r.passed and not r.skipped),
        skipped=sum(1 for r in results if r.skipped),
    )
    report.content_hash = hashlib.sha256(_canonical(report).encode("utf-8")).hexdigest()
    _write_report(report, config, artefact_dir)
    return report


def _write_report(
    report: FailsafeReport, config: AppConfig, artefact_dir: str | Path | None
) -> Path:
    """Write the RQ5 results table to ``artefacts/failsafe_report.json`` (F04/F10 idiom: sort_keys,
    indent=2, allow_nan=False, embedded content_hash, no timestamp). The ``content_hash`` certifies
    the OUTCOME (same inputs → same pass/fail); the ``measured`` timing in the file is per-run."""
    adir = Path(artefact_dir) if artefact_dir is not None else Path(config.paths.artefacts_dir)
    fname = config.failsafe.report_filename if config.failsafe else "failsafe_report.json"
    path = adir / fname
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report.model_dump(), sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8")
    return path
