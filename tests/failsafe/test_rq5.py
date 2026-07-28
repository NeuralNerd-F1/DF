"""F15 — the 7 §11.4 RQ5 failsafe tests, one pytest per master test + the suite/report + validators.

The offline 5 (Tests 2,3,4,5,7) run fully; Tests 1 & 6 assert the *mechanism* offline and carry an
``@pytest.mark.artefacts`` companion for the real *rate* (skipped without artefacts → F20).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from app.contracts.config import AppConfig, load_config
from app.eval.failsafe import (
    FailsafeTestResult,
    _errored,
    rq5_cache,
    rq5_circuit_breaker,
    rq5_concurrent,
    rq5_confidence_gate,
    rq5_database,
    rq5_null_tolerance,
    rq5_umap_ood,
    run_failsafe_suite,
)

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config.yaml"


@pytest.fixture
def cfg() -> AppConfig:
    return load_config(REPO_CONFIG)


# --- the 7 §11.4 tests -------------------------------------------------------------------------
async def test_rq5_1_confidence_gate_mechanism(cfg: AppConfig) -> None:
    r = await rq5_confidence_gate(cfg)
    assert r.passed  # a sub-gate prediction surfaces 200 LOW_CONFIDENCE
    assert r.measured["confidence_state"] == "LOW_CONFIDENCE"


async def test_rq5_2_circuit_breaker(cfg: AppConfig) -> None:
    r = await rq5_circuit_breaker(cfg)
    assert r.passed
    assert r.measured["breaker_opened"] and r.measured["stale_data"]
    assert r.measured["status"] == "DEGRADED"
    assert r.measured["elapsed_ms"] < cfg.failsafe.breaker_fallback_ms  # type: ignore[union-attr]


async def test_rq5_3_null_tolerance(cfg: AppConfig) -> None:
    r = await rq5_null_tolerance(cfg)
    assert r.passed  # 6 accepted+IMPUTED, 7 rejected LAP_NULL_EXCESS
    assert r.measured["7"] == "FAILED"


async def test_rq5_4_cache_unavailability(cfg: AppConfig) -> None:
    r = await rq5_cache(cfg)
    assert r.passed
    assert r.measured["failed"] == 0  # zero failed predictions with Redis down
    assert r.measured["cache_unavailable_logged"] is True


async def test_rq5_5_database_unavailability(cfg: AppConfig) -> None:
    r = await rq5_database(cfg)
    assert r.passed
    assert r.measured["status"] == 503 and r.measured["error_code"] == "DB_UNAVAILABLE"
    assert r.measured["rows_after_failed_write"] == 0  # no partial write


async def test_rq5_6_umap_ood_mechanism(cfg: AppConfig) -> None:
    r = await rq5_umap_ood(cfg)
    assert r.passed  # OUT_OF_DISTRIBUTION flagged, coords present, no crash
    assert r.measured["ood_flagged"] and r.measured["coords_present"]


async def test_rq5_7_concurrent(cfg: AppConfig) -> None:
    r = await rq5_concurrent(cfg)
    assert r.passed  # all complete + consistent (in-process; production p95 is F20)
    assert r.measured["completed"] == cfg.failsafe.n_concurrent  # type: ignore[union-attr]
    assert r.measured["consistent"] is True
    assert "p95_ms" in r.measured  # reported


# --- artefact-gated real-rate companions (Tests 1 & 6) — skipped without committed artefacts ---
# F20 now OWNS the real rate assertions: app/eval/evaluate.py::_test1_low_confidence_rate /
# _test6_ood_5sd_detection compute them, and tests/eval/test_evaluation.py::test_real_full_run
# drives them under the same @pytest.mark.artefacts gate. These two skip cleanly (they no longer
# raise) and point there — so a future artefacts-present run asserts in one place, not two.
def _artefact(cfg: AppConfig, filename: str) -> Path:
    return Path(cfg.paths.artefacts_dir) / filename


@pytest.mark.artefacts
async def test_rq5_1_real_gate_rate(cfg: AppConfig) -> None:
    art = _artefact(cfg, cfg.classifier.model_artefact_filename) if cfg.classifier else None
    if art is None or not art.is_file():
        pytest.skip("champion artefact absent — the 90% gate rate is evaluated in F20")
    pytest.skip(  # pragma: no cover - real rate is asserted by F20's test_real_full_run
        "real rate asserted by tests/eval/test_evaluation.py::test_real_full_run (F20 owns it)"
    )


@pytest.mark.artefacts
async def test_rq5_6_real_ood_detection(cfg: AppConfig) -> None:
    art = _artefact(cfg, cfg.stylespace.umap_model_filename) if cfg.stylespace else None
    if art is None or not art.is_file():
        pytest.skip("umap artefact absent — real 5-SD detection is evaluated in F20")
    pytest.skip(  # pragma: no cover - real detection is asserted by F20's test_real_full_run
        "real 5-SD detection asserted by tests/eval/test_evaluation.py::test_real_full_run (F20)"
    )


# --- suite + report ----------------------------------------------------------------------------
async def test_suite_runs_all_seven_and_writes_report(cfg: AppConfig, tmp_path: Path) -> None:
    report = await run_failsafe_suite(cfg, artefact_dir=tmp_path)
    assert report.total == 7
    assert report.passed == 7  # all 7 mechanisms pass offline
    fname = cfg.failsafe.report_filename  # type: ignore[union-attr]
    assert (tmp_path / fname).is_file()
    table = report.render_table()
    assert table.count("\n| ") >= 7  # 7 result rows


async def test_report_outcome_hash_is_stable(cfg: AppConfig, tmp_path: Path) -> None:
    r1 = await run_failsafe_suite(cfg, artefact_dir=tmp_path)
    r2 = await run_failsafe_suite(cfg, artefact_dir=tmp_path)
    assert r1.content_hash == r2.content_hash and r1.content_hash  # deterministic outcome hash


async def test_harness_never_pollutes_the_configured_cache_or_dq_log(
    cfg: AppConfig, tmp_path: Path
) -> None:
    """The harness must redirect internally — it must NOT write to config's cache_dir / dq log."""
    cfg.paths.cache_dir = str(tmp_path / "real_cache")
    cfg.paths.data_quality_log = str(tmp_path / "real_dq.jsonl")
    await run_failsafe_suite(cfg, artefact_dir=tmp_path / "art")
    assert not (tmp_path / "real_cache").exists()  # Test 2's fixture Parquet went to a temp dir
    assert not (tmp_path / "real_dq.jsonl").exists()  # Test 3/4 logging went to a temp dir


async def test_suite_survives_a_test_that_raises(
    cfg: AppConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A test raising even in its pre-try setup must not abort the suite (§4.0-h)."""
    import app.eval.failsafe as fs

    async def boom(config: AppConfig) -> FailsafeTestResult:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(fs, "rq5_confidence_gate", boom)
    report = await fs.run_failsafe_suite(cfg, artefact_dir=tmp_path)  # must not raise
    assert report.total == 7
    assert any(not r.passed and "kaboom" in " ".join(r.findings) for r in report.results)


# --- harness never fails silently (§4.0-h) -----------------------------------------------------
def test_errored_helper_marks_failed_with_finding() -> None:
    r = _errored(9, "demo", "expected", RuntimeError("boom"))
    assert r.passed is False and not r.skipped
    assert "boom" in r.findings[0]


async def test_every_test_returns_a_result_never_raises(cfg: AppConfig) -> None:
    for fn in (
        rq5_confidence_gate, rq5_circuit_breaker, rq5_null_tolerance, rq5_cache,
        rq5_database, rq5_umap_ood, rq5_concurrent,
    ):
        assert isinstance(await fn(cfg), FailsafeTestResult)


# --- config validators -------------------------------------------------------------------------
def _raw() -> dict:
    return yaml.safe_load(REPO_CONFIG.read_text(encoding="utf-8"))


def test_null_boundary_must_span_the_ceiling() -> None:
    raw = _raw()
    raw["failsafe"]["null_boundary"] = [1, 2, 3]  # missing 6 and 7
    with pytest.raises(Exception, match="null_boundary"):
        AppConfig(**raw)


def test_ood_injection_must_exceed_detection() -> None:
    raw = _raw()
    raw["failsafe"]["ood_injection_sd"] = 2.0  # <= stylespace.ood_sd_threshold (3.0)
    with pytest.raises(Exception, match="ood_injection_sd"):
        AppConfig(**raw)


def test_bad_rate_rejected() -> None:
    raw = _raw()
    raw["failsafe"]["ambiguous_low_conf_rate"] = 1.5
    with pytest.raises(Exception, match="ambiguous_low_conf_rate"):
        AppConfig(**raw)
