"""Best-effort startup (§8.1 / acceptance criterion #1).

With no model artefacts and no ``APEX_DATABASE_URL``, ``load_state`` must **degrade every handle to
``None`` without crashing** — the app still boots and ``/health`` reports the outage as 503 rather
than the process dying at startup. The degrade is NOT silent: each failed load writes a
``data_quality_log`` row.

The "no artefacts" precondition is CONSTRUCTED here (``artefacts_dir`` → tmp, DSN unset), not
assumed of the checkout. These tests previously redirected only the log path and passed because
``artefacts/`` happened to be empty; the moment a real bundle was promoted into it they failed,
asserting ``None`` against a genuinely loaded classifier. A test that only passes on an empty
working tree is not testing the degrade path.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from app.api.deps import AppState, load_state
from app.api.main import create_app
from app.contracts.config import load_config

DSN_ENV_VARS = ("APEX_DATABASE_URL", "APEX_REDIS_URL")


def _bare_config(tmp_path: Path, repo_config_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Real repo config with every external dependency pointed somewhere empty."""
    for var in DSN_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    config = load_config(repo_config_path)
    config.paths.data_quality_log = str(tmp_path / "dq.jsonl")
    config.paths.artefacts_dir = str(tmp_path / "artefacts")  # empty → nothing to load
    return config


def test_load_state_degrades_every_handle_to_none(
    tmp_path: Path, repo_config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fresh checkout (no artefacts, no DSN) → all handles/engine None, no exception raised."""
    config = _bare_config(tmp_path, repo_config_path, monkeypatch)

    state = load_state(config)  # must NOT raise

    assert isinstance(state, AppState)
    assert state.model_handle is None
    assert state.explainer is None
    assert state.style_handle is None
    assert state.monitor_handle is None
    assert state.personalisation_handle is None
    assert state.engine is None
    assert state.session_factory is None
    assert state.db_breaker is None


def test_startup_degrade_is_logged_not_silent(
    tmp_path: Path, repo_config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each missing component writes a data_quality_log row — the outage is observable (#1)."""
    config = _bare_config(tmp_path, repo_config_path, monkeypatch)
    dq_log = Path(config.paths.data_quality_log)

    load_state(config)

    assert dq_log.exists()
    lines = [ln for ln in dq_log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert lines, "a missing component must be logged, never silently swallowed"


def test_create_app_boots_without_artefacts(repo_config_path: Path) -> None:
    """create_app() builds the FastAPI app even when no artefacts exist (criterion #1)."""
    config = load_config(repo_config_path)
    app = create_app(config)  # must NOT raise
    assert app.title == "APEX API"
