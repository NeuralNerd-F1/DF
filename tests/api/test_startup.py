"""Best-effort startup (§8.1 / acceptance criterion #1).

No model artefacts are committed and no ``APEX_DATABASE_URL`` is set in the test env, so
``load_state`` against the real repo config must **degrade every handle to ``None`` without
crashing** — the app still boots and ``/health`` reports the outage as 503 rather than the process
dying at startup. The degrade is NOT silent: each failed load writes a ``data_quality_log`` row.
"""

from __future__ import annotations

from pathlib import Path

from app.api.deps import AppState, load_state
from app.api.main import create_app
from app.contracts.config import load_config


def test_load_state_degrades_every_handle_to_none(tmp_path: Path, repo_config_path: Path) -> None:
    """Fresh checkout (no artefacts, no DSN) → all handles/engine None, no exception raised."""
    config = load_config(repo_config_path)
    dq_log = tmp_path / "dq.jsonl"
    config.paths.data_quality_log = str(dq_log)

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


def test_startup_degrade_is_logged_not_silent(tmp_path: Path, repo_config_path: Path) -> None:
    """Each missing component writes a data_quality_log row — the outage is observable (#1)."""
    config = load_config(repo_config_path)
    dq_log = tmp_path / "dq.jsonl"
    config.paths.data_quality_log = str(dq_log)

    load_state(config)

    assert dq_log.exists()
    lines = [ln for ln in dq_log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert lines, "a missing component must be logged, never silently swallowed"


def test_create_app_boots_without_artefacts(repo_config_path: Path) -> None:
    """create_app() builds the FastAPI app even when no artefacts exist (criterion #1)."""
    config = load_config(repo_config_path)
    app = create_app(config)  # must NOT raise
    assert app.title == "APEX API"
