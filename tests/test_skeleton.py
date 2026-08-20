"""Structural conformance: the repo skeleton and dev stores match the contracts.

Covers the F00 acceptance criterion that the package layout matches the ARCHITECTURE.md
repo map and that docker-compose.yml pins the constitution's store versions.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

# The app package map from ARCHITECTURE.md.
EXPECTED_PACKAGES = (
    "app",
    "app.contracts",
    "app.pipeline",
    "app.ml",
    "app.monitoring",
    "app.personalisation",
    "app.api",
    "app.cache",
)


@pytest.mark.parametrize("pkg", EXPECTED_PACKAGES)
def test_package_importable(pkg: str) -> None:
    assert importlib.import_module(pkg) is not None


def test_artefacts_dir_exists() -> None:
    assert (REPO_ROOT / "artefacts").is_dir()


def test_config_yaml_exists() -> None:
    assert (REPO_ROOT / "config.yaml").is_file()


def _compose_services() -> dict[str, dict]:
    raw = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    return raw["services"]


def test_docker_compose_pins_postgres_15() -> None:
    assert _compose_services()["postgres"]["image"] == "postgres:15"


def test_docker_compose_pins_redis_7() -> None:
    assert _compose_services()["redis"]["image"] == "redis:7"
