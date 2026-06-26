"""Fixtures and synthetic source frames for the F01 ingestion tests.

Every external source is a plain callable here — no network, no ``fastf1`` import. The
breaker clock and the cache-freshness clock are driven by a controllable fake so the
window logic and the 6-day/8-day freshness tests never touch wall time.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from app.contracts.config import AppConfig, load_config
from app.pipeline.ingest import IngestSourceError, SessionRequest, SessionUnavailableError

# A fixed "now" (epoch seconds) the freshness tests reckon cache age against.
FIXED_NOW_EPOCH = 1_700_000_000.0
DAY_S = 86_400.0


@pytest.fixture
def request_obj() -> SessionRequest:
    return SessionRequest(season=2024, circuit="Bahrain", driver="VER")


@pytest.fixture
def ingest_config(repo_config_path: Path, tmp_path: Path) -> AppConfig:
    """Real repo config (so breaker values are the shipped ones) with paths in tmp."""
    cfg = load_config(repo_config_path)
    cfg.paths.cache_dir = str(tmp_path / "cache")
    cfg.paths.data_quality_log = str(tmp_path / "logs" / "dq.jsonl")
    return cfg


# --- Synthetic source frames -------------------------------------------------------


def telemetry_frame(laps: tuple[int, ...] = (1, 2, 3)) -> pd.DataFrame:
    """A FastF1-shaped telemetry frame: the 8 channels + a `lap` key, 2 samples/lap."""
    rows = []
    for lap in laps:
        for s in range(2):
            rows.append(
                {
                    "Speed": 250 + s,
                    "Throttle": 80,
                    "Brake": 0,
                    "RPM": 11000,
                    "Gear": 6,
                    "DRS": 0,
                    "X": float(lap * 10 + s),
                    "Y": float(lap),
                    "lap": lap,
                }
            )
    return pd.DataFrame(rows)


def openf1_frame(laps: tuple[int, ...] = (1, 2, 3)) -> pd.DataFrame:
    """OpenF1 per-lap compound + pit flags (pit-out on lap 1 = stint start)."""
    return pd.DataFrame(
        {
            "lap": list(laps),
            "compound": ["SOFT"] * len(laps),
            "pit_in": [False] * len(laps),
            "pit_out": [lap == laps[0] for lap in laps],
        }
    )


def track_status_frame(
    laps: tuple[int, ...] = (1, 2, 3), status: dict[int, str] | None = None
) -> pd.DataFrame:
    """FastF1 Laps.TrackStatus per lap: raw status code (default "1" = all-green)."""
    status = status or {}
    return pd.DataFrame(
        {"lap": list(laps), "track_status": [status.get(lap, "1") for lap in laps]}
    )


def ergast_frame(laps: tuple[int, ...] = (1, 2, 3), *, with_compound: bool = False) -> pd.DataFrame:
    """Ergast per-lap context: lap_time, stint, tyre_age (+ compound when it is the fallback)."""
    data = {
        "lap": list(laps),
        "lap_time": [92.0 + i for i, _ in enumerate(laps)],
        "stint": [1] * len(laps),
        "tyre_age": [i for i, _ in enumerate(laps)],
    }
    if with_compound:
        data["compound"] = ["MEDIUM"] * len(laps)
    return pd.DataFrame(data)


# --- Controllable clock + recording helpers ----------------------------------------


class FakeClock:
    """A monotonic clock under test control (seconds)."""

    def __init__(self, start: float = 0.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def no_sleep(_seconds: float) -> None:
    """Injected backoff sleeper that consumes no wall time (and no window time)."""


class Recorder:
    """Wraps a source callable, counting invocations and optionally raising/returning."""

    def __init__(self, behaviour) -> None:
        self.calls = 0
        self._behaviour = behaviour

    def __call__(self, request: SessionRequest) -> pd.DataFrame:
        self.calls += 1
        return self._behaviour(request)


def returns(frame: pd.DataFrame) -> Recorder:
    return Recorder(lambda _req: frame)


def raises_source_down() -> Recorder:
    def _b(_req: SessionRequest):
        raise IngestSourceError("source down")

    return Recorder(_b)


def raises_session_unavailable() -> Recorder:
    def _b(_req: SessionRequest):
        raise SessionUnavailableError("no such session")

    return Recorder(_b)


def returns_empty() -> Recorder:
    return Recorder(lambda _req: telemetry_frame(laps=()))


# --- F02 validation: synthetic wide session frames ---------------------------------
#
# F02 consumes F01's wide cache frame (one row per sample, per-lap context joined on
# `lap`). These builders produce that shape with one knob per filter so each test
# triggers exactly the condition it asserts.

# Nominal per-sample channel values for a clean, moving, green lap.
_CLEAN_SAMPLE = {
    "Speed": 250.0,
    "Throttle": 80.0,
    "Brake": 0.0,
    "RPM": 11000.0,
    "Gear": 6.0,
    "DRS": 0,
}


def lap_frame(
    lap: int = 1,
    *,
    n_samples: int = 40,
    lap_time: float = 2.0,
    track_status: object = "1",
    pit_in: bool = False,
    pit_out: bool = False,
    tyre_age: object = 5,
    stationary: int = 0,
    nan_gap: tuple[str, int] | None = None,
    all_nan: bool = False,
) -> pd.DataFrame:
    """One lap of F01-shaped wide telemetry, with knobs for the F02 filters.

    ``n_samples`` / ``lap_time`` set the effective rate (samples / lap_time — the
    LAP_004 proxy). ``stationary`` forces that many leading Speed samples to 0 km/h
    (LAP_005). ``nan_gap=(channel, length)`` punches an interior consecutive-NaN run
    (LAP_003 / LAP_INTERP). ``all_nan`` blanks every required channel.
    """
    df = pd.DataFrame(
        {
            **{ch: [val] * n_samples for ch, val in _CLEAN_SAMPLE.items()},
            "X": [float(s) for s in range(n_samples)],
            "Y": [float(lap)] * n_samples,
            "lap": [lap] * n_samples,
            "track_status": [track_status] * n_samples,
            "pit_in": [pit_in] * n_samples,
            "pit_out": [pit_out] * n_samples,
            "lap_time": [lap_time] * n_samples,
            "stint": [1] * n_samples,
            "tyre_age": [tyre_age] * n_samples,
        }
    )
    if all_nan:
        for ch in _CLEAN_SAMPLE:
            df[ch] = np.nan
    if stationary:
        df.loc[: stationary - 1, "Speed"] = 0.0
    if nan_gap is not None:
        channel, length = nan_gap
        start = 5  # interior, so a small gap is interpolatable
        df.loc[start : start + length - 1, channel] = np.nan
    return df


def session_frame(n_laps: int = 30, **lap_kwargs: object) -> pd.DataFrame:
    """A session of ``n_laps`` identical clean laps, numbered 1..n_laps."""
    return pd.concat(
        [lap_frame(lap=i, **lap_kwargs) for i in range(1, n_laps + 1)],  # type: ignore[arg-type]
        ignore_index=True,
    )


def read_dq(path: str | Path) -> list[dict]:
    """Parse the JSONL data_quality_log back into rows (empty if never written)."""
    p = Path(path)
    if not p.is_file():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture
def validate_config(repo_config_path: Path, tmp_path: Path) -> AppConfig:
    """Real repo config (shipped validate/threshold values) with the DQ log in tmp."""
    cfg = load_config(repo_config_path)
    cfg.paths.data_quality_log = str(tmp_path / "logs" / "dq.jsonl")
    return cfg


# --- F04 normalisation: config fixture + wide fit-input frame builder ----------------
#
# F04 consumes the assembled feature matrix (one row per accepted lap, the 28 canonical
# keys + a `circuit` and `split` column). `feature_frame` builds that shape with one knob
# per acceptance criterion so each test triggers exactly the condition it asserts, and with
# hand-computable distributions for the six normalised features so (x-mean)/std is checkable.

# The 28 canonical feature IDs (mirrors features.FEATURE_IDS without importing the module).
_FEATURE_IDS = tuple(f"F{n:02d}" for n in range(1, 29))

# The six F04-normalised IDs (mirrors config.NORMALISED_FEATURE_IDS).
_NORMALISED_IDS = ("F09", "F11", "F13", "F14", "F19", "F20")


@pytest.fixture
def normalise_config(repo_config_path: Path, tmp_path: Path) -> AppConfig:
    """Real repo config (shipped normalise/threshold values) with the DQ log + artefacts dir
    redirected under tmp, mirroring `validate_config`."""
    cfg = load_config(repo_config_path)
    cfg.paths.data_quality_log = str(tmp_path / "logs" / "dq.jsonl")
    cfg.paths.artefacts_dir = str(tmp_path / "artefacts")
    return cfg


def feature_frame(
    circuits: dict[str, int],
    *,
    split: str = "train",
    base: float = 100.0,
    step: float = 1.0,
    constant_feature: str | None = None,
    extra: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Wide F04 fit-input frame: one row per lap, columns `circuit`, `split`, `F01..F28`.

    ``circuits`` maps circuit name -> lap count. For each circuit the six normalised features
    are a deterministic arithmetic ramp ``base, base+step, base+2*step, ...`` (so mean/std are
    hand-computable); the other 22 keys are held at a constant (``1.0``) so a test can assert
    they pass through untouched. ``constant_feature`` forces one normalised feature to a single
    value across every lap (zero variance). ``extra`` is concatenated verbatim (e.g. pathological
    held-out rows for the leakage-zero-effect test).
    """
    frames: list[pd.DataFrame] = []
    for circuit, n in circuits.items():
        data: dict[str, object] = {"circuit": [circuit] * n, "split": [split] * n}
        ramp = [base + step * i for i in range(n)]
        for fid in _FEATURE_IDS:
            if fid in _NORMALISED_IDS:
                if constant_feature is not None and fid == constant_feature:
                    data[fid] = [base] * n
                else:
                    data[fid] = list(ramp)
            else:
                data[fid] = [1.0] * n
        frames.append(pd.DataFrame(data))
    if extra is not None:
        frames.append(extra)
    return pd.concat(frames, ignore_index=True)
