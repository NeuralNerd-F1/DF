"""F14 ``RedisJobStore`` — bounded (never hangs) + logs faults (never silently swallows). §1/HIGH-1.

Uses fakeredis for the happy path and a hanging/failing client for the failsafe paths — no server.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import fakeredis.aioredis
from app.contracts.errors import ErrorCode
from app.tasks.ingest import JobState, RedisJobStore


async def test_roundtrip_on_fakeredis() -> None:
    store = RedisJobStore(fakeredis.aioredis.FakeRedis(), key_prefix="apex", ttl_s=60)
    await store.create("j1")
    await store.update("j1", status=JobState.COMPLETE, progress=1.0, result={"laps_accepted": 9})
    got = await store.get("j1")
    assert got is not None and got.status is JobState.COMPLETE
    assert got.result == {"laps_accepted": 9}


async def test_unknown_id_is_none() -> None:
    store = RedisJobStore(fakeredis.aioredis.FakeRedis(), key_prefix="apex", ttl_s=60)
    assert await store.get("nope") is None


class _HangingClient:
    """Every op sleeps far past the store timeout — must be cut off, not hang the caller."""

    async def set(self, key: str, value: str, *, ex: int | None = None) -> None:
        await asyncio.sleep(30)

    async def get(self, key: str) -> object:
        await asyncio.sleep(30)


async def test_hung_backend_does_not_hang_and_is_logged(tmp_path: Path) -> None:
    log = tmp_path / "dq.jsonl"
    store = RedisJobStore(
        _HangingClient(), key_prefix="apex", ttl_s=60, timeout_s=0.05, log_path=str(log)
    )
    # Both bounded by timeout_s → return quickly, never hang.
    await asyncio.wait_for(store.update("j2", status=JobState.PENDING, progress=0.0), timeout=2.0)
    got = await asyncio.wait_for(store.get("j2"), timeout=2.0)
    assert got is None  # read fault degrades to "unknown"
    rows = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if x]
    codes = {r["error_code"] for r in rows}
    assert ErrorCode.CACHE_UNAVAILABLE.value in codes  # outage logged, NOT swallowed silently
