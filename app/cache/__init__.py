"""Redis cache layer (an optimisation, never a hard dependency)."""

from app.cache.redis import CacheResult, RedisCache, RedisClientProtocol, build_cache

__all__ = ["CacheResult", "RedisCache", "RedisClientProtocol", "build_cache"]
