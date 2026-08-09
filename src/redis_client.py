"""
Redis client singleton with graceful fallback.

Provides a single `get_redis()` function that returns a live Redis client
when REDIS_URL is configured, or None when it is not — so every caller can
do a simple `r = get_redis(); if r is None: <fallback>` without crashing.

Connection is created once and reused for the lifetime of the process.
"""

from typing import Optional
from loguru import logger

import redis

_redis_client: Optional[redis.Redis] = None
_redis_available: bool = False


def get_redis() -> Optional[redis.Redis]:
    """
    Return a live Redis client, or None if Redis is not configured / reachable.

    Thread-safe via module-level sentinel; first call initialises the client.
    Subsequent calls return the same singleton.
    """
    global _redis_client, _redis_available

    if _redis_client is not None:
        return _redis_client if _redis_available else None

    # Lazy import so config is already loaded before this module is imported
    from src.config import settings

    url = settings.redis_url
    if not url:
        logger.info("REDIS_URL not configured — Redis features will fall back to Supabase/in-memory.")
        _redis_available = False
        return None

    try:
        client = redis.Redis.from_url(
            url,
            decode_responses=True,  # always get str, not bytes
            socket_connect_timeout=3,
            socket_timeout=3,
            retry_on_timeout=True,
        )
        client.ping()  # verify connectivity at startup
        _redis_client = client
        _redis_available = True
        logger.info("Redis connected successfully.")
        return _redis_client
    except Exception as e:
        logger.warning(f"Redis connection failed ({e}) — falling back to Supabase/in-memory.")
        _redis_available = False
        return None
