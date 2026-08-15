"""
Stateful refresh-token store — 100% Redis based.

Strategy
--------
* LOGIN  (save_refresh_token):
    Write to Redis. Redis TTL = REFRESH_TOKEN_EXPIRE_DAYS.

* REFRESH (get_refresh_token):
    Read from Redis (<5 ms).

* ROTATE & REVOKE:
    Updates Redis in microseconds (<3ms).

Redis key schema
----------------
    rt:{token_hash}          → Hash  {user_id, email, expires_at, revoked}   TTL=30d
    user_rts:{user_id}       → Set   of token_hash strings                   TTL=30d
"""

from loguru import logger
from datetime import datetime, timedelta, timezone
from typing import Optional

from src.auth.constants import (
    REFRESH_TOKEN_EXPIRE_DAYS,
    REFRESH_TOKEN_KEY_PREFIX,
    USER_REFRESH_TOKENS_KEY_PREFIX,
    REFRESH_TOKEN_REDIS_TTL,
)
from src.redis_client import get_redis


def _rt_key(token_hash: str) -> str:
    return f"{REFRESH_TOKEN_KEY_PREFIX}{token_hash}"

def _user_rts_key(user_id: str) -> str:
    return f"{USER_REFRESH_TOKENS_KEY_PREFIX}{user_id}"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


# ── Save ──────────────────────────────────────────────────────────────────────

def save_refresh_token(
    user_id: str,
    token_hash: str,
    email: str = "",
) -> None:
    """
    Persist a new (un-revoked) refresh token in Redis.
    """
    expires_at = _now_utc() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    expires_iso = expires_at.isoformat()

    r = get_redis()
    if r is not None:
        try:
            key = _rt_key(token_hash)
            pipe = r.pipeline()
            pipe.hset(key, mapping={
                "user_id":    user_id,
                "email":      email or "",
                "expires_at": expires_iso,
                "revoked":    "0",
            })
            pipe.expire(key, REFRESH_TOKEN_REDIS_TTL)
            ukey = _user_rts_key(user_id)
            pipe.sadd(ukey, token_hash)
            pipe.expire(ukey, REFRESH_TOKEN_REDIS_TTL)
            pipe.execute()
        except Exception as e:
            logger.warning(f"Redis save_refresh_token failed: {e}")


# ── Read ──────────────────────────────────────────────────────────────────────

def get_refresh_token(token_hash: str) -> Optional[dict]:
    """
    Look up a refresh token by its hash from Redis (<5 ms).
    """
    r = get_redis()
    if r is not None:
        try:
            key = _rt_key(token_hash)
            data = r.hgetall(key)
            if data:
                data["revoked"] = data.get("revoked", "0") == "1"
                return data
        except Exception as e:
            logger.warning(f"Redis get_refresh_token failed: {e}")

    return None


# ── Revoke single token ───────────────────────────────────────────────────────

def revoke_refresh_token(token_hash: str) -> None:
    """Mark a single token as revoked in Redis."""
    r = get_redis()
    if r is not None:
        try:
            key = _rt_key(token_hash)
            r.hset(key, "revoked", "1")
        except Exception as e:
            logger.warning(f"Redis revoke_refresh_token failed: {e}")


# ── Revoke all tokens for a user ──────────────────────────────────────────────

def revoke_all_user_tokens(user_id: str) -> None:
    """Revoke every active refresh token for *user_id* (logout-everywhere) in Redis."""
    r = get_redis()
    if r is not None:
        try:
            ukey = _user_rts_key(user_id)
            hashes = r.smembers(ukey)
            if hashes:
                pipe = r.pipeline()
                for h in hashes:
                    pipe.hset(_rt_key(h), "revoked", "1")
                pipe.delete(ukey)
                pipe.execute()
        except Exception as e:
            logger.warning(f"Redis revoke_all_user_tokens failed: {e}")


# ── Validity check ────────────────────────────────────────────────────────────

def is_token_valid(row: dict) -> bool:
    """
    Return True if the row represents a currently-valid refresh token.
    """
    if not row:
        return False
    if row.get("revoked") in (True, "1", 1):
        return False
    expires_at_raw = row.get("expires_at")
    if not expires_at_raw:
        return False
    try:
        if isinstance(expires_at_raw, str):
            expires_at_raw = expires_at_raw.replace("Z", "+00:00")
            expires_at = datetime.fromisoformat(expires_at_raw)
        else:
            expires_at = expires_at_raw
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        return _now_utc() < expires_at
    except Exception:
        return False
