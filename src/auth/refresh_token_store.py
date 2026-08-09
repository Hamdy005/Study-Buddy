"""
Stateful refresh-token store — Redis-first, Supabase fallback.

Strategy
--------
* LOGIN  (save_refresh_token):
    Write to Redis (primary, fast reads) AND Supabase (audit log).
    Redis TTL = REFRESH_TOKEN_EXPIRE_DAYS.

* REFRESH (get_refresh_token):
    Read from Redis only (<5 ms).
    If Redis miss (e.g. first deploy after adding Redis), fall back to Supabase
    and repopulate Redis so subsequent reads are fast.

* REVOKE (revoke_refresh_token / revoke_all_user_tokens):
    Delete / mark revoked in Redis first, then mirror to Supabase.

Redis key schema
----------------
    rt:{token_hash}          → Hash  {user_id, expires_at, revoked}   TTL=30d
    user_rts:{user_id}       → Set   of token_hash strings             TTL=30d

Table DDL (run once in the Supabase SQL editor):

    CREATE TABLE IF NOT EXISTS refresh_tokens (
      id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      user_id     UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
      token_hash  TEXT NOT NULL UNIQUE,
      expires_at  TIMESTAMPTZ NOT NULL,
      revoked     BOOLEAN NOT NULL DEFAULT FALSE,
      created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    CREATE INDEX ON refresh_tokens(token_hash);
    CREATE INDEX ON refresh_tokens(user_id);
"""

from loguru import logger
from datetime import datetime, timedelta, timezone
from typing import Optional

from src.store import _table_supabase, _robust_execute
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

def save_refresh_token(user_id: str, token_hash: str) -> None:
    """
    Persist a new (un-revoked) refresh token.

    Writes to Redis (primary, fast reads) AND Supabase (audit log).
    Called once at login — the extra Supabase write here is acceptable
    because this path is only hit when the user actively signs in.
    """
    expires_at = _now_utc() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    expires_iso = expires_at.isoformat()

    # 1. Redis — primary store
    r = get_redis()
    if r is not None:
        try:
            key = _rt_key(token_hash)
            pipe = r.pipeline()
            pipe.hset(key, mapping={
                "user_id":    user_id,
                "expires_at": expires_iso,
                "revoked":    "0",
            })
            pipe.expire(key, REFRESH_TOKEN_REDIS_TTL)
            # Track all hashes per user so revoke_all_user_tokens can find them
            ukey = _user_rts_key(user_id)
            pipe.sadd(ukey, token_hash)
            pipe.expire(ukey, REFRESH_TOKEN_REDIS_TTL)
            pipe.execute()
        except Exception as e:
            logger.warning("Redis save_refresh_token failed: %s", e)

    # 2. Supabase — audit log / fallback
    try:
        _robust_execute(
            _table_supabase("refresh_tokens").insert({
                "user_id":    user_id,
                "token_hash": token_hash,
                "expires_at": expires_iso,
                "revoked":    False,
            })
        )
    except Exception as e:
        logger.error("Supabase save_refresh_token failed: %s", e)
        raise


# ── Read ──────────────────────────────────────────────────────────────────────

def get_refresh_token(token_hash: str) -> Optional[dict]:
    """
    Look up a refresh token by its hash.

    Reads from Redis first (<5 ms).  On a Redis miss, falls back to Supabase
    and back-populates Redis so the next read is fast.
    """
    r = get_redis()
    if r is not None:
        try:
            key = _rt_key(token_hash)
            data = r.hgetall(key)
            if data:
                # Normalise boolean — stored as "0"/"1" string
                data["revoked"] = data.get("revoked", "0") == "1"
                return data
        except Exception as e:
            logger.warning("Redis get_refresh_token failed: %s", e)

    # Redis miss or unavailable — fall back to Supabase
    try:
        result = _robust_execute(
            _table_supabase("refresh_tokens")
            .select("*")
            .eq("token_hash", token_hash)
        )
        rows = result.data
        if isinstance(rows, list):
            row = rows[0] if rows else None
        else:
            row = rows or None

        # Back-populate Redis so subsequent reads are fast
        if row and r is not None:
            try:
                key = _rt_key(token_hash)
                pipe = r.pipeline()
                pipe.hset(key, mapping={
                    "user_id":    str(row["user_id"]),
                    "expires_at": str(row.get("expires_at", "")),
                    "revoked":    "1" if row.get("revoked") else "0",
                })
                pipe.expire(key, REFRESH_TOKEN_REDIS_TTL)
                pipe.execute()
            except Exception as cache_err:
                logger.warning("Redis back-populate failed: %s", cache_err)

        return row
    except Exception as e:
        logger.error("get_refresh_token Supabase fallback failed: %s", e)
        return None


# ── Revoke single token ───────────────────────────────────────────────────────

def revoke_refresh_token(token_hash: str) -> None:
    """Mark a single token as revoked in Redis and Supabase."""
    r = get_redis()
    if r is not None:
        try:
            key = _rt_key(token_hash)
            r.hset(key, "revoked", "1")
        except Exception as e:
            logger.warning("Redis revoke_refresh_token failed: %s", e)

    try:
        _robust_execute(
            _table_supabase("refresh_tokens")
            .update({"revoked": True})
            .eq("token_hash", token_hash)
        )
    except Exception as e:
        logger.error("revoke_refresh_token Supabase failed: %s", e)


# ── Revoke all tokens for a user ──────────────────────────────────────────────

def revoke_all_user_tokens(user_id: str) -> None:
    """Revoke every active refresh token for *user_id* (logout-everywhere)."""
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
            logger.warning("Redis revoke_all_user_tokens failed: %s", e)

    try:
        _robust_execute(
            _table_supabase("refresh_tokens")
            .update({"revoked": True})
            .eq("user_id", user_id)
        )
    except Exception as e:
        logger.error("revoke_all_user_tokens Supabase failed for %s: %s", user_id, e)


# ── Validity check (unchanged — pure Python, no I/O) ─────────────────────────

def is_token_valid(row: dict) -> bool:
    """
    Return True if the row represents a currently-valid refresh token.

    Checks: row exists, not revoked, not expired.
    Works with both Supabase row dicts and Redis hgetall dicts.
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
