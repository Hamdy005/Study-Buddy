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

* ROTATE & REVOKE:
    Updates Redis in micro-seconds (<3ms) and offloads Supabase audit log writes
    to FastAPI BackgroundTasks to ensure refresh response time stays under ~15ms.

Redis key schema
----------------
    rt:{token_hash}          → Hash  {user_id, email, expires_at, revoked}   TTL=30d
    user_rts:{user_id}       → Set   of token_hash strings                   TTL=30d
"""

from loguru import logger
from datetime import datetime, timedelta, timezone
from typing import Optional
from fastapi import BackgroundTasks

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


def _supabase_insert_refresh_token(user_id: str, token_hash: str, expires_iso: str) -> None:
    """Helper function to insert refresh token into Supabase (runs in background)."""
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
        logger.error(f"Supabase background save_refresh_token failed: {e}")


def _supabase_revoke_refresh_token(token_hash: str) -> None:
    """Helper function to mark token revoked in Supabase (runs in background)."""
    try:
        _robust_execute(
            _table_supabase("refresh_tokens")
            .update({"revoked": True})
            .eq("token_hash", token_hash)
        )
    except Exception as e:
        logger.error(f"Supabase background revoke_refresh_token failed: {e}")


def _supabase_revoke_all_user_tokens(user_id: str) -> None:
    """Helper function to mark all user tokens revoked in Supabase (runs in background)."""
    try:
        _robust_execute(
            _table_supabase("refresh_tokens")
            .update({"revoked": True})
            .eq("user_id", user_id)
        )
    except Exception as e:
        logger.error(f"Supabase background revoke_all_user_tokens failed for {user_id}: {e}")


# ── Save ──────────────────────────────────────────────────────────────────────

def save_refresh_token(
    user_id: str,
    token_hash: str,
    email: str = "",
    background_tasks: Optional[BackgroundTasks] = None
) -> None:
    """
    Persist a new (un-revoked) refresh token.

    Writes to Redis synchronously (<3ms).
    Offloads Supabase audit write to background_tasks if provided, keeping refresh responses under 15ms.
    """
    expires_at = _now_utc() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    expires_iso = expires_at.isoformat()

    redis_ok = False
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
            redis_ok = True
        except Exception as e:
            logger.warning(f"Redis save_refresh_token failed: {e}")

    # If background_tasks is available and Redis succeeded, schedule DB write in background
    if background_tasks and redis_ok:
        background_tasks.add_task(_supabase_insert_refresh_token, user_id, token_hash, expires_iso)
    else:
        # Fallback / sync insert if Redis isn't available or background_tasks not passed
        _supabase_insert_refresh_token(user_id, token_hash, expires_iso)


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
                data["revoked"] = data.get("revoked", "0") == "1"
                return data
        except Exception as e:
            logger.warning(f"Redis get_refresh_token failed: {e}")

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
                    "email":      "",
                    "expires_at": str(row.get("expires_at", "")),
                    "revoked":    "1" if row.get("revoked") else "0",
                })
                pipe.expire(key, REFRESH_TOKEN_REDIS_TTL)
                pipe.execute()
            except Exception as cache_err:
                logger.warning(f"Redis back-populate failed: {cache_err}")

        return row
    except Exception as e:
        logger.error(f"get_refresh_token Supabase fallback failed: {e}")
        return None


# ── Revoke single token ───────────────────────────────────────────────────────

def revoke_refresh_token(token_hash: str, background_tasks: Optional[BackgroundTasks] = None) -> None:
    """Mark a single token as revoked in Redis and Supabase."""
    redis_ok = False
    r = get_redis()
    if r is not None:
        try:
            key = _rt_key(token_hash)
            r.hset(key, "revoked", "1")
            redis_ok = True
        except Exception as e:
            logger.warning(f"Redis revoke_refresh_token failed: {e}")

    if background_tasks and redis_ok:
        background_tasks.add_task(_supabase_revoke_refresh_token, token_hash)
    else:
        _supabase_revoke_refresh_token(token_hash)


# ── Revoke all tokens for a user ──────────────────────────────────────────────

def revoke_all_user_tokens(user_id: str, background_tasks: Optional[BackgroundTasks] = None) -> None:
    """Revoke every active refresh token for *user_id* (logout-everywhere)."""
    redis_ok = False
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
            redis_ok = True
        except Exception as e:
            logger.warning(f"Redis revoke_all_user_tokens failed: {e}")

    if background_tasks and redis_ok:
        background_tasks.add_task(_supabase_revoke_all_user_tokens, user_id)
    else:
        _supabase_revoke_all_user_tokens(user_id)


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
