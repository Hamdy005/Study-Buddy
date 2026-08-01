"""
Stateful refresh-token store.

Wraps the `refresh_tokens` Supabase table with simple CRUD helpers.
All functions are synchronous and use the existing `_table_supabase`
helper from `store.py`, so they get the same dev-mode fallback behaviour.

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

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from src.store import _table_supabase, _robust_execute
from src.auth.constants import REFRESH_TOKEN_EXPIRE_DAYS

logger = logging.getLogger(__name__)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def save_refresh_token(user_id: str, token_hash: str) -> None:
    """Persist a new (un-revoked) refresh token row for *user_id*."""
    expires_at = _now_utc() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    try:
        _robust_execute(
            _table_supabase("refresh_tokens").insert({
                "user_id": user_id,
                "token_hash": token_hash,
                "expires_at": expires_at.isoformat(),
                "revoked": False,
            })
        )
    except Exception as e:
        logger.error("save_refresh_token failed: %s", e)
        raise


def get_refresh_token(token_hash: str) -> Optional[dict]:
    """
    Look up a refresh token row by its hash.

    Returns the raw DB row dict, or None if not found.
    """
    try:
        result = _robust_execute(
            _table_supabase("refresh_tokens")
            .select("*")
            .eq("token_hash", token_hash)
        )
        rows = result.data
        if isinstance(rows, list):
            return rows[0] if rows else None
        return rows or None
    except Exception as e:
        logger.error("get_refresh_token failed: %s", e)
        return None


def revoke_refresh_token(token_hash: str) -> None:
    """Mark a single token as revoked."""
    try:
        _robust_execute(
            _table_supabase("refresh_tokens")
            .update({"revoked": True})
            .eq("token_hash", token_hash)
        )
    except Exception as e:
        logger.error("revoke_refresh_token failed: %s", e)


def revoke_all_user_tokens(user_id: str) -> None:
    """Revoke every active refresh token for *user_id* (logout-everywhere)."""
    try:
        _robust_execute(
            _table_supabase("refresh_tokens")
            .update({"revoked": True})
            .eq("user_id", user_id)
        )
    except Exception as e:
        logger.error("revoke_all_user_tokens failed for %s: %s", user_id, e)


def is_token_valid(row: dict) -> bool:
    """
    Return True if the DB row represents a currently-valid refresh token.

    Checks: row exists, not revoked, not expired.
    """
    if not row:
        return False
    if row.get("revoked"):
        return False
    expires_at_raw = row.get("expires_at")
    if not expires_at_raw:
        return False
    # Supabase returns ISO-8601 strings; parse and compare.
    try:
        if isinstance(expires_at_raw, str):
            # Handle both 'Z' suffix and '+00:00' offset
            expires_at_raw = expires_at_raw.replace("Z", "+00:00")
            expires_at = datetime.fromisoformat(expires_at_raw)
        else:
            expires_at = expires_at_raw
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        return _now_utc() < expires_at
    except Exception:
        return False
