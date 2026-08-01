"""
JWT and refresh-token utilities.

Responsibilities:
  - Sign / verify short-lived access tokens (15 min) with our own secret key.
  - Generate opaque refresh token strings and produce their SHA-256 hash for
    safe storage in the database.

No FastAPI or database imports — pure utility module.
"""
import hashlib
import secrets
from datetime import datetime, timedelta, timezone

import jwt

from src.config import settings
from src.auth.constants import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    REFRESH_TOKEN_EXPIRE_DAYS,
)


def create_access_token(user_id: str, email: str) -> str:
    """Return a signed JWT that expires in ACCESS_TOKEN_EXPIRE_MINUTES."""
    if not settings.jwt_secret_key:
        raise RuntimeError(
            "JWT_SECRET_KEY is not configured. "
            "Add it to config.env (generate with: openssl rand -hex 32)."
        )
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "email": email,
        "iat": now,
        "exp": now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict:
    """
    Decode and verify a JWT issued by create_access_token.

    Returns the payload dict on success.
    Raises jwt.ExpiredSignatureError if the token has expired.
    Raises jwt.InvalidTokenError (or subclass) for any other problem.
    """
    if not settings.jwt_secret_key:
        raise RuntimeError("JWT_SECRET_KEY is not configured.")
    return jwt.decode(
        token,
        settings.jwt_secret_key,
        algorithms=[settings.jwt_algorithm],
    )


# ── Refresh token ─────────────────────────────────────────────────────────────

REFRESH_TOKEN_EXPIRE_DAYS = 30


def create_refresh_token() -> str:
    """Return a cryptographically-secure 64-char hex string (32 random bytes)."""
    return secrets.token_hex(32)


def hash_token(raw: str) -> str:
    """SHA-256 hex digest — safe to store in the DB instead of the raw value."""
    return hashlib.sha256(raw.encode()).hexdigest()
