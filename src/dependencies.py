import time
from fastapi import HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer
from typing import Any, Optional

from src.database import get_supabase, get_auth_supabase

DEV_USER_ID = "00000000-0000-0000-0000-000000000001"
DEV_USER = {"id": DEV_USER_ID, "email": "dev@studymate.ai", "name": "Dev User"}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

_TOKEN_CACHE: dict = {}
_TOKEN_CACHE_TTL = 300  # 5 minutes


def _verify_token_cached(client, token: str) -> Any:
    now = time.time()

    # Simple cleanup to prevent unbounded growth
    if len(_TOKEN_CACHE) > 1000:
        expired = [k for k, v in _TOKEN_CACHE.items() if now - v[1] > _TOKEN_CACHE_TTL]
        for k in expired:
            del _TOKEN_CACHE[k]

    # Return cached user if valid
    if token in _TOKEN_CACHE:
        user, timestamp = _TOKEN_CACHE[token]
        if now - timestamp < _TOKEN_CACHE_TTL:
            return user

    # Not cached or expired — fetch from Supabase
    response = client.auth.get_user(token)
    user = getattr(response, "user", None) or response
    if user:
        _TOKEN_CACHE[token] = (user, now)
        return user

    raise ValueError("Invalid token response")


def _extract_token(request: Request) -> Optional[str]:
    """
    Extract JWT from headers — case-insensitive.
    Priority: X-Auth-Token → Authorization (skip HF space tokens)
    """
    headers = {k.lower(): v for k, v in request.headers.items()}

    # 1. Try X-Auth-Token first (our custom header)
    token = headers.get("x-auth-token")
    if token:
        return token

    # 2. Fallback to Authorization header
    auth = headers.get("authorization", "")
    if auth.startswith("Bearer "):
        token = auth[len("Bearer "):].strip()
        # Skip HF tokens — they're for space access, not user auth
        if token.startswith("hf_"):
            return None
        return token

    return None


async def get_current_user_id(request: Request) -> str:
    client = get_auth_supabase() or get_supabase()

    # Dev mode — no Supabase configured
    if client is None:
        return DEV_USER_ID

    token = _extract_token(request)

    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated")

    # ── Mode 1: our own stateless JWT (fast — no network call) ───────────────
    try:
        from src.auth.jwt_utils import decode_access_token
        payload = decode_access_token(token)
        user_id = payload.get("sub")
        if user_id:
            return str(user_id)
    except Exception:
        pass  # Fall through to Supabase validation

    # ── Mode 2: Supabase token (backwards-compat for Google OAuth sessions) ──
    try:
        user = _verify_token_cached(client, token)
        return str(user.id)
    except Exception as e:
        print(f"Token validation error: {e}")
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token")


async def get_current_user(request: Request) -> Any:
    client = get_auth_supabase() or get_supabase()

    # Dev mode
    if client is None:
        return DEV_USER

    token = _extract_token(request)

    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated")

    # ── Mode 1: our own stateless JWT ────────────────────────────────────────
    try:
        from src.auth.jwt_utils import decode_access_token
        payload = decode_access_token(token)
        user_id = payload.get("sub")
        if user_id:
            return {"id": user_id, "email": payload.get("email", "")}
    except Exception:
        pass

    # ── Mode 2: Supabase token (backwards compat) ────────────────────────────
    try:
        user = _verify_token_cached(client, token)
        if user:
            return user
    except Exception as e:
        print(f"Token validation error: {e}")

    raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated")