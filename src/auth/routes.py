import uuid
from loguru import logger
import cloudinary
import cloudinary.uploader
from datetime import timezone, datetime
from fastapi import APIRouter, HTTPException, UploadFile, File, Response, Request
from fastapi import Depends
from typing import Optional

from src.config import settings
from src.database import get_auth_supabase, get_supabase
from src.store import create_user, get_user_by_email, delete_user_data, update_user_profile, get_user_by_id
from src.dependencies import get_current_user_id, get_current_user, _verify_token_cached
from src.auth.jwt_utils import (
    create_access_token,
    create_refresh_token,
    hash_token,
)
from src.auth.refresh_token_store import (
    save_refresh_token,
    get_refresh_token,
    revoke_refresh_token,
    is_token_valid,
)
from .schemas import ProfileUpdateRequest, EmailRateLimitRequest
from .constants import (
    ALLOWED_MIME_TYPES,
    MAX_FILE_SIZE_BYTES,
    AVATAR_BUCKET,
    PLACEHOLDER_DOMAINS,
    EMAIL_RATE_LIMITS,
    REFRESH_COOKIE_NAME,
    REFRESH_COOKIE_PATH,
    REFRESH_COOKIE_MAX_AGE,
)
from .rate_limiter import (
    enforce_email_rate_limit,
    get_email_rate_limit_status,
    check_and_record_email_rate_limit,
)

router = APIRouter(prefix="/api/auth", tags=["Auth"])


# ── Cookie helper ─────────────────────────────────────────────────────────────
# Constants are defined in src/auth/constants.py


def _set_refresh_cookie(response: Response, raw_token: str) -> None:
    """Attach the refresh token as an HttpOnly cookie on *response*."""
    is_prod = settings.environment != "development"
    response.set_cookie(
        key=REFRESH_COOKIE_NAME,
        value=raw_token,
        httponly=True,
        secure=is_prod,
        samesite="none" if is_prod else "lax",
        max_age=REFRESH_COOKIE_MAX_AGE,
        path=REFRESH_COOKIE_PATH,
    )


def _clear_refresh_cookie(response: Response) -> None:
    """Remove the refresh token cookie."""
    is_prod = settings.environment != "development"
    response.delete_cookie(
        key=REFRESH_COOKIE_NAME,
        path=REFRESH_COOKIE_PATH,
        samesite="none" if is_prod else "lax",
        secure=is_prod,
    )


@router.post("/upload-avatar")
async def upload_avatar(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id),
):
    """Upload a profile avatar image to Cloudinary and return its public URL."""

    # 1. Validate MIME type
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            400,
            f"Unsupported file type '{file.content_type}'. "
            f"Allowed types: {', '.join(sorted(ALLOWED_MIME_TYPES))}",
        )

    # 2. Read file bytes and validate size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            400,
            f"File is too large ({len(content) // 1024} KB). Maximum allowed size is 6 MB.",
        )

    # 3. Configure Cloudinary
    if not settings.cloudinary_cloud_name or not settings.cloudinary_api_key or not settings.cloudinary_api_secret:
        raise HTTPException(503, "Cloudinary service is not configured")

    cloudinary.config(
        cloud_name=settings.cloudinary_cloud_name,
        api_key=settings.cloudinary_api_key,
        api_secret=settings.cloudinary_api_secret,
        secure=True,
    )

    # 4. Upload to Cloudinary with smart face-cropping & auto webp conversion
    try:
        response = cloudinary.uploader.upload(
            content,
            folder="avatars",
            public_id=f"avatar_{user_id}",
            overwrite=True,
            transformation=[
                {"width": 300, "height": 300, "crop": "fill", "gravity": "face"},
                {"fetch_format": "auto", "quality": "auto"}
            ]
        )
        public_url = response.get("secure_url") or response.get("url")
        return {"status": "success", "avatar_url": public_url}
    except Exception as e:
        raise HTTPException(500, f"Failed to upload avatar to Cloudinary: {e}")


@router.delete("/me")
async def delete_account(user_id: str = Depends(get_current_user_id)):
    # 1. Delete all DB data (materials, quizzes, profile, etc.)
    delete_user_data(user_id)

    # 2. Delete the Supabase Auth user so they can't sign in again
    admin_client = get_supabase()
    if admin_client:
        try:
            admin_client.auth.admin.delete_user(user_id)
        except Exception as e:
            raise HTTPException(500, f"Account data deleted but failed to remove auth user: {e}")

    return {"status": "success", "message": "Account deleted successfully"}


@router.get("/profile")
async def get_profile(
    user_id: str = Depends(get_current_user_id),
    current_user=Depends(get_current_user)
):
    user = get_user_by_id(user_id)

    # Fast-path: profile row exists and has a real email — return immediately
    email = user.get("email", "") if user else ""
    is_placeholder = (
        not user
        or not email
        or any(domain in email for domain in PLACEHOLDER_DOMAINS)
    )

    if is_placeholder:
        # First-ever login: pull real data from Supabase Auth admin API and persist it
        from src.database import get_supabase
        supabase = get_supabase()
        if supabase:
            try:
                res = supabase.auth.admin.get_user_by_id(user_id)
                if res.user and res.user.email and "@" in res.user.email:
                    real_email = res.user.email
                    real_name = res.user.user_metadata.get("name")

                    from src.store import _table_supabase, _map_profile
                    data = {"id": user_id, "email": real_email}
                    if real_name:
                        data["display_name"] = real_name

                    try:
                        # Use upsert so we NEVER overwrite daily_requests / last_request_date
                        # on subsequent sign-ins.  Only id/email/display_name are safe to set.
                        client = supabase  # already resolved above
                        res_upd = (
                            client.table("profiles")
                            .upsert(data, on_conflict="id", ignore_duplicates=False)
                            .execute()
                        )
                        if res_upd.data:
                            user = _map_profile(res_upd.data[0])
                    except Exception:
                        # Upsert failed — fall back to a plain update (never resets usage)
                        try:
                            # Strip the id from the update payload to avoid PK conflicts
                            update_data = {k: v for k, v in data.items() if k != "id"}
                            res_upd = (
                                _table_supabase("profiles")
                                .update(update_data)
                                .eq("id", user_id)
                                .execute()
                            )
                            if res_upd.data:
                                user = _map_profile(res_upd.data[0])
                        except Exception:
                            pass
            except Exception:
                pass

    if not user:
        # Last resort: synthetic profile from the JWT claims so the UI doesn't break.
        # We still try to fetch real usage from the DB to avoid resetting the counter.
        user_obj = current_user
        uid = getattr(user_obj, "id", None) or (user_obj.get("id") if isinstance(user_obj, dict) else None)
        if uid:
            from src.store import _map_profile, get_usage
            meta = getattr(user_obj, "user_metadata", {}) or {}
            # Fetch real usage so the fallback profile doesn't reset the counter to 0
            real_usage = get_usage(uid)
            user = _map_profile({
                "id": uid,
                "display_name": meta.get("full_name") or meta.get("name") or "User",
                "email": getattr(user_obj, "email", "") or "",
                "avatar_url": "",
                "daily_requests": real_usage.get("used", 0),
                "last_request_date": (
                    __import__('datetime').datetime.now(
                        __import__('datetime').timezone(__import__('datetime').timedelta(hours=3))
                    ).date().isoformat()
                ),
                "_is_fallback": True,
            })

    if not user:
        raise HTTPException(404, "Profile not found")
    return {"status": "success", "user": user}


@router.patch("/profile")
async def update_profile(body: ProfileUpdateRequest, user_id: str = Depends(get_current_user_id)):
    # Reject raw base64 image data — images must be uploaded via /upload-avatar first
    if body.avatar_url and body.avatar_url.startswith("data:"):
        raise HTTPException(
            400,
            "Storing raw image data is not allowed. "
            "Upload the image via POST /api/auth/upload-avatar and use the returned URL instead."
        )

    # Handle password update if password is provided
    if body.password is not None:
        # Verify current password if provided

        if body.current_password:
            user_data = get_user_by_id(user_id)
            user_email = user_data.get("email") if user_data else None
            if user_email and not any(domain in user_email for domain in PLACEHOLDER_DOMAINS):
                auth_client = get_auth_supabase()
                if auth_client:
                    try:
                        res = auth_client.auth.sign_in_with_password({
                            "email": user_email,
                            "password": body.current_password
                        })
                        if hasattr(res, "error") and res.error:
                            raise HTTPException(400, "Current password is incorrect.")
                    except HTTPException:
                        raise
                    except Exception:
                        raise HTTPException(400, "Current password is incorrect.")

        # Update password in Supabase Auth
        admin_client = get_supabase()
        if admin_client:
            try:
                admin_client.auth.admin.update_user_by_id(user_id, {"password": body.password})
            except Exception as e:
                raise HTTPException(500, f"Failed to update password: {e}")

    try:
        updated_user = update_user_profile(
            user_id,
            name=body.name,
            avatar_url=body.avatar_url,
            theme=body.theme
        )
        return {"status": "success", "user": updated_user}
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Failed to update profile: {e}")


@router.post("/check-email-rate-limit")
async def check_email_limit(body: EmailRateLimitRequest):
    """
    Enforce in-memory rate limit for email sending actions (3 emails per hour).
    Action must be one of: 'email_verification', 'forgot_password', or 'change_password_confirmation'.
    Raises HTTP 429 if limit is reached.
    """
    remaining = enforce_email_rate_limit(body.action, body.email)
    return {
        "status": "allowed",
        "action": body.action,
        "email": body.email,
        "remaining_attempts": remaining,
    }


@router.get("/email-rate-limit-status")
async def email_limit_status(action: str, email: str):
    """
    Get the status of an email rate limit window without recording a new attempt.
    """
    status_info = get_email_rate_limit_status(action, email)
    return {"status": "success", "action": action, "email": email, **status_info}


# ── Token Exchange & Refresh ───────────────────────────────────────────────────

@router.post("/session")
async def exchange_session(request: Request, response: Response):
    """
    Token exchange endpoint.

    Accepts a Supabase JWT in the Authorization header, validates it once
    (locally via the Supabase JWT secret — stateless, no network call), then issues:
      • A short-lived (15 min) JWT in the response body
      • A long-lived (30 day) opaque refresh token as an HttpOnly cookie

    The frontend should call this right after any Supabase sign-in event
    (onAuthStateChange fires with a session).

    IMPORTANT: We decode the Supabase JWT locally instead of calling
    client.auth.get_user() to avoid a 403 race condition — the Supabase JS
    SDK rotates the session internally immediately after sign-in, so a
    stateful get_user() call often fails before our backend can validate it.
    """
    import jwt as pyjwt

    # 1. Extract the Supabase token from the request.
    headers = {k.lower(): v for k, v in request.headers.items()}
    auth = headers.get("authorization", "")
    x_auth = headers.get("x-auth-token", "")
    raw_supabase_token = x_auth or (auth[len("Bearer "):].strip() if auth.startswith("Bearer ") else None)

    if not raw_supabase_token:
        content_type = headers.get("content-type", "")
        if content_type.startswith("text/plain"):
            body = await request.body()
            raw_supabase_token = body.decode("utf-8").strip() if body else None

    if not raw_supabase_token:
        raise HTTPException(401, "Authorization header with Supabase token required")

    # Skip HF space tokens — they're not user auth tokens
    if raw_supabase_token.startswith("hf_"):
        raise HTTPException(401, "HuggingFace space token is not a valid user auth token")

    # 2. Decode the Supabase JWT locally (stateless — no round-trip to Supabase)
    supabase_jwt_secret = settings.supabase_jwt_secret
    user_id = None
    email = ""
    user_metadata: dict = {}

    if supabase_jwt_secret:
        try:
            payload = pyjwt.decode(
                raw_supabase_token,
                supabase_jwt_secret,
                algorithms=["HS256", "HS384", "HS512"],
                options={"verify_aud": False},  # Supabase uses 'authenticated' as aud
            )
            user_id = payload.get("sub")
            email = payload.get("email", "")
            user_metadata = payload.get("user_metadata", {})
        except pyjwt.ExpiredSignatureError:
            raise HTTPException(401, "Supabase token has expired. Please sign in again.")
        except Exception:
            pass  # Fall through to Supabase API validation below

    # Fallback: validate via Supabase API (slower, but works if unverified decoding failed)
    if not user_id:
        client = get_auth_supabase() or get_supabase()
        if not client:
            raise HTTPException(503, "Auth service unavailable")
        try:
            sb_user = _verify_token_cached(client, raw_supabase_token)
            user_id = str(sb_user.id)
            email = getattr(sb_user, "email", "") or ""
            user_metadata = getattr(sb_user, "user_metadata", {}) or {}
        except Exception:
            raise HTTPException(401, "Invalid or expired Supabase token")

    if not user_id:
        raise HTTPException(401, "Could not extract user identity from token")

    # 3. Ensure a profile row exists (first login race-safe)
    profile = get_user_by_id(user_id)
    if not profile:
        name = (
            user_metadata.get("full_name")
            or user_metadata.get("name")
            or email.split("@")[0]
            or "User"
        )
        profile = create_user(name=name, email=email, password="", user_id=user_id)

    # 4. Issue our tokens
    access_token  = create_access_token(user_id, email)
    raw_refresh   = create_refresh_token()
    refresh_hash  = hash_token(raw_refresh)
    save_refresh_token(user_id, refresh_hash)

    _set_refresh_cookie(response, raw_refresh)

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": profile,
    }


@router.post("/refresh")
async def refresh_session(request: Request, response: Response):
    """
    Silently re-issue a new access token using the HttpOnly refresh token cookie.

    Rotates the refresh token on every use (old token is revoked, new one is issued)
    so a stolen token can only be used once before it's invalidated.
    """
    raw = request.cookies.get(REFRESH_COOKIE_NAME)
    if not raw:
        logger.warning("Refresh session failed: No refresh token cookie found in request. Cookies present: %s", list(request.cookies.keys()))
        raise HTTPException(401, "No refresh token cookie found")

    token_hash = hash_token(raw)
    row = get_refresh_token(token_hash)

    if not row:
        logger.warning("Refresh session failed: Token hash (%s...) not found in refresh_tokens table", token_hash[:8])
        _clear_refresh_cookie(response)
        raise HTTPException(401, "Refresh token is invalid, expired, or revoked")

    if not is_token_valid(row):
        logger.warning("Refresh session failed: Token for user %s is expired or revoked (expires_at=%s, revoked=%s)", row.get("user_id"), row.get("expires_at"), row.get("revoked"))
        _clear_refresh_cookie(response)
        raise HTTPException(401, "Refresh token is invalid, expired, or revoked")

    user_id = str(row["user_id"])

    # Fetch the user's email for the new access token payload
    profile = get_user_by_id(user_id)
    email = (profile or {}).get("email", "")

    # Rotate: revoke old, issue new refresh token
    revoke_refresh_token(token_hash)
    new_raw_refresh = create_refresh_token()
    new_hash        = hash_token(new_raw_refresh)
    save_refresh_token(user_id, new_hash)

    access_token = create_access_token(user_id, email)

    _set_refresh_cookie(response, new_raw_refresh)

    return {
        "access_token": access_token,
        "token_type": "bearer",
    }


@router.post("/logout")
async def logout(request: Request, response: Response):
    """
    Revoke the refresh token in the database and clear the cookie.

    This is the only true logout — do not rely on JWT expiry alone.
    The short-lived access token will expire naturally within 15 minutes.
    """
    raw = request.cookies.get(REFRESH_COOKIE_NAME)
    if raw:
        token_hash = hash_token(raw)
        revoke_refresh_token(token_hash)

    _clear_refresh_cookie(response)
    return {"status": "ok", "message": "Logged out successfully"}


@router.get("/me")
async def get_me(
    user_id: str = Depends(get_current_user_id),
    current_user=Depends(get_current_user),
):
    """
    Return the authenticated user's profile.

    Protected route — requires a valid JWT in Authorization: Bearer header.
    This is a thin wrapper over the existing get_profile logic so both
    /api/auth/profile and /api/auth/me return the same shape.
    """
    user = get_user_by_id(user_id)
    if not user:
        # Fallback: build minimal profile from JWT payload
        uid = (
            getattr(current_user, "id", None)
            or (current_user.get("id") if isinstance(current_user, dict) else None)
        )
        if uid:
            from src.store import _map_profile, get_usage
            email = (
                getattr(current_user, "email", "") or
                (current_user.get("email") if isinstance(current_user, dict) else "") or ""
            )
            real_usage = get_usage(uid)
            user = _map_profile({
                "id": uid,
                "display_name": email.split("@")[0] or "User",
                "email": email,
                "avatar_url": "",
                "daily_requests": real_usage.get("used", 0),
                "last_request_date": (
                    datetime.now(timezone.utc).date().isoformat()
                ),
            })

    if not user:
        raise HTTPException(404, "Profile not found")
    return {"status": "success", "user": user}
