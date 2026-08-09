ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/avif",
    "image/svg+xml",
}

MAX_FILE_SIZE_BYTES = 6 * 1024 * 1024  # 6 MB

AVATAR_BUCKET = "avatars"

PLACEHOLDER_DOMAINS = ["@placeholder.ai", "@studymate.ai"]

# Rate limits per email action (3 emails per hour)
EMAIL_RATE_LIMITS = {
    "email_verification": {"limit": 3, "window_seconds": 3600},
    "forgot_password": {"limit": 3, "window_seconds": 3600},
    "change_password_confirmation": {"limit": 3, "window_seconds": 3600},
}

# ── JWT / Access token ────────────────────────────────────────────────────────

# Short-lived JWT lifetime. Stateless: verified via signature only, no DB lookup.
ACCESS_TOKEN_EXPIRE_MINUTES = 15

# ── Refresh token ─────────────────────────────────────────────────────────────

# Long-lived opaque token lifetime. Stateful: looked up in the DB on every use.
REFRESH_TOKEN_EXPIRE_DAYS = 30

# ── HttpOnly cookie settings ──────────────────────────────────────────────────

REFRESH_COOKIE_NAME    = "refresh_token"
# Scope the cookie to the auth sub-path so it is NOT sent to /api/materials etc.
REFRESH_COOKIE_PATH    = "/api/auth"
REFRESH_COOKIE_MAX_AGE = REFRESH_TOKEN_EXPIRE_DAYS * 24 * 3600  # seconds

# ── Redis Auth Keys & TTLs ─────────────────────────────────────────────────────
REFRESH_TOKEN_KEY_PREFIX = "rt:"
USER_REFRESH_TOKENS_KEY_PREFIX = "user_rts:"
REFRESH_TOKEN_REDIS_TTL = REFRESH_TOKEN_EXPIRE_DAYS * 86400  # seconds

