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
