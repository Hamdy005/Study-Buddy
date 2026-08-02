import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


ENV_PATH = Path(__file__).resolve().parents[1] / "config.env"
load_dotenv(ENV_PATH)


class Settings:
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    mistral_api_key: str = os.getenv("MISTRAL_API_KEY", "")
    # Accept both plain and NEXT_PUBLIC_ prefixed names (config.env uses NEXT_PUBLIC_)
    supabase_url: str = (
        os.getenv("SUPABASE_URL")
        or os.getenv("NEXT_PUBLIC_SUPABASE_URL", "")
    )
    supabase_key: str = (
        os.getenv("SUPABASE_KEY")
        or os.getenv("NEXT_PUBLIC_SUPABASE_KEY", "")
    )
    supabase_anon_key: str = (
        os.getenv("SUPABASE_ANON_KEY")
        or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY", "")
    )
    supabase_jwt_secret: str = (
        os.getenv("SUPABASE_JWT_SECRET")
        or os.getenv("JWT_SECRET", "")
    )
    cloudinary_cloud_name: str = (
        os.getenv("CLOUDINARY_CLOUD_NAME")
        or os.getenv("CLOUD_NAME", "")
    )
    cloudinary_api_key: str = (
        os.getenv("CLOUDINARY_API_KEY")
        or os.getenv("CLOUD_API_KEY", "")
    )
    cloudinary_api_secret: str = (
        os.getenv("CLOUDINARY_API_SECRET")
        or os.getenv("CLOUD_SECRET", "")
    )
    model_name: str = os.getenv("MODEL_NAME", "gemini-3.1-flash-lite")
    transformers_no_tf: str = os.getenv("TRANSFORMERS_NO_TF", "1")
    cors_allowed_origins: list[str] = [
        origin.strip()
        for origin in os.getenv("CORS_ALLOWED_ORIGINS", "").split(",")
        if origin.strip()
    ]
    local_arabic_nsfw_words: list[str] = [
        w.strip()
        for w in os.getenv("LOCAL_ARABIC_NSFW_WORDS", "").split(",")
        if w.strip()
    ]
    local_nsfw_words: list[str] = [
        w.strip()
        for w in os.getenv("LOCAL_NSFW_WORDS", "").split(",")
        if w.strip()
    ]

    # ── JWT / Refresh-Token Auth ───────────────────────────────────────────────
    jwt_secret_key: str = os.getenv("JWT_SECRET_KEY", "")
    jwt_algorithm: str  = os.getenv("JWT_ALGORITHM", "HS256")
    # Set to 'development' locally so Secure cookie flag is not required over HTTP
    environment: str    = os.getenv("ENVIRONMENT", "production")
    # Supabase JWT secret — used to verify Supabase-issued tokens locally (stateless).
    # Found in: Supabase Dashboard → Project Settings → API → JWT Settings → JWT Secret
    # This avoids the 403 "Session does not exist" error from stateful get_user() calls.
    supabase_jwt_secret: str = os.getenv("SUPABASE_JWT_SECRET", "")


@lru_cache()
def get_settings() -> Settings:
    s = Settings()
    if s.gemini_api_key:
        os.environ["GEMINI_API_KEY"] = s.gemini_api_key
    if s.mistral_api_key:
        os.environ["MISTRAL_API_KEY"] = s.mistral_api_key
    if "TRANSFORMERS_NO_TF" not in os.environ and s.transformers_no_tf:
        os.environ["TRANSFORMERS_NO_TF"] = s.transformers_no_tf
    return s


settings = get_settings()
