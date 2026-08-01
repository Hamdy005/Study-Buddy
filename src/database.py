import logging
from typing import Optional
from supabase import Client, create_client
from src.config import settings

logger = logging.getLogger(__name__)

# Singletons — created once, reused on every request
_supabase_client: Optional[Client] = None
_auth_supabase_client: Optional[Client] = None


def get_supabase() -> Optional[Client]:
    global _supabase_client
    if _supabase_client is None:
        if not settings.supabase_url or not settings.supabase_key:
            return None
        _supabase_client = create_client(settings.supabase_url, settings.supabase_key)
    return _supabase_client


def get_auth_supabase() -> Optional[Client]:
    global _auth_supabase_client
    if _auth_supabase_client is None:
        if not settings.supabase_url or not settings.supabase_anon_key:
            return None
        _auth_supabase_client = create_client(settings.supabase_url, settings.supabase_anon_key)
    return _auth_supabase_client


def warmup_database() -> None:
    """Eagerly instantiate Supabase clients and pre-warm TLS connections during server boot."""
    db_client = get_supabase()
    get_auth_supabase()
    if db_client:
        try:
            db_client.table("profiles").select("id").limit(1).execute()
            logger.info("Supabase database connection warmed up successfully.")
        except Exception as e:
            logger.warning(f"Database warmup query failed (safe to ignore if offline/testing): {e}")