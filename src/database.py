from typing import Optional
from supabase import Client, create_client
from src.config import settings

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