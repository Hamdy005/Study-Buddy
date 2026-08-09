import os
import time
import json
from httpx import RemoteProtocolError
from typing import Optional
import logging
from datetime import datetime, timezone, date, timedelta
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from src.database import get_supabase
from src.redis_client import get_redis

# ── Redis Key Prefixes & TTLs ──────────────────────────────────────────────────
MEMORY_KEY_PREFIX = "mem:"
MEMORY_REDIS_TTL = 48 * 3600             # 48 hours — keeps active sessions warm
MEMORY_MAX_MESSAGES = 20                 # Max message history retained in Redis cache (10 turns)

DAILY_RATE_LIMIT_KEY_PREFIX = "rate:"
DAILY_RATE_LIMIT_REDIS_TTL = 86400       # 24 hours — daily usage counter auto-expiration

def _get_today_date_str() -> str:
    # Shift UTC time by 3 hours to match Egypt timezone (UTC+3), so daily limits reset at 12 AM Egypt time.
    egypt_tz = timezone(timedelta(hours=3))
    return datetime.now(egypt_tz).date().isoformat()

_in_memory: dict = {
    "materials": {},
    "material_chunks": {},
    "summaries": {},
    "quizzes": {},
    "users": {},
    "next_id": 0,
}

ADMIN_EMAILS = set(
    email.strip() 
    for email in os.environ.get("ADMIN_EMAILS", "").split(",")
    if email.strip()
)


logger = logging.getLogger(__name__)


def _get_next_id() -> str:
    _in_memory["next_id"] += 1
    return str(_in_memory["next_id"])


_supabase_client = None

def _db():  
    global _supabase_client
    if _supabase_client is None:
        _supabase_client = get_supabase()
    if _supabase_client is not None:
        return _supabase_client
    return None


class _FakeTable:
    def __init__(self, name):
        self.name = name
        self._pending_insert: list | None = None
        self._eq_field: str | None = None
        self._eq_value = None
        self._update_data: dict | None = None
        self._single = False

    def insert(self, data):
        if isinstance(data, list):
            for item in data:
                item["id"] = item.get("id", _get_next_id())
                _in_memory.setdefault(self.name, {})[item["id"]] = item
            self._pending_insert = data
        else:
            data["id"] = data.get("id", _get_next_id())
            _in_memory.setdefault(self.name, {})[data["id"]] = data
            self._pending_insert = [data]
        return self

    def select(self, *args):
        return self

    def eq(self, field, value):
        self._eq_field = field
        self._eq_value = value
        return self

    def order(self, field):
        return self

    def maybe_single(self):
        self._single = True
        return self

    def update(self, data):
        self._update_data = data
        return self

    def delete(self):
        """Mark this query for deletion."""
        self._delete = True
        return self

    def execute(self):
        if getattr(self, '_delete', False):
            store = _in_memory.get(self.name, {})
            if self._eq_field:
                keys = [k for k, v in store.items() if v.get(self._eq_field) == self._eq_value]
                for k in keys:
                    store.pop(k, None)
            return self._make_response([])
        if self._pending_insert is not None:
            return self._make_response(self._pending_insert)
        records = list(_in_memory.get(self.name, {}).values())
        if self._eq_field:
            records = [r for r in records if r.get(self._eq_field) == self._eq_value]
        if self._update_data is not None:
            for r in records:
                r.update(self._update_data)
        
        if self._single:
            data = records[0] if records else None
        else:
            data = records
            
        return self._make_response(data)

    def _make_response(self, data):
        class R:
            def execute(self):
                return self
        r = R()
        r.data = data
        return r


def _table_supabase(table: str):
    client = _db()
    if client is not None:
        return client.table(table)
    return _FakeTable(table)

def _robust_execute(query):
    for attempt in range(3):
        try:
            return query.execute()
        except RemoteProtocolError as e:
            if attempt == 2:
                raise e
            time.sleep(0.5 * (attempt + 1))
    return query.execute()


# ── Materials ──────────────────────────────────────────

def list_materials(user_id: str) -> list[dict]:
    client = _db()
    if client is not None:
        try:
            result = _robust_execute(client.table("materials").select("*").eq("user_id", user_id).order("created_at"))
            data = list(reversed(result.data))
            for r in data:
                if r.get("source_type") == "url" and not r.get("url"):
                    r["source_type"] = "topic"
            return data
        except Exception:
            pass
    records = list(_in_memory.get("materials", {}).values())
    data = list(reversed([r for r in records if r.get("user_id") == user_id]))
    for r in data:
        if r.get("source_type") == "url" and not r.get("url"):
            r["source_type"] = "topic"
    return data


def is_title_taken(title: str, exclude_id: Optional[str] = None, user_id: Optional[str] = None) -> bool:
    """Check whether *title* is already used by *user_id* (or globally when user_id is None)."""
    normalized = title.strip().lower()
    if not normalized:
        return False
    try:
        query = _table_supabase("materials").select("id,title")
        if user_id:
            query = query.eq("user_id", user_id)
        result = _robust_execute(query)
        for row in result.data:
            if exclude_id and row.get("id") == exclude_id:
                continue
            if row.get("title", "").strip().lower() == normalized:
                return True
    except Exception:
        pass
    for row in _in_memory.get("materials", {}).values():
        if exclude_id and row.get("id") == exclude_id:
            continue
        if user_id and row.get("user_id") != user_id:
            continue
        if row.get("title", "").strip().lower() == normalized:
            return True
    return False


def create_material(user_id: str, source_type: str, title: str,
                    file_path: Optional[str] = None,
                    url: Optional[str] = None,
                    topic: Optional[str] = None) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    
    # Ensure profile exists to avoid foreign key violations (Key (user_id) not present in table "profiles")
    try:
        # We use a direct check to avoid circular imports or complex logic
        client = _db()
        if client:
            res = client.table("profiles").select("id").eq("id", user_id).execute()
            if not res.data:
                # Create a minimal profile if missing
                client.table("profiles").insert({
                    "id": user_id,
                    "display_name": f"User_{user_id[:8]}",
                    "email": f"{user_id}@placeholder.ai"
                }).execute()
    except Exception as e:
        logger.error(f"Failed to ensure profile for user {user_id}: {e}")

    # Workaround for DB check constraint that restricts source_type to 'pdf' or 'url'
    actual_source_type = source_type
    if source_type == "topic":
        actual_source_type = "url"

    # Auto-resolve duplicate titles PER USER so each user's material list
    original_title = title
    counter = 1
    while is_title_taken(title, user_id=user_id):
        title = f"{original_title} ({counter})"
        counter += 1

    data = {"user_id": user_id, "source_type": actual_source_type, "title": title, "status": "pending",
            "created_at": now, "updated_at": now}
    if file_path:
        data["file_path"] = file_path
    if url:
        data["url"] = url

    # Insert assuming the global constraint on `title` has been replaced with a per-user one
    result = _robust_execute(_table_supabase("materials").insert(data))

    ret_data = result.data[0]
    if ret_data.get("source_type") == "url" and not ret_data.get("url"):
        ret_data["source_type"] = "topic"
    return ret_data


def update_material_status(material_id: str, status: str,
                           error_message: Optional[str] = None):
    data = {"status": status}
    if error_message:
        data["error_message"] = error_message
    _robust_execute(_table_supabase("materials").update(data).eq("id", material_id))


def get_material(material_id: str) -> Optional[dict]:
    if material_id.startswith("temp-"):
        return None
    result = _robust_execute(_table_supabase("materials").select("*").eq("id", material_id))
    if result.data:
        data = result.data[0]
        if data.get("source_type") == "url" and not data.get("url"):
            data["source_type"] = "topic"
        return data
    return None


def rename_material(material_id: str, title: str):
    if material_id.startswith("temp-"):
        return
    _robust_execute(_table_supabase("materials").update({
        "title": title,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", material_id))


def delete_material(material_id: str):
    if material_id.startswith("temp-"):
        return
    # Due to cascading or manual deletion, we delete child records first
    
    # chat_messages don't have material_id, so we must fetch session_ids first
    sessions_res = _robust_execute(_table_supabase("chat_sessions").select("id").eq("material_id", material_id))
    session_ids = [s["id"] for s in sessions_res.data] if sessions_res.data else []

    for sid in session_ids:
        _robust_execute(_table_supabase("chat_messages").delete().eq("session_id", sid))

    _robust_execute(_table_supabase("chat_sessions").delete().eq("material_id", material_id))
    _robust_execute(_table_supabase("summaries").delete().eq("material_id", material_id))
    _robust_execute(_table_supabase("quizzes").delete().eq("material_id", material_id))
    # Delete embeddings before chunks (FK dependency)
    _robust_execute(_table_supabase("material_embeddings").delete().eq("material_id", material_id))
    _robust_execute(_table_supabase("material_chunks").delete().eq("material_id", material_id))
    _robust_execute(_table_supabase("materials").delete().eq("id", material_id))


# ── Material Chunks ────────────────────────────────────

def save_chunks(material_id: str, chunks: list[str]) -> list[str]:
    cleaned_chunks = [c.replace("\x00", "").replace("\u0000", "") for c in chunks if c]
    records = [
        {"material_id": material_id, "chunk_index": i, "content": c}
        for i, c in enumerate(cleaned_chunks)
    ]
    result = _robust_execute(_table_supabase("material_chunks").insert(records))
    return [r["id"] for r in result.data]


def get_chunks(material_id: str) -> list[dict]:
    result = (
        _table_supabase("material_chunks")
        .select("*")
        .eq("material_id", material_id)
        .order("chunk_index")
        .execute()
    )
    return result.data


# ── Summaries ──────────────────────────────────────────

def save_summary(material_id: str, user_id: str, summary: str,
                 time_taken: float, model_name: str = ""):
    data = {
        "material_id": material_id,
        "user_id": user_id,
        "summary": summary,
        "status": "completed",
        "time_taken": time_taken,
        "model_name": model_name,
    }
    # Single atomic upsert — eliminates the race between select→insert/update
    # under concurrent summarization requests (both see no row → both insert → 500).
    client = _db()
    if client is not None:
        _robust_execute(client.table("summaries").upsert(data, on_conflict="material_id"))
    else:
        # Offline / dev fallback: manual check-then-write (no concurrency risk in dev)
        existing = _FakeTable("summaries").select("*").eq("material_id", material_id).execute()
        if existing.data:
            _FakeTable("summaries").update(data).eq("material_id", material_id).execute()
        else:
            _FakeTable("summaries").insert(data).execute()


def get_summary(material_id: str) -> Optional[dict]:
    try:
        result = (
            _table_supabase("summaries")
            .select("*")
            .eq("material_id", material_id)
            .execute()
        )
        if not result or not result.data:
            return None
        # Return the most recent one if duplicates exist
        return result.data[0]
    except Exception:
        return None


# ── Quizzes ────────────────────────────────────────────

def save_quiz(user_id: str, material_id: Optional[str], source_type: str,
              difficulty: str, mcq_count: int, tf_count: int,
              quiz_data: dict, model_name: str = "") -> dict:
    data = {
        "user_id": user_id,
        "source_type": source_type,
        "difficulty": difficulty,
        "mcq_count": mcq_count,
        "tf_count": tf_count,
        "quiz_data": quiz_data,
        "status": "completed",
        "model_name": model_name,
    }
    if material_id:
        data["material_id"] = material_id
    result = _table_supabase("quizzes").insert(data).execute()
    return result.data[0]


def get_quizzes(material_id: Optional[str] = None, user_id: Optional[str] = None) -> list[dict]:
    query = _table_supabase("quizzes").select("*")
    if user_id:
        query = query.eq("user_id", user_id)
    if material_id:
        query = query.eq("material_id", material_id)
    result = _robust_execute(query)
    return result.data


# ── Users (maps to Supabase `profiles` table) ─────────

def _map_profile(profile: dict) -> dict:
    today = _get_today_date_str()
    used = profile.get("daily_requests", 0) if profile.get("last_request_date") == today else 0
    return {
        "id": profile["id"],
        "name": profile.get("display_name", ""),
        "email": profile.get("email", ""),
        "avatar": profile.get("avatar_url", ""),
        "theme": profile.get("theme", "system"),
        "usage": {
            "used": used,
            "limit": 20,
            "remaining": max(0, 20 - used)
        }
    }


def create_user(name: str, email: str, password: str, user_id: Optional[str] = None) -> dict:
    existing = get_user_by_email(email)
    if existing:
        return existing
    
    data = {"display_name": name, "email": email}
    if user_id:
        data["id"] = user_id
        
    try:
        result = _table_supabase("profiles").insert(data).execute()
        # insert().execute().data is always a list
        return _map_profile(result.data[0] if isinstance(result.data, list) and result.data else result.data)
    except Exception:
        result = _FakeTable("profiles").insert(data).execute()
        return _map_profile(result.data[0] if isinstance(result.data, list) and result.data else result.data)


def get_user_by_email(email: str) -> Optional[dict]:
    try:
        # Standard select instead of maybe_single to be more robust against 406 errors
        result = _table_supabase("profiles").select("*").eq("email", email).execute()
        if result.data:
            return _map_profile(result.data[0])
    except Exception:
        pass
    fake = _FakeTable("profiles")
    result = fake.select("*").eq("email", email).execute()
    if result.data:
        return _map_profile(result.data[0])
    return None


def get_user_by_id(user_id: str) -> Optional[dict]:
    try:
        # Standard select instead of maybe_single to be more robust against 406 errors
        result = _table_supabase("profiles").select("*").eq("id", user_id).execute()
        if result.data:
            return _map_profile(result.data[0])
    except Exception:
        pass
    fake = _FakeTable("profiles")
    result = fake.select("*").eq("id", user_id).execute()
    if result.data:
        return _map_profile(result.data[0])
    return None


def update_user_profile(user_id: str, name: Optional[str] = None, avatar_url: Optional[str] = None, theme: Optional[str] = None) -> dict:
    """
    Updates the user profile in the database.
    """
    data = {}
    if name is not None:
        data["display_name"] = name
    if avatar_url is not None:
        data["avatar_url"] = avatar_url
    if theme is not None:
        data["theme"] = theme
    
    if not data:
        user = get_user_by_id(user_id)
        if not user:
            raise ValueError("User not found")
        return user

    try:
        result = _robust_execute(_table_supabase("profiles").update(data).eq("id", user_id))
        if result.data:
            return _map_profile(result.data[0])
    except Exception:
        pass
    
    # Fake fallback
    store = _in_memory.get("profiles", {})
    if user_id in store:
        store[user_id].update(data)
        return _map_profile(store[user_id])
    
    raise ValueError("User not found")


# ── Chat Messages (persistent) ──────────────────────────

def save_chat_messages(material_id: str, user_id: str, messages: list[dict]):
    existing = get_chat_messages(material_id)
    data = {
        "material_id": material_id,
        "user_id": user_id,
        "messages": messages,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    tbl = _table_supabase("chat_messages")
    if existing:
        tbl.update(data).eq("material_id", material_id).execute()
    else:
        data["created_at"] = data["updated_at"]
        tbl.insert(data).execute()


def get_chat_messages(material_id: str) -> list[dict]:
    result = (
        _table_supabase("chat_messages")
        .select("*")
        .eq("material_id", material_id)
        
        .execute()
    )
    if isinstance(result.data, list):
        data = result.data[0] if result.data else {}
    else:
        data = result.data or {}
    return data.get("messages", [])


# ── Quiz Results ────────────────────────────────────────

def save_quiz_result(quiz_id: str, user_id: str, result_data: dict):
    data = {
        "quiz_id": quiz_id,
        "user_id": user_id,
        "score":   int(result_data.get("score", 0)),
        "total":   int(result_data.get("total", 0)),
        "results": result_data,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _table_supabase("quiz_attempts").insert(data).execute()


def get_quiz_results(quiz_id: str) -> list[dict]:
    result = (
        _table_supabase("quiz_attempts")
        .select("*")
        .eq("quiz_id", quiz_id)
        .order("created_at", desc=False)
        .execute()
    )
    return result.data


# ── Chat Sessions (proper DB structure) ───────────────

def create_chat_session(user_id: str, material_id: str, title: str = "New Chat") -> dict:
    now = datetime.now(timezone.utc).isoformat()
    data = {
        "user_id": user_id,
        "material_id": material_id,
        "title": title,
        "created_at": now,
        "updated_at": now,
    }
    result = _table_supabase("chat_sessions").insert(data).execute()
    return result.data[0]


def list_chat_sessions(material_id: str, user_id: str) -> list[dict]:
    result = (
        _table_supabase("chat_sessions")
        .select("*")
        .eq("material_id", material_id)
        .eq("user_id", user_id)
        .order("created_at")
        .execute()
    )
    return result.data


def get_chat_session(session_id: str) -> Optional[dict]:
    result = _table_supabase("chat_sessions").select("*").eq("id", session_id).execute()
    if isinstance(result.data, list):
        return result.data[0] if result.data else None
    return result.data or None


def rename_chat_session(session_id: str, title: str):
    _table_supabase("chat_sessions").update({
        "title": title,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", session_id).execute()


def delete_chat_session(session_id: str):
    _robust_execute(_table_supabase("chat_messages").delete().eq("session_id", session_id))
    _robust_execute(_table_supabase("chat_sessions").delete().eq("id", session_id))


def append_session_message(session_id: str, role: str, content: str) -> dict:
    data = {
        "session_id": session_id,
        "role": role,
        "content": content,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    result = _table_supabase("chat_messages").insert(data).execute()
    # Update session's updated_at
    _table_supabase("chat_sessions").update({
        "updated_at": data["created_at"]
    }).eq("id", session_id).execute()
    return result.data[0]


def get_session_messages(session_id: str) -> list[dict]:
    result = (
        _table_supabase("chat_messages")
        .select("*")
        .eq("session_id", session_id)
        .order("created_at")
        .execute()
    )
    return result.data


# ── Conversation Memory — Redis-backed, Python-dict fallback ─────────────────

import uuid as _uuid

# Fallback in-process dict for when Redis is unavailable
_memories: dict[str, ConversationBufferMemory] = {}


def _mem_redis_key(memory_id: str) -> str:
    return f"{MEMORY_KEY_PREFIX}{memory_id}"


def _load_memory_from_messages(messages: list[dict]) -> ConversationBufferWindowMemory:
    """Build a fresh ConversationBufferWindowMemory from a flat message list."""
    mem = ConversationBufferWindowMemory(
        input_key="input", memory_key="chat_history", return_messages=True, k=5
    )
    for i in range(0, len(messages) - 1, 2):
        user_msg = messages[i]
        ai_msg = messages[i + 1] if i + 1 < len(messages) else None
        if user_msg.get("role") == "user" and ai_msg and ai_msg.get("role") == "assistant":
            mem.save_context(
                {"input": user_msg["content"]},
                {"output": ai_msg["content"]},
            )
    return mem


def get_or_create_memory(memory_id: Optional[str] = None, seed_messages: list[dict] | None = None):
    """
    Get or create a ConversationBufferWindowMemory.

    Redis path (fast):
        Checks `mem:{memory_id}` in Redis first.  If present, deserialises the
        cached message list and builds memory from it — no Supabase query needed.
        TTL is refreshed on each access so active conversations stay warm.

    Fallback path (Supabase seed / in-process dict):
        Falls back to `seed_messages` from Supabase (as before) and caches the
        result in Redis so the next call skips Supabase entirely.
    """
    mid = memory_id or str(_uuid.uuid4())
    r = get_redis()

    # ── Redis path ────────────────────────────────────────────────────────────
    if r is not None:
        try:
            rkey = _mem_redis_key(mid)
            raw = r.get(rkey)
            if raw:
                cached_msgs: list[dict] = json.loads(raw)
                mem = _load_memory_from_messages(cached_msgs)
                r.expire(rkey, MEMORY_REDIS_TTL)   # refresh TTL on each use
                return mem, mid
        except Exception as e:
            logger.warning("Redis get_or_create_memory read failed: %s", e)

    # ── Seed from DB messages (Supabase / provided list) ─────────────────────
    source_msgs: list[dict] = seed_messages or []
    mem = _load_memory_from_messages(source_msgs)

    # Cache in Redis so next request skips Supabase
    if r is not None and source_msgs:
        try:
            rkey = _mem_redis_key(mid)
            r.set(rkey, json.dumps(source_msgs), ex=MEMORY_REDIS_TTL)
        except Exception as e:
            logger.warning("Redis get_or_create_memory write failed: %s", e)

    # Also keep the in-process fallback dict warm
    _memories[mid] = mem
    return mem, mid


def append_memory_message(memory_id: str, role: str, content: str) -> None:
    """
    Append a single message to the Redis-cached message list for a session.

    Called after each AI turn so Redis stays in sync without a full Supabase
    round-trip.  Silently no-ops if Redis is unavailable.
    """
    r = get_redis()
    if r is None:
        return
    try:
        rkey = _mem_redis_key(memory_id)
        raw = r.get(rkey)
        msgs: list[dict] = json.loads(raw) if raw else []
        msgs.append({"role": role, "content": content})
        # Keep only the last N messages to cap memory usage
        msgs = msgs[-MEMORY_MAX_MESSAGES:]
        r.set(rkey, json.dumps(msgs), ex=MEMORY_REDIS_TTL)
    except Exception as e:
        logger.warning("append_memory_message Redis failed: %s", e)


def check_daily_limit(user_id: str, email: Optional[str] = None, limit: int = 20) -> bool:
    """
    Checks if the user is under the daily limit. Returns True if allowed, False if exceeded.
    Does NOT increment the count.

    Redis path: atomic GET on `rate:{user_id}:{date}` (<5 ms, no Supabase hit).
    Fallback: existing Supabase profiles query.
    """
    if email and email in ADMIN_EMAILS:
        return True

    today = _get_today_date_str()

    # ── Redis path ────────────────────────────────────────────────────────────
    r = get_redis()
    if r is not None:
        try:
            rkey = f"{DAILY_RATE_LIMIT_KEY_PREFIX}{user_id}:{today}"
            val = r.get(rkey)
            count = int(val) if val is not None else 0
            return count < limit
        except Exception as e:
            logger.warning("Redis check_daily_limit failed: %s — falling back to Supabase", e)

    # ── Supabase fallback ─────────────────────────────────────────────────────
    try:
        result = _robust_execute(
            _table_supabase("profiles")
            .select("daily_requests, last_request_date")
            .eq("id", user_id)
        )
        if not result.data:
            return True
        profile = result.data[0] if result.data else None
        if not profile:
            return True

        last_date = profile.get("last_request_date")
        count = profile.get("daily_requests", 0) or 0
        if last_date != today:
            count = 0
        return count < limit
    except Exception as e:
        logger.error(f"Rate limit check failed: {e}")
        return True


def increment_daily_usage(user_id: str) -> None:
    """
    Increments the daily request count for the user.

    Redis path: atomic INCR + EXPIRE on `rate:{user_id}:{date}` (microseconds).
    Always also updates Supabase so the dashboard & DB stay in sync.
    """
    today = _get_today_date_str()

    # ── Redis path — atomic INCR ──────────────────────────────────────────────
    r = get_redis()
    if r is not None:
        try:
            rkey = f"{DAILY_RATE_LIMIT_KEY_PREFIX}{user_id}:{today}"
            pipe = r.pipeline()
            pipe.incr(rkey)
            pipe.expire(rkey, DAILY_RATE_LIMIT_REDIS_TTL)   # auto-expire at next calendar day
            pipe.execute()
        except Exception as e:
            logger.warning("Redis increment_daily_usage failed: %s", e)

    # ── Supabase — keep DB in sync for audit / dashboard ─────────────────────
    try:
        result = _robust_execute(
            _table_supabase("profiles")
            .select("daily_requests, last_request_date")
            .eq("id", user_id)
        )
        if not result.data:
            return
        profile = result.data[0] if result.data else None
        if not profile:
            return

        last_date = profile.get("last_request_date")
        count = profile.get("daily_requests", 0) or 0
        if last_date != today:
            count = 0

        _robust_execute(
            _table_supabase("profiles")
            .update({"daily_requests": count + 1, "last_request_date": today})
            .eq("id", user_id)
        )
    except Exception as e:
        logger.error(f"Failed to increment daily usage: {e}")


def check_and_increment_daily_limit(user_id: str, email: Optional[str] = None, limit: int = 20) -> bool:
    """
    Check limit and increment if allowed. Deprecated/kept for compatibility.
    """
    allowed = check_daily_limit(user_id, email, limit)
    if allowed and not (email and email in ADMIN_EMAILS):
        increment_daily_usage(user_id)
    return allowed


def get_usage(user_id: str) -> dict:
    """
    Returns current usage for a user.

    Redis path: read rate counter directly (<5 ms).
    Fallback: Supabase profiles query.
    """
    today = _get_today_date_str()

    # ── Redis path ────────────────────────────────────────────────────────────
    r = get_redis()
    if r is not None:
        try:
            rkey = f"{DAILY_RATE_LIMIT_KEY_PREFIX}{user_id}:{today}"
            val = r.get(rkey)
            used = int(val) if val is not None else 0
            return {"used": used, "limit": 20, "remaining": max(0, 20 - used)}
        except Exception as e:
            logger.warning("Redis get_usage failed: %s — falling back to Supabase", e)

    # ── Supabase fallback ─────────────────────────────────────────────────────
    try:
        result = _robust_execute(
            _table_supabase("profiles")
            .select("daily_requests, last_request_date")
            .eq("id", user_id)
        )
        if not result.data:
            return {"used": 0, "limit": 20, "remaining": 20}

        profile = result.data[0]
        if not profile:
            return {"used": 0, "limit": 20, "remaining": 20}

        used = profile.get("daily_requests", 0) if profile.get("last_request_date") == today else 0
        return {
            "used": used,
            "limit": 20,
            "remaining": max(0, 20 - used)
        }
    except Exception:
        return {"used": 0, "limit": 20, "remaining": 20}


def delete_user_data(user_id: str):
    """
    Deletes all data associated with a user.
    """
    # 1. Get all materials for this user and delete them one by one to ensure cascading deletes
    materials_res = _robust_execute(_table_supabase("materials").select("id").eq("user_id", user_id))
    material_ids = [m["id"] for m in materials_res.data] if materials_res.data else []
    for mid in material_ids:
        delete_material(mid)
    
    # 2. Delete any quizzes that might not be tied to a specific material
    _robust_execute(_table_supabase("quizzes").delete().eq("user_id", user_id))
    
    # 3. Delete the user profile
    _robust_execute(_table_supabase("profiles").delete().eq("id", user_id))
