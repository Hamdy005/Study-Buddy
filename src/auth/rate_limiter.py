import time
from collections import defaultdict
from typing import Dict, List, Tuple
from threading import Timer, Lock
from fastapi import HTTPException

# In-Memory Email Rate Limits per action (3 emails per 1 hour window, 60s per-send cooldown)
EMAIL_ACTION_LIMITS = {
    "email_verification": {"limit": 3, "window_seconds": 3600, "cooldown_seconds": 60},
    "forgot_password": {"limit": 3, "window_seconds": 3600, "cooldown_seconds": 60},
    "change_password_confirmation": {"limit": 3, "window_seconds": 3600, "cooldown_seconds": 60},
}

# In-memory stores & thread lock
_store: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
_cooldowns: Dict[str, Dict[str, float]] = defaultdict(dict)
_lock = Lock()


def _cleanup_store():
    """Periodically purges expired entries from _store and _cooldowns to prevent memory leaks."""
    now = time.time()
    with _lock:
        for action, users in list(_store.items()):
            config = EMAIL_ACTION_LIMITS.get(action, {})
            window = config.get("window_seconds", 3600)
            cutoff = now - window
            for identifier, timestamps in list(users.items()):
                filtered = [ts for ts in timestamps if ts > cutoff]
                if filtered:
                    _store[action][identifier] = filtered
                else:
                    del _store[action][identifier]

        for action, users in list(_cooldowns.items()):
            for identifier, until in list(users.items()):
                if now >= until:
                    del _cooldowns[action][identifier]

    # Schedule next cleanup run in 30 minutes
    t = Timer(1800, _cleanup_store)
    t.daemon = True
    t.start()


# Start background cleanup timer (daemon thread so it doesn't block process exit)
_cleanup_timer = Timer(1800, _cleanup_store)
_cleanup_timer.daemon = True
_cleanup_timer.start()


def check_and_record_email_rate_limit(
    action: str, identifier: str, cooldown_seconds: int = 60
) -> Tuple[bool, int, int]:
    """
    Check and record an in-memory rate limit and per-send cooldown for a specific email action.

    :param action: 'email_verification', 'forgot_password', or 'change_password_confirmation'
    :param identifier: Email address or user ID
    :param cooldown_seconds: Minimum seconds between individual sends (default 60s)
    :return: (is_allowed, remaining_attempts, retry_after_seconds)
    """
    if action not in EMAIL_ACTION_LIMITS:
        raise ValueError(f"Unknown action '{action}'. Allowed actions: {list(EMAIL_ACTION_LIMITS.keys())}")

    config = EMAIL_ACTION_LIMITS[action]
    limit = config["limit"]
    window = config["window_seconds"]
    cooldown = config.get("cooldown_seconds", cooldown_seconds)

    now = time.time()
    clean_id = identifier.lower().strip()

    with _lock:
        # 1. Check per-send cooldown first
        cooldown_until = _cooldowns[action].get(clean_id, 0)
        if now < cooldown_until:
            wait = int(cooldown_until - now) + 1
            cutoff = now - window
            timestamps = [ts for ts in _store[action][clean_id] if ts > cutoff]
            remaining = max(0, limit - len(timestamps))
            return False, remaining, max(1, wait)

        # 2. Check 1-hour window limit
        cutoff = now - window
        timestamps = [ts for ts in _store[action][clean_id] if ts > cutoff]

        if len(timestamps) >= limit:
            oldest = timestamps[0]
            retry_after = int(oldest + window - now) + 1
            _store[action][clean_id] = timestamps
            return False, 0, max(1, retry_after)

        # Allowed: record timestamp and set next cooldown
        timestamps.append(now)
        _store[action][clean_id] = timestamps
        _cooldowns[action][clean_id] = now + cooldown

        remaining = limit - len(timestamps)
        return True, remaining, 0


def enforce_email_rate_limit(action: str, identifier: str) -> int:
    """
    Enforces in-memory rate limit and per-send cooldown. Raises HTTP 429 if violated.

    :return: Number of remaining attempts in the current window.
    """
    allowed, remaining, retry_after = check_and_record_email_rate_limit(action, identifier)
    if not allowed:
        action_name = action.replace("_", " ").title()
        if retry_after <= 60:
            msg = f"Please wait {retry_after} seconds before requesting another {action_name} email."
        else:
            minutes = (retry_after + 59) // 60
            msg = f"Rate limit exceeded for {action_name}. Maximum 3 emails per hour allowed. Please try again in {minutes} minute(s)."

        raise HTTPException(
            status_code=429,
            detail=msg,
            headers={"Retry-After": str(retry_after)}
        )
    return remaining


def get_email_rate_limit_status(action: str, identifier: str) -> Dict[str, int]:
    """
    Get current usage and cooldown status without recording a new attempt.
    """
    if action not in EMAIL_ACTION_LIMITS:
        raise ValueError(f"Unknown action '{action}'. Allowed actions: {list(EMAIL_ACTION_LIMITS.keys())}")

    config = EMAIL_ACTION_LIMITS[action]
    limit = config["limit"]
    window = config["window_seconds"]
    cooldown = config.get("cooldown_seconds", 60)

    now = time.time()
    cutoff = now - window
    clean_id = identifier.lower().strip()

    with _lock:
        timestamps = [ts for ts in _store[action][clean_id] if ts > cutoff]
        used = len(timestamps)
        remaining = max(0, limit - used)

        cooldown_until = _cooldowns[action].get(clean_id, 0)
        cooldown_remaining = max(0, int(cooldown_until - now) + 1) if now < cooldown_until else 0

        window_retry_after = 0
        if used >= limit and timestamps:
            window_retry_after = max(1, int(timestamps[0] + window - now) + 1)

        retry_after = max(cooldown_remaining, window_retry_after)

    return {
        "limit": limit,
        "used": used,
        "remaining": remaining,
        "window_seconds": window,
        "cooldown_seconds": cooldown,
        "cooldown_remaining_seconds": cooldown_remaining,
        "retry_after_seconds": retry_after
    }
