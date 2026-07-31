import time
import pytest
from fastapi import HTTPException
from src.auth.rate_limiter import (
    check_and_record_email_rate_limit,
    enforce_email_rate_limit,
    get_email_rate_limit_status,
    _cleanup_store,
    _store,
    _cooldowns,
)

def test_email_rate_limiter_cooldown_blocks_immediate_retry():
    email = "test@example.com"
    action = "email_verification"

    # 1st send succeeds
    allowed, remaining, retry_after = check_and_record_email_rate_limit(action, email)
    assert allowed is True
    assert remaining == 2
    assert retry_after == 0

    # Immediate 2nd send should be blocked by 60s cooldown
    allowed_2, remaining_2, retry_after_2 = check_and_record_email_rate_limit(action, email)
    assert allowed_2 is False
    assert remaining_2 == 2  # still 2 remaining out of 3, but blocked by cooldown
    assert 1 <= retry_after_2 <= 60


def test_email_rate_limiter_allows_retry_after_cooldown_expires():
    email = "test@example.com"
    action = "forgot_password"

    # 1st attempt
    check_and_record_email_rate_limit(action, email)

    # Fast forward cooldown in store
    _cooldowns[action][email] = time.time() - 1

    # 2nd attempt should now succeed
    allowed, remaining, _ = check_and_record_email_rate_limit(action, email)
    assert allowed is True
    assert remaining == 1


def test_cleanup_store_removes_expired_records():
    email = "old@example.com"
    action = "email_verification"

    # Insert old timestamp (2 hours ago) and expired cooldown
    _store[action][email] = [time.time() - 7200]
    _cooldowns[action][email] = time.time() - 3600

    # Run cleanup
    _cleanup_store()

    # Expired entries should be purged from memory
    assert email not in _store[action]
    assert email not in _cooldowns[action]


def test_different_actions_and_emails_have_separate_limits():
    email1 = "user1@example.com"
    email2 = "user2@example.com"

    check_and_record_email_rate_limit("email_verification", email1)

    # email1 for change_password_confirmation should still be allowed (separate action cooldown)
    allowed, remaining, _ = check_and_record_email_rate_limit("change_password_confirmation", email1)
    assert allowed is True

    # email2 for email_verification should also be allowed (separate user cooldown)
    allowed, remaining, _ = check_and_record_email_rate_limit("email_verification", email2)
    assert allowed is True


def test_get_email_rate_limit_status():
    email = "status@example.com"
    action = "change_password_confirmation"

    status = get_email_rate_limit_status(action, email)
    assert status["used"] == 0
    assert status["remaining"] == 3
    assert status["cooldown_remaining_seconds"] == 0

    enforce_email_rate_limit(action, email)
    status = get_email_rate_limit_status(action, email)
    assert status["used"] == 1
    assert status["remaining"] == 2
    assert status["cooldown_remaining_seconds"] > 0


def test_email_verification_three_sends_with_cooldown_and_fourth_fails():
    email = "limit_test@example.com"
    action = "email_verification"

    # Send 1
    allowed, remaining, _ = check_and_record_email_rate_limit(action, email)
    assert allowed is True
    assert remaining == 2

    # Fast-forward 60s cooldown for Send 2
    _cooldowns[action][email] = time.time() - 1

    # Send 2
    allowed, remaining, _ = check_and_record_email_rate_limit(action, email)
    assert allowed is True
    assert remaining == 1

    # Fast-forward 60s cooldown for Send 3
    _cooldowns[action][email] = time.time() - 1

    # Send 3
    allowed, remaining, _ = check_and_record_email_rate_limit(action, email)
    assert allowed is True
    assert remaining == 0

    # Fast-forward 60s cooldown before Send 4
    _cooldowns[action][email] = time.time() - 1

    # Send 4 (within the 1-hour window) must fail because 3/3 attempts were used
    with pytest.raises(HTTPException) as exc_info:
        enforce_email_rate_limit(action, email)

    assert exc_info.value.status_code == 429
    assert "Rate limit exceeded" in exc_info.value.detail or "Maximum 3 emails per hour" in exc_info.value.detail

