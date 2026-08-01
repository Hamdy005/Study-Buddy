"""
Comprehensive tests for JWT & Refresh Token authentication endpoints:

  POST /api/auth/session — Exchange Supabase JWT for custom JWT + HttpOnly refresh cookie
  POST /api/auth/refresh — Silent token rotation via HttpOnly cookie
  POST /api/auth/logout  — Server-side refresh token revocation + cookie clear
  GET  /api/auth/me       — Protected profile retrieval

All DB writes and external Supabase calls are mocked.
"""

import time
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.main import app
from src.dependencies import get_current_user
from src.auth.jwt_utils import (
    create_access_token,
    create_refresh_token,
    decode_access_token,
    hash_token,
)
from src.auth.constants import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    REFRESH_COOKIE_NAME,
)
from tests.conftest import (
    FAKE_USER_ID,
    FAKE_EMAIL,
    FAKE_PROFILE,
    fake_sb_user,
    future_expires_at,
    past_expires_at,
)

# Endpoint URLs
SESSION_URL = "/api/auth/session"
REFRESH_URL = "/api/auth/refresh"
LOGOUT_URL  = "/api/auth/logout"
ME_URL      = "/api/auth/me"


# ── Cookie Helper ─────────────────────────────────────────────────────────────

def _get_cookie_attrs(response, cookie_name: str = REFRESH_COOKIE_NAME) -> dict[str, str | bool]:
    """Extract and parse Set-Cookie attributes for a specific cookie name."""
    headers = [v for k, v in response.headers.multi_items() if k.lower() == "set-cookie"]
    cookie_str = next((h for h in headers if h.startswith(f"{cookie_name}=")), None)
    if not cookie_str:
        return {}
    parts = [p.strip() for p in cookie_str.split(";")]
    attrs: dict[str, str | bool] = {}
    for part in parts:
        if "=" in part:
            k, v = part.split("=", 1)
            attrs[k.strip().lower()] = v.strip()
        else:
            attrs[part.lower()] = True
    return attrs


# ═══════════════════════════════════════════════════════════════════════════════
# 1. JWT & HASH UNIT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestJwtUtils:
    def test_jwt_creation_and_decoding(self):
        token = create_access_token(FAKE_USER_ID, FAKE_EMAIL)
        payload = decode_access_token(token)
        assert payload["sub"] == FAKE_USER_ID
        assert payload["email"] == FAKE_EMAIL

    def test_jwt_expiration(self):
        before = time.time()
        token = create_access_token(FAKE_USER_ID, FAKE_EMAIL)
        payload = decode_access_token(token)
        expected = before + ACCESS_TOKEN_EXPIRE_MINUTES * 60
        assert abs(payload["exp"] - expected) < 5

    def test_jwt_tampering(self):
        import jwt as _jwt
        token = create_access_token(FAKE_USER_ID, FAKE_EMAIL)
        with pytest.raises(_jwt.InvalidTokenError):
            decode_access_token(token[:-4] + "XXXX")

    def test_hash_properties(self):
        raw = create_refresh_token()
        assert hash_token(raw) == hash_token(raw)
        assert len(hash_token(raw)) == 64
        assert create_refresh_token() != create_refresh_token()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SESSION EXCHANGE (POST /api/auth/session)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSessionExchange:
    @pytest.mark.anyio
    async def test_exchange_success(self, client: AsyncClient):
        """Valid Supabase token returns JWT in body, HttpOnly refresh cookie, and isolates tokens."""
        sb_user = fake_sb_user()
        raw_refresh = create_refresh_token()

        with (
            patch("src.auth.routes._verify_token_cached", return_value=sb_user),
            patch("src.auth.routes.get_user_by_id", return_value=FAKE_PROFILE),
            patch("src.auth.routes.save_refresh_token") as mock_save,
            patch("src.auth.routes.create_refresh_token", return_value=raw_refresh),
        ):
            response = await client.post(SESSION_URL, headers={"X-Auth-Token": "sb-jwt"})

        assert response.status_code == 200
        data = response.json()

        # JWT in body
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert decode_access_token(data["access_token"])["sub"] == FAKE_USER_ID

        # Refresh cookie attributes & security
        attrs = _get_cookie_attrs(response)
        assert attrs.get("httponly") is True
        assert "samesite" in attrs
        assert REFRESH_COOKIE_NAME in response.cookies

        # Isolation checks: JWT not in cookie, raw refresh token not in JSON body
        all_set_cookies = " ".join([v for k, v in response.headers.multi_items() if k.lower() == "set-cookie"])
        assert data["access_token"] not in all_set_cookies
        assert raw_refresh not in response.text

        mock_save.assert_called_once()

    @pytest.mark.anyio
    async def test_exchange_new_user_profile(self, client: AsyncClient):
        """Creates user profile automatically if user doesn't exist yet."""
        sb_user = fake_sb_user()
        with (
            patch("src.auth.routes._verify_token_cached", return_value=sb_user),
            patch("src.auth.routes.get_user_by_id", return_value=None),
            patch("src.auth.routes.create_user", return_value=FAKE_PROFILE) as mock_create,
            patch("src.auth.routes.save_refresh_token"),
        ):
            response = await client.post(SESSION_URL, headers={"X-Auth-Token": "sb-jwt"})

        assert response.status_code == 200
        mock_create.assert_called_once()

    @pytest.mark.anyio
    async def test_exchange_auth_headers(self, client: AsyncClient):
        """Accepts Authorization: Bearer header format."""
        sb_user = fake_sb_user()
        with (
            patch("src.auth.routes._verify_token_cached", return_value=sb_user),
            patch("src.auth.routes.get_user_by_id", return_value=FAKE_PROFILE),
            patch("src.auth.routes.save_refresh_token"),
        ):
            response = await client.post(SESSION_URL, headers={"Authorization": "Bearer sb-jwt"})

        assert response.status_code == 200

    @pytest.mark.anyio
    async def test_exchange_unauthorized(self, client: AsyncClient):
        """Missing or invalid Supabase token returns 401."""
        r1 = await client.post(SESSION_URL)
        assert r1.status_code == 401

        with patch("src.auth.routes._verify_token_cached", side_effect=Exception("Invalid")):
            r2 = await client.post(SESSION_URL, headers={"X-Auth-Token": "bad-token"})
        assert r2.status_code == 401


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SILENT TOKEN REFRESH (POST /api/auth/refresh)
# ═══════════════════════════════════════════════════════════════════════════════

class TestRefreshSession:
    @pytest.mark.anyio
    async def test_refresh_success(
        self, client: AsyncClient, raw_refresh_token: str, valid_refresh_row: dict
    ):
        """Valid refresh cookie rotates refresh token, issues new JWT, and preserves HttpOnly flags."""
        revoke_mock = MagicMock()
        save_mock = MagicMock()

        with (
            patch("src.auth.routes.get_refresh_token", return_value=valid_refresh_row),
            patch("src.auth.routes.get_user_by_id", return_value=FAKE_PROFILE),
            patch("src.auth.routes.revoke_refresh_token", revoke_mock),
            patch("src.auth.routes.save_refresh_token", save_mock),
        ):
            response = await client.post(REFRESH_URL, cookies={REFRESH_COOKIE_NAME: raw_refresh_token})

        assert response.status_code == 200
        data = response.json()

        # New JWT
        assert "access_token" in data
        assert decode_access_token(data["access_token"])["sub"] == FAKE_USER_ID

        # Rotated HttpOnly refresh cookie
        attrs = _get_cookie_attrs(response)
        assert attrs.get("httponly") is True

        # Stateful rotation in DB
        revoke_mock.assert_called_once_with(hash_token(raw_refresh_token))
        save_mock.assert_called_once()

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "db_row, expected_status",
        [
            (None, 401),                                                        # Unknown token
            ({"user_id": FAKE_USER_ID, "revoked": True, "expires_at": future_expires_at(30)}, 401),  # Revoked token
            ({"user_id": FAKE_USER_ID, "revoked": False, "expires_at": past_expires_at(60)}, 401),   # Expired token
        ],
    )
    async def test_refresh_invalid_tokens(
        self, client: AsyncClient, raw_refresh_token: str, db_row, expected_status
    ):
        """Invalid, revoked, or expired refresh tokens return 401."""
        with patch("src.auth.routes.get_refresh_token", return_value=db_row):
            response = await client.post(REFRESH_URL, cookies={REFRESH_COOKIE_NAME: raw_refresh_token})
        assert response.status_code == expected_status

    @pytest.mark.anyio
    async def test_refresh_missing_cookie(self, client: AsyncClient):
        """Missing refresh cookie returns 401."""
        response = await client.post(REFRESH_URL)
        assert response.status_code == 401


# ═══════════════════════════════════════════════════════════════════════════════
# 4. LOGOUT (POST /api/auth/logout)
# ═══════════════════════════════════════════════════════════════════════════════

class TestLogout:
    @pytest.mark.anyio
    async def test_logout(self, client: AsyncClient, raw_refresh_token: str):
        """Revokes token in DB and clears HttpOnly cookie."""
        revoke_mock = MagicMock()
        with patch("src.auth.routes.revoke_refresh_token", revoke_mock):
            response = await client.post(LOGOUT_URL, cookies={REFRESH_COOKIE_NAME: raw_refresh_token})

        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        revoke_mock.assert_called_once_with(hash_token(raw_refresh_token))

    @pytest.mark.anyio
    async def test_logout_idempotent(self, client: AsyncClient):
        """Logout without cookie or repeated logout succeeds silently."""
        r1 = await client.post(LOGOUT_URL)
        assert r1.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════════
# 5. PROTECTED ROUTE (GET /api/auth/me)
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetMe:
    @pytest.mark.anyio
    async def test_get_me_success(self, client: AsyncClient, raw_access_token: str):
        """Valid JWT returns current user profile."""
        with patch("src.auth.routes.get_user_by_id", return_value=FAKE_PROFILE):
            response = await client.get(ME_URL, headers={"X-Auth-Token": raw_access_token})

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["user"]["id"] == FAKE_USER_ID

    @pytest.mark.anyio
    async def test_get_me_unauthorized(self):
        """Missing or tampered JWT returns 401."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            r1 = await ac.get(ME_URL)
            assert r1.status_code == 401

            token = create_access_token(FAKE_USER_ID, FAKE_EMAIL)
            r2 = await ac.get(ME_URL, headers={"X-Auth-Token": token[:-6] + "BADJWT"})
            assert r2.status_code == 401


# ═══════════════════════════════════════════════════════════════════════════════
# 6. COMPLETE AUTH LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════════════

class TestFullAuthFlow:
    @pytest.mark.anyio
    async def test_complete_lifecycle(self, client: AsyncClient):
        """Tests exchange -> profile fetch -> token refresh -> logout -> blocked refresh."""
        sb_user = fake_sb_user()
        raw_refresh = create_refresh_token()

        # 1. Exchange
        with (
            patch("src.auth.routes._verify_token_cached", return_value=sb_user),
            patch("src.auth.routes.get_user_by_id", return_value=FAKE_PROFILE),
            patch("src.auth.routes.save_refresh_token"),
            patch("src.auth.routes.create_refresh_token", return_value=raw_refresh),
        ):
            r1 = await client.post(SESSION_URL, headers={"X-Auth-Token": "sb-jwt"})

        assert r1.status_code == 200
        access_token = r1.json()["access_token"]
        cookie = r1.cookies.get(REFRESH_COOKIE_NAME)

        # 2. Get Me using JWT
        with patch("src.auth.routes.get_user_by_id", return_value=FAKE_PROFILE):
            r2 = await client.get(ME_URL, headers={"X-Auth-Token": access_token})
        assert r2.status_code == 200

        # 3. Refresh
        new_refresh = create_refresh_token()
        valid_row = {"user_id": FAKE_USER_ID, "token_hash": hash_token(cookie), "expires_at": future_expires_at(30), "revoked": False}
        with (
            patch("src.auth.routes.get_refresh_token", return_value=valid_row),
            patch("src.auth.routes.get_user_by_id", return_value=FAKE_PROFILE),
            patch("src.auth.routes.revoke_refresh_token"),
            patch("src.auth.routes.save_refresh_token"),
            patch("src.auth.routes.create_refresh_token", return_value=new_refresh),
        ):
            r3 = await client.post(REFRESH_URL, cookies={REFRESH_COOKIE_NAME: cookie})
        assert r3.status_code == 200
        rotated_cookie = r3.cookies.get(REFRESH_COOKIE_NAME)

        # 4. Logout
        with patch("src.auth.routes.revoke_refresh_token"):
            r4 = await client.post(LOGOUT_URL, cookies={REFRESH_COOKIE_NAME: rotated_cookie})
        assert r4.status_code == 200

        # 5. Refresh after logout fails
        with patch("src.auth.routes.get_refresh_token", return_value=None):
            r5 = await client.post(REFRESH_URL, cookies={REFRESH_COOKIE_NAME: rotated_cookie})
        assert r5.status_code == 401
