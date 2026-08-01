"""
Simulation Test: Hugging Face Spaces Deployment & Cross-Origin Auth Flow

Simulates the exact production deployment environment on Hugging Face Spaces:
  • Behind Cloudflare/Nginx reverse proxy (X-Forwarded-Proto: https, X-Forwarded-For)
  • ENVIRONMENT="production" (HF Space Secrets)
  • Cross-Origin request from frontend (Origin: https://studybuddyai.dev)
  • Supabase JWT Secret configured locally (stateless verification)
  • Token Exchange (/api/auth/session) → Refresh Rotation (/api/auth/refresh) → Protected Route (/api/auth/me)
"""

import pytest
import jwt as pyjwt
from unittest.mock import patch
from httpx import ASGITransport, AsyncClient

from src.main import app
from src.config import settings
from src.auth.jwt_utils import hash_token
from src.auth.refresh_token_store import save_refresh_token, get_refresh_token

SESSION_URL = "/api/auth/session"
REFRESH_URL = "/api/auth/refresh"
ME_URL      = "/api/auth/me"

FAKE_USER_ID = "48e0fa0e-85c8-40c4-a5a3-c51c5eb53b5e"
FAKE_EMAIL   = "hf_test_user@example.com"
FAKE_SECRET  = "test-supabase-jwt-secret-key-32-chars-long"

FAKE_PROFILE = {
    "id": FAKE_USER_ID,
    "email": FAKE_EMAIL,
    "display_name": "HF Test User",
    "avatar_url": "",
    "daily_requests": 0,
    "last_request_date": "2026-08-01",
}


def create_fake_supabase_jwt() -> str:
    """Create a valid Supabase-issued JWT payload signed with FAKE_SECRET."""
    payload = {
        "sub": FAKE_USER_ID,
        "email": FAKE_EMAIL,
        "role": "authenticated",
        "aud": "authenticated",
        "user_metadata": {"full_name": "HF Test User"},
        "exp": 9999999999,
    }
    return pyjwt.encode(payload, FAKE_SECRET, algorithm="HS256")


@pytest.mark.anyio
class TestHFDeploymentSimulation:

    async def test_full_hf_production_simulation(self):
        """
        Simulate the complete Hugging Face Space auth flow:
          1. ENVIRONMENT="production"
          2. Reverse proxy headers (X-Forwarded-Proto: https)
          3. Cross-origin request from studybuddyai.dev
          4. POST /api/auth/session returns access_token + HttpOnly Secure SameSite=None cookie
          5. POST /api/auth/refresh uses cookie and successfully rotates tokens
          6. GET /api/auth/me authenticates statelessly
        """
        supabase_jwt = create_fake_supabase_jwt()

        # Database store for simulated DB operations
        db_store: dict[str, dict] = {}

        def mock_save_refresh(uid: str, thash: str):
            row = {
                "id": "ref-123",
                "user_id": uid,
                "token_hash": thash,
                "expires_at": "2099-01-01T00:00:00Z",
                "revoked": False,
            }
            db_store[thash] = row
            return row

        def mock_get_refresh(thash: str):
            return db_store.get(thash)

        def mock_revoke_refresh(thash: str):
            if thash in db_store:
                db_store[thash]["revoked"] = True

        with (
            patch.object(settings, "environment", "production"),
            patch.object(settings, "supabase_jwt_secret", FAKE_SECRET),
            patch("src.auth.routes.get_user_by_id", return_value=FAKE_PROFILE),
            patch("src.auth.routes.save_refresh_token", side_effect=mock_save_refresh),
            patch("src.auth.routes.get_refresh_token", side_effect=mock_get_refresh),
            patch("src.auth.routes.revoke_refresh_token", side_effect=mock_revoke_refresh),
        ):
            # Headers sent by Cloudflare/HF Proxy & Browser cross-origin request
            proxy_headers = {
                "X-Forwarded-Proto": "https",
                "X-Forwarded-For": "10.16.17.73",
                "Origin": "https://studybuddyai.dev",
                "X-Auth-Token": supabase_jwt,
            }

            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="https://hamdy005-study-buddy.hf.space"
            ) as client:

                # ── STEP 1: Session Exchange ──────────────────────────────────
                res1 = await client.post(SESSION_URL, headers=proxy_headers)
                assert res1.status_code == 200, f"Session exchange failed: {res1.text}"
                data1 = res1.json()
                assert "access_token" in data1
                assert data1["user"]["id"] == FAKE_USER_ID

                # Check cookie header flags in production mode
                set_cookie_headers = res1.headers.get_list("set-cookie")
                assert len(set_cookie_headers) > 0, "No set-cookie header returned"
                cookie_str = set_cookie_headers[0].lower()

                assert "refresh_token=" in cookie_str
                assert "httponly" in cookie_str
                assert "secure" in cookie_str
                assert "samesite=none" in cookie_str

                # Extract raw refresh token from set-cookie
                raw_cookie_val = set_cookie_headers[0].split(";")[0].split("=")[1]
                access_token_1 = data1["access_token"]

                # ── STEP 2: Silent Refresh Rotation ──────────────────────────
                refresh_headers = {
                    "X-Forwarded-Proto": "https",
                    "Origin": "https://studybuddyai.dev",
                }
                # Attach the HttpOnly cookie as client would send it
                res2 = await client.post(
                    REFRESH_URL,
                    headers=refresh_headers,
                    cookies={"refresh_token": raw_cookie_val}
                )
                assert res2.status_code == 200, f"Refresh failed: {res2.text}"
                data2 = res2.json()
                assert "access_token" in data2
                access_token_2 = data2["access_token"]

                # Verify cookie was rotated (new set-cookie header returned)
                set_cookie_headers_2 = res2.headers.get_list("set-cookie")
                assert len(set_cookie_headers_2) > 0
                assert "refresh_token=" in set_cookie_headers_2[0].lower()

                # ── STEP 3: Protected Route Request ───────────────────────────
                me_headers = {
                    "X-Forwarded-Proto": "https",
                    "Authorization": f"Bearer {access_token_2}",
                }
                res3 = await client.get(ME_URL, headers=me_headers)
                assert res3.status_code == 200
                data3 = res3.json()
                assert data3["status"] == "success"
                assert data3["user"]["id"] == FAKE_USER_ID
