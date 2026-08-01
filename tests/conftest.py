import os
from collections.abc import AsyncGenerator
from io import BytesIO
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

# Environment Variables
os.environ["SUPABASE_URL"]    = os.getenv("SUPABASE_URL",    "https://placeholder-project.supabase.co")
os.environ["SUPABASE_KEY"]    = os.getenv("SUPABASE_KEY",    "placeholder-service-key")
os.environ["SUPABASE_ANON_KEY"] = os.getenv("SUPABASE_ANON_KEY", "placeholder-anon-key")
os.environ["SECRET_KEY"]      = os.getenv("SECRET_KEY",      "test-secret-key-for-testing-only")
# JWT settings used by src.auth.jwt_utils
os.environ["JWT_SECRET_KEY"]  = os.getenv("JWT_SECRET_KEY",  "test-jwt-secret-key-for-testing-only-32x")
os.environ["JWT_ALGORITHM"]   = os.getenv("JWT_ALGORITHM",   "HS256")
os.environ["ENVIRONMENT"]     = os.getenv("ENVIRONMENT",     "development")

from src.dependencies import DEV_USER_ID, get_current_user_id
from src.main import app
from src.auth.rate_limiter import _store, _cooldowns
from src.auth.jwt_utils import create_access_token, create_refresh_token, hash_token

pytest_plugins = ["anyio"]


# ── Rate Limiter Clean Fixture ──────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clear_rate_limiter_store():
    _store.clear()
    _cooldowns.clear()
    yield
    _store.clear()
    _cooldowns.clear()


# ── Backend & session ────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


# ── Supabase Storage mock ────────────────────────────────────────────────────

@pytest.fixture
def mocked_supabase() -> MagicMock:
    """
    Replaces get_supabase() in the auth routes with a fake Supabase client.
    Tracks uploaded file bytes in an in-memory dict keyed by storage path.
    Mirrors the role of `mocked_aws` (moto) from S3-based tests.
    """
    uploaded_files: dict[str, bytes] = {}

    mock_client = MagicMock()

    def fake_upload(path: str, file: bytes, file_options=None):
        uploaded_files[path] = file
        return MagicMock()  # simulate a successful response

    def fake_get_public_url(path: str) -> str:
        return f"https://placeholder-project.supabase.co/storage/v1/object/public/avatars/{path}"

    bucket_mock = MagicMock()
    bucket_mock.upload.side_effect = fake_upload
    bucket_mock.get_public_url.side_effect = fake_get_public_url
    mock_client.storage.from_.return_value = bucket_mock

    # Attach the uploaded_files dict so tests can inspect it
    mock_client._uploaded_files = uploaded_files

    with patch("src.auth.routes.get_supabase", return_value=mock_client):
        yield mock_client


# ── HTTP client ──────────────────────────────────────────────────────────────

@pytest.fixture
async def client(mocked_supabase) -> AsyncGenerator[AsyncClient, None]:
    """
    Async HTTP client wired to the FastAPI app.
    Bypasses JWT auth by overriding get_current_user_id with a fixed dev user ID.
    Automatically uses the mocked Supabase client.
    """
    async def override_get_current_user_id() -> str:
        return DEV_USER_ID

    app.dependency_overrides[get_current_user_id] = override_get_current_user_id

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as ac:
        yield ac

    app.dependency_overrides.clear()


# ── Helper functions ─────────────────────────────────────────────────────────

def auth_header(token: str) -> dict[str, str]:
    return {"X-Auth-Token": token}


def make_image_bytes(size_bytes: int = 1024) -> bytes:
    """Create a minimal valid PNG header + random bytes for a given size."""
    # Minimal 1x1 PNG header (valid enough to pass MIME check)
    png_header = bytes([
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  # PNG signature
    ])
    padding = b"\x00" * max(0, size_bytes - len(png_header))
    return png_header + padding


# ── JWT / Auth shared constants ───────────────────────────────────────────────
# Used by test_jwt_auth.py and any future auth-related test files.

FAKE_USER_ID = DEV_USER_ID
FAKE_EMAIL   = "test@example.com"
FAKE_NAME    = "Test User"

FAKE_PROFILE = {
    "id":    FAKE_USER_ID,
    "name":  FAKE_NAME,
    "email": FAKE_EMAIL,
    "avatar": "",
    "theme": "system",
    "usage": {"used": 0, "limit": 20, "remaining": 20},
}


def fake_sb_user(
    user_id: str = FAKE_USER_ID,
    email: str = FAKE_EMAIL,
) -> MagicMock:
    """A minimal mock that looks like a Supabase User object."""
    m = MagicMock()
    m.id            = user_id
    m.email         = email
    m.user_metadata = {"full_name": FAKE_NAME}
    return m


def future_expires_at(days: int = 30) -> str:
    """ISO-8601 timestamp that is `days` from now (for valid refresh token rows)."""
    return (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()


def past_expires_at(seconds: int = 10) -> str:
    """ISO-8601 timestamp that is `seconds` in the past (for expired token rows)."""
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()


# ── JWT fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def raw_access_token() -> str:
    """A freshly-signed 15-min JWT for the dev user."""
    return create_access_token(FAKE_USER_ID, FAKE_EMAIL)


@pytest.fixture
def raw_refresh_token() -> str:
    """A cryptographically-random opaque refresh token string."""
    return create_refresh_token()


@pytest.fixture
def valid_refresh_row(raw_refresh_token: str) -> dict:
    """A DB row representing a valid (not revoked, not expired) refresh token."""
    return {
        "user_id":    FAKE_USER_ID,
        "token_hash": hash_token(raw_refresh_token),
        "expires_at": future_expires_at(30),
        "revoked":    False,
    }
