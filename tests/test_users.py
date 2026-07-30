"""
Tests for POST /api/auth/upload-avatar
"""
from io import BytesIO

import pytest
from httpx import AsyncClient

from tests.conftest import make_image_bytes
from src.auth.constants import MAX_FILE_SIZE_BYTES, AVATAR_BUCKET
from src.dependencies import DEV_USER_ID

UPLOAD_URL = "/api/auth/upload-avatar"


from unittest.mock import patch, MagicMock

# ── 1. Successful upload ─────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_image_success(client: AsyncClient, mocked_supabase):
    """A valid PNG image should be uploaded and return a public Cloudinary URL."""
    image_bytes = make_image_bytes(size_bytes=1024)  # 1 KB — well within the 6 MB limit

    mock_response = {
        "secure_url": f"https://res.cloudinary.com/demo/image/upload/v1/avatars/avatar_{DEV_USER_ID}.png"
    }

    with patch("cloudinary.uploader.upload", return_value=mock_response) as mock_upload:
        response = await client.post(
            UPLOAD_URL,
            files={"file": ("avatar.png", BytesIO(image_bytes), "image/png")},
        )

        data = response.json()

        assert response.status_code == 200, f"Unexpected response: {response.text}"
        assert data["status"] == "success"
        assert "avatar_url" in data
        assert data["avatar_url"].startswith("https://res.cloudinary.com/")
        mock_upload.assert_called_once()


# ── 2. Failure — wrong file type ─────────────────────────────────────────────

@pytest.mark.anyio
async def test_image_failure_wrong_type(client: AsyncClient, mocked_supabase):
    """Uploading a non-image file type (PDF) should be rejected with HTTP 400."""
    fake_pdf = b"%PDF-1.4 fake pdf content"

    response = await client.post(
        UPLOAD_URL,
        files={"file": ("document.pdf", BytesIO(fake_pdf), "application/pdf")},
    )

    data = response.json()

    assert response.status_code == 400, f"Expected 400, got {response.status_code}"
    # Nothing should have been uploaded to storage
    assert len(mocked_supabase._uploaded_files) == 0
    # The error message should explain what went wrong
    detail = data.get("detail", "")
    assert "application/pdf" in detail or "Unsupported file type" in detail


# ── 3. Failure — file too large ───────────────────────────────────────────────

@pytest.mark.anyio
async def test_image_failure_too_large(client: AsyncClient, mocked_supabase):
    """Uploading an image that exceeds MAX_FILE_SIZE_BYTES should be rejected with HTTP 400."""
    oversized_image = make_image_bytes(size_bytes=MAX_FILE_SIZE_BYTES + 1)  # 1 byte over the limit

    response = await client.post(
        UPLOAD_URL,
        files={"file": ("huge_photo.png", BytesIO(oversized_image), "image/png")},
    )

    data = response.json()

    assert response.status_code == 400, f"Expected 400, got {response.status_code}"
    # Nothing should have been uploaded to storage
    assert len(mocked_supabase._uploaded_files) == 0
    # The error message should mention size
    detail = data.get("detail", "")
    assert "too large" in detail.lower() or "maximum" in detail.lower()


# ── 4. PATCH /api/auth/profile — Password Updates ────────────────────────────

PROFILE_URL = "/api/auth/profile"

@pytest.mark.anyio
async def test_update_password_too_short(client: AsyncClient, mocked_supabase):
    """Updating password with less than 8 characters should return 422 Unprocessable Entity via Pydantic."""
    response = await client.patch(
        PROFILE_URL,
        json={"password": "12345"}
    )
    assert response.status_code == 422
    data = response.json()
    assert "at least 8 characters" in str(data.get("detail", ""))



@pytest.mark.anyio
async def test_update_password_success(client: AsyncClient, mocked_supabase):
    """Updating password with valid data should return 200 success."""
    mocked_supabase.auth.admin.update_user_by_id.return_value = {"id": DEV_USER_ID}
    mocked_supabase.auth.sign_in_with_password.return_value = MagicMock(error=None)

    with patch("src.auth.routes.get_user_by_id", return_value={"id": DEV_USER_ID, "email": "test@example.com"}), \
         patch("src.auth.routes.update_user_profile", return_value={"id": DEV_USER_ID, "email": "test@example.com"}), \
         patch("src.auth.routes.get_auth_supabase", return_value=mocked_supabase):
        response = await client.patch(
            PROFILE_URL,
            json={
                "current_password": "oldpassword123",
                "password": "newpassword123"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        mocked_supabase.auth.admin.update_user_by_id.assert_called_once_with(
            DEV_USER_ID, {"password": "newpassword123"}
        )



@pytest.mark.anyio
async def test_update_password_invalid_current_password(client: AsyncClient, mocked_supabase):
    """Updating password with incorrect current_password should return 400."""
    mocked_supabase.auth.sign_in_with_password.side_effect = Exception("Invalid credentials")

    with patch("src.auth.routes.get_user_by_id", return_value={"id": DEV_USER_ID, "email": "test@example.com"}), \
         patch("src.auth.routes.get_auth_supabase", return_value=mocked_supabase):
        response = await client.patch(
            PROFILE_URL,
            json={
                "current_password": "wrongpassword",
                "password": "newpassword123"
            }
        )

        assert response.status_code == 400
        data = response.json()
        assert "Current password is incorrect" in data.get("detail", "")

