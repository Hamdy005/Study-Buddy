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


# ── 1. Successful upload ─────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_image_success(client: AsyncClient, mocked_supabase):
    """A valid PNG image should be uploaded and return a public Supabase URL."""
    image_bytes = make_image_bytes(size_bytes=1024)  # 1 KB — well within the 6 MB limit

    response = await client.post(
        UPLOAD_URL,
        files={"file": ("avatar.png", BytesIO(image_bytes), "image/png")},
    )

    data = response.json()

    assert response.status_code == 200, f"Unexpected response: {response.text}"
    assert data["status"] == "success"
    assert "avatar_url" in data
    assert data["avatar_url"].startswith("https://")
    assert AVATAR_BUCKET in data["avatar_url"]

    # Confirm the file was stored in the mocked bucket under the user's folder
    uploaded = mocked_supabase._uploaded_files
    assert len(uploaded) == 1
    stored_path = list(uploaded.keys())[0]
    assert stored_path.startswith(DEV_USER_ID)
    assert stored_path.endswith(".png")
    assert uploaded[stored_path] == image_bytes


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
