"""Tests for input validation, security guard clauses, and error handling."""

import io
from starlette.testclient import TestClient


def test_missing_files_returns_422(client: TestClient):
    """Test that submitting without files returns 422 Unprocessable Entity."""
    response = client.post("/verify")
    assert response.status_code == 422


def test_missing_one_file_returns_422(client: TestClient, genuine_signature_bytes_1: bytes):
    """Test that submitting only one file returns 422."""
    files = {
        "file_asli": ("specimen.png", io.BytesIO(genuine_signature_bytes_1), "image/png"),
    }
    response = client.post("/verify", files=files)
    assert response.status_code == 422


def test_empty_file_returns_400(client: TestClient, genuine_signature_bytes_1: bytes):
    """Test that submitting an empty file (0 bytes) returns 400 Bad Request."""
    files = {
        "file_asli": ("specimen.png", io.BytesIO(genuine_signature_bytes_1), "image/png"),
        "file_uji": ("empty.png", io.BytesIO(b""), "image/png"),
    }
    response = client.post("/verify", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["error"].lower()


def test_unsupported_media_type_magic_bytes_returns_415(
    client: TestClient, genuine_signature_bytes_1: bytes, corrupt_bytes: bytes
):
    """Test that non-image payloads (spoofed MIME) return 415 Unsupported Media Type."""
    files = {
        "file_asli": ("specimen.png", io.BytesIO(genuine_signature_bytes_1), "image/png"),
        "file_uji": ("spoofed.png", io.BytesIO(corrupt_bytes), "image/png"),
    }
    response = client.post("/verify", files=files)
    assert response.status_code == 415
    assert "valid jpeg, png, or webp" in response.json()["error"].lower()


def test_oversized_file_returns_413(client: TestClient, genuine_signature_bytes_1: bytes):
    """Test that files exceeding 5MB return 413 Request Entity Too Large."""
    # Create a 5.5 MB payload with a valid PNG magic byte prefix
    oversized_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * (6 * 1024 * 1024)
    files = {
        "file_asli": ("specimen.png", io.BytesIO(genuine_signature_bytes_1), "image/png"),
        "file_uji": ("huge.png", io.BytesIO(oversized_data), "image/png"),
    }
    response = client.post("/verify", files=files)
    assert response.status_code == 413
    assert "exceeds maximum allowed size" in response.json()["error"].lower()
