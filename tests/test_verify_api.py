"""End-to-end integration tests for the /verify endpoint."""

import io
from starlette.testclient import TestClient


def test_verify_identical_signatures(
    client: TestClient, genuine_signature_bytes_1: bytes
):
    """Test verification of identical signature specimens yields an AUTHENTIC verdict."""
    files = {
        "file_asli": ("genuine_ref.png", io.BytesIO(genuine_signature_bytes_1), "image/png"),
        "file_uji": ("genuine_test.png", io.BytesIO(genuine_signature_bytes_1), "image/png"),
    }
    response = client.post("/verify", files=files)
    assert response.status_code == 200
    data = response.json()

    assert "verification" in data
    result = data["verification"]
    assert result["verdict"] == "AUTHENTIC"
    assert "AUTHENTIC" in result["status"]
    assert result["similarity_score"] >= 0.95
    assert result["normalized_percentage"] >= 97.0
    assert result["latency_ms"] > 0.0
    assert "LEGAL DISCLAIMER" in result["disclaimer"]


def test_verify_genuine_variant_signatures(
    client: TestClient, genuine_signature_bytes_1: bytes, genuine_signature_bytes_2: bytes
):
    """Test verification of slightly perturbed specimens from the same signer."""
    files = {
        "file_asli": ("specimen_1.png", io.BytesIO(genuine_signature_bytes_1), "image/png"),
        "file_uji": ("specimen_2.png", io.BytesIO(genuine_signature_bytes_2), "image/png"),
    }
    response = client.post("/verify", files=files)
    assert response.status_code == 200
    result = response.json()["verification"]

    assert result["verdict"] in ["AUTHENTIC", "INCONCLUSIVE"]
    assert result["similarity_score"] > 0.50
    assert isinstance(result["latency_ms"], float)


def test_verify_forged_specimen(
    client: TestClient, genuine_signature_bytes_1: bytes, forged_signature_bytes: bytes
):
    """Test verification of genuine specimen vs. an angular forgery specimen."""
    files = {
        "file_asli": ("specimen_genuine.png", io.BytesIO(genuine_signature_bytes_1), "image/png"),
        "file_uji": ("specimen_forged.png", io.BytesIO(forged_signature_bytes), "image/png"),
    }
    response = client.post("/verify", files=files)
    assert response.status_code == 200
    result = response.json()["verification"]

    # Forgery score must be markedly lower than identical match
    assert result["similarity_score"] < 0.85
    assert "analysis" in result


def test_verify_response_schema_completeness(
    client: TestClient, genuine_signature_bytes_1: bytes
):
    """Test that all required contract fields are populated in response."""
    files = {
        "file_asli": ("asli.png", io.BytesIO(genuine_signature_bytes_1), "image/png"),
        "file_uji": ("uji.png", io.BytesIO(genuine_signature_bytes_1), "image/png"),
    }
    response = client.post("/verify", files=files)
    assert response.status_code == 200
    result = response.json()["verification"]

    expected_keys = {
        "verdict",
        "status",
        "similarity_score",
        "normalized_percentage",
        "system_threshold",
        "confidence_band",
        "analysis",
        "disclaimer",
        "latency_ms",
    }
    assert expected_keys.issubset(result.keys())
