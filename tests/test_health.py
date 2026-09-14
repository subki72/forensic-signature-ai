"""Integration tests for monitoring and probe endpoints."""

from starlette.testclient import TestClient


def test_health_liveness_probe(client: TestClient):
    """Test shallow liveness endpoint returns HTTP 200."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "service" in data


def test_readiness_probe(client: TestClient):
    """Test deep readiness probe returns HTTP 200 and model metadata."""
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["model_loaded"] is True
    assert data["model_file_size_bytes"] > 0
    assert "calibrated_threshold" in data


def test_root_landing(client: TestClient):
    """Test root endpoint returns navigation metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["docs"] == "/docs"
    assert data["health"] == "/health"
