"""Unit tests for the ML inference engine and SiameseNetwork."""

import pytest
import torch

from api.engine import SiameseNetwork, model_manager
from api.schemas import VerdictEnum


def test_siamese_network_forward_shape():
    """Test SiameseNetwork forward pass produces (B, 128) embeddings."""
    model = SiameseNetwork(embedding_dim=128)
    model.eval()

    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        embeddings = model.forward_once(dummy_input)

    assert embeddings.shape == (2, 128)


def test_siamese_network_l2_normalization():
    """Test that all output embeddings are strictly constrained to the L2 unit sphere."""
    model = SiameseNetwork(embedding_dim=128)
    model.eval()

    dummy_input = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        embeddings = model.forward_once(dummy_input)

    # Calculate L2 norm for each embedding in batch
    l2_norms = torch.norm(embeddings, p=2, dim=1)
    # Each norm must equal 1.0 within floating point tolerance
    for norm in l2_norms:
        assert pytest.approx(norm.item(), rel=1e-3) == 1.0


def test_compute_similarity_identical_vectors():
    """Test that identical embeddings have a cosine similarity of 1.0."""
    v1 = torch.randn(128)
    v1 = v1 / torch.norm(v1, p=2)

    similarity = model_manager.compute_similarity(v1, v1)
    assert pytest.approx(similarity, abs=1e-3) == 1.0


def test_compute_similarity_opposite_vectors():
    """Test that antipodal (opposite) embeddings have a cosine similarity of -1.0."""
    v1 = torch.randn(128)
    v1 = v1 / torch.norm(v1, p=2)
    v2 = -v1

    similarity = model_manager.compute_similarity(v1, v2)
    assert pytest.approx(similarity, abs=1e-3) == -1.0


def test_verdict_evaluation_authentic():
    """Test 3-tier verdict logic for high similarity (>= 0.70)."""
    verdict, status_label, confidence, analysis = model_manager.evaluate_verdict(0.85)
    assert verdict == VerdictEnum.AUTHENTIC
    assert "AUTHENTIC" in status_label
    assert "High Confidence" in confidence


def test_verdict_evaluation_inconclusive():
    """Test 3-tier verdict logic for borderline similarity (0.55 - 0.70)."""
    # Test at midpoint 0.58
    verdict, status_label, confidence, analysis = model_manager.evaluate_verdict(0.58)
    assert verdict == VerdictEnum.INCONCLUSIVE
    assert "INCONCLUSIVE" in status_label
    assert "HUMAN REVIEW REQUIRED" in status_label


def test_verdict_evaluation_forgery():
    """Test 3-tier verdict logic for low similarity (< 0.55)."""
    verdict, status_label, confidence, analysis = model_manager.evaluate_verdict(0.30)
    assert verdict == VerdictEnum.FORGERY
    assert "FORGERY" in status_label


def test_model_manager_readiness_status():
    """Test that ModelManager returns healthy readiness status."""
    readiness = model_manager.get_readiness()
    assert readiness["status"] == "ready"
    assert readiness["model_loaded"] is True
    assert readiness["model_file_size_bytes"] > 0
