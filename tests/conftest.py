"""Pytest configuration and synthetic fixtures for Forensic Signature AI."""

import cv2
import numpy as np
import pytest
from starlette.testclient import TestClient

from api.main import app


def _create_synthetic_signature(seed: int = 42, style: str = "curved") -> bytes:
    """Generate a realistic synthetic signature image in memory and return PNG bytes.

    Args:
        seed: Random seed for deterministic reproducibility.
        style: 'curved' for standard signature or 'angular' for forgery specimen.

    Returns:
        Encoded PNG bytes.
    """
    np.random.seed(seed)
    # White background canvas
    canvas = np.ones((300, 600), dtype=np.uint8) * 255

    if style == "curved":
        # Draw smooth handwriting ink strokes
        points = [
            (50, 200), (90, 80), (140, 220), (190, 110),
            (240, 180), (300, 140), (380, 230), (450, 160),
            (520, 210), (550, 190)
        ]
        pts = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], isClosed=False, color=20, thickness=3, lineType=cv2.LINE_AA)

        # Add underline stroke
        cv2.line(canvas, (40, 245), (540, 240), color=25, thickness=2, lineType=cv2.LINE_AA)
        # Add dot
        cv2.circle(canvas, (555, 235), radius=3, color=20, thickness=-1)

    elif style == "variant":
        # Slightly perturbed genuine signature
        points = [
            (52, 198), (92, 83), (139, 218), (192, 112),
            (238, 182), (302, 138), (378, 228), (452, 162),
            (518, 208), (548, 192)
        ]
        pts = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], isClosed=False, color=22, thickness=3, lineType=cv2.LINE_AA)
        cv2.line(canvas, (42, 246), (538, 241), color=25, thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (553, 236), radius=3, color=22, thickness=-1)

    else:
        # Distinctly different angular forgery stroke pattern
        points = [
            (50, 120), (120, 240), (200, 70), (280, 250),
            (360, 80), (440, 240), (520, 100)
        ]
        pts = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], isClosed=False, color=15, thickness=4, lineType=cv2.LINE_AA)

    success, buffer = cv2.imencode(".png", canvas)
    assert success, "Failed to encode synthetic signature image to PNG."
    return buffer.tobytes()


@pytest.fixture(scope="session")
def client() -> TestClient:
    """FastAPI TestClient fixture."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def genuine_signature_bytes_1() -> bytes:
    """Primary reference signature specimen bytes."""
    return _create_synthetic_signature(seed=42, style="curved")


@pytest.fixture(scope="session")
def genuine_signature_bytes_2() -> bytes:
    """Slight variant of genuine signature from same signer."""
    return _create_synthetic_signature(seed=42, style="variant")


@pytest.fixture(scope="session")
def forged_signature_bytes() -> bytes:
    """Questioned forged signature with distinct stroke trajectory."""
    return _create_synthetic_signature(seed=999, style="angular")


@pytest.fixture(scope="session")
def corrupt_bytes() -> bytes:
    """Corrupt non-image byte payload."""
    return b"This is a corrupt payload that is not a valid JPEG or PNG file."
