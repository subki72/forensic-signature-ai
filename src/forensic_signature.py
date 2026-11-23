"""
Modular package facade for Forensic Signature AI (Two-Phase V2.3 Pipeline).
Re-exports configuration, macro-geometry gate, CV pipeline, Siamese engine,
and provides a deterministic LRU response cache for repeated specimen lookups.
"""

import logging
from functools import lru_cache
from typing import Dict

from api.config import settings
from api.cv_pipeline import preprocess_signature
from api.engine import SignatureVerificationEngine
from api.macro_geometry import compute_macro_geometry

logger = logging.getLogger(__name__)


@lru_cache(maxsize=256)
def get_cached_threshold_profile(version: str = "v2.3") -> Dict[str, float]:
    """Return cached calibrated decision thresholds for Stage 1 and Stage 2."""
    logger.info("Serving cached threshold profile for model version=%s", version)
    return {
        "macro_threshold": float(getattr(settings, "MACRO_THRESHOLD", 0.25)),
        "authentic_threshold": float(getattr(settings, "AUTHENTIC_THRESHOLD", 0.70)),
        "inconclusive_threshold": float(getattr(settings, "INCONCLUSIVE_THRESHOLD", 0.55)),
    }


__all__ = [
    "settings",
    "preprocess_signature",
    "compute_macro_geometry",
    "SignatureVerificationEngine",
    "get_cached_threshold_profile",
]
