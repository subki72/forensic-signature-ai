"""Macro-Geometric Screening Pipeline (Stage 1 of 2-Phase Verification).

Performs fast, deterministic geometric analysis of two binary signature images
using Projection Profile Correlation and ORB Keypoint Matching. This acts as
a fail-fast gate BEFORE the computationally expensive Siamese ResNet (Stage 2).

Design Rationale:
- Two signatures from completely different people (e.g., "Saniya" vs "Preethi")
  will be REJECTED here, within milliseconds, because their structural layout
  and projection density profiles are fundamentally different.
- Only signatures that pass the geometry gate proceed to the deep learning stage.

Algorithms:
1. Horizontal & Vertical Projection Profile (HPP/VPP) Correlation:
   - Projects ink density along each axis to capture overall shape rhythm.
   - Uses Pearson-like cosine correlation between the two 1D profiles.
2. ORB (Oriented FAST and Rotated BRIEF) Keypoint Match Ratio:
   - Detects distinctive stroke intersections and curvature anchors.
   - Measures the fraction of matched keypoints using BFMatcher + ratio test.
"""

import cv2
import numpy as np

from api.logging_config import logger


def _extract_projection_profiles(binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute horizontal and vertical ink density projection profiles.

    Args:
        binary: Binary image with white ink strokes on black background (uint8).

    Returns:
        Tuple of (hpp, vpp) — 1D float32 arrays normalized to [0.0, 1.0].
    """
    # Horizontal Projection Profile (HPP): sum ink pixels per row
    hpp = binary.sum(axis=1).astype(np.float32)
    # Vertical Projection Profile (VPP): sum ink pixels per column
    vpp = binary.sum(axis=0).astype(np.float32)

    # Normalize to [0, 1] to remove scale bias from image size differences
    hpp_max = hpp.max()
    vpp_max = vpp.max()
    if hpp_max > 0:
        hpp /= hpp_max
    if vpp_max > 0:
        vpp /= vpp_max

    return hpp, vpp


def _cosine_correlation_1d(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1D arrays after resampling to equal length.

    Args:
        a: First profile array.
        b: Second profile array.

    Returns:
        Cosine similarity in range [0.0, 1.0] (clamped; negative treated as 0).
    """
    target_len = max(len(a), len(b))
    if target_len == 0:
        return 0.0

    # Resample both profiles to the same length for direct comparison
    a_resized = cv2.resize(a.reshape(-1, 1), (1, target_len), interpolation=cv2.INTER_LINEAR).flatten()
    b_resized = cv2.resize(b.reshape(-1, 1), (1, target_len), interpolation=cv2.INTER_LINEAR).flatten()

    norm_a = np.linalg.norm(a_resized)
    norm_b = np.linalg.norm(b_resized)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    cosine_sim = float(np.dot(a_resized, b_resized) / (norm_a * norm_b))
    return max(0.0, cosine_sim)  # Clamp negative cosines to 0


def _orb_match_ratio(binary_a: np.ndarray, binary_b: np.ndarray) -> float:
    """Compute ORB keypoint match ratio between two binary images.

    Uses Lowe's ratio test (threshold 0.75) to filter out ambiguous matches.
    Returns the ratio of good inlier matches to the maximum number of keypoints
    detected in either image.

    Args:
        binary_a: First binary signature image.
        binary_b: Second binary signature image.

    Returns:
        Match ratio in range [0.0, 1.0]. Returns 0.0 if too few keypoints.
    """
    orb = cv2.ORB_create(nfeatures=500)
    kp_a, des_a = orb.detectAndCompute(binary_a, None)
    kp_b, des_b = orb.detectAndCompute(binary_b, None)

    # Guard: too few keypoints to make a meaningful comparison
    if des_a is None or des_b is None or len(kp_a) < 5 or len(kp_b) < 5:
        logger.debug("ORB: insufficient keypoints (%d vs %d). Returning 0.0.", len(kp_a), len(kp_b))
        return 0.0

    # Brute-Force Matcher with Hamming distance for binary ORB descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    try:
        matches = bf.knnMatch(des_a, des_b, k=2)
    except cv2.error:
        return 0.0

    # Lowe's ratio test: keep only unambiguous, discriminative matches
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    max_kp = max(len(kp_a), len(kp_b))
    ratio = len(good_matches) / max_kp if max_kp > 0 else 0.0
    return float(min(ratio, 1.0))


def compute_macro_score(binary_a: np.ndarray, binary_b: np.ndarray) -> dict:
    """Compute composite macro-geometric similarity score between two signature images.

    Combines:
    - HPP cosine correlation (weight 40%): vertical layout rhythm
    - VPP cosine correlation (weight 40%): horizontal stroke distribution
    - ORB keypoint match ratio (weight 20%): structural anchor point overlap

    Args:
        binary_a: Binary image of reference signature (white ink on black).
        binary_b: Binary image of questioned signature (white ink on black).

    Returns:
        Dictionary with:
            'macro_score' (float): Weighted composite score in [0.0, 1.0].
            'hpp_corr' (float): Horizontal projection profile correlation.
            'vpp_corr' (float): Vertical projection profile correlation.
            'orb_ratio' (float): ORB inlier match ratio.
    """
    hpp_a, vpp_a = _extract_projection_profiles(binary_a)
    hpp_b, vpp_b = _extract_projection_profiles(binary_b)

    hpp_corr = _cosine_correlation_1d(hpp_a, hpp_b)
    vpp_corr = _cosine_correlation_1d(vpp_a, vpp_b)
    orb_ratio = _orb_match_ratio(binary_a, binary_b)

    # Weighted composite: projection profiles are the primary signal,
    # ORB provides a local structure cross-check.
    macro_score = (0.40 * hpp_corr) + (0.40 * vpp_corr) + (0.20 * orb_ratio)

    logger.debug(
        "MacroGeometry: HPP=%.4f | VPP=%.4f | ORB=%.4f | Composite=%.4f",
        hpp_corr,
        vpp_corr,
        orb_ratio,
        macro_score,
    )

    return {
        "macro_score": round(float(macro_score), 4),
        "hpp_corr": round(float(hpp_corr), 4),
        "vpp_corr": round(float(vpp_corr), 4),
        "orb_ratio": round(float(orb_ratio), 4),
    }
