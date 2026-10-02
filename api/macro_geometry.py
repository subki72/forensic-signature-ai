"""Macro-Geometric Screening Pipeline (Stage 1 of 2-Phase Verification).

Performs fast, deterministic shape analysis of two NORMALIZED 224x224
signature images using direct 2D spatial comparison methods. Acts as a
fail-fast gate BEFORE the Siamese ResNet (Stage 2).

CRITICAL DESIGN: Inputs must be the padded, centered, 224x224 single-channel
binary images (from pad_and_resize). This ensures both signatures are:
  1. Cropped to bounding box  (paper noise removed)
  2. Aspect-ratio padded      (no shape distortion)
  3. Resized to 224x224       (same scale for pixel comparison)
  4. Centered on canvas       (aligned for overlap measurement)

Previous HPP/VPP approach FAILED because:
  - 1D projection profiles are NOT shape-discriminative.
  - Two completely different signatures with similar horizontal extent
    produce near-identical VPP profiles (e.g., VPP corr was 93.2% for
    two visually distinct signatures).
  - Projection profiles measure ink DISTRIBUTION, not ink SHAPE.

Current approach uses three 2D shape-aware metrics:
  1. Pixel IoU        — directly measures stroke pixel overlap
  2. Hu Moments       — rotation/scale invariant shape topology comparison
  3. Pixel Correlation — 2D normalized cross-correlation of the full image
"""

import cv2
import numpy as np

from api.logging_config import logger


def _compute_pixel_iou(bin_a: np.ndarray, bin_b: np.ndarray) -> float:
    """Compute Intersection over Union of foreground (stroke) pixels.

    This is the most direct measure of whether two signatures share the same
    physical stroke positions on the canvas. For two different people's
    signatures (different letterforms in different positions), IoU will be
    very low (typically 0.02-0.10). For same-person genuine pairs with
    natural variation, IoU is moderate (typically 0.15-0.40).

    Args:
        bin_a: First binary image (0 or 255, uint8).
        bin_b: Second binary image (0 or 255, uint8).

    Returns:
        IoU score in [0.0, 1.0].
    """
    bool_a = bin_a > 0
    bool_b = bin_b > 0

    intersection = np.logical_and(bool_a, bool_b).sum()
    union = np.logical_or(bool_a, bool_b).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)


def _compute_hu_similarity(bin_a: np.ndarray, bin_b: np.ndarray) -> float:
    """Compute shape similarity using Hu Moments on the stroke contours.

    Hu Moments are seven invariant moments that characterize shape topology
    independent of translation, rotation, and scale. Two fundamentally
    different signature shapes (e.g., "SJ" vs "ew") will have very different
    Hu moment vectors.

    Uses cosine similarity on log-transformed Hu moments for numerical
    stability (Hu moments span many orders of magnitude).

    Args:
        bin_a: First binary image (0 or 255, uint8).
        bin_b: Second binary image (0 or 255, uint8).

    Returns:
        Similarity score in [0.0, 1.0].
    """
    moments_a = cv2.moments(bin_a)
    moments_b = cv2.moments(bin_b)

    # Guard: if either image has no mass, shapes are incomparable
    if moments_a["m00"] == 0 or moments_b["m00"] == 0:
        return 0.0

    hu_a = cv2.HuMoments(moments_a).flatten()
    hu_b = cv2.HuMoments(moments_b).flatten()

    # Log-transform for numerical stability (moments span 10^-1 to 10^-15)
    # Use sign-preserving log: -sign(h) * log10(|h|)
    eps = 1e-12
    log_hu_a = -np.sign(hu_a) * np.log10(np.abs(hu_a) + eps)
    log_hu_b = -np.sign(hu_b) * np.log10(np.abs(hu_b) + eps)

    # Cosine similarity of the log-transformed Hu vectors
    norm_a = np.linalg.norm(log_hu_a)
    norm_b = np.linalg.norm(log_hu_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    cosine = float(np.dot(log_hu_a, log_hu_b) / (norm_a * norm_b))
    return max(0.0, cosine)  # Clamp negatives to 0


def _compute_pixel_correlation(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Compute 2D Normalized Cross-Correlation between two images.

    Unlike 1D projection profiles, this compares the FULL 2D pixel pattern.
    Mean-subtraction ensures the metric is not dominated by the shared black
    background — it measures how well the stroke PATTERNS align.

    For two sparse binary images where strokes are in different locations,
    the cross-correlation will be low because white pixels don't coincide
    with white pixels in the other image.

    Args:
        img_a: First image (uint8, same dimensions as img_b).
        img_b: Second image (uint8, same dimensions as img_b).

    Returns:
        Correlation coefficient in [0.0, 1.0] (negatives clamped to 0).
    """
    a = img_a.astype(np.float32)
    b = img_b.astype(np.float32)

    a_centered = a - a.mean()
    b_centered = b - b.mean()

    norm_a = np.linalg.norm(a_centered)
    norm_b = np.linalg.norm(b_centered)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    ncc = float(np.sum(a_centered * b_centered) / (norm_a * norm_b))
    return max(0.0, ncc)


def compute_macro_score(padded_a: np.ndarray, padded_b: np.ndarray) -> dict:
    """Compute composite macro-geometric similarity using 2D shape comparison.

    INPUTS MUST BE the 224x224 single-channel padded images from pad_and_resize
    (extract via rgb_output[:, :, 0]). Both signatures are already cropped,
    aspect-ratio padded, and centered at the same scale.

    Combines three complementary shape metrics:
      - Pixel IoU        (50%): Direct stroke pixel overlap. Most discriminative
                                for sparse binary images because it ignores the
                                shared background entirely.
      - Hu Moments       (25%): Topological shape invariants. Captures whether
                                the overall contour "shape family" matches.
      - Pixel Correlation (25%): Full 2D pattern correlation. Captures spatial
                                arrangement of ink density.

    Args:
        padded_a: 224x224 uint8 single-channel padded image (reference).
        padded_b: 224x224 uint8 single-channel padded image (questioned).

    Returns:
        Dictionary with:
            'macro_score'   (float): Weighted composite [0.0–1.0].
            'pixel_iou'     (float): Stroke pixel IoU [0.0–1.0].
            'hu_similarity' (float): Hu Moments cosine similarity [0.0–1.0].
            'pixel_corr'    (float): Normalized cross-correlation [0.0–1.0].
    """
    # Re-binarize: INTER_AREA resize may have introduced anti-aliasing
    _, bin_a = cv2.threshold(padded_a, 127, 255, cv2.THRESH_BINARY)
    _, bin_b = cv2.threshold(padded_b, 127, 255, cv2.THRESH_BINARY)

    # ── Metric 1: Pixel IoU (50%) ──────────────────────────────────────────
    pixel_iou = _compute_pixel_iou(bin_a, bin_b)

    # ── Metric 2: Hu Moments shape similarity (25%) ────────────────────────
    hu_similarity = _compute_hu_similarity(bin_a, bin_b)

    # ── Metric 3: 2D Normalized Cross-Correlation (25%) ────────────────────
    pixel_corr = _compute_pixel_correlation(padded_a, padded_b)

    # ── Weighted Composite ─────────────────────────────────────────────────
    # IoU is weighted highest because it is the most discriminative metric
    # for sparse binary images (ignores the dominant shared background).
    macro_score = (0.50 * pixel_iou) + (0.25 * hu_similarity) + (0.25 * pixel_corr)

    logger.debug(
        "MacroGeometry: IoU=%.4f | Hu=%.4f | NCC=%.4f | Composite=%.4f",
        pixel_iou,
        hu_similarity,
        pixel_corr,
        macro_score,
    )

    return {
        "macro_score": round(float(macro_score), 4),
        "pixel_iou": round(float(pixel_iou), 4),
        "hu_similarity": round(float(hu_similarity), 4),
        "pixel_corr": round(float(pixel_corr), 4),
    }
