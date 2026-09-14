"""Computer vision preprocessing pipeline for signature ink stroke extraction.

Implements pure, deterministic transformations:
1. Decoding with decompression bomb protection.
2. Gaussian blurring and Otsu binarization (foreground ink = 255).
3. Morphological closing to seal pen stroke micro-fissures.
4. Contour detection and bounding box merging (retaining dots, underlines, and signatures).
5. Aspect-ratio aware padding to square and resizing to 224x224.
6. ImageNet normalization and PyTorch tensor conversion.
"""

import cv2
import numpy as np
import torch

from api.config import settings
from api.logging_config import logger

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ImageProcessingError(Exception):
    """Raised when image decoding or processing fails validation."""
    pass


def decode_image(
    image_bytes: bytes,
    max_dimension: int = settings.MAX_IMAGE_DIMENSION,
) -> np.ndarray:
    """Decode raw bytes into a grayscale NumPy image array with guardrails.

    Args:
        image_bytes: Raw binary bytes of uploaded image.
        max_dimension: Maximum permissible height or width in pixels.

    Returns:
        Grayscale uint8 NumPy array of shape (H, W).

    Raises:
        ImageProcessingError: If decoding fails or image exceeds dimension limits.
    """
    if not image_bytes:
        raise ImageProcessingError("Image byte payload is empty.")

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ImageProcessingError(
            "Failed to decode image. File may be corrupted or an unsupported format."
        )

    height, width = img.shape
    if height == 0 or width == 0:
        raise ImageProcessingError("Decoded image has zero dimension.")

    if height > max_dimension or width > max_dimension:
        raise ImageProcessingError(
            f"Image dimensions ({width}x{height}) exceed maximum allowed dimension "
            f"({max_dimension}x{max_dimension}) pixels (decompression protection)."
        )

    return img


def isolate_ink_strokes(img: np.ndarray) -> np.ndarray:
    """Isolate ink strokes from document background using Otsu & morphology.

    Args:
        img: Grayscale NumPy array (H, W).

    Returns:
        Binary uint8 image where foreground ink strokes are 255 and background is 0.
    """
    # 5x5 Gaussian blur to suppress scanner sensor noise and paper grain
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Inverted Otsu thresholding: ink strokes become white (255), paper becomes black (0)
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Morphological closing (dilation followed by erosion) to link stroke micro-gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return closed


def extract_contour_bbox(
    binary: np.ndarray,
    min_area: int = 50,
) -> tuple[int, int, int, int] | None:
    """Extract merged bounding box containing all significant signature strokes.

    Filters out tiny speckles (< min_area pixels) while merging multi-part
    signature strokes (e.g. underlines, cross-bars, detached initials).

    Args:
        binary: Binary image with white ink strokes.
        min_area: Minimum contour area threshold in pixels.

    Returns:
        Tuple (x, y, w, h) or None if no valid contours were found.
    """
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]

    if not valid_contours:
        return None

    # Merge all contour coordinates into a single point cloud
    all_points = np.vstack(valid_contours)
    x, y, w, h = cv2.boundingRect(all_points)
    return (int(x), int(y), int(w), int(h))


def pad_and_resize(
    img: np.ndarray,
    bbox: tuple[int, int, int, int] | None = None,
    padding: int = 15,
    target_size: int = 224,
) -> np.ndarray:
    """Crop, pad to 1:1 aspect ratio square, and resize to target dimension.

    Padding prevents aspect ratio distortion that would otherwise warp
    ink stroke angles and curvature metrics.

    Args:
        img: Binary or grayscale image.
        bbox: Optional (x, y, w, h) bounding box to crop.
        padding: Pixel padding border added around bounding box.
        target_size: Target square dimension (default 224 for ResNet).

    Returns:
        RGB uint8 NumPy array of shape (target_size, target_size, 3).
    """
    height, width = img.shape[:2]

    if bbox is not None:
        x, y, w, h = bbox
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(width, x + w + padding)
        y2 = min(height, y + h + padding)
        cropped = img[y1:y2, x1:x2]
    else:
        cropped = img

    ch, cw = cropped.shape[:2]
    if ch == 0 or cw == 0:
        cropped = img
        ch, cw = cropped.shape[:2]

    # Center cropped signature in a square canvas of dimension max(ch, cw)
    max_dim = max(ch, cw)
    padded = np.zeros((max_dim, max_dim), dtype=np.uint8)
    y_off = (max_dim - ch) // 2
    x_off = (max_dim - cw) // 2
    padded[y_off : y_off + ch, x_off : x_off + cw] = cropped

    # Resize to 224x224 using area interpolation
    resized = cv2.resize(padded, (target_size, target_size), interpolation=cv2.INTER_AREA)

    # Convert single channel to 3-channel RGB
    rgb = np.stack([resized, resized, resized], axis=-1)
    return rgb


def normalize_image_to_tensor(rgb_image: np.ndarray) -> torch.Tensor:
    """Apply ImageNet normalization and convert RGB array to PyTorch Tensor.

    Args:
        rgb_image: RGB uint8 array (224, 224, 3) with values in [0, 255].

    Returns:
        PyTorch float32 Tensor of shape (1, 3, 224, 224).
    """
    # Scale to [0.0, 1.0]
    normalized = rgb_image.astype(np.float32) / 255.0

    # ImageNet standard normalization: (x - mean) / std
    normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD

    # Transpose from (H, W, C) to (C, H, W)
    transposed = np.transpose(normalized, (2, 0, 1))

    # Add batch dimension: (1, C, H, W)
    tensor = torch.from_numpy(transposed).unsqueeze(0)
    return tensor


def preprocess_signature_bytes(image_bytes: bytes) -> torch.Tensor:
    """End-to-end computer vision preprocessing from image bytes to PyTorch tensor.

    Args:
        image_bytes: Raw bytes of uploaded signature image.

    Returns:
        Normalized tensor (1, 3, 224, 224) ready for model forward pass.
    """
    img = decode_image(image_bytes)
    binary = isolate_ink_strokes(img)
    bbox = extract_contour_bbox(binary)
    rgb = pad_and_resize(binary, bbox=bbox, target_size=224)
    tensor = normalize_image_to_tensor(rgb)
    return tensor
