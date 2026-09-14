"""Unit tests for the computer vision preprocessing pipeline."""

import numpy as np
import pytest
import torch

from api.cv_pipeline import (
    ImageProcessingError,
    decode_image,
    extract_contour_bbox,
    isolate_ink_strokes,
    normalize_image_to_tensor,
    pad_and_resize,
    preprocess_signature_bytes,
)


def test_decode_image_success(genuine_signature_bytes_1: bytes):
    """Test successful decoding of valid PNG image bytes."""
    img = decode_image(genuine_signature_bytes_1)
    assert isinstance(img, np.ndarray)
    assert img.ndim == 2
    assert img.shape[0] > 0 and img.shape[1] > 0


def test_decode_image_empty():
    """Test that empty bytes raise ImageProcessingError."""
    with pytest.raises(ImageProcessingError, match="Image byte payload is empty"):
        decode_image(b"")


def test_decode_image_corrupt(corrupt_bytes: bytes):
    """Test that corrupt non-image bytes raise ImageProcessingError."""
    with pytest.raises(ImageProcessingError, match="Failed to decode image"):
        decode_image(corrupt_bytes)


def test_decode_image_decompression_guard(genuine_signature_bytes_1: bytes):
    """Test that dimensions exceeding max_dimension are rejected."""
    with pytest.raises(ImageProcessingError, match="exceed maximum allowed dimension"):
        decode_image(genuine_signature_bytes_1, max_dimension=50)


def test_isolate_ink_strokes(genuine_signature_bytes_1: bytes):
    """Test Otsu binarization and morphological closing produce binary ink strokes."""
    img = decode_image(genuine_signature_bytes_1)
    binary = isolate_ink_strokes(img)

    assert binary.shape == img.shape
    unique_vals = set(np.unique(binary))
    # Must only contain 0 (background) and 255 (ink strokes)
    assert unique_vals.issubset({0, 255})
    assert 255 in unique_vals  # Must have extracted some foreground ink


def test_extract_contour_bbox(genuine_signature_bytes_1: bytes):
    """Test extraction of merged bounding box around signature strokes."""
    img = decode_image(genuine_signature_bytes_1)
    binary = isolate_ink_strokes(img)
    bbox = extract_contour_bbox(binary, min_area=20)

    assert bbox is not None
    x, y, w, h = bbox
    assert x >= 0 and y >= 0
    assert w > 10 and h > 10


def test_extract_contour_bbox_empty_canvas():
    """Test bounding box extraction on blank canvas returns None."""
    blank = np.zeros((200, 200), dtype=np.uint8)
    bbox = extract_contour_bbox(blank, min_area=50)
    assert bbox is None


def test_pad_and_resize(genuine_signature_bytes_1: bytes):
    """Test padding to 1:1 square aspect ratio and resizing to 224x224 RGB."""
    img = decode_image(genuine_signature_bytes_1)
    binary = isolate_ink_strokes(img)
    bbox = extract_contour_bbox(binary)
    rgb = pad_and_resize(binary, bbox=bbox, target_size=224)

    assert isinstance(rgb, np.ndarray)
    assert rgb.shape == (224, 224, 3)
    assert rgb.dtype == np.uint8


def test_normalize_image_to_tensor():
    """Test ImageNet normalization and tensor conversion."""
    sample_rgb = np.full((224, 224, 3), fill_value=128, dtype=np.uint8)
    tensor = normalize_image_to_tensor(sample_rgb)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32


def test_preprocess_signature_bytes_pipeline(genuine_signature_bytes_1: bytes):
    """Test complete end-to-end preprocessing pipeline."""
    tensor = preprocess_signature_bytes(genuine_signature_bytes_1)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 224, 224)
