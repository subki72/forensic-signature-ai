"""Signature verification endpoint with two-phase pipeline, threadpool offload,
strict validation, and rate limiting.

Pipeline:
  Stage 1 (fast, < 10ms): Macro-Geometric Screening via HPP/VPP + ORB.
                          Input: bbox-CROPPED binary (paper background stripped).
  Stage 2 (deep, ~80ms):  Siamese ResNet micro-stroke kinematic analysis.
                          Only runs when Stage 1 passes.
"""

import time
from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from api.config import settings
from api.cv_pipeline import (
    ImageProcessingError,
    decode_image,
    isolate_ink_strokes,
    extract_contour_bbox,
    pad_and_resize,
    normalize_image_to_tensor,
)
from api.engine import model_manager
from api.logging_config import logger
from api.schemas import (
    ErrorResponse,
    VerificationResponse,
    VerificationResult,
)
from api.limiter import limiter

router = APIRouter(tags=["Verification"])

# Magic byte signatures for image validation
IMAGE_MAGIC_HEADERS = [
    b"\xff\xd8\xff",  # JPEG / JPG
    b"\x89PNG\r\n\x1a\n",  # PNG
    b"RIFF",  # WEBP (starts with RIFF....WEBP)
]


def _validate_image_bytes(data: bytes, filename: str) -> None:
    """Validate image payload size and magic byte signatures."""
    if not data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Uploaded file '{filename}' is empty (0 bytes).",
        )

    if len(data) > settings.MAX_UPLOAD_SIZE_BYTES:
        max_mb = settings.MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File '{filename}' exceeds maximum allowed size of {max_mb} MB.",
        )

    # Magic byte validation to prevent MIME spoofing
    has_valid_magic = any(data.startswith(magic) for magic in IMAGE_MAGIC_HEADERS)
    if not has_valid_magic:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"File '{filename}' does not contain valid JPEG, PNG, or WEBP image headers.",
        )


def _crop_to_signature(binary, bbox, padding: int = 10):
    """Crop a binary image to the signature bounding box + padding.

    Purpose: Stage 1 (HPP/VPP projection profiles + ORB) should compare ONLY
    the ink strokes, not the surrounding paper. Without cropping, ruled-paper
    lines, different page margins, and different photo framing corrupt the
    projection profiles and cause false-FORGERY on genuine same-person
    signatures taken from different papers or at different zoom levels.

    Args:
        binary: Full-frame binary uint8 image (white ink on black background).
        bbox: (x, y, w, h) tuple from extract_contour_bbox, or None.
        padding: Extra pixels to retain around the signature region.

    Returns:
        Cropped binary image, or original image if bbox is None.
    """
    if bbox is None:
        return binary

    h, w = binary.shape[:2]
    x, y, bw, bh = bbox
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w, x + bw + padding)
    y2 = min(h, y + bh + padding)
    cropped = binary[y1:y2, x1:x2]

    # Guard: degenerate crop falls back to original
    if cropped.size == 0:
        return binary
    return cropped


def _run_two_phase_pipeline(raw_reference: bytes, raw_questioned: bytes) -> dict:
    """CPU-bound worker executing the complete 2-phase forensic pipeline.

    Executes in a threadpool worker to avoid starving the asyncio event loop.

    Stage 1 receives bbox-CROPPED binary images (paper/background stripped).
    Stage 2 receives padded 224x224 tensors (standard Siamese preprocessing).

    Returns a dict with all verdict + telemetry fields.
    """
    # Decode + preprocess reference
    img_ref = decode_image(raw_reference)
    binary_ref = isolate_ink_strokes(img_ref)
    bbox_ref = extract_contour_bbox(binary_ref)
    binary_ref_for_stage1 = _crop_to_signature(binary_ref, bbox_ref)   # <-- Stage 1
    rgb_ref = pad_and_resize(binary_ref, bbox=bbox_ref, target_size=224)
    tensor_ref = normalize_image_to_tensor(rgb_ref)                     # <-- Stage 2

    # Decode + preprocess questioned
    img_que = decode_image(raw_questioned)
    binary_que = isolate_ink_strokes(img_que)
    bbox_que = extract_contour_bbox(binary_que)
    binary_que_for_stage1 = _crop_to_signature(binary_que, bbox_que)    # <-- Stage 1
    rgb_que = pad_and_resize(binary_que, bbox=bbox_que, target_size=224)
    tensor_que = normalize_image_to_tensor(rgb_que)                     # <-- Stage 2

    # Execute 2-phase pipeline
    return model_manager.run_two_stage_pipeline(
        binary_reference=binary_ref_for_stage1,
        binary_questioned=binary_que_for_stage1,
        tensor_reference=tensor_ref,
        tensor_questioned=tensor_que,
    )


@router.post(
    "/verify",
    response_model=VerificationResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image format or decoding failure"},
        413: {"model": ErrorResponse, "description": "Uploaded image exceeds size limits"},
        415: {"model": ErrorResponse, "description": "Unsupported media type"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded (10 requests/minute per IP)"},
        500: {"model": ErrorResponse, "description": "Internal verification error"},
    },
    summary="Verify Signature Authenticity (Two-Phase Pipeline)",
    description=(
        "Runs a 2-phase forensic verification: "
        "Stage 1 rejects cross-signer pairs via macro-geometric screening (HPP/VPP + ORB). "
        "Stage 2 detects skilled forgeries via Siamese ResNet micro-stroke analysis."
    ),
)
@limiter.limit("10/minute")
async def verify_signature(
    request: Request,
    file_asli: UploadFile = File(
        ...,
        description="Reference (authentic) specimen image file (JPEG/PNG/WEBP, max 5MB)",
    ),
    file_uji: UploadFile = File(
        ...,
        description="Questioned signature image file (JPEG/PNG/WEBP, max 5MB)",
    ),
) -> VerificationResponse:
    """Execute two-phase forensic signature comparison between reference and questioned documents."""
    start_time = time.perf_counter()

    # 1. Read byte streams asynchronously
    try:
        bytes_reference = await file_asli.read()
        bytes_questioned = await file_uji.read()
    except Exception as e:
        logger.error("Failed to read uploaded files: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read uploaded file streams.",
        )

    # 2. Input validation guard clauses (size limits & magic bytes)
    _validate_image_bytes(bytes_reference, file_asli.filename or "file_asli")
    _validate_image_bytes(bytes_questioned, file_uji.filename or "file_uji")

    # 3. Execute CPU-bound 2-phase pipeline in threadpool (prevents event loop freeze)
    try:
        pipeline_result = await run_in_threadpool(
            _run_two_phase_pipeline, bytes_reference, bytes_questioned
        )
    except ImageProcessingError as ipe:
        logger.warning("Image processing error during verification: %s", ipe)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ipe),
        )
    except Exception as exc:
        logger.error("Unexpected error during forensic comparison: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during signature feature extraction and analysis.",
        )

    elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

    logger.info(
        "Verification completed in %.2f ms | Macro=%.4f | Micro=%.4f | "
        "Stage1_rejected=%s | Verdict: %s",
        elapsed_ms,
        pipeline_result["macro_score"],
        pipeline_result["micro_score"],
        pipeline_result["stage_rejected"],
        pipeline_result["verdict"].value,
    )

    result = VerificationResult(
        verdict=pipeline_result["verdict"],
        status=pipeline_result["status"],
        similarity_score=pipeline_result["similarity_score"],
        normalized_percentage=pipeline_result["normalized_percentage"],
        system_threshold=round(model_manager.active_threshold, 4),
        confidence_band=pipeline_result["confidence_band"],
        analysis=pipeline_result["analysis"],
        latency_ms=elapsed_ms,
        # Two-phase telemetry
        macro_score=pipeline_result["macro_score"],
        hpp_corr=pipeline_result["hpp_corr"],
        vpp_corr=pipeline_result["vpp_corr"],
        orb_ratio=pipeline_result["orb_ratio"],
        micro_score=pipeline_result["micro_score"],
        stage_rejected=pipeline_result["stage_rejected"],
        rejection_stage=pipeline_result["rejection_stage"],
    )

    return VerificationResponse(verification=result)
