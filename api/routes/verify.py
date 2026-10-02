"""Signature verification endpoint with two-phase pipeline, threadpool offload,
strict validation, and rate limiting.

Pipeline:
  Stage 1 (fast, < 15ms): Macro-Geometric Screening via IoU + Hu Moments + NCC.
                          Input: padded, centered, 224x224 single-channel binary.
  Stage 2 (deep, ~80ms):  Siamese ResNet micro-stroke kinematic analysis.
                          Input: ImageNet-normalized 224x224 tensor.
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


def _run_two_phase_pipeline(raw_reference: bytes, raw_questioned: bytes) -> dict:
    """CPU-bound worker executing the complete 2-phase forensic pipeline.

    Executes in a threadpool worker to avoid starving the asyncio event loop.

    PREPROCESSING (critical for accuracy):
      pad_and_resize() outputs a 224x224x3 RGB image where:
        1. The signature is cropped to its bounding box
        2. Padded to a 1:1 square (aspect-ratio preserved)
        3. Resized to 224x224 with INTER_AREA interpolation
        4. Centered on the canvas

      For Stage 1: We extract the single channel ([:, :, 0]) to get a 224x224
      grayscale image where both signatures are at the same scale, position,
      and canvas size. This is ESSENTIAL for IoU (pixel overlap) to work —
      without normalization, two photos of the same signature at different
      zoom levels would have zero overlap.

      For Stage 2: We apply ImageNet normalization to the RGB output to
      produce the standard Siamese ResNet input tensor.

    Returns a dict with all verdict + telemetry fields.
    """
    # -- Reference image preprocessing ----------------------------------------
    img_ref = decode_image(raw_reference)
    binary_ref = isolate_ink_strokes(img_ref)
    bbox_ref = extract_contour_bbox(binary_ref)
    rgb_ref = pad_and_resize(binary_ref, bbox=bbox_ref, target_size=224)
    padded_ref = rgb_ref[:, :, 0]                     # 224x224 single-channel for Stage 1
    tensor_ref = normalize_image_to_tensor(rgb_ref)    # (1, 3, 224, 224) for Stage 2

    # -- Questioned image preprocessing ----------------------------------------
    img_que = decode_image(raw_questioned)
    binary_que = isolate_ink_strokes(img_que)
    bbox_que = extract_contour_bbox(binary_que)
    rgb_que = pad_and_resize(binary_que, bbox=bbox_que, target_size=224)
    padded_que = rgb_que[:, :, 0]                      # 224x224 single-channel for Stage 1
    tensor_que = normalize_image_to_tensor(rgb_que)     # (1, 3, 224, 224) for Stage 2

    # -- Execute 2-phase pipeline ----------------------------------------------
    return model_manager.run_two_stage_pipeline(
        padded_reference=padded_ref,
        padded_questioned=padded_que,
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
        "Stage 1 rejects cross-signer pairs via macro-geometric screening (IoU + Hu Moments + NCC). "
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
        pixel_iou=pipeline_result["pixel_iou"],
        hu_similarity=pipeline_result["hu_similarity"],
        pixel_corr=pipeline_result["pixel_corr"],
        micro_score=pipeline_result["micro_score"],
        stage_rejected=pipeline_result["stage_rejected"],
        rejection_stage=pipeline_result["rejection_stage"],
    )

    return VerificationResponse(verification=result)
