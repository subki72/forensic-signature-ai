"""Signature verification endpoint with threadpool offload and strict validation."""

import time
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from api.config import settings
from api.cv_pipeline import ImageProcessingError, preprocess_signature_bytes
from api.engine import model_manager
from api.logging_config import logger
from api.schemas import (
    ErrorResponse,
    VerificationResponse,
    VerificationResult,
)

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


def _compute_forensic_similarity(
    raw_reference: bytes, raw_questioned: bytes
) -> float:
    """CPU-bound worker executing CV preprocessing and neural inference.

    Executes in a threadpool worker to avoid starving the asyncio event loop.
    """
    # 1. Computer vision preprocessing (Otsu binarization, bounding box, padding)
    tensor_reference = preprocess_signature_bytes(raw_reference)
    tensor_questioned = preprocess_signature_bytes(raw_questioned)

    # 2. Extract L2-normalized 128-dimensional embeddings
    emb_reference = model_manager.extract_embedding(tensor_reference)
    emb_questioned = model_manager.extract_embedding(tensor_questioned)

    # 3. Calculate Cosine Similarity on unit hypersphere
    similarity = model_manager.compute_similarity(emb_reference, emb_questioned)
    return similarity


@router.post(
    "/verify",
    response_model=VerificationResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image format or decoding failure"},
        413: {"model": ErrorResponse, "description": "Uploaded image exceeds size limits"},
        415: {"model": ErrorResponse, "description": "Unsupported media type"},
        500: {"model": ErrorResponse, "description": "Internal verification error"},
    },
    summary="Verify Signature Authenticity",
    description="Compares a reference (genuine) specimen against a questioned signature using a Siamese Network.",
)
async def verify_signature(
    file_asli: UploadFile = File(
        ...,
        description="Reference (authentic) specimen image file (JPEG/PNG/WEBP, max 5MB)",
    ),
    file_uji: UploadFile = File(
        ...,
        description="Questioned signature image file (JPEG/PNG/WEBP, max 5MB)",
    ),
) -> VerificationResponse:
    """Execute forensic signature comparison between reference and questioned documents."""
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

    # 3. Execute CPU-bound inference in threadpool worker (prevents event loop freeze)
    try:
        similarity = await run_in_threadpool(
            _compute_forensic_similarity, bytes_reference, bytes_questioned
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

    # 4. Evaluate 3-tier verdict and normalized percentage
    verdict, status_label, confidence_band, analysis = model_manager.evaluate_verdict(similarity)
    normalized_pct = round(((similarity + 1.0) / 2.0) * 100.0, 2)
    elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

    logger.info(
        "Verification completed in %.2f ms | Similarity: %.4f (%s%%) | Verdict: %s",
        elapsed_ms,
        similarity,
        normalized_pct,
        verdict.value,
    )

    result = VerificationResult(
        verdict=verdict,
        status=status_label,
        similarity_score=similarity,
        normalized_percentage=normalized_pct,
        system_threshold=round(model_manager.active_threshold, 4),
        confidence_band=confidence_band,
        analysis=analysis,
        latency_ms=elapsed_ms,
    )

    return VerificationResponse(verification=result)
