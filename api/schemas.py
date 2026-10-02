"""Pydantic data transfer objects (DTOs) and API schemas.

Provides explicit request and response models for verification,
health monitoring, and error handling.

Version: 2.3.0 -- Two-Phase Verification Pipeline.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class VerdictEnum(str, Enum):
    """3-Tier biometric verification verdict."""

    AUTHENTIC = "AUTHENTIC"
    INCONCLUSIVE = "INCONCLUSIVE"
    FORGERY = "FORGERY"


class VerificationResult(BaseModel):
    """Detailed verification verdict and analysis payload (Two-Phase Pipeline)."""

    verdict: VerdictEnum = Field(
        ...,
        description="Categorical verdict: AUTHENTIC, INCONCLUSIVE, or FORGERY.",
    )
    status: str = Field(
        ...,
        description="Human-readable status label with formatting.",
        example="AUTHENTIC (VERIFIED)",
    )
    similarity_score: float = Field(
        ...,
        description="Raw cosine similarity score (-1.0 to 1.0).",
        ge=-1.0,
        le=1.0,
        example=0.8245,
    )
    normalized_percentage: float = Field(
        ...,
        description="UI-normalized similarity percentage (0% to 100%).",
        ge=0.0,
        le=100.0,
        example=91.23,
    )
    system_threshold: float = Field(
        ...,
        description="Calibrated threshold score used for primary classification.",
        example=0.70,
    )
    confidence_band: str = Field(
        ...,
        description="Confidence band classification (High Confidence, Marginal, Inconclusive).",
        example="High Confidence Match",
    )
    analysis: str = Field(
        ...,
        description="Forensic ink stroke and morphological anatomy assessment.",
    )
    disclaimer: str = Field(
        default=(
            "LEGAL DISCLAIMER: This automated AI verification provides supplemental "
            "comparative screening and does NOT constitute certified forensic legal testimony. "
            "Borderline or questioned documents must be verified by a qualified forensic document examiner."
        ),
        description="Statutory legal and evidentiary limitation disclaimer.",
    )
    latency_ms: float = Field(
        ...,
        description="Total inference pipeline latency in milliseconds.",
        example=124.5,
    )

    # -- Two-Phase Pipeline Telemetry -----------------------------------------
    macro_score: float = Field(
        ...,
        description=(
            "Stage 1 (Macro-Geometric) composite score [0.0-1.0]. "
            "Weighted composite of HPP correlation, VPP correlation, and ORB inlier ratio."
        ),
        ge=0.0,
        le=1.0,
        example=0.82,
    )
    hpp_corr: float = Field(
        ...,
        description="Horizontal Projection Profile cosine correlation [0.0-1.0].",
        ge=0.0,
        le=1.0,
        example=0.78,
    )
    vpp_corr: float = Field(
        ...,
        description="Vertical Projection Profile cosine correlation [0.0-1.0].",
        ge=0.0,
        le=1.0,
        example=0.85,
    )
    orb_ratio: float = Field(
        ...,
        description="ORB keypoint Lowe's ratio-test inlier match ratio [0.0-1.0].",
        ge=0.0,
        le=1.0,
        example=0.22,
    )
    micro_score: float = Field(
        ...,
        description=(
            "Stage 2 (Micro-Stroke Siamese ResNet) raw cosine similarity [-1.0 to 1.0]. "
            "Set to -1.0 if Stage 1 rejected the pair (fail-fast, Siamese not executed)."
        ),
        ge=-1.0,
        le=1.0,
        example=0.82,
    )
    stage_rejected: bool = Field(
        ...,
        description="True if the pair was rejected by Stage 1 (macro geometry gate) without running Stage 2.",
        example=False,
    )
    rejection_stage: Optional[str] = Field(
        default=None,
        description=(
            "Which pipeline stage produced the FORGERY verdict. "
            "'macro' = Stage 1 shape mismatch, 'micro' = Stage 2 stroke kinematic anomaly, "
            "None = verdict is AUTHENTIC or INCONCLUSIVE."
        ),
        example="macro",
    )


class VerificationResponse(BaseModel):
    """Standard response model for POST /verify endpoint."""

    verification: VerificationResult


class HealthResponse(BaseModel):
    """Liveness probe response model for GET /health."""

    status: str = Field(default="healthy", example="healthy")
    version: str = Field(default="2.3.0", example="2.3.0")
    environment: str = Field(default="production", example="production")
    service: str = Field(
        default="Forensic Signature Verification API",
        example="Forensic Signature Verification API",
    )


class ReadinessResponse(BaseModel):
    """Readiness probe response model for GET /ready."""

    status: str = Field(default="ready", example="ready")
    model_loaded: bool = Field(..., description="Whether model weights are loaded in RAM.")
    model_file_size_bytes: int = Field(..., description="Size of model weight artifact on disk.")
    weights_verified: bool = Field(..., description="Whether weights passed integrity validation.")
    device: str = Field(default="cpu", example="cpu")
    calibrated_threshold: float = Field(..., description="Active decision threshold in use.")
    model_version: str = Field(default="v2.1", example="v2.1")


class ErrorResponse(BaseModel):
    """Standardized API error response payload."""

    error: str = Field(..., description="Error message title.")
    detail: Optional[str] = Field(default=None, description="Detailed diagnostic context if safe to expose.")
