"""Health and readiness monitoring endpoints.

Enables container orchestrators (Docker, Kubernetes, Hugging Face Spaces)
to assess liveness and model initialization status.
"""

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from api.config import settings
from api.engine import model_manager
from api.schemas import HealthResponse, ReadinessResponse

router = APIRouter(tags=["Monitoring"])


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Shallow Liveness Probe",
    description="Returns HTTP 200 immediately if the API web process is alive.",
)
async def liveness_probe() -> HealthResponse:
    """Check application server liveness."""
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
        service=settings.PROJECT_NAME,
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    status_code=status.HTTP_200_OK,
    summary="Deep Readiness Probe",
    description="Verifies whether ML model weights and thresholds are loaded in RAM and ready for inference.",
)
async def readiness_probe():
    """Check deep system and model weight readiness."""
    readiness = model_manager.get_readiness()

    if not readiness["model_loaded"]:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=readiness,
        )

    return ReadinessResponse(**readiness)
