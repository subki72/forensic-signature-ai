"""FastAPI Application Entrypoint for Forensic Signature AI.

Provides production-ready configuration, lifespan management, secure CORS,
exception handling, and modular route inclusion.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.config import settings
from api.engine import model_manager
from api.logging_config import logger
from api.routes.health import router as health_router
from api.routes.verify import router as verify_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for startup and shutdown hooks."""
    logger.info("Starting up %s (v%s)...", settings.PROJECT_NAME, settings.VERSION)
    readiness = model_manager.get_readiness()
    logger.info(
        "Engine Status: %s | Model Loaded: %s | Weights Verified: %s | Threshold: %.4f",
        readiness["status"],
        readiness["model_loaded"],
        readiness["weights_verified"],
        readiness["calibrated_threshold"],
    )
    yield
    logger.info("Shutting down %s...", settings.PROJECT_NAME)


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "Production-Ready Forensic Signature Verification System using a "
        "Siamese Neural Network (ResNet-18) and computer vision ink stroke analysis."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Secure CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Standardize HTTP exceptions into clean JSON response."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catch unhandled server errors without exposing internal stack traces."""
    logger.error("Unhandled exception at %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "An unexpected internal server error occurred."},
    )


# Include modular route handlers
app.include_router(health_router)
app.include_router(verify_router)


@app.get("/", tags=["General"], include_in_schema=False)
async def root():
    """Root landing endpoint providing service metadata."""
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs": "/docs",
        "health": "/health",
        "ready": "/ready",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )