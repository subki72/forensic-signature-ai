"""Configuration settings for Forensic Signature AI backend.

Uses Pydantic Settings for type-safe environment variable management
and validation.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings and configuration parameters."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Project metadata
    PROJECT_NAME: str = "Legal Document AI - Forensic Signature Verification"
    VERSION: str = "2.2.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False

    # Server binding
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # CORS configuration
    CORS_ORIGINS: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "https://huggingface.co",
        ]
    )

    # File validation & safety limits
    MAX_UPLOAD_SIZE_BYTES: int = 5 * 1024 * 1024  # 5 MB maximum per image
    MAX_IMAGE_DIMENSION: int = 4096  # Max width/height to prevent decompression bombs
    ALLOWED_MIME_TYPES: set[str] = Field(
        default_factory=lambda: {
            "image/jpeg",
            "image/jpg",
            "image/png",
            "image/webp",
        }
    )

    # ML Model & Threshold Configuration
    MODEL_PATH: str = "models/forensic_signature_v2.pt"
    CONFIG_PATH: str = "models/model_config.json"
    EMBEDDING_DIM: int = 128
    
    # 3-Tier Biometric Classification Thresholds
    AUTHENTIC_THRESHOLD: float = 0.70  # >= 0.70 Cosine Sim -> Authentic (UI >= 85%)
    INCONCLUSIVE_THRESHOLD: float = 0.55  # 0.55 - 0.70 -> Inconclusive (UI 77.5% - 85%)
    # < 0.55 -> Forgery / Non-identical (UI < 77.5%)

    # Security & Optional Authentication
    API_KEY: str | None = None
    REQUIRE_API_KEY: bool = False


# Global settings singleton
settings = Settings()
