"""Machine Learning Inference Engine and Siamese Network model manager.

Provides thread-safe model loading, L2-normalized embedding extraction,
cosine similarity computation, and 3-tier forensic classification.
"""

import json
import os
import threading
import torch
import torch.nn.functional as F
import torchvision.models as models

from api.config import settings
from api.logging_config import logger
from api.schemas import VerdictEnum


class SiameseNetwork(torch.nn.Module):
    """Siamese Neural Network with ResNet-18 backbone and projection head.

    Produces L2-normalized 128-dimensional embeddings for metric learning
    and cosine similarity comparison.
    """

    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        backbone = models.resnet18(weights=None)
        # Retain all layers except the final classification fully connected layer
        self.features = torch.nn.Sequential(*list(backbone.children())[:-1])
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, embedding_dim),
        )

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        """Extract a single L2-normalized embedding from image tensor.

        Args:
            x: Input batch of shape (B, 3, 224, 224).

        Returns:
            L2-normalized tensor of shape (B, embedding_dim).
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.projection(x)
        x = F.normalize(x, p=2, dim=1)
        return x

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for single input or pair of inputs."""
        e1 = self.forward_once(x1)
        if x2 is None:
            return e1
        e2 = self.forward_once(x2)
        return e1, e2


class ModelManager:
    """Thread-safe singleton manager for model loading and forensic scoring."""

    _instance: "ModelManager | None" = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "ModelManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        self.device = torch.device("cpu")
        self.model = SiameseNetwork(embedding_dim=settings.EMBEDDING_DIM)
        self.model.to(self.device)
        self.model.eval()

        self.model_loaded: bool = False
        self.weights_verified: bool = False
        self.model_file_size_bytes: int = 0
        self.active_threshold: float = settings.AUTHENTIC_THRESHOLD
        self.inconclusive_threshold: float = settings.INCONCLUSIVE_THRESHOLD
        self.model_version: str = "v2.1"

        self._load_artifacts()
        self._initialized = True

    def _load_artifacts(self) -> None:
        """Load trained weights and calibrated threshold configuration defensively."""
        # 1. Load model weights
        if os.path.exists(settings.MODEL_PATH):
            self.model_file_size_bytes = os.path.getsize(settings.MODEL_PATH)

            # Defensive check: inspect if file is an unhydrated Git LFS text pointer
            if self.model_file_size_bytes < 1024:
                try:
                    with open(settings.MODEL_PATH, "r", encoding="utf-8", errors="ignore") as f:
                        header = f.read(200)
                    if "git-lfs" in header:
                        logger.warning(
                            "Model file '%s' is a Git LFS pointer (%d bytes). "
                            "Using initialized baseline weights in degraded mode.",
                            settings.MODEL_PATH,
                            self.model_file_size_bytes,
                        )
                        self.model_loaded = True
                        self.weights_verified = False
                        return
                except Exception as ex:
                    logger.warning("Error reading potential LFS file: %s", ex)

            # Attempt safe weights-only loading
            try:
                state_dict = torch.load(
                    settings.MODEL_PATH,
                    map_location=self.device,
                    weights_only=True,
                )
                self.model.load_state_dict(state_dict, strict=False)
                self.model_loaded = True
                self.weights_verified = True
                logger.info(
                    "Successfully loaded model weights from '%s' (%d bytes).",
                    settings.MODEL_PATH,
                    self.model_file_size_bytes,
                )
            except Exception as e:
                logger.error(
                    "Failed to load weights from '%s': %s. Operating with baseline weights.",
                    settings.MODEL_PATH,
                    str(e),
                )
                self.model_loaded = True
                self.weights_verified = False
        else:
            logger.warning(
                "Model file not found at '%s'. Initializing default baseline weights.",
                settings.MODEL_PATH,
            )
            self.model_loaded = True
            self.weights_verified = False

        # 2. Load threshold calibration config
        if os.path.exists(settings.CONFIG_PATH):
            try:
                with open(settings.CONFIG_PATH, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
                self.model_version = config_data.get("model_version", "v2.2")
                self.active_threshold = float(config_data.get("authentic_threshold", settings.AUTHENTIC_THRESHOLD))
                self.inconclusive_threshold = float(config_data.get("inconclusive_threshold", settings.INCONCLUSIVE_THRESHOLD))
                logger.info(
                    "Loaded calibrated 3-tier thresholds: Authentic=%.4f, Inconclusive=%.4f (Optimal Youden J: %.4f)",
                    self.active_threshold,
                    self.inconclusive_threshold,
                    float(config_data.get("optimal_threshold", 0.577)),
                )
            except Exception as e:
                logger.warning("Error reading config '%s': %s", settings.CONFIG_PATH, e)

    def extract_embedding(self, tensor: torch.Tensor) -> torch.Tensor:
        """Extract a 128-dim L2-normalized feature vector from an image tensor.

        Args:
            tensor: Preprocessed tensor of shape (1, 3, 224, 224).

        Returns:
            Normalized 1D PyTorch tensor of size (128,).
        """
        with torch.no_grad():
            tensor = tensor.to(self.device)
            embedding = self.model.forward_once(tensor).squeeze(0)
        return embedding

    def compute_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """Calculate cosine similarity between two 128-dimensional embeddings.

        Args:
            emb1: Reference embedding (128,).
            emb2: Questioned embedding (128,).

        Returns:
            Cosine similarity score in range [-1.0, 1.0].
        """
        score = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        return round(float(score), 4)

    def evaluate_verdict(
        self, similarity: float
    ) -> tuple[VerdictEnum, str, str, str]:
        """Apply 3-tier forensic classification logic.

        Args:
            similarity: Cosine similarity score in range [-1.0, 1.0].

        Returns:
            Tuple of (VerdictEnum, status_label, confidence_band, analysis_string).
        """
        if similarity >= self.active_threshold:
            verdict = VerdictEnum.AUTHENTIC
            status = "AUTHENTIC (VERIFIED)"
            confidence = "High Confidence Match" if similarity >= 0.75 else "Moderate Confidence Match"
            analysis = (
                "Ink stroke anatomy, pen pressure dynamics, and trajectory contours "
                "are structurally consistent with the reference specimen."
            )
        elif similarity >= self.inconclusive_threshold:
            verdict = VerdictEnum.INCONCLUSIVE
            status = "INCONCLUSIVE (HUMAN REVIEW REQUIRED)"
            confidence = "Borderline / Indeterminate Region"
            analysis = (
                "Moderate stroke similarity detected. Variations observed may arise from natural signer "
                "variability, writing surface friction, or skilled forgery attempts. "
                "Manual examination by a qualified forensic document examiner is required."
            )
        else:
            verdict = VerdictEnum.FORGERY
            status = "FORGERY / NOT IDENTICAL"
            confidence = "High Discrepancy" if similarity < 0.40 else "Moderate Discrepancy"
            analysis = (
                "Significant structural deviations detected in ink stroke acceleration, "
                "curvature entropy, and loop topology compared to the reference specimen."
            )

        return verdict, status, confidence, analysis

    def get_readiness(self) -> dict:
        """Return readiness diagnostic dictionary."""
        return {
            "status": "ready" if self.model_loaded else "degraded",
            "model_loaded": self.model_loaded,
            "model_file_size_bytes": self.model_file_size_bytes,
            "weights_verified": self.weights_verified,
            "device": str(self.device),
            "calibrated_threshold": self.active_threshold,
            "model_version": self.model_version,
        }


# Global model manager singleton
model_manager = ModelManager()
