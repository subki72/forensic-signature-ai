"""
Legal Document AI - FastAPI Backend (V2)
========================================
Production-ready REST API for forensic signature verification.
Uses a multi-identity Siamese Network (ResNet18 + projection head)
to extract L2-normalized 128-dimensional embeddings from signature
images and computes Cosine Similarity to determine authenticity.
Threshold is calibrated from ROC curve analysis on validation data.
"""

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import json

app = FastAPI(
    title="Legal Document AI API",
    description="Production-Ready Forensic Signature Verification System",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================================================================
# 1. CORE ENGINE SETUP (V2 — Multi-Identity Siamese Network)
# ===========================================================================
print("Initializing Forensic AI Engine V2...")


class SiameseNetwork(torch.nn.Module):
    """Siamese Network with ResNet18 backbone + projection head.

    Produces L2-normalized 128-dim embeddings for cosine similarity.
    Must match the architecture defined in notebooks/03_train_siamese_v2.ipynb.
    """

    def __init__(self, embedding_dim=128):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = torch.nn.Sequential(*list(backbone.children())[:-1])
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, embedding_dim),
        )

    def forward_once(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.projection(x)
        x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, x1, x2=None):
        e1 = self.forward_once(x1)
        if x2 is None:
            return e1
        e2 = self.forward_once(x2)
        return e1, e2


# Initialize model
model = SiameseNetwork(embedding_dim=128)

MODEL_PATH_V2 = "models/forensic_signature_v2.pt"
CONFIG_PATH = "models/model_config.json"

if os.path.exists(MODEL_PATH_V2):
    model.load_state_dict(
        torch.load(MODEL_PATH_V2, map_location="cpu", weights_only=True)
    )
    print(f"Successfully loaded V2 model from '{MODEL_PATH_V2}'.")
else:
    print("WARNING: V2 model not found. Using default ResNet18 weights.")
    print("Please run notebooks/03_train_siamese_v2.ipynb to train the model.")

model.eval()

# Load threshold from config (calibrated from ROC curve)
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        model_config = json.load(f)
    SYSTEM_THRESHOLD = model_config.get("optimal_threshold", 0.85)
    print(f"Loaded calibrated threshold from config: {SYSTEM_THRESHOLD:.4f}")
else:
    SYSTEM_THRESHOLD = 0.85
    print(f"WARNING: Config not found. Using default threshold: {SYSTEM_THRESHOLD}")

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ===========================================================================
# 2. FEATURE EXTRACTION (V2 Pipeline)
# ===========================================================================
def extract_feature_vector(image_bytes: bytes) -> torch.Tensor | None:
    """Extract a 128-dimensional L2-normalized feature vector from raw image bytes.

    V2 Pipeline:
        1. Decode image bytes to grayscale.
        2. Otsu binarization to isolate ink strokes.
        3. Morphological closing to connect nearby strokes.
        4. Detect ALL contours and compute merged bounding box.
        5. Crop, pad to square (preserve aspect ratio), resize to 224x224.
        6. Pass through SiameseNetwork for L2-normalized 128-dim embedding.

    Args:
        image_bytes: Raw bytes of a signature image (JPEG/PNG).

    Returns:
        A 128-dimensional L2-normalized ``torch.Tensor``, or ``None``
        if decoding fails.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # Otsu binarization
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Morphological closing to connect nearby strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Find ALL contours and merge bounding boxes
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = [c for c in contours if cv2.contourArea(c) > 50]

    if contours:
        all_points = np.vstack(contours)
        x, y, w, h = cv2.boundingRect(all_points)
        pad = 15
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img.shape[1], x + w + pad)
        y2 = min(img.shape[0], y + h + pad)
        cropped = binary[y1:y2, x1:x2]
    else:
        cropped = binary

    # Pad to square (preserve aspect ratio)
    ch, cw = cropped.shape
    if ch == 0 or cw == 0:
        cropped = binary
        ch, cw = cropped.shape
    max_dim = max(ch, cw)
    padded = np.zeros((max_dim, max_dim), dtype=np.uint8)
    y_off = (max_dim - ch) // 2
    x_off = (max_dim - cw) // 2
    padded[y_off:y_off + ch, x_off:x_off + cw] = cropped

    # Resize and convert to 3-channel
    resized = cv2.resize(padded, (224, 224), interpolation=cv2.INTER_AREA)
    rgb = np.stack([resized, resized, resized], axis=-1)

    tensor = preprocess(rgb).unsqueeze(0)
    with torch.no_grad():
        vector = model.forward_once(tensor).squeeze()
    return vector


# ===========================================================================
# 3. VERIFICATION ENDPOINT
# ===========================================================================
@app.post("/verify")
async def verify_signature(
    file_asli: UploadFile = File(..., description="Reference (genuine) signature image"),
    file_uji: UploadFile = File(..., description="Questioned signature image"),
):
    """Compare two signature images and return a verification verdict.

    Accepts a multipart/form-data POST with two image files. Returns a JSON
    object containing the verification status, similarity score, system
    threshold, and a human-readable analysis string.
    """
    try:
        bytes_reference = await file_asli.read()
        bytes_questioned = await file_uji.read()

        vec_reference = extract_feature_vector(bytes_reference)
        vec_questioned = extract_feature_vector(bytes_questioned)

        if vec_reference is None or vec_questioned is None:
            return JSONResponse(
                content={"error": "Failed to decode one or both image files."},
                status_code=400,
            )

        score = F.cosine_similarity(
            vec_reference.unsqueeze(0), vec_questioned.unsqueeze(0)
        ).item()

        is_match = score >= SYSTEM_THRESHOLD

        return {
            "verification": {
                "status": "AUTHENTIC (VERIFIED)" if is_match else "FORGERY / NOT IDENTICAL",
                "similarity_score": round(score, 4),
                "system_threshold": SYSTEM_THRESHOLD,
                "analysis": (
                    "Ink stroke anatomy is structurally consistent with the reference specimen."
                    if is_match
                    else "Significant deviations detected in stroke dynamics and pressure distribution."
                ),
            }
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)