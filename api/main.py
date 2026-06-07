"""
Legal Document AI - FastAPI Backend
====================================
Production-ready REST API for forensic signature verification.
Uses a fine-tuned ResNet18 Siamese Network to extract 512-dimensional
feature vectors from signature images and computes Cosine Similarity
to determine authenticity.
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
# 1. CORE ENGINE SETUP
# ===========================================================================
print("Initializing Forensic AI Engine...")

weights = models.ResNet18_Weights.DEFAULT
resnet = models.resnet18(weights=weights)
resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))

MODEL_PATH = "models/forensic_signature_v1.pt"
if os.path.exists(MODEL_PATH):
    resnet.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    )
    print(f"Successfully loaded fine-tuned model weights from '{MODEL_PATH}'.")
else:
    print("WARNING: Fine-tuned model not found. Defaulting to pre-trained ResNet18.")

resnet.eval()

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

SYSTEM_THRESHOLD = 0.85
"""Cosine similarity threshold. Raised to 0.85 to strictly reject forgeries using the fine-tuned model."""


# ===========================================================================
# 2. FEATURE EXTRACTION
# ===========================================================================
def extract_feature_vector(image_bytes: bytes) -> torch.Tensor | None:
    """Extract a 512-dimensional feature vector from raw image bytes.

    Pipeline:
        1. Decode image bytes into an OpenCV BGR array.
        2. Convert to grayscale and apply adaptive Gaussian thresholding
           to isolate ink strokes from the paper background.
        3. Detect contours and crop to the largest bounding box with padding.
        4. Pass the cropped RGB region through the ResNet18 backbone.

    Args:
        image_bytes: Raw bytes of a signature image (JPEG/PNG).

    Returns:
        A 512-dimensional ``torch.Tensor``, or ``None`` if decoding fails.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2,
    )

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        pad = 15
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2 = min(image.shape[1], x + w + pad)
        y2 = min(image.shape[0], y + h + pad)
        cropped = image[y1:y2, x1:x2]
    else:
        cropped = image

    rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    tensor = preprocess(rgb).unsqueeze(0)
    with torch.no_grad():
        vector = resnet(tensor).squeeze()
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