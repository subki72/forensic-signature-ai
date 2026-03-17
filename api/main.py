from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import os

app = FastAPI(
    title="Legal Document AI API", 
    description="Production-Ready Forensic Signature Verification System"
)

# ==========================================
# 1. CORE ENGINE SETUP
# ==========================================
print("Initializing Forensic AI Engine...")

# Initialize ResNet18 Architecture
weights = models.ResNet18_Weights.DEFAULT
resnet = models.resnet18(weights=weights)
resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))

# Load Fine-Tuned Weights
model_path = "models/detektif_forensik_v1.pt"
if os.path.exists(model_path):
    resnet.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    print("Successfully loaded fine-tuned model weights (v1).")
else:
    print("WARNING: Fine-tuned model not found. Defaulting to pre-trained ResNet18.")

resnet.eval()

# Image Preprocessing Pipeline
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ==========================================
# 2. FORENSIC EXTRACTION LOGIC
# ==========================================
def extract_dna_from_bytes(image_bytes):
    # Convert incoming bytes to OpenCV format
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None: return None
    
    # Preprocessing: Grayscale & Adaptive Thresholding
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Contour detection and bounding box extraction with padding
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        pad = 15
        x_p, y_p = max(0, x-pad), max(0, y-pad)
        w_p, h_p = min(image.shape[1]-x_p, w+(pad*2)), min(image.shape[0]-y_p, h+(pad*2))
        cropped = image[y_p:y_p+h_p, x_p:x_p+w_p]
    else:
        cropped = image
        
    # Feature extraction (512-Dimensional Vector)
    sig_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    input_tensor = preprocess(sig_rgb).unsqueeze(0)
    with torch.no_grad():
        dna = resnet(input_tensor).squeeze()
    return dna

# ==========================================
# 3. RESTRICTED VERIFICATION ENDPOINT
# ==========================================
@app.post("/verify")
async def verify_signature(
    file_asli: UploadFile = File(...), 
    file_uji: UploadFile = File(...)
):
    try:
        # Read raw image bytes
        bytes_asli = await file_asli.read()
        bytes_uji = await file_uji.read()
        
        # Extract Signature DNA
        dna_asli = extract_dna_from_bytes(bytes_asli)
        dna_uji = extract_dna_from_bytes(bytes_uji)
        
        if dna_asli is None or dna_uji is None:
            return JSONResponse(content={"error": "Failed to decode image data"}, status_code=400)
        
        # Calculate Cosine Similarity
        score = F.cosine_similarity(dna_asli.unsqueeze(0), dna_uji.unsqueeze(0)).item()
        
        # System Threshold (Derived from baseline forensic audit)
        system_threshold = 0.73
        
        is_match = score >= system_threshold
        
        # Output values kept in Indonesian to maintain frontend API contract
        status = "ASLI (TERVERIFIKASI)" if is_match else "PALSU / TIDAK IDENTIK"
        
        return {
            "verifikasi": {
                "status": status,
                "skor_kemiripan": round(score, 4),
                "threshold_sistem": system_threshold,
                "hasil_analisa": "Tanda tangan identik" if is_match else "Perbedaan tarikan tinta terdeteksi"
            }
        }
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)