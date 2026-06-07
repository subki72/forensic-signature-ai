---
title: Forensic Signature AI
emoji: 🖋️
colorFrom: yellow
colorTo: red
sdk: docker
pinned: false
---

# Legal Document AI - Forensic Signature Verification

An end-to-end artificial intelligence system for verifying the authenticity of handwritten signatures on legal documents. The system combines computer vision preprocessing with a fine-tuned Siamese Network to detect forgeries through structural comparison of ink stroke anatomy.

## Architecture V2.1

| Layer | Technology | Purpose |
|---|---|---|
| Computer Vision | OpenCV | Otsu binarization, morphological closing, contour merging, and aspect-ratio aware padding |
| Deep Learning | PyTorch | Siamese Network (ResNet-18 backbone) with a 128-dimensional projection head |
| Backend API | FastAPI | Asynchronous REST endpoint for model serving |
| Frontend | React + Vite | Interactive web interface with normalized Cosine Similarity scoring (0-100%) |

## Performance

The system utilizes a Cosine Similarity metric to compare the 128-dimensional embeddings of two signatures. In the frontend UI, this raw score (-1.0 to 1.0) is normalized to a 0% to 100% scale for better UX.

| Metric | Raw Cosine Similarity | UI Normalized (0-100%) |
|---|---|---|
| Authentic Signatures (Mean) | ~0.75 | ~87.5% |
| Forged Signatures (Mean) | ~0.27 | ~63.5% |
| **System Threshold** | **0.60** | **80.0%** |
| Score Gap | 0.48 | 24.0% |

*Thresholds are dynamically loaded from `models/model_config.json`.*

## Repository Structure

```
.
├── api/
│   └── main.py                    # FastAPI application, preprocessing, and model serving
├── frontend/                      # React (Vite) web interface
│   ├── src/
│   │   ├── App.jsx                # Main application component (handles score normalization)
│   │   └── index.css              # Design system and styling
├── models/                        # Trained model weights (Tracked via Git LFS)
│   ├── forensic_signature_v2.pt   # V2.1 Siamese Network weights
│   └── model_config.json          # Calibrated thresholds and metrics
├── data/                          # Raw and processed datasets (git-ignored)
├── notebooks/
│   ├── 01_thresholding_test.ipynb # OpenCV pipeline R&D
│   ├── 02_train_siamese.ipynb     # Initial V1 training
│   └── 03_train_siamese_v2.ipynb  # V2.1 Two-phase training with AMP & Data Augmentation
├── test_api.py                    # Automated API endpoint tests
├── requirements.txt               # Python dependencies (CPU-only PyTorch for deployment)
├── Dockerfile                     # Hugging Face Spaces deployment config
├── .dockerignore
└── .gitignore
```

## Quick Start

### 1. Environment Setup

```bash
conda create -n legal_ai_env python=3.10
conda activate legal_ai_env
pip install -r requirements.txt
```

*Note: For local training with a GPU, install the CUDA version of PyTorch instead of the CPU version listed in `requirements.txt`.*

### 2. Running the Backend

```bash
uvicorn api.main:app --reload
```

The API will be available at `http://localhost:8000/docs`.

### 3. Running the Frontend

```bash
cd frontend
npm install
npm run dev
```

The web interface will be available at `http://localhost:5173`.

## Deployment

The system is designed for a split deployment architecture:

- **Frontend** is deployed to [Vercel](https://vercel.com/) as a static React application.
- **Backend** is deployed to [Hugging Face Spaces](https://huggingface.co/spaces) using the included Dockerfile.
- **Model Weights** are tracked using Git Large File Storage (LFS).

### Environment Variables

| Variable | Where | Value |
|---|---|---|
| `VITE_API_URL` | Vercel | The public URL of the Hugging Face Space (e.g. `https://user-space.hf.space`) |

## API Reference

### `POST /verify`

Accepts a `multipart/form-data` request with two image files:

| Field | Type | Description |
|---|---|---|
| `file_asli` | `UploadFile` | Reference (genuine) signature image |
| `file_uji` | `UploadFile` | Questioned signature image |

**Response:**

```json
{
  "verification": {
    "status": "AUTHENTIC (VERIFIED)",
    "similarity_score": 0.8542,
    "system_threshold": 0.601,
    "analysis": "Ink stroke anatomy is structurally consistent with the reference specimen."
  }
}
```

## Training

The model is trained using the CEDAR Signature Dataset. The training pipeline has been upgraded to a robust two-phase approach in `notebooks/03_train_siamese_v2.ipynb`:

1. **Phase 1 (Frozen Backbone):** Trains only the 128-dim projection head to adapt to the signature domain.
2. **Phase 2 (Full Fine-tuning):** Unfreezes the ResNet-18 backbone with a lower learning rate.
3. **Optimizations:** Utilizes Automatic Mixed Precision (AMP), aggressive Gaussian Blur augmentations, and a CosineEmbeddingLoss with a margin of 0.3.

After training, the model weights and threshold configuration are exported to the `models/` directory.

## License

This project is intended for educational and research purposes.
