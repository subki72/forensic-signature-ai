# Legal Document AI - Forensic Signature Verification

An end-to-end production-grade biometric artificial intelligence system for verifying handwritten signature authenticity on legal documents. The architecture combines computer vision ink stroke isolation (OpenCV) with a Siamese Neural Network (ResNet-18) to detect forgeries through structural anatomical comparison.

---

## 1. Architecture Overview (V2.2 Production Standard)

| Layer | Technology | Architectural Responsibility |
|---|---|---|
| **Computer Vision** | OpenCV (`cv2`) | Otsu binarization, morphological closing, contour point-cloud merging, aspect-ratio padding (224x224). |
| **Deep Learning** | PyTorch / Torchvision | Siamese Network (ResNet-18 backbone + projection head) trained via 2-phase Triplet Margin Loss with Cosine Distance. |
| **Backend API** | FastAPI / Uvicorn | Asynchronous REST gateway with threadpool offloading, Pydantic validation, CORS, and health probes. |
| **Frontend** | React 19 + Vite 8 | Glassmorphism UI with real-time telemetry, 3-tier verdict display, and legal disclaimers. |
| **DevOps & Testing** | Docker, Pytest | Non-root container hardening, Docker Compose local stack, and automated unit/integration tests (30/30 passing). |

### Non-Blocking Asynchronous Inference
To prevent blocking the single-threaded Python asyncio event loop during compute-heavy operations, all computer vision preprocessing (`cv_pipeline.py`) and neural network forward passes (`engine.py`) are dispatched to an asynchronous worker threadpool using `fastapi.concurrency.run_in_threadpool`. This guarantees predictable API response times, high throughput, and zero request starvation under concurrent loads.

---

## 2. Empirical Performance Benchmarks (V2.2 vs Baseline)

Trained and audited across **687 signers (14,626 total specimens: 7,313 genuine, 7,313 skilled forgeries)** with strict disjoint signer-level 80/20 splitting (zero data leakage):

| Evaluation Metric | ImageNet Baseline (V1) | Multi-Signer Triplet AI (V2.2) | Gain / Improvement | Status |
|---|:---:|:---:|:---:|:---:|
| **Score Separation Gap** | `0.0231` | **`0.5354`** | **+2,217% (>23x wider margin)** | **PASS** ($\ge 0.40$) |
| **ROC-AUC** | ~`0.5200` (Near Random) | **`0.8933` (~89.3%)** | **+71.8% Discriminative Power** | **High** |
| **Equal Error Rate (EER)** | ~`48.0%` | **`18.63%`** | **-61.2% Error Reduction** | Calibrated @ `0.5575` |
| **Optimal Threshold (Youden J)** | N/A | **`0.5770`** | **TPR: 80.2%, FPR: 16.8%** | Calibrated |
| **Genuine Pair Similarity (Mean)** | `0.9264` | **`0.7195`** | Tight intra-class consistency | Verified |
| **Skilled Forgery Similarity (Mean)** | `0.9033` | **`0.1841`** | Suppressed forgery response | Verified |
| **Training Regime** | Off-the-shelf ImageNet | 2-Phase Triplet Margin Loss | 10 Epochs Head + 30 Epochs Full | Complete |

---

## 3. Decision Framework (3-Tier Biometric Verdict)

To eliminate legal liability from binary false classifications, the system implements a calibrated 3-tier decision protocol:

| Raw Cosine Similarity | UI Normalized | Verdict Category | Operational Meaning |
|---|:---:|:---:|---|
| >= 0.70 | >= 85.0% | **`AUTHENTIC`** | High-confidence structural ink stroke match. |
| 0.55 - 0.70 | 77.5% - 85.0% | **`INCONCLUSIVE`** | **Human Review Required**: Natural variance or skilled forgery; requires forensic examiner review. |
| < 0.55 | < 77.5% | **`FORGERY`** | Significant stroke anatomy, loop curvature, and acceleration discrepancy. |

---

## 4. Repository Structure

```text
.
├── api/
│   ├── __init__.py                # Package initializer
│   ├── main.py                    # FastAPI application, CORS, lifespan, and error handlers
│   ├── config.py                  # Type-safe Pydantic Settings
│   ├── schemas.py                 # Pydantic DTOs, Enums, and response models
│   ├── logging_config.py          # Structured logging configuration
│   ├── engine.py                  # SiameseNetwork, ModelManager singleton, similarity logic
│   ├── cv_pipeline.py             # Pure OpenCV ink stroke isolation and padding functions
│   └── routes/
│       ├── health.py              # GET /health (liveness) and GET /ready (readiness)
│       └── verify.py              # POST /verify with threadpool offload & MIME validation
├── frontend/                      # React 19 (Vite 8) Single Page Application
│   ├── public/
│   │   └── favicon.svg            # Custom HD legal shield & signature vector
│   ├── src/
│   │   ├── components/
│   │   │   ├── DropzoneCard.jsx   # Drag-and-drop file upload with live preview
│   │   │   ├── ScoreBar.jsx       # Similarity metric bar, telemetry chips, and legal notice
│   │   │   ├── TelemetryHeader.jsx# Live backend liveness & readiness probe banner
│   │   │   └── VerdictBadge.jsx   # 3-tier categorical verdict badge
│   │   ├── services/
│   │   │   └── api.js             # Encapsulated REST client with AbortController timeout
│   │   ├── utils/
│   │   │   └── verdict.jsx        # Verdict theme mapping and Lucide icon configuration
│   │   ├── App.jsx                # Main forensic application layout and state machine
│   │   ├── index.css              # Custom Glassmorphism design tokens & styles
│   │   └── main.jsx               # Application DOM entry point
│   ├── package.json               # Frontend dependencies (React 19, Lucide React)
│   ├── vite.config.js             # Vite build pipeline configuration
│   └── Dockerfile                 # Multi-stage production Nginx frontend container
├── models/                        # Serialized ML artifacts
│   ├── forensic_signature_v2.pt   # Calibrated PyTorch Siamese Network state dict (~45.4 MB)
│   ├── model_config.json          # ROC-AUC optimal threshold (0.577) and EER metrics
│   ├── training_history.json      # Training loss and validation loss curves (40 epochs)
│   └── evaluation_plots.png       # ROC-AUC curve & score distribution histogram
├── notebooks/                     # Exploratory analysis & training pipelines
│   ├── 01_thresholding_test.ipynb # Baseline CV stroke isolation & threshold audit
│   ├── 02_train_siamese.ipynb     # Single-identity Siamese fine-tuning
│   └── 03_train_siamese_v2.ipynb  # Multi-signer general-purpose Siamese V2.2 (Triplet Loss + Hard Mining)
├── tests/                         # Automated Pytest suite (30 test cases)
│   ├── __init__.py
│   ├── conftest.py                # Synthetic signature generator & TestClient fixtures
│   ├── test_cv_pipeline.py        # OpenCV binarization and padding unit tests
│   ├── test_engine.py             # SiameseNetwork and similarity unit tests
│   ├── test_health.py             # Liveness and readiness probe tests
│   ├── test_validation.py         # File size (5MB), magic byte, and empty payload tests
│   └── test_verify_api.py         # End-to-end verification endpoint tests
├── Dockerfile                     # Hardened backend container with non-root user & health check
├── docker-compose.yml             # Orchestrated multi-service local stack
├── pytest.ini                     # Pytest configuration
├── requirements.txt               # Pinned production and test dependencies
└── README.md                      # Project documentation
```

---

## 5. Quick Start

### A. Local Environment

```bash
# 1. Install Python dependencies (choose one):
# For backend API & testing only (lightweight CPU):
pip install -r requirements.txt

# OR for full training pipeline & Jupyter notebooks (CUDA GPU recommended):
pip install -r requirements-dev.txt

# 2. Run automated test suite (100% pass target)
python -m pytest tests/ -v

# 3. Start backend API server (port 8000)
python -m uvicorn api.main:app --reload --port 8000

# 4. Start frontend development server (port 5173)
cd frontend
npm install
npm run dev
```

### B. Docker Compose (Unified Stack)

```bash
docker compose up -d --build
```
- Frontend UI: `http://localhost:5173`
- Backend Swagger Docs: `http://localhost:8000/docs`
- Health Probe: `http://localhost:8000/health`
- Readiness Probe: `http://localhost:8000/ready`

---

## 6. API Reference

### `GET /health`
Shallow liveness probe used by container orchestrators (e.g., Kubernetes, ECS) to determine process viability.

**Response (`200 OK`)**:
```json
{
  "status": "healthy",
  "service": "Forensic Signature AI",
  "version": "2.2.0"
}
```

---

### `GET /ready`
Deep readiness probe evaluating whether model weights are loaded and calibrated inference thresholds are verified.

**Response (`200 OK`)**:
```json
{
  "status": "ready",
  "model_loaded": true,
  "weights_verified": true,
  "calibrated_threshold": 0.7000
}
```

---

### `POST /verify`
Compares a reference specimen image against a questioned document signature using non-blocking threadpool inference.

**Payload (`multipart/form-data`)**:
- `file_asli`: Reference authentic signature image (JPEG, PNG, or WEBP, maximum 5 MB).
- `file_uji`: Questioned signature image (JPEG, PNG, or WEBP, maximum 5 MB).

**Validation Safeguards**:
- **Zero-Byte Inspection**: Reject empty files immediately with `400 Bad Request`.
- **Payload Size Enforcement**: Reject files exceeding 5 MB with `413 Request Entity Too Large`.
- **Magic Byte Validation**: Validate binary headers (`FF D8 FF` for JPEG, `89 50 4E 47` for PNG, `52 49 46 46` for WEBP) to prevent MIME-type spoofing with `415 Unsupported Media Type`.

**Response (`200 OK`)**:
```json
{
  "verification": {
    "verdict": "AUTHENTIC",
    "status": "AUTHENTIC (VERIFIED)",
    "similarity_score": 0.9912,
    "normalized_percentage": 99.56,
    "system_threshold": 0.7000,
    "confidence_band": "High Confidence Match",
    "analysis": "Ink stroke anatomy, pen pressure dynamics, and trajectory contours are structurally consistent with the reference specimen.",
    "disclaimer": "LEGAL DISCLAIMER: This automated AI verification provides supplemental comparative screening and does NOT constitute certified forensic legal testimony. Borderline or questioned documents must be verified by a qualified forensic document examiner.",
    "latency_ms": 112.4
  }
}
```

---

## 7. License & Legal Evidentiary Disclaimer

This project is intended for educational, research, and supplemental forensic screening purposes. Automated signature verification algorithms must not be used as sole or conclusive judicial evidence in court proceedings without corroboration by a certified forensic document examiner.
