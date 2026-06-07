# Legal Document AI - Forensic Signature Verification

An end-to-end artificial intelligence system for verifying the authenticity of
handwritten signatures on legal documents. The system combines computer vision
preprocessing with a fine-tuned Siamese Network (ResNet-18 backbone) to detect
forgeries through structural comparison of ink stroke anatomy.


## Architecture

| Layer | Technology | Purpose |
|---|---|---|
| Computer Vision | OpenCV | Adaptive thresholding, contour detection, signature isolation |
| Deep Learning | PyTorch (ResNet-18) | 512-dimensional feature extraction via Siamese architecture |
| Backend API | FastAPI | Asynchronous REST endpoint for model serving |
| Frontend | React + Vite | Interactive web interface with drag-and-drop upload |


## Performance

| Comparison | Cosine Similarity |
|---|---|
| Genuine vs. Genuine | ~0.98 |
| Genuine vs. Forged | ~0.07 |
| System Threshold | 0.73 |


## Repository Structure

```
.
├── api/
│   └── main.py                    # FastAPI application and model serving
├── frontend/                      # React (Vite) web interface
│   ├── src/
│   │   ├── App.jsx                # Main application component
│   │   ├── index.css              # Design system and styling
│   │   └── main.jsx               # React entry point
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── models/                        # Trained model weights (.pt)
├── data/                          # Raw and processed datasets (git-ignored)
├── notebooks/
│   ├── 01_thresholding_test.ipynb # OpenCV pipeline R&D and threshold calibration
│   └── 02_train_siamese.ipynb     # Siamese network fine-tuning loop
├── test_api.py                    # Automated API endpoint tests
├── requirements.txt               # Python dependencies
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
    "similarity_score": 0.9842,
    "system_threshold": 0.73,
    "analysis": "Ink stroke anatomy is structurally consistent with the reference specimen."
  }
}
```


## Training

The model is trained using the CEDAR Signature Dataset. The training pipeline is
documented in the Jupyter notebooks under `notebooks/`:

1. **01_thresholding_test.ipynb** -- Establishes the computer vision pipeline and
   calibrates the cosine similarity threshold through a baseline forensic audit.
2. **02_train_siamese.ipynb** -- Fine-tunes ResNet-18 using contrastive pairs
   (genuine vs. forged) with `CosineEmbeddingLoss`.

After training, the model weights are saved to `models/forensic_signature_v1.pt`.


## License

This project is intended for educational and research purposes.
