# Legal Document AI - Signature Verification System

![Streamlit Web Interface](assets/web_ui.png)

## Overview
This repository contains an end-to-end artificial intelligence system designed to verify the authenticity of signatures on legal documents. Built to address the critical need for automated forensic document analysis, this system utilizes computer vision for precise signature extraction and a fine-tuned Siamese Network architecture to detect forgeries with high accuracy.

## Technical Architecture
The system is divided into four main components:
1. **Computer Vision Pipeline (OpenCV):** Implements adaptive Gaussian thresholding and contour detection to automatically isolate, crop, and pad signatures from raw document scans, removing background noise and varying paper textures.
2. **Deep Learning Engine (PyTorch):** Utilizes a ResNet18 backbone modified into a Siamese architecture. It extracts 512-dimensional feature vectors (DNA) from signatures and computes Cosine Similarity to measure anatomical structural differences.
3. **Backend API (FastAPI):** A high-performance, asynchronous REST API serving the machine learning model in a production-ready environment.
4. **Interactive Frontend (Streamlit):** A user-friendly web interface allowing non-technical users to upload master and questioned documents for real-time verification.

## Performance Metrics
Following rigorous fine-tuning using contrasting data pairs (Genuine vs. Forged/Others), the system achieves exceptional margin separation:
- **Genuine vs. Genuine Match:** ~0.98 Cosine Similarity
- **Genuine vs. Forged Match:** ~0.07 Cosine Similarity
- **System Threshold:** Set strictly at 0.73 to eliminate false positives in legal contexts.

## Repository Structure
```text
.
├── api/
│   └── main.py                 # FastAPI application and model serving
├── data/                       # Directory for raw and processed datasets (ignored in git)
├── models/                     # Directory for compiled .pt model weights (ignored in git)
├── notebooks/
│   ├── 01_thresholding_test.ipynb  # R&D for OpenCV extraction and baseline audit
│   └── 02_train_siamese.ipynb      # Fine-tuning loop and data augmentation
├── app.py                      # Streamlit frontend application
├── run_all.py                  # Automation script to launch both API and Web UI
├── requirements.txt            # Python dependencies
└── README.md
```

## Quick Start

### 1. Environment Setup
Create a virtual environment and install dependencies:
```bash
conda create -n legal_ai_env python=3.10
conda activate legal_ai_env
pip install -r requirements.txt
```

### 2. Running the Application
A conductor script is provided to initialize both the FastAPI backend and the Streamlit frontend concurrently from a single terminal.
```bash
python run_all.py
```
- The API will be available at: `http://localhost:8000/docs`
- The Web UI will be available at: `http://localhost:8501`

## API Documentation
The `/verify` endpoint accepts a `multipart/form-data` POST request containing two image files:
- `file_asli`: The verified master signature.
- `file_uji`: The questioned signature document.

Returns a JSON response containing the verification status, similarity score, threshold, and analytical conclusion.
```
