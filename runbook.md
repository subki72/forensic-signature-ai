# Operational Runbook — Forensic Signature AI (V2.3)

## 1. Service Overview
- **API Gateway**: FastAPI + Uvicorn (`uvicorn api.main:app --host 0.0.0.0 --port 8000`).
- **Probes**: `GET /health` (liveness) and `GET /ready` (verifies `forensic_signature_v2.pt` weights loaded and threshold profile calibrated).
- **Two-Phase Inference**: Stage 1 2D Macro-Geometry Gate (`macro_score >= 0.25`) -> Stage 2 Siamese ResNet-18 Cosine Similarity (`>= 0.70 AUTHENTIC`, `0.55-0.70 INCONCLUSIVE`, `< 0.55 FORGERY`).

## 2. Standard Operations
- **Local Startup**: `make run` or `docker compose up --build`
- **Verification**: `make test` and `make lint`

## 3. Incident Response
| Symptom | Probable Cause | Remediation |
|---|---|---|
| `HTTP 415 Unsupported Media Type` | Spoofed image extension or invalid binary magic bytes | Confirm client uploads valid PNG/JPEG/WEBP files (`<= 5 MB`). |
| `HTTP 429 Too Many Requests` | Client exceeded `slowapi` 10 req/min rate limit | Inspect client retry backoff or adjust rate limit threshold in `api/config.py`. |
| `weights_verified: false` on `/ready` | Missing or LFS pointer-only `models/forensic_signature_v2.pt` | Pull Git LFS object (`git lfs pull`) and restart container. |

## 4. Rollback Procedure
Revert container image and deployment to pinned release tag `v1.0.0` (`make docker-build`).
