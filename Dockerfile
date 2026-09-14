# Production Dockerfile for Forensic Signature AI Backend
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Install system runtime dependencies for OpenCV and health check tooling
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root application user for container hardening
RUN groupadd -g 10001 appuser && \
    useradd -u 10001 -g appuser -s /bin/bash -m appuser

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application package and trained models
COPY api/ ./api/
COPY models/ ./models/

# Ensure non-root ownership of app directory
RUN chown -R appuser:appuser /app

# Switch to non-root execution
USER appuser

# Expose default HTTP port (8000 for standard deployments, 7860 for HF Spaces)
EXPOSE 8000 7860

# Container liveness health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Launch FastAPI ASGI server with Uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
