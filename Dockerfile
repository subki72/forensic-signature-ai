FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application directories
COPY api/ ./api/
COPY models/ ./models/

# Hugging Face Spaces expects the app to run on port 7860
EXPOSE 7860

# Command to run the FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
