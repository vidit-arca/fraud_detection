# Use a slim, stable Python 3.10 base image
FROM python:3.10-slim

# Set system-level environment variables
# - PYTHONDONTWRITEBYTECODE: Prevents Python from writing .pyc files
# - PYTHONUNBUFFERED: Ensures application logs are flushed immediately to stdout/stderr
# - TORCH_HOME: Directs PyTorch to cache model weights in /app/.cache/torch
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_HOME=/app/.cache/torch

# Set working directory inside the container
WORKDIR /app

# Install system-level dependencies
# - poppler-utils: Required by the pdf2image library to convert uploaded PDF documents to images
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to maximize Docker layer caching
COPY requirements.txt .

# Install Python packages:
# 1. Install CPU-only wheels for torch & torchvision (saves ~1.5GB of GPU/CUDA packages)
# 2. Install all other requirements
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download and cache the pre-trained EfficientNet-B0 weights during build.
# This makes container startup instantaneous and allows the container to run completely offline.
RUN python -c "import torchvision.models as models; models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)"

# Copy the application source code
COPY . .

# Ensure standard runtime directories exist
RUN mkdir -p /app/models /app/uploads

# Expose FastAPI default port
EXPOSE 8000

# Run the FastAPI server via uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
