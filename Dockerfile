# Qwen3-TTS Voice Cloning WebUI
# Optimized for Runpod with RTX 4090

FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/huggingface
ENV TRANSFORMERS_CACHE=/runpod-volume/huggingface
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
ENV USE_FLASH_ATTENTION=true

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Flash Attention 2 (for RTX 4090)
RUN pip install --no-cache-dir flash-attn --no-build-isolation || \
    echo "Flash Attention installation failed, will use eager attention"

# Copy application code
COPY src/ ./src/
COPY app.py .

# Create cache directory
RUN mkdir -p /runpod-volume/huggingface

# Expose Gradio port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Start the application
CMD ["python", "app.py"]
