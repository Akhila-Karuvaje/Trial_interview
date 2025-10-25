FROM python:3.10.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
    portaudio19-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data at build time
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /tmp/uploads /tmp/whisper_cache /tmp/transformers_cache

# Set environment variables
ENV TRANSFORMERS_CACHE=/tmp/transformers_cache \
    TORCH_HOME=/tmp/torch_cache \
    PYTHONUNBUFFERED=1

# Expose port (Render provides PORT env var)
EXPOSE 10000

# CRITICAL: Use $PORT variable and increase timeout
CMD gunicorn app:app \
    --bind 0.0.0.0:$PORT \
    --timeout 600 \
    --workers 1 \
    --worker-class sync \
    --preload \
    --log-level info \
    --access-logfile - \
    --error-logfile -
