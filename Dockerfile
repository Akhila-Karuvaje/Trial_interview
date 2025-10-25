# Base image
FROM python:3.10.13-slim

# Set working directory
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

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Upgrade pip and install Python packages
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data at build time
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"

# Copy all application code
COPY . .

# Create necessary directories for uploads and caches
RUN mkdir -p /tmp/uploads /tmp/whisper_cache /tmp/transformers_cache

# Set environment variables for caches
ENV TRANSFORMERS_CACHE=/tmp/transformers_cache \
    TORCH_HOME=/tmp/torch_cache \
    PYTHONUNBUFFERED=1

# Expose a port (Render will override with $PORT)
EXPOSE 10000

# Use Gunicorn to serve Flask app and bind to Render's $PORT
CMD exec gunicorn app:app \
    --bind 0.0.0.0:${PORT:-5000} \
    --timeout 600 \
    --workers 1 \
    --worker-class sync \
    --preload \
    --log-level info \
    --access-logfile - \
    --error-logfile -
