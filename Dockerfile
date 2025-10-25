FROM python:3.10-slim        # Base Python image

WORKDIR /app                 # Set working directory

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    portaudio19-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY . /app                  # Copy project files

# Install Python dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader punkt stopwords

EXPOSE 5000                  # Expose Flask port

# Environment variables
ENV PORT=5000 PYTHONUNBUFFERED=1 GOOGLE_API_KEY="" SECRET_KEY=""

# Run app with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
