FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    portaudio19-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader punkt stopwords

EXPOSE 5000

ENV PORT=5000 PYTHONUNBUFFERED=1 GOOGLE_API_KEY="" SECRET_KEY=""

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
