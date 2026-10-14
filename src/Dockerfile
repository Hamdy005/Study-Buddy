FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for audio processing (soundfile & ffmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt && \
    pip install --no-cache-dir --no-deps nemo_toolkit==2.2.1

COPY . /app/src

EXPOSE 7860

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "7860"]
