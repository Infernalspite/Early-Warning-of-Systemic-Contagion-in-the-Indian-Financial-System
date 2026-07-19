FROM python:3.12-slim

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# HF Spaces expects port 7860
ENV PORT=7860
ENV YF_CACHE_DIR=/tmp/yfinance_cache

EXPOSE 7860

CMD ["gunicorn", "app:app", \
     "--bind", "0.0.0.0:7860", \
     "--timeout", "120", \
     "--workers", "1", \
     "--log-level", "info"]
