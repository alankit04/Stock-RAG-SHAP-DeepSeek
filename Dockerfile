# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# system deps needed to compile some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Tesseract OCR + runtime shared libs (no build tools needed here)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy compiled packages from builder
COPY --from=builder /install /usr/local

# Copy application source + model artifacts
COPY app.py             .
COPY xgb_model_subset.pkl .
COPY X_test_subset.csv  .
COPY y_test_subset.csv  .
COPY "Shap Plot.png"    .

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# Ollama lives in its own container; point to it via OLLAMA_HOST env var
ENV OLLAMA_HOST=http://ollama:11434
ENV OLLAMA_MODEL=deepseek-r1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
