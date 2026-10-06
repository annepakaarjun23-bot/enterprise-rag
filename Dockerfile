FROM python:3.12-slim AS deps

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /install

RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.2.0"

COPY requirements.txt ./
RUN pip install -r requirements.txt


FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    HF_HOME=/app/models/hf \
    SENTENCE_TRANSFORMERS_HOME=/app/models/hf \
    FASTEMBED_CACHE_PATH=/app/models/fastembed

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash app

WORKDIR /app

COPY --from=deps /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY evals/ ./evals/
COPY alembic.ini ./
COPY migrations/ ./migrations/

RUN mkdir -p /app/data/models && chown -R app:app /app

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "rag.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--loop", "asyncio:SelectorEventLoop", \
     "--log-level", "info"]