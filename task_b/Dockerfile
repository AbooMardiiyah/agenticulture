# Build context: repo root (set in docker-compose.yml)
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv

# Install CPU-only torch first to avoid pulling the 2GB GPU build
RUN uv pip install --system --no-cache \
    torch --index-url https://download.pytorch.org/whl/cpu

COPY task_b/requirements.txt ./requirements.txt
RUN uv pip install --system --no-cache -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

COPY core/ ./core/
COPY task_b/ ./task_b/

ENV PYTHONPATH=/app
ENV PORT=8002

EXPOSE 8002
CMD ["python", "-m", "uvicorn", "task_b.main:app", "--host", "0.0.0.0", "--port", "8002"]
