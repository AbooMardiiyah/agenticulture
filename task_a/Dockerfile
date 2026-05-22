FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv

RUN uv pip install --system --no-cache \
    torch --index-url https://download.pytorch.org/whl/cpu

COPY task_a/requirements.txt ./requirements.txt
RUN uv pip install --system --no-cache -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

COPY core/ ./core/
COPY task_a/ ./task_a/

ENV PYTHONPATH=/app
ENV PORT=8001

EXPOSE 8001
CMD ["python", "-m", "uvicorn", "task_a.main:app", "--host", "0.0.0.0", "--port", "8001"]
