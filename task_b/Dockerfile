FROM python:3.11-slim

WORKDIR /app

COPY task_b/requirements.txt ./requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ curl \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv \
    && uv pip install --system --no-cache \
        torch --index-url https://download.pytorch.org/whl/cpu \
    && uv pip install --system --no-cache -r requirements.txt \
    && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')" \
    && rm -rf /usr/local/lib/python3.11/site-packages/torch/bin \
              /usr/local/lib/python3.11/site-packages/torch/include \
              /usr/local/lib/python3.11/site-packages/torch/test \
              /usr/local/lib/python3.11/site-packages/torch/testing \
    && apt-get purge -y gcc g++ curl \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /root/.cache/pip

COPY core/ ./core/
COPY task_b/ ./task_b/

ENV PYTHONPATH=/app
ENV PORT=8002

EXPOSE 8002
CMD ["python", "-m", "uvicorn", "task_b.main:app", "--host", "0.0.0.0", "--port", "8002"]
