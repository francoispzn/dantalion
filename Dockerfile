# A minimal runtime image. The agent itself needs no model in the image; point it
# at an Ollama/OpenAI-compatible endpoint at run time, or mount a GGUF and install
# the optional llama-cpp extra.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN pip install --no-cache-dir uv

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

COPY examples ./examples

ENTRYPOINT ["uv", "run", "dantalion"]
CMD ["--help"]
