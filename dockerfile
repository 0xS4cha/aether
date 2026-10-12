FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libxrender1 \
    libxext6 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml README.md uv.lock ./
COPY src/ src/
COPY config/ config/

RUN uv sync --frozen --no-dev
RUN uv pip install -e .

ENTRYPOINT ["uv", "run", "aether"]
CMD ["visualize", "--sample", "--headless", "--screenshot", "data/outputs/docker_render.png"]
