FROM python:3.12-slim

WORKDIR /app

# packages needed by psycopg and python wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
  && rm -rf /var/lib/apt/lists/*

# install uv into the base image
RUN pip install --no-cache-dir uv

ENV UV_PYTHON_DOWNLOADS=never
ENV UV_PROJECT_ENVIRONMENT=/app/.venv

# install dependencies before copying app code for better layer caching
COPY pyproject.toml uv.lock* /app/

RUN uv sync --frozen --no-dev

# copy app code after dependencies
COPY . /app

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
