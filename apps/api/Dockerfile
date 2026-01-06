FROM python:3.12-slim

WORKDIR /app

# System deps for psycopg + builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
  && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

ENV UV_PYTHON_DOWNLOADS=never
ENV UV_PROJECT_ENVIRONMENT=/app/.venv

# Copy only dependency files first (better caching)
COPY pyproject.toml uv.lock* /app/

# Create venv + install deps
RUN uv sync --frozen --no-dev

# Now copy the rest of the repo
COPY . /app

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
