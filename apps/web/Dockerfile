FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
  && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

ENV UV_PYTHON_DOWNLOADS=never
ENV UV_PROJECT_ENVIRONMENT=/app/.venv

COPY pyproject.toml uv.lock* /app/
RUN uv sync --frozen --no-dev

COPY . /app

EXPOSE 8501
CMD ["uv", "run", "streamlit", "run", "apps/web/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
