FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
  && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock* /app/
RUN uv sync --frozen --no-dev

COPY . /app

EXPOSE 8501
CMD ["uv", "run", "streamlit", "run", "apps/web/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
