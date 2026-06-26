# RAG-Powered Enterprise Knowledge Base Chatbot

A permission-aware RAG chatbot for querying internal company-style documents. It ingests markdown, text, and PDF files, stores embeddings in Postgres with pgvector, and serves authenticated chat through a FastAPI backend and React UI.

The main project path is pgvector-backed retrieval with access checks in SQL. A FAISS prototype is kept under `rag/experiments/faiss` as a learning/comparison path, not as the running app backend.

> This is a prototype for the RAG pipeline and access-control model. It is not a production security system.

## Demo

https://github.com/user-attachments/assets/03f57462-72c0-4072-bbca-13587ef4a4f0

## What It Does

- Ingests documents from `data/raw/`
- Chunks documents and stores metadata
- Embeds chunks with `sentence-transformers`
- Retrieves top-k matches from Postgres + pgvector
- Filters retrieval by user access level before anything reaches the LLM
- Optionally asks Ollama to generate a grounded answer from retrieved context
- Returns answer text plus source citations
- Provides a React UI for login, chat, filters, modes, and source inspection

## Architecture

```text
React UI (nginx)
  login + chat + sources
        |
        | /api proxy
        v
FastAPI
  /auth/login
  /chat
  jwt auth
  permission-aware pgvector query
        |                         |
        | SQL                     | HTTP
        v                         v
Postgres + pgvector            Ollama
documents/chunks               local LLM
```

## Repository Layout

```text
apps/api/                  FastAPI app, auth, chat, retrieval
apps/web/                  React + TypeScript UI
data/raw/                  sample knowledge-base documents
infra/docker/              compose stack and postgres init SQL
rag/ingest/                document loading, chunking, pgvector ingest
rag/experiments/faiss/     older FAISS retrieval experiment
scripts/seed_users.py      creates demo users
scripts/smoke_api.py       login/chat/access-level smoke check
```

## Quickstart With Docker

Prerequisites:

- Docker with Compose
- enough disk space for the Python image, model dependencies, and Ollama model

Create an env file:

```bash
cp infra/docker/.env.example infra/docker/.env
```

Edit `infra/docker/.env` and set a real `JWT_SECRET`.

Start the stack:

```bash
docker compose --env-file infra/docker/.env -f infra/docker/docker-compose.yml up -d --build
```

Pull the Ollama model:

```bash
docker compose --env-file infra/docker/.env -f infra/docker/docker-compose.yml exec ollama ollama pull llama3.1:8b
```

Seed users and ingest the sample docs:

```bash
docker compose --env-file infra/docker/.env -f infra/docker/docker-compose.yml exec api uv run python -m scripts.seed_users
docker compose --env-file infra/docker/.env -f infra/docker/docker-compose.yml exec api uv run python -m rag.ingest.pg_ingest --input_dir data/raw --reset
```

Open:

- Web UI: http://localhost:3000
- API docs: http://localhost:8000/docs

## Local Development

Backend dependencies:

```bash
uv sync
```

Frontend dependencies:

```bash
npm --prefix apps/web ci
```

Run Postgres/pgvector and Ollama with Docker or local services, then set environment variables:

```bash
export DATABASE_URL='postgresql+psycopg://rag:rag@127.0.0.1:5432/rag_kb'
export JWT_SECRET='replace-with-a-long-random-secret'
export OLLAMA_URL='http://127.0.0.1:11434'
export USE_LLM='false'
```

If you are not using the Docker DB service, apply `infra/docker/postgres/init.sql` to your local Postgres database before seeding.

Seed and ingest:

```bash
uv run python -m scripts.seed_users
uv run python -m rag.ingest.pg_ingest --input_dir data/raw --reset
```

Run backend and frontend:

```bash
uv run uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload
npm --prefix apps/web run dev
```

In local dev, set the UI API URL to `http://localhost:8000` in the settings panel.

## Environment Variables

Backend:

- `DATABASE_URL`: Postgres connection string
- `JWT_SECRET`: required token signing secret
- `JWT_EXPIRE_MINUTES`: token lifetime, default `120`
- `EMBEDDING_DEVICE`: `auto`, `cpu`, or `cuda`
- `OLLAMA_URL`: Ollama endpoint
- `OLLAMA_MODEL`: default `llama3.1:8b`
- `USE_LLM`: `true` for Ollama answers, `false` for citations-only answers
- `OLLAMA_TIMEOUT_S`: Ollama request timeout
- `CORS_ORIGINS`: comma-separated allowed frontend origins

Frontend:

- `VITE_API_URL`: API base URL baked into the Vite build, default `/api`

## Demo Users

Run `scripts.seed_users` to create:

```text
public@demo.com      public123      access 0
internal@demo.com    internal123    access 1
restricted@demo.com  restricted123  access 2
admin@demo.com       admin123       access 2
```

The access model is intentionally simple:

```sql
chunk.access_level <= user.max_access_level
document.access_level <= user.max_access_level
```

This check runs in the retrieval SQL before context is sent to Ollama.

## Verification

Backend syntax check:

```bash
python -m py_compile apps/api/main.py apps/api/models.py scripts/seed_users.py scripts/smoke_api.py rag/ingest/pg_ingest.py
```

Frontend build:

```bash
npm --prefix apps/web run build
```

API smoke check, after the API is running and data is seeded:

```bash
python scripts/smoke_api.py
```

The smoke script logs in as each demo user, calls `/chat`, and fails if any returned source is above that user's access level.

## FAISS Experiment

The FAISS code under `rag/experiments/faiss` is preserved as an earlier retrieval prototype. It is useful for explaining the learning path from local vector search to pgvector-backed retrieval.

Build the FAISS experiment index:

```bash
uv run python -m rag.experiments.faiss.build_index --input_dir data/raw
```

The running app does not use this path.

## Limitations

- Demo credentials are for local testing only.
- Access control is numeric and document-level/chunk-level, not a full enterprise RBAC model.
- JWT auth is intentionally minimal.
- Ollama runs locally and must have the configured model pulled.
- The app does not include document upload or admin workflows.
