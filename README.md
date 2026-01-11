# RAG-Powered Enterprise Knowledge Base (KB) Chatbot

A **permission-aware Retrieval-Augmented Generation (RAG)** chatbot for querying a company-style knowledge base (policies, IT runbooks, onboarding docs, FAQs). The system combines **vector retrieval (pgvector)** with a **local LLM (Ollama)** to generate concise, context-aware answers **grounded in retrieved sources** and **filtered by user permissions**.

**Flow**: End-to-end ingestion → embeddings → vector search → authenticated chat API → LLM answer generation → UI, with a containerized workflow for repeatable runs.
> Note: I have not audited the code for security vulnerabilities. Please review further if you want to deploy in a production environment.
---

## Why this project exists

In my previous internship, I noticed that we were handling our sensitive internal documentation/materials in an unsafe way that it may get compromised. So, I researched how small scale companies handle their data and I found out that majority of them were handling their data similarly. So, this is my attempt for the problem.

In many companies, institutional knowledge is scattered across wikis, docs, PDFs, and internal portals. A lot of employees waste time searching and asking around, and restricted information must remain protected.

This project solves that by:
- indexing internal documents into a vector database,
- retrieving the best matching chunks for a question,
- using an LLM to synthesize a grounded answer,
- enforcing **access controls at retrieval time** so restricted content never reaches the LLM for unauthorized users.

> This project prioritizes the RAG pipeline and security logic; it is fully containerized to support flexible deployment across various server infrastructures.

---
## Demo

https://github.com/user-attachments/assets/b3f1e7be-1ea6-4e86-b7ba-b570bd4b4a0f

---

## Key features

### RAG pipeline (end-to-end)
- **Ingestion**: load docs from a directory, chunk them, store rich metadata.
- **Embeddings**: embed chunks with a sentence-transformer model.
- **Vector search**: top-k similarity search using **pgvector** inside Postgres.
- **Grounded answers**: retrieved chunks are passed into the LLM, which answers using only that context.
- **Citations**: responses include a source list (doc title/path, metadata, similarity score).

### LLM answer generation (Ollama)
- Runs a LLM via Ollama.
- Prompts are structured to:
  - answer **only using retrieved context**,
  - return **concise**, actionable outputs,
  - refuse when context is insufficient (avoid hallucinations).
- Configurable model + endpoint via environment variables.

### Enterprise access controls
- **Auth**: users log in and receive a bearer token.
- **Access levels**: both content and users have access levels (`public`, `internal`, `restricted`).
- **Hard enforcement in SQL**: retrieval query includes `access_level <= user.max_access_level`, preventing leakage.
- **Metadata filters** (e.g., department) supported to narrow retrieval.

### Usable interface
- **Streamlit UI**: login + chat interface + sources view.

### Containerized stack 
Docker Compose stack with:
- FastAPI backend
- Streamlit frontend
- Postgres + pgvector
- Ollama (LLM)

---

## Architecture

```text
                    ┌──────────────────────────┐
                    │        Streamlit UI      │
                    │  - Login (JWT)           │
User ──[Server]───► │  - Chat + Sources        │
                    └───────────┬──────────────┘
                                │ HTTP
                                ▼
                    ┌──────────────────────────┐
                    │        FastAPI API       │
                    │  /auth/login             │
                    │  /chat (RAG + LLM)       │
                    │  - permission-aware SQL  │
                    │  - LLM prompt + response │
                    └───────┬─────────┬────────┘
                            │         │
                            │ SQL     │ HTTP
                            ▼         ▼
                ┌────────────────┐  ┌────────────────┐
                │ Postgres+pgvec │  │     Ollama     │
                │ documents      │  │ local LLM model│
                │ chunks (vector)│  │ /api/chat      │
                └────────────────┘  └────────────────┘
```

---

## Tech stack

- **Backend:** FastAPI, SQLModel/SQLAlchemy
- **Frontend:** Streamlit
- **Vector store:** PostgreSQL + **pgvector**
- **Embeddings:** sentence-transformers (Hugging Face)
- **LLM:** llama3.1:8b
- **Auth:** token-based (login → bearer token)
- **Dev tooling:** `uv`
- **Containers:** Docker + Docker Compose

---

## Repository layout (high-level)

```text
rag-enterprise-kb/
├─ apps/
│  ├─ api/            # FastAPI service (auth, retrieval, LLM chat)
│  └─ web/            # Streamlit UI (login + chat)
├─ rag/
│  └─ ingest/         # ingestion pipelines (pgvector ingestion)
├─ scripts/           # helper scripts (seed users, debug, etc.)
├─ data/
│  └─ raw/            # sample docs (your KB corpus)
├─ infra/
│  └─ docker/         # docker compose + Dockerfiles
├─ pyproject.toml     # dependencies (uv)
└─ README.md
```

---

## Quickstart: Docker Compose

### 1) Prerequisites
- Docker + Docker Compose
- (Optional for local dev) Python 3.11+ + `uv`

### 2) Start the stack
From repo root:

```bash
docker compose -f infra/docker/docker-compose.yml up -d --build
docker compose -f infra/docker/docker-compose.yml ps
```

### 3) Pull an Ollama model (LLM)
Inside the Ollama container:

```bash
docker compose -f infra/docker/docker-compose.yml exec ollama ollama pull llama3.1:8b
```

> You can choose a smaller model if needed (e.g., `phi3`, `llama3.2:3b`) for faster CPU-only runs.

### 4) Initialize DB schema (first run)
```bash
docker compose -f infra/docker/docker-compose.yml exec api uv run python -c "from apps.api.core.db import engine; from sqlmodel import SQLModel; import apps.api.models as m; SQLModel.metadata.create_all(engine); print('tables created')"
```

### 5) Seed demo users
```bash
docker compose -f infra/docker/docker-compose.yml exec api uv run python -m scripts.seed_users
```

### 6) Ingest documents
Put docs into `data/raw/` and run:

```bash
docker compose -f infra/docker/docker-compose.yml exec api uv run python -m rag.ingest.pg_ingest --input_dir data/raw --reset
```

### 7) Open the UI
- Streamlit: http://localhost:8501  
- API docs: http://localhost:8000/docs  

---

## Local development (no Docker)

### 1) Install dependencies with uv
```bash
uv sync
```

### 2) Start Postgres (pgvector) + Ollama
- Use Docker for DB/Ollama **or** run them locally.
- Ensure your environment variables point to the correct endpoints.

### 3) Seed + ingest
```bash
uv run python -m scripts.seed_users
uv run python -m rag.ingest.pg_ingest --input_dir data/raw --reset
```

### 4) Run backend + frontend
```bash
uv run uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload
uv run streamlit run apps/web/app.py
```

---

## Environment variables

Typical configuration (set in `.env` for Docker Compose):

- `DATABASE_URL` — Postgres connection string
- `JWT_SECRET` — signing secret for tokens
- `OLLAMA_URL` — e.g. `http://ollama:11434` (in Docker)
- `OLLAMA_MODEL` — e.g. `llama3.1:8b`
- `CORS_ORIGINS` — allowed UI origins

---

## Demo accounts and permissions

The project uses numeric access levels:

- `public` = 0
- `internal` = 1
- `restricted` = 2

Example demo users:
- `public@demo.com`
- `internal@demo.com`
- `restricted@demo.com`

The retrieval layer enforces:

- `chunk.access_level <= user.max_access_level`

So restricted users can retrieve everything; public users only retrieve public chunks.

---

## How chat works (request flow)

1) User logs in → receives token  
2) UI sends `/chat` request with:
   - query
   - optional metadata filters (e.g., department)
3) API:
   - embeds the query
   - runs permission-aware pgvector SQL top-k retrieval
   - builds an LLM prompt using the retrieved chunks
   - calls Ollama to generate the answer
   - returns answer + citations

---

## Grounding and hallucination control

The prompt is designed to:
- **only** use retrieved context,
- say “I don’t know” when context is insufficient,
- keep answers short and actionable,
- include citations so users can verify.

For additional safety:
- retrieval is filtered by access level **before** sending to the LLM.

---

## Troubleshooting

### “no configuration file provided”
You ran docker compose without specifying a file. Use:

```bash
docker compose -f infra/docker/docker-compose.yml <command>
```

### “permission denied while trying to connect to the Docker daemon socket”
Add your user to the docker group (Linux):

```bash
sudo usermod -aG docker $USER
newgrp docker
```

### Ollama model not found
Pull it in the container:

```bash
docker compose -f infra/docker/docker-compose.yml exec ollama ollama pull llama3.1:8b
```

### DB schema missing (relation does not exist)
Run the schema init command shown in Quickstart step 4.

---
