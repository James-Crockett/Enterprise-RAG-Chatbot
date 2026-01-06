from __future__ import annotations

from typing import Any, Dict, Optional, List
from uuid import UUID

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlmodel import Session, select
from sqlalchemy import text as sql_text
from sentence_transformers import SentenceTransformer

from apps.api.core.db import get_session
from apps.api.core.security import verify_password, create_access_token
from apps.api.core.deps import get_current_user
from apps.api.models import User

from rag.generation.ollama_client import ollama_generate
import os
import json
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

app = FastAPI(title="RAG Enterprise KB (pgvector)", version="0.2.0")

# Embedder is loaded once (fast for repeated queries)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBED_MODEL)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_TIMEOUT_S = float(os.getenv("OLLAMA_TIMEOUT_S", "60"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Schemas ----------
class LoginRequest(BaseModel):
    email: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


# class ChatRequest(BaseModel):
#     query: str
#     top_k: int = 5
#     filters: Optional[Dict[str, Any]] = None
#     mode: str = "citations_only"

class ChatRequest(BaseModel):
    query: str
    top_k: int = 5
    filters: Optional[Dict[str, Any]] = None  # stored in chunks.metadata JSON
    mode: str = "rag"  # "rag" or "citations_only"

class Citation(BaseModel):
    source_path: Optional[str] = None
    title: Optional[str] = None
    department: Optional[str] = None
    page: Optional[int] = None
    access_level: Optional[int] = None


# class ChunkOut(BaseModel):
#     chunk_id: int
#     score: float
#     text: str
#     citation: Citation

class ChunkOut(BaseModel):
    chunk_id: int
    score: float
    vector_score: float = 0.0
    keyword_score: float = 0.0
    text: str
    citation: Citation


class ChatResponse(BaseModel):
    query: str
    answer: str = ""
    mode: str = "citations_only"
    results: List[ChunkOut] = []


# ---------- Helpers ----------
def citations_only_answer(query: str, chunks: List[str]) -> str:
    """
    Very simple answer for now:
    - Take the most relevant chunk and return the first ~2 sentences.
    We'll improve this (sentence scoring) later like before.
    """
    if not chunks:
        return "I couldn't find relevant information in the knowledge base."
    text = chunks[0].replace("\n", " ").strip()
    # crude sentence split
    parts = text.split(". ")
    return ". ".join(parts[:2]).strip() + ("." if len(parts) > 0 and not text.endswith(".") else "")

def build_rag_prompt(question: str, retrieved: List[ChunkOut]) -> str:
    ctx = "\n\n".join(
        f"[{c.citation.source_path or 'unknown'}]\n{c.text}"
        for c in retrieved[:5]
    )
    return f"""You are a company knowledge base assistant.
Use ONLY the context below. If the answer isn't in the context, say you couldn't find it.
Cite sources like (source_path).

Question: {question}

Context:
{ctx}

Answer:
"""
def _val(obj, key, default=None):
    # Works for dict-like and object-like results
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

def build_context(results, max_chars: int = 12000) -> str:
    parts = []
    used = 0

    for r in results:
        meta = _val(r, "citation", {}) or {}
        if not isinstance(meta, dict):
            # if citation is a model, convert to dict if possible
            meta = meta.model_dump() if hasattr(meta, "model_dump") else dict(meta)

        header = (
            f"[chunk:{_val(r,'chunk_id')}] "
            f"title={meta.get('title')} "
            f"dept={meta.get('department')} "
            f"access={meta.get('access_level')} "
            f"path={meta.get('source_path')}"
        )

        body = (_val(r, "text", "") or "").strip()
        block = header + "\n" + body + "\n"

        if used + len(block) > max_chars:
            break

        parts.append(block)
        used += len(block)

    return "\n---\n".join(parts)



def ollama_chat(messages: List[dict]) -> str:
    """
    Calls Ollama's /api/chat endpoint and returns the assistant text.

    We set temperature low for enterprise Q&A (more factual, less creative).
    """
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.2},
    }

    req = Request(
        url=f"{OLLAMA_URL}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(req, timeout=OLLAMA_TIMEOUT_S) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return (data.get("message") or {}).get("content", "").strip()
    except HTTPError as e:
        raise RuntimeError(f"Ollama HTTPError {e.code}: {e.read().decode('utf-8', errors='ignore')}")
    except URLError as e:
        raise RuntimeError(f"Ollama URLError: {e}")


def rag_answer(query: str, results: List[dict]) -> str:
    """
    RAG = Retrieval + Generation.

    We give the model the retrieved context and require:
    - answer ONLY from context
    - if not enough info, say so
    - cite chunk ids like [chunk:123]
    """
    context = build_context(results)

    system = (
        "You are an enterprise knowledge base assistant.\n"
        "Answer using ONLY the provided CONTEXT.\n"
        "If the answer is not in the context, say you don't know based on the KB.\n"
        "Keep answers concise and actionable.\n"
        "When you use a fact, cite it with [chunk:<id>] at the end of the sentence.\n"
    )

    user = f"QUESTION:\n{query}\n\nCONTEXT:\n{context}"

    return ollama_chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )


# ---------- Routes ----------
@app.get("/health")
def health():
    return {"ok": True, "embed_model": EMBED_MODEL}


@app.post("/auth/login", response_model=LoginResponse)
def login(req: LoginRequest, session: Session = Depends(get_session)):
    user = session.exec(select(User).where(User.email == req.email)).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"sub": str(user.id)})
    return LoginResponse(access_token=token)


@app.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    # 1) Embed the query (normalize for cosine)
    qvec_list = embedder.encode([req.query], normalize_embeddings=True)[0].tolist()
    qvec = "[" + ",".join(str(x) for x in qvec_list) + "]"

    # 2) Build SQL:
    #    - enforce permission: chunks.access_level <= user.max_access_level
    #    - optional JSON metadata filters
    where_clauses = ["access_level <= :max_level"]
    alpha = 0.15  # keyword boost strength (tune 0.05–0.3)
    params = {
        "qvec": qvec,
        "qtext": req.query,
        "alpha": alpha,
        "k": req.top_k,
        "max_level": user.max_access_level,
    }

    # params = {"qvec": qvec, "k": req.top_k, "max_level": user.max_access_level}


    if req.filters:
        for key, val in req.filters.items():
            where_clauses.append(f"(metadata->>:k_{key}) = :v_{key}")
            params[f"k_{key}"] = key
            params[f"v_{key}"] = str(val)

    where_sql = " AND ".join(where_clauses)

    stmt = sql_text(f"""
    WITH q AS (
    SELECT
        CAST(:qvec AS vector) AS qvec,
        plainto_tsquery('simple', :qtext) AS tsq
    )
    SELECT
    c.id,
    c.text,
    c.metadata AS meta,
    c.access_level,
    (1 - (c.embedding <=> q.qvec)) AS vscore,
    ts_rank_cd(to_tsvector('simple', c.text), q.tsq) AS tscore,
    ((1 - (c.embedding <=> q.qvec)) + (:alpha * ts_rank_cd(to_tsvector('english', c.text), q.tsq))) AS score
    FROM chunks c, q
    WHERE {where_sql}
    ORDER BY score DESC
    LIMIT :k
    """)

    # stmt = sql_text(f"""
    # SELECT
    # id,
    # text,
    # metadata AS meta,
    # access_level,
    # (1 - (embedding <=> CAST(:qvec AS vector))) AS score
    # FROM chunks
    # WHERE {where_sql}
    # ORDER BY embedding <=> CAST(:qvec AS vector)
    # LIMIT :k
    # """)

    rows = session.execute(stmt, params).all()

    results: List[ChunkOut] = []
    chunk_texts: List[str] = []

    for r in rows:
        meta = r.meta or {}
        chunk_texts.append(r.text)

        citation = Citation(
        source_path=meta.get("source_path"),
        title=meta.get("title"),
        department=meta.get("department"),
        page=meta.get("page"),
        access_level=int(r.access_level),
        )

        results.append(
            ChunkOut(
                chunk_id=int(r.id),
                score=float(r.score),
                vector_score=float(r.vscore),
                keyword_score=float(r.tscore),
                text=r.text,
                citation=citation,
            )
        )

    if req.mode == "citations_only":
        answer = citations_only_answer(req.query, chunk_texts)
        mode = "citations_only"
    else:
        try:
            answer = rag_answer(req.query, results)
            mode = "rag"
        except Exception as e:
            # If Ollama is down/unreachable, fall back to citations_only
            answer = citations_only_answer(req.query, chunk_texts) + f"\n\n(LLM fallback: {e})"
            mode = "citations_only"

    return ChatResponse(query=req.query, answer=answer, mode=mode, results=results)


    #     citation = Citation(
    #         source_path=meta.get("source_path"),
    #         title=meta.get("title"),
    #         department=meta.get("department"),
    #         page=meta.get("page"),
    #         access_level=int(r.access_level),
    #     )
    #     results.append({
    #     "chunk_id": r.id,
    #     "score": float(r.score),
    #     "vector_score": float(r.vscore),
    #     "keyword_score": float(r.tscore),
    #     "text": r.text,
    #     "citation": meta,
    # })

        # results.append(
        #     ChunkOut(
        #         chunk_id=int(r.id),
        #         score=float(r.score),
        #         text=r.text,
        #         citation=citation,
        #     )
        # )

    # answer = citations_only_answer(req.query, chunk_texts)

    # return ChatResponse(query=req.query, answer=answer, mode="citations_only", results=results)
   