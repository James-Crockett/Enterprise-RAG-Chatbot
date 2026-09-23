from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

import torch
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sqlalchemy import text as sql_text
from sqlmodel import Session, col, select
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from apps.api.core.db import get_session
from apps.api.core.deps import get_current_user
from apps.api.core.security import create_access_token, verify_password
from apps.api.models import Conversation, Message, User

app = FastAPI(title="RAG Enterprise KB (pgvector)", version="0.2.0")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "auto").lower()
if EMBEDDING_DEVICE == "auto":
    EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# load the embedder once so requests only encode the query.
embedder = SentenceTransformer(EMBED_MODEL, device=EMBEDDING_DEVICE)
print(f"Embedding device: {EMBEDDING_DEVICE}")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
USE_LLM = os.getenv("USE_LLM", "false").lower() in ("1", "true", "yes")
OLLAMA_TIMEOUT_S = float(os.getenv("OLLAMA_TIMEOUT_S", "60"))

# keep filters tied to stored chunk metadata, not arbitrary sql fields.
ALLOWED_FILTER_KEYS = {"department", "source_path", "source_type"}

cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://localhost:5173",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["POST", "GET", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)


class LoginRequest(BaseModel):
    email: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=10)
    filters: dict[str, Any] | None = None
    mode: Literal["rag", "citations_only"] = "rag"
    # omit to start a new conversation.
    conversation_id: UUID | None = None


class CitationOut(BaseModel):
    document_id: str
    title: str
    source_path: str
    department: str
    access_level: int


class ChunkOut(BaseModel):
    chunk_id: int
    text: str
    score: float
    citation: CitationOut


class ChatResponse(BaseModel):
    query: str
    answer: str = ""
    mode: Literal["rag", "citations_only"] = "citations_only"
    results: list[ChunkOut] = Field(default_factory=list)
    conversation_id: str


class ConversationOut(BaseModel):
    id: str
    title: str
    updated_at: datetime


class MessageOut(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    mode: str | None = None
    results: list[ChunkOut] = Field(default_factory=list)


class ConversationDetail(ConversationOut):
    messages: list[MessageOut]


def citations_only_answer(chunks: list[str]) -> str:
    if not chunks:
        return "I couldn't find relevant information in the knowledge base."

    text = chunks[0].replace("\n", " ").strip()
    parts = text.split(". ")
    answer = ". ".join(parts[:2]).strip()
    if answer and not answer.endswith("."):
        answer += "."
    return answer


def build_context(results: list[ChunkOut], max_chars: int = 12000) -> str:
    parts: list[str] = []
    used = 0

    for result in results:
        citation = result.citation
        block = (
            f"[chunk:{result.chunk_id}] "
            f"title={citation.title} "
            f"dept={citation.department} "
            f"access={citation.access_level} "
            f"path={citation.source_path}\n"
            f"{result.text.strip()}\n"
        )

        if used + len(block) > max_chars:
            break

        parts.append(block)
        used += len(block)

    return "\n---\n".join(parts)


def ollama_chat(messages: list[dict[str, str]]) -> str:
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
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Ollama HTTPError {exc.code}: {detail}")
    except URLError as exc:
        raise RuntimeError(f"Ollama URLError: {exc}")


def rag_answer(query: str, results: list[ChunkOut]) -> str:
    context = build_context(results)
    system = (
        "You are an enterprise knowledge base assistant.\n"
        "Answer using ONLY the provided CONTEXT.\n"
        "If the answer is not in the context, say you don't know based on the KB.\n"
        "Keep answers concise and actionable.\n"
        "When you use a fact, cite it with [chunk:<id>] at the end of the sentence.\n"
    )

    return ollama_chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": f"QUESTION:\n{query}\n\nCONTEXT:\n{context}"},
        ]
    )


def retrieve_chunks(
    session: Session,
    query: str,
    top_k: int,
    filters: dict[str, Any] | None,
    max_access_level: int,
) -> list[ChunkOut]:
    query_vector = embedder.encode([query], normalize_embeddings=True)[0].tolist()
    qvec = "[" + ",".join(str(value) for value in query_vector) + "]"

    where_clauses = [
        "c.access_level <= :max_level",
        "d.access_level <= :max_level",
    ]
    params: dict[str, Any] = {
        "qvec": qvec,
        "k": top_k,
        "max_level": max_access_level,
    }

    if filters:
        for index, (key, value) in enumerate(filters.items()):
            if key not in ALLOWED_FILTER_KEYS:
                raise HTTPException(status_code=400, detail=f"Unsupported filter key: {key}")

            where_clauses.append(f"(c.metadata ->> :key{index}) = :value{index}")
            params[f"key{index}"] = key
            params[f"value{index}"] = str(value)

    # enforce access in sql before chunks can reach the prompt.
    stmt = sql_text(
        f"""
        SELECT
            c.id AS chunk_id,
            c.text AS text,
            d.id AS document_id,
            d.title AS title,
            d.source_path AS source_path,
            d.department AS department,
            d.access_level AS doc_access_level,
            (1 - (c.embedding <=> CAST(:qvec AS vector))) AS score
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE {" AND ".join(where_clauses)}
        ORDER BY c.embedding <=> CAST(:qvec AS vector)
        LIMIT :k
        """
    )

    rows = session.execute(stmt, params).all()
    results: list[ChunkOut] = []

    for row in rows:
        data = row._mapping
        results.append(
            ChunkOut(
                chunk_id=int(data["chunk_id"]),
                text=str(data["text"]),
                score=float(data["score"]),
                citation=CitationOut(
                    document_id=str(data["document_id"]),
                    title=str(data["title"]),
                    source_path=str(data["source_path"]),
                    department=str(data["department"]),
                    access_level=int(data["doc_access_level"]),
                ),
            )
        )

    return results


def owned_conversation(session: Session, conversation_id: UUID, user: User) -> Conversation:
    conversation = session.get(Conversation, conversation_id)
    # 404 for other users' chats too, so ids can't be probed.
    if not conversation or conversation.user_id != user.id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


def conversation_title(query: str, max_chars: int = 60) -> str:
    title = " ".join(query.split())
    if len(title) <= max_chars:
        return title
    return title[: max_chars - 3].rstrip() + "..."


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
    # check ownership before spending time on retrieval and the llm.
    if req.conversation_id:
        conversation = owned_conversation(session, req.conversation_id, user)
    else:
        conversation = Conversation(user_id=user.id, title=conversation_title(req.query))

    results = retrieve_chunks(
        session=session,
        query=req.query,
        top_k=req.top_k,
        filters=req.filters,
        max_access_level=user.max_access_level,
    )
    chunk_texts = [result.text for result in results]

    mode: Literal["rag", "citations_only"] = "citations_only"
    if req.mode == "citations_only" or not USE_LLM:
        answer = citations_only_answer(chunk_texts)
    else:
        try:
            answer = rag_answer(req.query, results)
            mode = "rag"
        except Exception:
            # keep the app usable when ollama is disabled or unavailable.
            answer = citations_only_answer(chunk_texts)

    conversation.updated_at = datetime.utcnow()
    session.add(conversation)
    session.add(Message(conversation_id=conversation.id, role="user", content=req.query, sources=[]))
    session.add(
        Message(
            conversation_id=conversation.id,
            role="assistant",
            content=answer,
            mode=mode,
            sources=[result.model_dump(mode="json") for result in results],
        )
    )
    session.commit()

    return ChatResponse(
        query=req.query,
        answer=answer,
        mode=mode,
        results=results,
        conversation_id=str(conversation.id),
    )


@app.get("/departments", response_model=list[str])
def departments(
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    # only list departments the user can actually retrieve from.
    rows = session.execute(
        sql_text(
            "SELECT DISTINCT department FROM documents "
            "WHERE access_level <= :max_level ORDER BY department"
        ),
        {"max_level": user.max_access_level},
    ).all()
    return [row[0] for row in rows]


@app.get("/conversations", response_model=list[ConversationOut])
def list_conversations(
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    conversations = session.exec(
        select(Conversation)
        .where(Conversation.user_id == user.id)
        .order_by(col(Conversation.updated_at).desc())
    ).all()
    return [
        ConversationOut(id=str(c.id), title=c.title, updated_at=c.updated_at)
        for c in conversations
    ]


@app.get("/conversations/{conversation_id}", response_model=ConversationDetail)
def get_conversation(
    conversation_id: UUID,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    conversation = owned_conversation(session, conversation_id, user)
    rows = session.exec(
        select(Message)
        .where(Message.conversation_id == conversation.id)
        .order_by(col(Message.id))
    ).all()

    messages: list[MessageOut] = []
    for row in rows:
        # drop saved sources above the user's current clearance, in case it was lowered.
        results = [
            ChunkOut.model_validate(source)
            for source in row.sources
            if source["citation"]["access_level"] <= user.max_access_level
        ]
        messages.append(
            MessageOut(role=row.role, content=row.content, mode=row.mode, results=results)
        )

    return ConversationDetail(
        id=str(conversation.id),
        title=conversation.title,
        updated_at=conversation.updated_at,
        messages=messages,
    )


@app.delete("/conversations/{conversation_id}", status_code=204)
def delete_conversation(
    conversation_id: UUID,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    conversation = owned_conversation(session, conversation_id, user)
    session.delete(conversation)
    session.commit()
