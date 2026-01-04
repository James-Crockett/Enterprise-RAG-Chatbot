from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, List

from fastapi import FastAPI
from pydantic import BaseModel

from rag.retrieval.vectorstore import LocalVectorStore, RetrievedChunk
from rag.generation.citations_only import build_citations_only_answer



INDEX_PATH = Path(os.getenv("INDEX_PATH", "storage/faiss/index.faiss"))
CHUNKS_PATH = Path(os.getenv("CHUNKS_PATH", "storage/docstore/chunks.jsonl"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

app = FastAPI(title="RAG Enterprise KB (Retrieval API)", version="0.1.0")

store: Optional[LocalVectorStore] = None


class ChatRequest(BaseModel):
    query: str
    top_k: int = 5
    filters: Optional[Dict[str, Any]] = None  # e.g. {"department": "hr"} or {"confidentiality": "restricted"}


class Citation(BaseModel):
    source_path: str
    page: Optional[int] = None
    title: Optional[str] = None
    department: Optional[str] = None
    confidentiality: Optional[str] = None


class ChunkOut(BaseModel):
    chunk_id: int
    score: float
    text: str
    citation: Citation


class ChatResponse(BaseModel):
    query: str
    mode: str  # retrieval_only / citations_only
    answer: str = ""  
    results: List[ChunkOut] = []


@app.on_event("startup")
def _startup():
    global store
    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        print("Index not found. Run: python -m rag.ingest.build_index --input_dir data/raw")
        return
    store = LocalVectorStore(INDEX_PATH, CHUNKS_PATH, EMBED_MODEL)
    print("VectorStore loaded")


@app.get("/health")
def health():
    ok = store is not None
    return {"ok": ok, "index_path": str(INDEX_PATH), "chunks_path": str(CHUNKS_PATH)}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if store is None:
        return ChatResponse(query=req.query, mode="retrieval_only", results=[])

    hits = store.retrieve(req.query, k=req.top_k, filters=req.filters)

    results = []
    for h in hits:
        meta = h.metadata
        citation = Citation(
            source_path=meta.get("source_path", ""),
            page=meta.get("page"),
            title=meta.get("title"),
            department=meta.get("department"),
            confidentiality=meta.get("confidentiality"),
        )
        results.append(ChunkOut(chunk_id=h.chunk_id, score=h.score, text=h.text, citation=citation))

    return ChatResponse(query=req.query, mode="retrieval_only", results=results)

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if store is None:
        return ChatResponse(query=req.query, mode="retrieval_only", answer="", results=[])

    hits = store.retrieve(req.query, k=req.top_k, filters=req.filters)

    # Build a citations-only answer using the SAME embedder as retrieval (store.model)
    answer, used_chunk_ids = build_citations_only_answer(
        query=req.query,
        chunks=hits,
        embedder=store.model,
        max_sentences=3,
    )

    results = []
    for h in hits:
        meta = h.metadata
        citation = Citation(
            source_path=meta.get("source_path", ""),
            page=meta.get("page"),
            title=meta.get("title"),
            department=meta.get("department"),
            confidentiality=meta.get("confidentiality"),
        )

        # Optional: mark whether this chunk contributed to answer
        text = h.text
        results.append(ChunkOut(chunk_id=h.chunk_id, score=h.score, text=text, citation=citation))

    return ChatResponse(query=req.query, mode="citations_only", answer=answer, results=results)
