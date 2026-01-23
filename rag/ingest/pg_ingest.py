from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from uuid import uuid4

import numpy as np
from sqlmodel import Session
from sqlalchemy import text as sql_text
from sentence_transformers import SentenceTransformer

from apps.api.core.db import engine
from apps.api.models import Document, Chunk
from rag.ingest.loaders import load_documents, Document as LoadedDocument
from rag.ingest.chunking import chunk_document


def infer_access_level_from_path(path_str: str) -> int:
    """
    Simple clearance model:
    - public -> 0
    - internal -> 1
    - restricted/confidential -> 2
    Default: 1 (internal)
    """
    p = path_str.lower()
    if "/public/" in p or "\\public\\" in p:
        return 0
    if "/restricted/" in p or "\\restricted\\" in p or "/confidential/" in p or "\\confidential\\" in p:
        return 2
    if "/internal/" in p or "\\internal\\" in p:
        return 1
    return 1


def reset_tables(session: Session) -> None:
    """
    Wipes documents + chunks (for dev).
    SQLAlchemy 2.0 requires raw SQL strings be wrapped in text().
    """
    session.exec(sql_text("TRUNCATE TABLE chunks RESTART IDENTITY CASCADE;"))
    session.exec(sql_text("TRUNCATE TABLE documents CASCADE;"))
    session.commit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data/raw")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--max_chars", type=int, default=1200)
    parser.add_argument("--overlap_chars", type=int, default=200)
    parser.add_argument("--reset", action="store_true", help="Delete existing documents/chunks before ingesting")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Input dir not found: {input_dir.resolve()}")

    # Load docs (MD/TXT/PDF)
    loaded_docs: List[LoadedDocument] = load_documents(input_dir)
    if not loaded_docs:
        raise SystemExit(f"No documents found in {input_dir.resolve()}")

    # Build chunks with metadata
    # store key citation fields in Chunk.meta so /chat can display them.
    chunk_rows: List[Dict[str, Any]] = []
    document_rows: List[Document] = []

    for d in loaded_docs:
        meta = dict(d.metadata)
        source_path = meta.get("source_path", "")
        access_level = infer_access_level_from_path(source_path)

        doc = Document(
            id=uuid4(),
            title=meta.get("title", "untitled"),
            source_path=source_path,
            department=meta.get("department", "general"),
            access_level=access_level,
        )
        document_rows.append(doc)

        chunks = chunk_document(
            text=d.text,
            base_metadata=meta,
            max_chars=args.max_chars,
            overlap_chars=args.overlap_chars,
        )

        for ch in chunks:
            # keep same access_level as doc (or override later if needed)
            chunk_meta = dict(ch.metadata)
            chunk_meta["access_level"] = access_level

            chunk_rows.append(
                {
                    "document_id": doc.id,
                    "chunk_index": chunk_meta.get("chunk_index", 0),
                    "page": chunk_meta.get("page"),
                    "text": ch.text,
                    "meta": chunk_meta,  # NOTE: model field is "meta", DB column is "metadata"
                    "access_level": access_level,
                }
            )

    # Embed all chunk texts (batch)
    embedder = SentenceTransformer(args.model)
    texts = [r["text"] for r in chunk_rows]
    embs = embedder.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=True)
    embs = np.asarray(embs, dtype="float32")

    if embs.shape[1] != 384:
        raise SystemExit(f"Unexpected embedding dim {embs.shape[1]} (expected 384). Are you using MiniLM?")

    # Insert into Postgres
    with Session(engine) as session:
        if args.reset:
            reset_tables(session)

        # Insert documents first
        for doc in document_rows:
            session.add(doc)
        session.commit()

        # Insert chunks
        for i, r in enumerate(chunk_rows):
            emb = embs[i].tolist()
            c = Chunk(
                document_id=r["document_id"],
                chunk_index=r["chunk_index"],
                page=r["page"],
                text=r["text"],
                meta=r["meta"],
                access_level=r["access_level"],
                embedding=emb,
            )
            session.add(c)

        session.commit()

    print("\n Ingestion complete")
    print(f"- Documents inserted: {len(document_rows)}")
    print(f"- Chunks inserted:    {len(chunk_rows)}")
    print(f"- Input dir:          {input_dir.resolve()}")
    print(f"- Access model:       public=0, internal=1, restricted=2")


if __name__ == "__main__":
    main()
