from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from rag.ingest.loaders import load_documents
from rag.ingest.chunking import chunk_document


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data/raw")
    parser.add_argument("--index_dir", type=str, default="storage/faiss")
    parser.add_argument("--docstore_dir", type=str, default="storage/docstore")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--max_chars", type=int, default=1200)
    parser.add_argument("--overlap_chars", type=int, default=200)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    index_dir = Path(args.index_dir)
    docstore_dir = Path(args.docstore_dir)

    docs = load_documents(input_dir)
    if not docs:
        raise SystemExit(f"No documents found in {input_dir.resolve()}")

    # Chunk all documents
    chunks = []
    for d in docs:
        chunks.extend(
            chunk_document(
                text=d.text,
                base_metadata=d.metadata,
                max_chars=args.max_chars,
                overlap_chars=args.overlap_chars,
            )
        )

    if not chunks:
        raise SystemExit("No chunks produced. Check your loaders/chunking.")

    # Embed chunks
    model = SentenceTransformer(args.model)
    texts = [c.text for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    # Build FAISS index (cosine via normalized + inner product)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Persist index
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / "index.faiss"
    faiss.write_index(index, str(index_path))

    # Persist docstore (chunk text + metadata + id)
    docstore_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = docstore_dir / "chunks.jsonl"

    rows = []
    for i, c in enumerate(chunks):
        meta = dict(c.metadata)
        meta["chunk_id"] = i
        rows.append({"chunk_id": i, "text": c.text, "metadata": meta})

    write_jsonl(chunks_path, rows)

    print("\nBuilt index!")
    print(f"- Docs loaded:   {len(docs)}")
    print(f"- Chunks created:{len(chunks)}")
    print(f"- FAISS index:   {index_path.resolve()}")
    print(f"- Docstore:      {chunks_path.resolve()}")


if __name__ == "__main__":
    main()
