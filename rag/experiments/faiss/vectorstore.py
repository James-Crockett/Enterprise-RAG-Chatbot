from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


@dataclass
class RetrievedChunk:
    chunk_id: int
    score: float
    text: str
    metadata: Dict[str, Any]


class LocalVectorStore:
    def __init__(self, index_path: Path, chunks_path: Path, model_name: str):
        self.index = faiss.read_index(str(index_path))
        self.model = SentenceTransformer(model_name)
        self.chunks = self._load_chunks(chunks_path)

    def _load_chunks(self, chunks_path: Path) -> List[Dict[str, Any]]:
        out = []
        with chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                out.append(json.loads(line))
        return out

    def _match_filters(self, meta: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
        if not filters:
            return True
        for k, v in filters.items():
            if meta.get(k) != v:
                return False
        return True

    def retrieve(self, query: str, k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[RetrievedChunk]:
        # Over-fetch then filter, because FAISS itself doesn't filter metadata in this simple setup.
        overfetch = max(k * 5, 20)

        q = self.model.encode([query], normalize_embeddings=True).astype("float32")
        scores, ids = self.index.search(q, overfetch)

        results: List[RetrievedChunk] = []
        for idx, score in zip(ids[0], scores[0]):
            if idx < 0:
                continue
            row = self.chunks[int(idx)]
            meta = row["metadata"]
            if not self._match_filters(meta, filters):
                continue
            results.append(
                RetrievedChunk(
                    chunk_id=row["chunk_id"],
                    score=float(score),
                    text=row["text"],
                    metadata=meta,
                )
            )
            if len(results) >= k:
                break

        return results
