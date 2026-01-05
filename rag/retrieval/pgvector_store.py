from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sqlmodel import Session, select
from sqlalchemy import text as sql_text
from sentence_transformers import SentenceTransformer

from apps.api.models import Chunk


@dataclass
class Retrieved:
    id: int
    score: float
    text: str
    metadata: Dict[str, Any]


class PgVectorStore:
    def __init__(self, embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embed_model)

    def embed(self, query: str) -> List[float]:
        # normalize so cosine distance behaves consistently
        vec = self.model.encode([query], normalize_embeddings=True)[0]
        return vec.tolist()

    def retrieve(
        self,
        session: Session,
        query: str,
        k: int,
        user_roles: List[str],
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Retrieved]:
        """
        Uses pgvector cosine distance.
        RBAC: only chunks whose allowed_roles overlaps user_roles are eligible.
        """
        qvec = self.embed(query)

        where_clauses = ["allowed_roles && :user_roles"]
        params: Dict[str, Any] = {"qvec": qvec, "k": k, "user_roles": user_roles}

        if filters:
            # support common filters stored in metadata JSONB
            for key, val in filters.items():
                where_clauses.append(f"(metadata->>:k_{key}) = :v_{key}")
                params[f"k_{key}"] = key
                params[f"v_{key}"] = str(val)

        where_sql = " AND ".join(where_clauses)

        #cosine distance
        stmt = sql_text(f"""
            SELECT id, text, metadata, (1 - (embedding <=> :qvec)) AS score
            FROM chunks
            WHERE {where_sql}
            ORDER BY embedding <=> :qvec
            LIMIT :k
        """)

        rows = session.exec(stmt, params).all()

        out: List[Retrieved] = []
        for r in rows:
            out.append(Retrieved(id=r.id, score=float(r.score), text=r.text, metadata=r.metadata))
        return out
