from __future__ import annotations

import re
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from rag.retrieval.vectorstore import RetrievedChunk


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> List[str]:
    """
    Break text into sentences.
    This is a simple regex-based splitter; good enough for MVP.
    """
    text = text.replace("\n", " ").strip()
    if not text:
        return []
    sents = _SENT_SPLIT.split(text)
    # Keep only non-trivial sentences
    return [s.strip() for s in sents if len(s.strip()) >= 20]


def build_citations_only_answer(
    query: str,
    chunks: List[RetrievedChunk],
    embedder: SentenceTransformer,
    max_sentences: int = 3,
) -> Tuple[str, List[int]]:
    """
    Creates an answer without any LLM:
    - Take retrieved chunks
    - Split into sentences
    - Score each sentence by semantic similarity to the query
    - Return top-N sentences as the answer

    Returns:
      (answer_text, used_chunk_ids)
    """
    if not chunks:
        return "I couldn't find relevant information in the knowledge base.", []

    # Collect candidate sentences with a pointer to which chunk they came from
    candidates: List[Tuple[str, int]] = []
    for ch in chunks:
        for s in split_sentences(ch.text):
            candidates.append((s, ch.chunk_id))

    if not candidates:
        return "I found relevant sources, but couldn't extract a clear answer.", [c.chunk_id for c in chunks]

    sentences = [s for s, _ in candidates]

    # Embed query + sentences in the same vector space
    q_emb = embedder.encode([query], normalize_embeddings=True).astype("float32")  # (1, d)
    s_emb = embedder.encode(sentences, normalize_embeddings=True).astype("float32")  # (n, d)

    # Cosine similarity (because vectors are normalized) = dot product
    scores = (s_emb @ q_emb.T).reshape(-1)  # (n,)

    # Pick top scoring sentences, deduplicate, and keep it concise
    top_idx = np.argsort(-scores)[: max_sentences * 3]  # over-pick then dedupe
    chosen = []
    used_chunk_ids = []
    seen = set()

    for i in top_idx:
        sent = sentences[int(i)]
        if sent in seen:
            continue
        seen.add(sent)
        chosen.append(sent)
        used_chunk_ids.append(candidates[int(i)][1])
        if len(chosen) >= max_sentences:
            break

    # Small cleanup: keep unique chunk ids in order
    unique_used = []
    for cid in used_chunk_ids:
        if cid not in unique_used:
            unique_used.append(cid)

    return " ".join(chosen), unique_used
