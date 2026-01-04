from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Chunk:
    text: str
    metadata: Dict[str, Any]


def _split_into_paragraphs(text: str) -> List[str]:
    # Basic paragraph split: blank lines separate paragraphs
    paras = []
    buf = []
    for line in text.splitlines():
        if line.strip():
            buf.append(line.strip())
        else:
            if buf:
                paras.append(" ".join(buf))
                buf = []
    if buf:
        paras.append(" ".join(buf))
    return paras


def chunk_document(
    text: str,
    base_metadata: Dict[str, Any],
    max_chars: int = 1200,
    overlap_chars: int = 200,
) -> List[Chunk]:
    """
    Simple chunking:
    - break into paragraphs
    - pack paragraphs until max_chars
    - add overlap between chunks to preserve context
    """
    paras = _split_into_paragraphs(text)
    chunks: List[str] = []

    current = ""
    for p in paras:
        if len(current) + len(p) + 1 <= max_chars:
            current = (current + " " + p).strip()
        else:
            if current:
                chunks.append(current)
            current = p

    if current:
        chunks.append(current)

    # add overlap by taking tail of previous chunk
    overlapped: List[str] = []
    for i, ch in enumerate(chunks):
        if i == 0:
            overlapped.append(ch)
            continue
        prev_tail = chunks[i - 1][-overlap_chars:] if overlap_chars > 0 else ""
        overlapped.append((prev_tail + "\n" + ch).strip())

    out: List[Chunk] = []
    for idx, ch in enumerate(overlapped):
        meta = dict(base_metadata)
        meta["chunk_index"] = idx
        out.append(Chunk(text=ch, metadata=meta))
    return out
