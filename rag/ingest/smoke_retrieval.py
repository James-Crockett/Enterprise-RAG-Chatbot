from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DATA_PATH = Path("data/raw/policies.md")

def chunk_text(text: str, max_chars: int = 500):
    chunks, buf, cur = [], [], 0
    for line in text.splitlines():
        if not line.strip():
            continue
        if cur + len(line) > max_chars and buf:
            chunks.append("\n".join(buf))
            buf, cur = [], 0
        buf.append(line)
        cur += len(line)
    if buf:
        chunks.append("\n".join(buf))
    return chunks

def main():
    assert DATA_PATH.exists(), f"Missing {DATA_PATH}"
    text = DATA_PATH.read_text(encoding="utf-8")
    chunks = chunk_text(text)

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb = model.encode(chunks, normalize_embeddings=True)
    emb = np.asarray(emb, dtype="float32")

    index = faiss.IndexFlatIP(emb.shape[1])  # cosine when normalized
    index.add(emb)

    query = "How do I request PTO?"
    q = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q, k=min(3, len(chunks)))

    print("\nQuery:", query)
    for rank, (i, s) in enumerate(zip(ids[0], scores[0]), start=1):
        print(f"\n#{rank} score={s:.4f}\n{chunks[int(i)]}")

if __name__ == "__main__":
    main()
