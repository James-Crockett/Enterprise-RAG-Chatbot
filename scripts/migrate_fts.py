from sqlmodel import Session
from apps.api.core.db import engine
from sqlalchemy import text

SQL = """
CREATE INDEX IF NOT EXISTS chunks_text_fts_idx
ON chunks
USING GIN (to_tsvector('english', text));
"""

def main():
    with Session(engine) as s:
        s.exec(text(SQL))
        s.commit()
    print("FTS index created (or already exists).")

if __name__ == "__main__":
    main()
