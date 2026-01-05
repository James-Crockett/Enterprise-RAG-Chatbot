CREATE EXTENSION IF NOT EXISTS vector;

-- Users: login + access level
-- max_access_level: 0=public, 1=internal, 2=restricted
-- is_admin: can upload docs / reindex
CREATE TABLE IF NOT EXISTS users (
  id UUID PRIMARY KEY,
  email TEXT UNIQUE NOT NULL,
  hashed_password TEXT NOT NULL,
  max_access_level INT NOT NULL DEFAULT 0,
  is_admin BOOLEAN NOT NULL DEFAULT FALSE,
  is_active BOOLEAN NOT NULL DEFAULT TRUE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Documents: uploaded files and their default sensitivity
CREATE TABLE IF NOT EXISTS documents (
  id UUID PRIMARY KEY,
  title TEXT NOT NULL,
  source_path TEXT,
  department TEXT NOT NULL DEFAULT 'general',
  access_level INT NOT NULL DEFAULT 1,  -- default internal
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Chunks: retrieval unit
-- MiniLM embeddings are 384-dim
CREATE TABLE IF NOT EXISTS chunks (
  id BIGSERIAL PRIMARY KEY,
  document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
  chunk_index INT NOT NULL,
  page INT,
  text TEXT NOT NULL,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  access_level INT NOT NULL DEFAULT 1,
  embedding VECTOR(384) NOT NULL
);

-- Indexes
CREATE INDEX IF NOT EXISTS chunks_doc_id_idx ON chunks(document_id);
CREATE INDEX IF NOT EXISTS chunks_access_level_idx ON chunks(access_level);

-- Vector index (good practice; still fine for MVP)
CREATE INDEX IF NOT EXISTS chunks_embedding_cos_idx
  ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
