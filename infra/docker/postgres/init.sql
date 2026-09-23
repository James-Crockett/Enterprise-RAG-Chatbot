CREATE EXTENSION IF NOT EXISTS vector;

-- users: login plus access level
-- max_access_level: 0=public, 1=internal, 2=restricted
CREATE TABLE IF NOT EXISTS users (
  id UUID PRIMARY KEY,
  email TEXT UNIQUE NOT NULL,
  hashed_password TEXT NOT NULL,
  max_access_level INT NOT NULL DEFAULT 0,
  is_active BOOLEAN NOT NULL DEFAULT TRUE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- documents: uploaded files and their default sensitivity
CREATE TABLE IF NOT EXISTS documents (
  id UUID PRIMARY KEY,
  title TEXT NOT NULL,
  source_path TEXT,
  department TEXT NOT NULL DEFAULT 'general',
  access_level INT NOT NULL DEFAULT 1,  -- default internal
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- chunks: retrieval unit
-- minilm embeddings are 384-dim
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

-- indexes
CREATE INDEX IF NOT EXISTS chunks_doc_id_idx ON chunks(document_id);
CREATE INDEX IF NOT EXISTS chunks_access_level_idx ON chunks(access_level);

-- no ann index on embeddings: an ivfflat index built on an empty table only
-- searches one near-empty list per query and drops results. an exact scan is
-- fast at this size; add hnsw after loading data if the corpus grows large.

-- conversations: saved chat threads, one owner each
CREATE TABLE IF NOT EXISTS conversations (
  id UUID PRIMARY KEY,
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  title TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS conversations_user_updated_idx
  ON conversations(user_id, updated_at DESC);

-- messages: one row per turn; assistant rows keep the sources they cited
CREATE TABLE IF NOT EXISTS messages (
  id BIGSERIAL PRIMARY KEY,
  conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
  content TEXT NOT NULL,
  mode TEXT,
  sources JSONB NOT NULL DEFAULT '[]'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS messages_conversation_idx ON messages(conversation_id, id);
