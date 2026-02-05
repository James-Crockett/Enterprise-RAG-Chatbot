export type ChatRole = "user" | "assistant";

export type Citation = {
  document_id: number;
  title: string;
  source_path: string;
  department: string;
  access_level: number;
};

export type ChunkResult = {
  chunk_id: number;
  text: string;
  score: number;
  citation: Citation;
};

export type ChatMessage = {
  role: ChatRole;
  content: string;
  results?: ChunkResult[];
};

export type ChatResponse = {
  query: string;
  answer: string;
  mode: string;
  results: ChunkResult[];
};
