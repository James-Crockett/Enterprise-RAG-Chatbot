export type ChatRole = "user" | "assistant";

export type Citation = {
  document_id: string;
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
  error?: boolean;
};

export type ChatResponse = {
  query: string;
  answer: string;
  mode: string;
  results: ChunkResult[];
  conversation_id: string;
};

export type ConversationSummary = {
  id: string;
  title: string;
  updated_at: string;
};

export type ConversationDetail = ConversationSummary & {
  messages: {
    role: ChatRole;
    content: string;
    mode: string | null;
    results: ChunkResult[];
  }[];
};
