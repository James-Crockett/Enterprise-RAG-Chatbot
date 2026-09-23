import type {
  ChatResponse,
  ConversationDetail,
  ConversationSummary,
} from "./types";

export class ApiError extends Error {
  constructor(message: string, public status: number) {
    super(message);
  }
}

async function request<T>(
  apiUrl: string,
  path: string,
  options: { method?: string; token?: string; body?: unknown } = {}
): Promise<T> {
  const headers: Record<string, string> = {};
  if (options.body !== undefined) headers["Content-Type"] = "application/json";
  if (options.token) headers.Authorization = `Bearer ${options.token}`;

  const resp = await fetch(`${apiUrl}${path}`, {
    method: options.method || (options.body !== undefined ? "POST" : "GET"),
    headers,
    body: options.body !== undefined ? JSON.stringify(options.body) : undefined,
  });

  if (!resp.ok) {
    const text = await resp.text();
    throw new ApiError(text || `Request failed (${resp.status})`, resp.status);
  }

  if (resp.status === 204) return undefined as T;
  return (await resp.json()) as T;
}

export function login(apiUrl: string, email: string, password: string) {
  return request<{ access_token: string; token_type: string }>(
    apiUrl,
    "/auth/login",
    { body: { email, password } }
  );
}

export function chat(
  apiUrl: string,
  token: string,
  payload: {
    query: string;
    top_k: number;
    filters?: Record<string, string>;
    mode?: "rag" | "citations_only";
    conversation_id?: string;
  }
) {
  return request<ChatResponse>(apiUrl, "/chat", { token, body: payload });
}

export function listDepartments(apiUrl: string, token: string) {
  return request<string[]>(apiUrl, "/departments", { token });
}

export function listConversations(apiUrl: string, token: string) {
  return request<ConversationSummary[]>(apiUrl, "/conversations", { token });
}

export function getConversation(apiUrl: string, token: string, id: string) {
  return request<ConversationDetail>(apiUrl, `/conversations/${id}`, { token });
}

export function deleteConversation(apiUrl: string, token: string, id: string) {
  return request<void>(apiUrl, `/conversations/${id}`, {
    token,
    method: "DELETE",
  });
}
