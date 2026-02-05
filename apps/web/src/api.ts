import type { ChatResponse } from "./types";

export async function login(apiUrl: string, email: string, password: string) {
  const resp = await fetch(`${apiUrl}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(text || "Login failed");
  }

  return (await resp.json()) as { access_token: string; token_type: string };
}

export async function chat(
  apiUrl: string,
  token: string,
  payload: {
    query: string;
    top_k: number;
    filters?: Record<string, string>;
    mode?: "rag" | "citations_only";
  }
) {
  const resp = await fetch(`${apiUrl}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify(payload),
  });

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(text || "Chat request failed");
  }

  return (await resp.json()) as ChatResponse;
}
