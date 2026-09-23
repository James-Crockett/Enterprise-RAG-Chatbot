import { useEffect, useRef, useState } from "react";
import type { FormEvent, KeyboardEvent, ReactNode } from "react";
import {
  ApiError,
  chat,
  deleteConversation,
  getConversation,
  listConversations,
  listDepartments,
  login,
} from "./api";
import type { ChatMessage, ChunkResult, ConversationSummary } from "./types";

const API_URL = import.meta.env.VITE_API_URL || "/api";
const STORAGE_KEY = "rag_kb_settings";

type Mode = "rag" | "citations_only";

type Settings = {
  topK: number;
  department: string;
  mode: Mode;
  email: string;
};

const defaultSettings: Settings = {
  topK: 5,
  department: "",
  mode: "rag",
  email: "",
};

function loadSettings(): Settings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return defaultSettings;
    const saved = { ...defaultSettings, ...JSON.parse(raw) } as Settings;
    // older builds stored "(none)" for no department filter.
    if (saved.department === "(none)") saved.department = "";
    return saved;
  } catch {
    return defaultSettings;
  }
}

function saveSettings(settings: Settings) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  } catch {
    // settings just won't persist.
  }
}

// the api returns fastapi error bodies like {"detail": "..."} as plain text.
function errorText(err: unknown, fallback: string) {
  if (err instanceof ApiError && (err.status === 502 || err.status === 504)) {
    return "The server took too long to answer. The reply may still show up in this chat's history in a minute.";
  }
  if (!(err instanceof Error) || !err.message) return fallback;
  try {
    const detail = JSON.parse(err.message).detail;
    return typeof detail === "string" ? detail : fallback;
  } catch {
    return err.message.startsWith("<") ? fallback : err.message;
  }
}

export default function App() {
  const [settings, setSettings] = useState<Settings>(loadSettings);
  const [token, setToken] = useState<string | null>(null);

  const update = (patch: Partial<Settings>) => {
    setSettings((prev) => {
      const next = { ...prev, ...patch };
      saveSettings(next);
      return next;
    });
  };

  if (!token) {
    return (
      <Login
        email={settings.email}
        onEmail={(email) => update({ email })}
        onToken={setToken}
      />
    );
  }

  return (
    <Chat
      token={token}
      settings={settings}
      onUpdate={update}
      onLogout={() => setToken(null)}
    />
  );
}

function Login({
  email,
  onEmail,
  onToken,
}: {
  email: string;
  onEmail: (email: string) => void;
  onToken: (token: string) => void;
}) {
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const result = await login(API_URL, email, password);
      onToken(result.access_token);
    } catch (err) {
      setError(errorText(err, "Login failed."));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="login">
      <form className="login-form" onSubmit={submit}>
        <h1>Sign in</h1>
        <p className="muted">Enterprise knowledge base</p>

        <label>
          Email
          <input
            type="email"
            autoComplete="username"
            value={email}
            onChange={(e) => onEmail(e.target.value)}
            required
            autoFocus={!email}
          />
        </label>
        <label>
          Password
          <input
            type="password"
            autoComplete="current-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            autoFocus={Boolean(email)}
          />
        </label>

        {error ? <p className="error-text">{error}</p> : null}

        <button type="submit" className="primary" disabled={busy}>
          {busy ? "Signing in..." : "Continue"}
        </button>
      </form>
    </div>
  );
}

function Chat({
  token,
  settings,
  onUpdate,
  onLogout,
}: {
  token: string;
  settings: Settings;
  onUpdate: (patch: Partial<Settings>) => void;
  onLogout: () => void;
}) {
  const [conversations, setConversations] = useState<ConversationSummary[]>(
    []
  );
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [departments, setDepartments] = useState<string[]>([]);
  const [prompt, setPrompt] = useState("");
  const [busy, setBusy] = useState(false);
  const [loadingThread, setLoadingThread] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  // the thread the user last asked for, so slow loads can't overwrite a newer pick.
  const requestedId = useRef<string | null>(null);

  // an expired token makes every call fail, so send the user back to login.
  const onAuthError = (err: unknown) => {
    if (err instanceof ApiError && err.status === 401) {
      onLogout();
      return true;
    }
    return false;
  };

  const refreshConversations = async () => {
    try {
      setConversations(await listConversations(API_URL, token));
    } catch (err) {
      onAuthError(err);
    }
  };

  useEffect(() => {
    refreshConversations();
    listDepartments(API_URL, token)
      .then(setDepartments)
      .catch(onAuthError);
    // load once per login.
  }, [token]);

  // a saved filter can point at a department this account can't see.
  useEffect(() => {
    if (
      departments.length > 0 &&
      settings.department &&
      !departments.includes(settings.department)
    ) {
      onUpdate({ department: "" });
    }
  }, [departments, settings.department, onUpdate]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "end" });
  }, [messages, busy]);

  // grow the textarea with its content; css caps the height.
  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
  }, [prompt]);

  const newChat = () => {
    requestedId.current = null;
    setActiveId(null);
    setMessages([]);
    setLoadingThread(false);
    setSidebarOpen(false);
    inputRef.current?.focus();
  };

  const openConversation = async (id: string) => {
    setSidebarOpen(false);
    if (id === activeId) return;
    requestedId.current = id;
    setActiveId(id);
    setMessages([]);
    setLoadingThread(true);
    try {
      const detail = await getConversation(API_URL, token, id);
      if (requestedId.current !== id) return;
      setMessages(
        detail.messages.map((m) => ({
          role: m.role,
          content: m.content,
          results: m.results,
        }))
      );
    } catch (err) {
      if (requestedId.current !== id || onAuthError(err)) return;
      setMessages([
        {
          role: "assistant",
          content: errorText(err, "Could not load this chat."),
          error: true,
        },
      ]);
    } finally {
      if (requestedId.current === id) setLoadingThread(false);
    }
  };

  const removeConversation = async (id: string) => {
    const target = conversations.find((c) => c.id === id);
    if (!window.confirm(`Delete "${target?.title || "this chat"}"?`)) return;
    try {
      await deleteConversation(API_URL, token, id);
      setConversations((prev) => prev.filter((c) => c.id !== id));
      if (id === activeId) newChat();
    } catch (err) {
      if (!onAuthError(err)) window.alert(errorText(err, "Delete failed."));
    }
  };

  const send = async () => {
    const query = prompt.trim();
    if (!query || busy || loadingThread) return;

    setMessages((prev) => [...prev, { role: "user", content: query }]);
    setPrompt("");
    setBusy(true);

    try {
      const response = await chat(API_URL, token, {
        query,
        top_k: settings.topK,
        filters: settings.department
          ? { department: settings.department }
          : undefined,
        mode: settings.mode,
        conversation_id: activeId || undefined,
      });
      requestedId.current = response.conversation_id;
      setActiveId(response.conversation_id);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: response.answer || "No answer returned.",
          results: response.results || [],
        },
      ]);
      refreshConversations();
    } catch (err) {
      if (onAuthError(err)) return;
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: errorText(err, "Something went wrong."),
          error: true,
        },
      ]);
      // a proxy timeout doesn't stop the api, which may still save the chat.
      refreshConversations();
    } finally {
      setBusy(false);
      inputRef.current?.focus();
    }
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      send();
    }
  };

  const summary = [
    `top ${settings.topK}`,
    settings.department || "all departments",
    settings.mode === "rag" ? "LLM answer" : "excerpt only",
  ].join(" · ");

  const activeTitle = conversations.find((c) => c.id === activeId)?.title;

  return (
    <div className={`app${sidebarOpen ? " sidebar-open" : ""}`}>
      <aside className="sidebar">
        <div className="sidebar-top">
          <span className="brand">Enterprise KB</span>
          <button className="new-chat" onClick={newChat} disabled={busy}>
            New chat
          </button>
        </div>

        <nav className="history" aria-label="Previous chats">
          {conversations.length === 0 ? (
            <p className="history-empty muted">No chats yet.</p>
          ) : (
            groupByDate(conversations).map(([label, items]) => (
              <div key={label} className="history-group">
                <div className="history-label">{label}</div>
                {items.map((c) => (
                  <div
                    key={c.id}
                    className={`history-item${
                      c.id === activeId ? " active" : ""
                    }`}
                  >
                    <button
                      className="history-open"
                      onClick={() => openConversation(c.id)}
                      disabled={busy}
                      title={c.title}
                    >
                      {c.title}
                    </button>
                    <button
                      className="history-delete"
                      onClick={() => removeConversation(c.id)}
                      disabled={busy}
                      aria-label={`Delete ${c.title}`}
                    >
                      <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">
                        <path
                          d="M4 4l8 8M12 4l-8 8"
                          stroke="currentColor"
                          strokeWidth="1.6"
                          strokeLinecap="round"
                        />
                      </svg>
                    </button>
                  </div>
                ))}
              </div>
            ))
          )}
        </nav>

        <div className="sidebar-bottom">
          <span className="account" title={settings.email}>
            {settings.email}
          </span>
          <button className="plain" onClick={onLogout}>
            Log out
          </button>
        </div>
      </aside>

      <div
        className="backdrop"
        onClick={() => setSidebarOpen(false)}
        aria-hidden="true"
      />

      <div className="main">
        <header className="topbar">
          <button
            className="plain menu-toggle"
            onClick={() => setSidebarOpen(true)}
            aria-label="Show chats"
          >
            <svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true">
              <path
                d="M2.5 4h11M2.5 8h11M2.5 12h11"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinecap="round"
              />
            </svg>
          </button>
          <span className="thread-title">{activeTitle || ""}</span>
          <SettingsMenu
            settings={settings}
            departments={departments}
            onUpdate={onUpdate}
          />
        </header>

        <main className="thread">
          <div className="thread-inner">
            {loadingThread ? (
              <p className="muted pending">Loading chat...</p>
            ) : messages.length === 0 ? (
              <div className="empty">
                <h2>What do you want to know?</h2>
                <p className="muted">
                  Answers come only from documents your account can access.
                </p>
              </div>
            ) : (
              messages.map((m, idx) => <Message key={idx} message={m} />)
            )}
            {busy ? (
              <div className="msg assistant">
                <p className="muted pending">Searching documents...</p>
              </div>
            ) : null}
            <div ref={endRef} />
          </div>
        </main>

        <footer className="composer-wrap">
          <div className="composer">
            <textarea
              ref={inputRef}
              rows={1}
              placeholder="Ask a question"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={onKeyDown}
              autoFocus
            />
            <button
              className="send"
              onClick={send}
              disabled={busy || loadingThread || !prompt.trim()}
              aria-label="Send"
            >
              <svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true">
                <path
                  d="M8 13V3M3.5 7.5 8 3l4.5 4.5"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </button>
          </div>
          <p className="composer-note">{summary}</p>
        </footer>
      </div>
    </div>
  );
}

function groupByDate(conversations: ConversationSummary[]) {
  const startOfToday = new Date();
  startOfToday.setHours(0, 0, 0, 0);
  const today = startOfToday.getTime();
  const day = 24 * 60 * 60 * 1000;

  const groups: [string, ConversationSummary[]][] = [
    ["Today", []],
    ["Yesterday", []],
    ["Previous 7 days", []],
    ["Previous 30 days", []],
    ["Older", []],
  ];

  for (const c of conversations) {
    const t = new Date(c.updated_at).getTime();
    const index =
      t >= today ? 0 : t >= today - day ? 1 : t >= today - 7 * day ? 2 : t >= today - 30 * day ? 3 : 4;
    groups[index][1].push(c);
  }

  return groups.filter(([, items]) => items.length > 0);
}

function SettingsMenu({
  settings,
  departments,
  onUpdate,
}: {
  settings: Settings;
  departments: string[];
  onUpdate: (patch: Partial<Settings>) => void;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onPointer = (e: MouseEvent) => {
      if (!ref.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: globalThis.KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onPointer);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onPointer);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <div className="menu-anchor" ref={ref}>
      <button
        className="plain"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        Settings
      </button>
      {open ? (
        <div className="menu">
          <label>
            <span>Sources</span>
            <select
              value={settings.topK}
              onChange={(e) => onUpdate({ topK: Number(e.target.value) })}
            >
              {Array.from({ length: 10 }, (_, i) => i + 1).map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>Department</span>
            <select
              value={settings.department}
              onChange={(e) => onUpdate({ department: e.target.value })}
            >
              <option value="">All</option>
              {departments.map((dept) => (
                <option key={dept} value={dept}>
                  {dept}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>Answer</span>
            <select
              value={settings.mode}
              onChange={(e) => onUpdate({ mode: e.target.value as Mode })}
            >
              <option value="rag">LLM answer</option>
              <option value="citations_only">Excerpt only</option>
            </select>
          </label>
        </div>
      ) : null}
    </div>
  );
}

function Message({ message }: { message: ChatMessage }) {
  if (message.role === "user") {
    return (
      <div className="msg user">
        <div className="user-text">{message.content}</div>
      </div>
    );
  }

  const results = message.results || [];
  return (
    <div className={`msg assistant${message.error ? " failed" : ""}`}>
      {renderMessage(message.content, results)}
      <Sources results={results} />
    </div>
  );
}

function Sources({ results }: { results: ChunkResult[] }) {
  if (results.length === 0) return null;

  return (
    <details className="sources">
      <summary>
        {results.length} {results.length === 1 ? "source" : "sources"}
      </summary>
      <ol>
        {results.map((r) => (
          <li key={r.chunk_id}>
            <div className="source-head">
              <span className="source-title">
                {r.citation.title || "Untitled"}
              </span>
              <span className="muted">
                {r.citation.department} · level {r.citation.access_level} ·{" "}
                {r.score.toFixed(2)}
              </span>
            </div>
            <div className="source-path">{r.citation.source_path}</div>
            <p className="source-snippet">
              {r.text.slice(0, 320)}
              {r.text.length > 320 ? "..." : ""}
            </p>
          </li>
        ))}
      </ol>
    </details>
  );
}

function renderMessage(raw: string, results: ChunkResult[]) {
  const lines = normalizeBullets(raw).split(/\n+/);
  const blocks: JSX.Element[] = [];
  let listItems: string[] = [];
  let ordered = false;

  const pushList = () => {
    if (listItems.length === 0) return;
    const items = listItems.map((item, idx) => (
      <li key={idx}>{renderInline(item, results)}</li>
    ));
    blocks.push(
      ordered ? (
        <ol key={`list-${blocks.length}`}>{items}</ol>
      ) : (
        <ul key={`list-${blocks.length}`}>{items}</ul>
      )
    );
    listItems = [];
  };

  lines.forEach((line) => {
    const trimmed = line.trim();
    if (!trimmed) {
      pushList();
      return;
    }
    const bullet = trimmed.match(/^[*+-]\s+(.*)$/);
    const numbered = trimmed.match(/^\d+[.)]\s+(.*)$/);
    const item = bullet || numbered;
    if (item) {
      // a switch between bullets and numbers starts a new list.
      if (listItems.length > 0 && ordered !== Boolean(numbered)) pushList();
      ordered = Boolean(numbered);
      listItems.push(item[1]);
    } else {
      pushList();
      blocks.push(
        <p key={`p-${blocks.length}`}>{renderInline(trimmed, results)}</p>
      );
    }
  });

  pushList();

  return <div className="message-content">{blocks}</div>;
}

function normalizeBullets(text: string) {
  if (text.includes("\n*") || text.includes("\n-")) return text;
  if (text.includes(" * ")) {
    return text.replace(/\s\*\s+/g, "\n* ");
  }
  return text;
}

// turns **bold** and `code` into markup and the model's [chunk:<id>] tags
// into numbers that match the source list.
function renderInline(text: string, results: ChunkResult[]) {
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`|\s*\[chunk:\s*\d+\])/g);
  return parts.map((part, idx): ReactNode => {
    if (part.length > 2 && part.startsWith("`") && part.endsWith("`")) {
      return <code key={idx}>{part.slice(1, -1)}</code>;
    }
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={idx}>{part.slice(2, -2)}</strong>;
    }
    const cite = part.match(/^\s*\[chunk:\s*(\d+)\]$/);
    if (cite) {
      const pos = results.findIndex((r) => r.chunk_id === Number(cite[1]));
      if (pos < 0) return null;
      return (
        <sup key={idx} className="cite" title={results[pos].citation.title}>
          {pos + 1}
        </sup>
      );
    }
    return part;
  });
}
