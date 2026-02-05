import { useMemo, useState } from "react";
import { chat, login } from "./api";
import type { ChatMessage, ChunkResult } from "./types";

const DEPARTMENTS = [
  "(none)",
  "general",
  "hr",
  "it",
  "engineering",
  "research",
  "finance",
  "security",
];

function resolveDefaultApi() {
  const fromEnv = import.meta.env.VITE_API_URL;
  if (fromEnv) return fromEnv;
  return "/api";
}

const DEFAULT_API = resolveDefaultApi();
const STORAGE_KEY = "rag_kb_settings";

type Settings = {
  apiUrl: string;
  topK: number;
  department: string;
  mode: "rag" | "citations_only";
  email: string;
};

const defaultSettings: Settings = {
  apiUrl: DEFAULT_API,
  topK: 5,
  department: "(none)",
  mode: "rag",
  email: "internal@demo.com",
};

function loadSettings(): Settings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return defaultSettings;
    const merged = { ...defaultSettings, ...JSON.parse(raw) } as Settings;
    // If the UI is running in the browser, avoid docker-only hostnames.
    if (
      typeof window !== "undefined" &&
      merged.apiUrl.startsWith("http://api:")
    ) {
      merged.apiUrl = "/api";
    }
    return merged;
  } catch {
    return defaultSettings;
  }
}

function persistSettings(settings: Settings) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
}

export default function App() {
  const [settings, setSettings] = useState<Settings>(loadSettings);
  const [password, setPassword] = useState("internal123");
  const [token, setToken] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [prompt, setPrompt] = useState("");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [debug, setDebug] = useState<string | null>(null);

  const filters = useMemo(() => {
    if (settings.department === "(none)") return undefined;
    return { department: settings.department };
  }, [settings.department]);

  const onUpdate = (patch: Partial<Settings>) => {
    setSettings((prev) => {
      const next = { ...prev, ...patch };
      persistSettings(next);
      return next;
    });
  };

  const doLogin = async () => {
    setStatus(null);
    setDebug(null);
    setBusy(true);
    try {
      const result = await login(settings.apiUrl, settings.email, password);
      setToken(result.access_token);
      setStatus("Logged in successfully.");
    } catch (err) {
      const message = err instanceof Error ? err.message : "Login failed.";
      setStatus(message);
      setDebug(
        `Login failed at ${new Date().toLocaleTimeString()} using ${settings.apiUrl}.`
      );
    } finally {
      setBusy(false);
    }
  };

  const doLogout = () => {
    setToken(null);
    setMessages([]);
    setStatus("Logged out.");
  };

  const sendPrompt = async () => {
    if (!prompt.trim() || !token) return;

    const userMessage: ChatMessage = { role: "user", content: prompt.trim() };
    setMessages((prev) => [...prev, userMessage]);
    setPrompt("");
    setBusy(true);
    setStatus(null);
    setDebug(null);

    try {
      const response = await chat(settings.apiUrl, token, {
        query: userMessage.content,
        top_k: settings.topK,
        filters,
        mode: settings.mode,
      });

      const assistant: ChatMessage = {
        role: "assistant",
        content: response.answer || "(No answer returned.)",
        results: response.results || [],
      };

      setMessages((prev) => [...prev, assistant]);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Request failed.";
      setStatus(message);
      setDebug(
        `Chat request failed at ${new Date().toLocaleTimeString()} using ${settings.apiUrl}.`
      );
    } finally {
      setBusy(false);
    }
  };

  const isAuthed = Boolean(token);

  if (!isAuthed) {
    return (
      <div className="auth-shell">
        <div className="auth-card">
          <div>
            <p className="eyebrow">Enterprise KB</p>
            <h1>Sign in to access the knowledge base.</h1>
            <p className="subtext">
              Authenticate first, then continue to the chat workspace.
            </p>
          </div>

          <label className="field">
            <span>API URL</span>
            <input
              value={settings.apiUrl}
              onChange={(e) => onUpdate({ apiUrl: e.target.value })}
              placeholder="/api"
            />
          </label>
          <p className="helper-text">
            This is the FastAPI backend the UI will call for login and chat. In
            Docker, the web container proxies `/api` to the API service.
          </p>

          <label className="field">
            <span>Email</span>
            <input
              value={settings.email}
              onChange={(e) => onUpdate({ email: e.target.value })}
              placeholder="internal@demo.com"
            />
          </label>
          <label className="field">
            <span>Password</span>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </label>

          <div className="button-row">
            <button onClick={doLogin} disabled={busy}>
              {busy ? "Signing in..." : "Login"}
            </button>
          </div>

          {status ? <p className="status-text">{status}</p> : null}
          {debug ? <p className="debug-text">{debug}</p> : null}
        </div>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <p className="eyebrow">Enterprise KB</p>
          <h1>Permission-aware knowledge search.</h1>
          <p className="subtext">
            Authenticated RAG chat backed by pgvector and Ollama.
          </p>
        </div>
        <div className="status-card">
          <div className="status-label">API</div>
          <div className="status-value">{settings.apiUrl}</div>
          <div className={`status-pill ${token ? "live" : "idle"}`}>
            {token ? "Authenticated" : "Not logged in"}
          </div>
        </div>
      </header>

      <main className="app-main">
        <section className="panel settings">
          <h2>Retrieval controls</h2>
          <p className="panel-lede">Tune search behavior for this session.</p>

          <label className="field">
            <span>API URL</span>
            <input
              value={settings.apiUrl}
              onChange={(e) => onUpdate({ apiUrl: e.target.value })}
              placeholder="/api"
            />
          </label>
          <p className="helper-text">
            Backend URL for login and chat requests.
          </p>

          <label className="field">
            <span>Top-K sources</span>
            <input
              type="range"
              min={1}
              max={10}
              value={settings.topK}
              onChange={(e) =>
                onUpdate({ topK: Number(e.target.value) || 5 })
              }
            />
            <div className="range-value">{settings.topK}</div>
          </label>

          <label className="field">
            <span>Department filter</span>
            <select
              value={settings.department}
              onChange={(e) => onUpdate({ department: e.target.value })}
            >
              {DEPARTMENTS.map((dept) => (
                <option key={dept} value={dept}>
                  {dept}
                </option>
              ))}
            </select>
          </label>

          <label className="field">
            <span>Mode</span>
            <select
              value={settings.mode}
              onChange={(e) =>
                onUpdate({ mode: e.target.value as Settings["mode"] })
              }
            >
              <option value="rag">rag</option>
              <option value="citations_only">citations_only</option>
            </select>
          </label>

          <div className="divider" />

          <h3>Session</h3>
          <div className="button-row">
            <button onClick={doLogout} className="ghost">
              Logout
            </button>
          </div>
          {status ? <p className="status-text">{status}</p> : null}
          {debug ? <p className="debug-text">{debug}</p> : null}
        </section>

        <section className="panel chat">
          <div className="chat-scroll">
            {!isAuthed ? (
              <div className="empty-state">
                <h3>Sign in to continue</h3>
                <p>
                  You need to authenticate before you can query the knowledge
                  base.
                </p>
              </div>
            ) : messages.length === 0 ? (
              <div className="empty-state">
                <h3>Start a new question</h3>
                <p>
                  Ask about policies, onboarding, or incident response. We will
                  only answer using retrieved knowledge base context.
                </p>
              </div>
            ) : (
              messages.map((m, idx) => (
                <div key={`${m.role}-${idx}`} className={`msg ${m.role}`}>
                  <div className="bubble">
                    {renderMessage(m.content)}
                    {m.role === "assistant" ? (
                      <Sources results={m.results || []} />
                    ) : null}
                  </div>
                </div>
              ))
            )}
          </div>

          <div className="composer">
            <textarea
              placeholder="Ask a question about your company knowledge base..."
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  sendPrompt();
                }
              }}
              disabled={!isAuthed || busy}
            />
            <div className="composer-actions">
              <span className="hint">
                {isAuthed ? "Press Enter to send" : "Login to chat"}
              </span>
              <button onClick={sendPrompt} disabled={!isAuthed || busy}>
                {busy ? "Searching..." : "Send"}
              </button>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

function Sources({ results }: { results: ChunkResult[] }) {
  if (!results || results.length === 0) {
    return <p className="sources-empty">No sources returned.</p>;
  }

  return (
    <details className="sources">
      <summary>Sources ({results.length})</summary>
      <div className="sources-list">
        {results.map((r, idx) => (
          <div key={r.chunk_id} className="source-card">
            <div className="source-title">
              {idx + 1}. {r.citation.title || "(no title)"}
            </div>
            <div className="source-meta">
              <span>path: {r.citation.source_path}</span>
              <span>
                dept: {r.citation.department} | access: {r.citation.access_level}
              </span>
              <span>
                chunk_id: {r.chunk_id} | score: {r.score.toFixed(4)}
              </span>
            </div>
            <p className="source-snippet">
              {r.text.slice(0, 320)}{r.text.length > 320 ? "..." : ""}
            </p>
          </div>
        ))}
      </div>
    </details>
  );
}

function renderMessage(raw: string) {
  const normalized = normalizeBullets(raw);
  const lines = normalized.split(/\n+/);
  const blocks: JSX.Element[] = [];
  let listItems: string[] = [];

  const pushList = () => {
    if (listItems.length === 0) return;
    blocks.push(
      <ul key={`list-${blocks.length}`}>
        {listItems.map((item, idx) => (
          <li key={`li-${blocks.length}-${idx}`}>{renderInline(item)}</li>
        ))}
      </ul>
    );
    listItems = [];
  };

  lines.forEach((line) => {
    const trimmed = line.trim();
    if (!trimmed) {
      pushList();
      return;
    }
    if (trimmed.startsWith("* ") || trimmed.startsWith("- ")) {
      listItems.push(trimmed.slice(2));
    } else {
      pushList();
      blocks.push(
        <p key={`p-${blocks.length}`}>{renderInline(trimmed)}</p>
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

function renderInline(text: string) {
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return parts.map((part, idx) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={`b-${idx}`}>{part.slice(2, -2)}</strong>;
    }
    return <span key={`t-${idx}`}>{part}</span>;
  });
}
