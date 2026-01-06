import requests
import streamlit as st
import os

DEFAULT_API = os.getenv("API_URL", "http://127.0.0.1:8000")
API_URL = st.sidebar.text_input("API URL", DEFAULT_API).rstrip("/")

st.title("Enterprise KB Chatbot")
st.caption("Permission-aware retrieval backed by Postgres (pgvector)")

# -----------------------
# Auth (JWT) in sidebar
# -----------------------
if "token" not in st.session_state:
    st.session_state.token = None

with st.sidebar.expander("Login", expanded=True):
    email = st.text_input("Email", value="internal@demo.com")
    password = st.text_input("Password", value="internal123", type="password")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Login"):
            try:
                r = requests.post(
                    f"{API_URL}/auth/login",
                    json={"email": email, "password": password},
                    timeout=20,
                )
                if r.status_code != 200:
                    st.error(f"Login failed: {r.text}")
                else:
                    st.session_state.token = r.json()["access_token"]
                    st.success("Logged in")
            except requests.RequestException as e:
                st.error(f"Login error: {e}")
    with col2:
        if st.button("Logout"):
            st.session_state.token = None
            st.info("Logged out")

if not st.session_state.token:
    st.warning("Please login to use chat.")
    st.stop()

# -----------------------
# Controls
# -----------------------
st.sidebar.subheader("Retrieval settings")
top_k = st.sidebar.slider("Top-K sources", 1, 10, 5)

# Simple optional filters (these match JSON metadata keys your ingest stores)
dept = st.sidebar.selectbox(
    "Department filter (optional)",
    ["(none)", "general", "hr", "it", "engineering", "research", "finance", "security"],
)

filters = {}
if dept != "(none)":
    filters["department"] = dept

# -----------------------
# Chat UI
# -----------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask a question about your company knowledge base...")

if prompt:
    # show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # call API
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    payload = {
        "query": prompt,
        "top_k": top_k,
        "filters": (filters or None),
    }

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            try:
                resp = requests.post(
                    f"{API_URL}/chat",
                    json=payload,
                    headers=headers,
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as e:
                st.error(f"Request failed: {e}")
                st.stop()

    answer = data.get("answer", "")
    st.markdown(answer if answer else "_No answer returned._")

    results = data.get("results", [])
    if results:
        with st.expander("Sources"):
            for i, r in enumerate(results, start=1):
                c = r.get("citation", {}) or {}
                st.markdown(
                    f"**{i}. {c.get('title', '(no title)')}**  \n"
                    f"- path: `{c.get('source_path', '')}`  \n"
                    f"- department: `{c.get('department', '')}`, access: `{c.get('access_level', '')}`  \n"
                    f"- chunk_id: `{r.get('chunk_id', '')}`, score: `{r.get('score', 0):.4f}`"
                )
                text = r.get("text", "")
                st.caption(text[:300] + ("..." if len(text) > 300 else ""))
                st.divider()
    else:
        st.caption("No sources returned.")


    # store assistant message in history
    st.session_state.messages.append({"role": "assistant", "content": answer})
