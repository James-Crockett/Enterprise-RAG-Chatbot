import requests
import streamlit as st

DEFAULT_API = "http://127.0.0.1:8000"
API_URL = st.sidebar.text_input("API URL", DEFAULT_API).rstrip("/")

st.title("Enterprise Knowledge Base Chatbot (RAG)")
st.caption("Retrieval + citations-only answering (no LLM yet)")

top_k = st.sidebar.slider("Top-K sources", 1, 10, 5)
dept = st.sidebar.selectbox("Department filter", ["(none)", "general", "hr", "it", "engineering", "research"])
conf = st.sidebar.selectbox("Confidentiality filter", ["(none)", "public", "internal", "restricted"])

filters = {}
if dept != "(none)":
    filters["department"] = dept
if conf != "(none)":
    filters["confidentiality"] = conf

query = st.text_input("Ask a question")

def api_health() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except requests.RequestException:
        return False

ok = api_health()
if not ok:
    st.error(
        "API is not reachable. Start it in another terminal:\n\n"
        "uv run uvicorn apps.api.main:app --reload --port 8000 --host 127.0.0.1"
    )
    st.stop()

if st.button("Ask") and query.strip():
    payload = {"query": query, "top_k": top_k, "filters": (filters or None)}
    try:
        resp = requests.post(f"{API_URL}/chat", json=payload, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as e:
        st.error(f"Request failed: {e}")
        st.stop()

    data = resp.json()

    st.subheader("Answer")
    st.write(data.get("answer", ""))

    st.subheader("Sources")
    for r in data.get("results", []):
        c = r.get("citation", {})
        label = f"{c.get('title','(no title)')} • {c.get('source_path','')} • page={c.get('page')}"
        with st.expander(label):
            st.write(f"Score: {r.get('score', 0):.4f}")
            st.write(r.get("text", ""))
