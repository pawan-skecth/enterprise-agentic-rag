"""
Streamlit chat UI for the Enterprise RAG system.
Runs the RAG graph directly (no FastAPI dependency).
"""

import sys
from pathlib import Path

# Add project root to sys.path so 'workflow', 'retrieval', 'agents' are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from workflow.rag_graph import build_rag_graph
import config

# ── Page config ────────────────────────────────────
st.set_page_config(
    page_title="Enterprise RAG Assistant",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Enterprise RAG Assistant")
st.caption("Hybrid search + reranking + agentic self-correction + guardrails")


# ── Source dedup helper (handles None / int / str page, whitespace, casing) ──
def dedupe_sources(sources):
    """Return sources with duplicate (source, page) removed. Preserves order."""
    if not sources:
        return []

    def norm(s):
        src = (s.get("source") or "?").strip().lower()
        page = s.get("page")
        # Normalize page: None / "" → None; numeric str → int
        if page is None or page == "":
            page = None
        else:
            try:
                page = int(page)
            except (ValueError, TypeError):
                page = str(page).strip() or None
        return src, page

    seen = set()
    unique = []
    for s in sources:
        key = norm(s)
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique


# ── Load graph once (cached) ───────────────────────
@st.cache_resource
def get_graph():
    return build_rag_graph()

graph = get_graph()

# ── Sidebar ────────────────────────────────────────
st.sidebar.success("✅ RAG graph loaded")
st.sidebar.write(f"**Model:** `{config.GROQ_MODEL}`")
st.sidebar.write(f"**Collection:** `{config.QDRANT_COLLECTION}`")

# ── Chat history ───────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            unique = dedupe_sources(msg["sources"])
            st.caption(
                "📚 Sources: "
                + ", ".join(
                    f"{s['source']} (p.{s['page']})" if s.get("page") else s["source"]
                    for s in unique
                )
            )

# ── Input ──────────────────────────────────────────
prompt = st.chat_input("Ask a question about your documents...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                state = graph.invoke({
                    "question": prompt,
                    "query": prompt,
                    "attempts": 0,
                })
                answer = state.get("answer", "")
                sources = [
                    {"source": c.get("source", "?"), "page": c.get("page")}
                    for c in state.get("reranked_chunks", [])
                ]
                grade = state.get("grade", "")
                blocked = (grade == "blocked")
            except Exception as e:
                answer = f"❌ Error: {e}"
                sources = []
                grade = "error"
                blocked = False

        if blocked:
            st.warning(answer)
        else:
            st.markdown(answer)
            unique_sources = dedupe_sources(sources)
            if unique_sources:
                st.caption(
                    "📚 Sources: "
                    + ", ".join(
                        f"{s['source']} (p.{s['page']})" if s.get("page") else s["source"]
                        for s in unique_sources
                    )
                )
                st.caption(f"Grade: `{grade}` | Attempts: `{state.get('attempts', 0)}`")

    # ⚠️ Store DEDUPED sources in session state so history re-render is clean too
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": dedupe_sources(sources),
    })