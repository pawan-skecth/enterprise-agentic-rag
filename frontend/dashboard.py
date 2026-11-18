"""
Streamlit chat UI for the Enterprise RAG system.

Calls the FastAPI backend at http://127.0.0.1:8000/ask
"""

import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

# ── Page config ────────────────────────────────────
st.set_page_config(
    page_title="Enterprise RAG Assistant",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Enterprise RAG Assistant")
st.caption("Hybrid search + reranking + agentic self-correction + guardrails")

# ── Health check ───────────────────────────────────
try:
    health = requests.get(f"{API_URL}/health", timeout=5).json()
    st.sidebar.success(f"✅ API online")
    st.sidebar.write(f"**Model:** `{health['model']}`")
    st.sidebar.write(f"**Collection:** `{health['collection']}`")
except Exception:
    st.sidebar.error("❌ API not reachable")
    st.stop()

# ── Chat history ───────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption(
                "📚 Sources: "
                + ", ".join(
                    f"{s['source']} (p.{s['page']})" if s.get("page") else s["source"]
                    for s in msg["sources"]
                )
            )

# ── Input ──────────────────────────────────────────
prompt = st.chat_input("Ask a question about your documents...")

if prompt:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    f"{API_URL}/ask",
                    json={"question": prompt},
                    timeout=120,
                ).json()
                answer = resp.get("answer", "")
                sources = resp.get("sources", [])
                grade = resp.get("grade", "")
                blocked = resp.get("blocked", False)
            except Exception as e:
                answer = f"❌ API error: {e}"
                sources = []
                grade = "error"
                blocked = False

        # Show answer
        if blocked:
            st.warning(answer)
        else:
            st.markdown(answer)
            if sources:
                st.caption(
                    "📚 Sources: "
                    + ", ".join(
                        f"{s['source']} (p.{s['page']})" if s.get("page") else s["source"]
                        for s in sources
                    )
                )
                st.caption(f"Grade: `{grade}` | Attempts: `{resp.get('attempts', 0)}`")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })