"""
LangGraph nodes for the RAG workflow.
Each node is a function that takes RAGState and returns a partial update dict.
"""

from groq import Groq

import config
from retrieval.hybrid_search import hybrid_search
from retrieval.reranker import rerank
from agents.answer_generator import generate_answer
from workflow.guardrail import is_query_safe
from workflow.state import RAGState


# ── Retrieval budget ───────────────────────────────
DENSE_K = 20
BM25_K = 20
RERANK_TOP_K = 5
MAX_ATTEMPTS = 2


# ── Groq client ─────────────────────────────────────
_llm_client = Groq(api_key=config.GROQ_API_KEY)


# ── Helper: safely extract text from reasoning-model output ────
def _extract_text(response) -> str:
    """Return content if present, else fall back to reasoning."""
    msg = response.choices[0].message
    content = (msg.content or "").strip()
    if content:
        return content
    reasoning = getattr(msg, "reasoning", None) or ""
    reasoning = reasoning.strip()
    if not reasoning:
        return ""
    lines = [l.strip() for l in reasoning.splitlines() if l.strip()]
    return lines[-1] if lines else ""


# ══════════════════════════════════════════════════
# NODE 0 — GUARDRAIL (input safety check)
# ══════════════════════════════════════════════════
def guardrail_node(state: RAGState) -> dict:
    """Check if the user query is safe before proceeding."""
    question = state.get("question", "")
    if not is_query_safe(question):
        print("[guardrail] Query blocked as unsafe.")
        return {
            "answer": "I cannot process this request as it violates safety policies.",
            "grade": "blocked",
        }
    print("[guardrail] Query passed safety check.")
    return {}


# ══════════════════════════════════════════════════
# NODE 1 — RETRIEVE + RERANK
# ══════════════════════════════════════════════════
def retrieve_node(state: RAGState) -> dict:
    """Hybrid search + cross-encoder rerank."""
    query = state.get("query") or state.get("question", "")
    attempts = state.get("attempts", 0) + 1

    print(f"[retrieve] Attempt {attempts} | query='{query}'")

    candidates = hybrid_search(
        query=query,
        top_k=DENSE_K,
        dense_k=DENSE_K,
        bm25_k=BM25_K,
    )
    print(f"[retrieve]   hybrid returned {len(candidates)} candidates")

    top_chunks = rerank(query=query, candidates=candidates, top_k=RERANK_TOP_K)
    print(f"[retrieve]   reranked to {len(top_chunks)} chunks")

    return {
        "chunks": candidates,
        "reranked_chunks": top_chunks,
        "attempts": attempts,
    }


# ══════════════════════════════════════════════════
# NODE 2 — GRADE
# ══════════════════════════════════════════════════
_GRADER_PROMPT = """You are a relevance grader for a RAG system.

Given a QUESTION and retrieved CONTEXT chunks, decide whether the chunks can plausibly answer the question.

The bar is "can the question be answered using these chunks?" — NOT "do they contain the exact answer verbatim". Paraphrased or partial info counts as relevant.

Respond with EXACTLY one word:
- "relevant"
- "weak"
"""


def grade_node(state: RAGState) -> dict:
    """Grade retrieval quality → 'relevant' or 'weak'."""
    question = state.get("question", "")
    chunks = state.get("reranked_chunks", [])

    if not chunks:
        print("[grade] No chunks → weak")
        return {"grade": "weak"}

    context_snippets = []
    for i, c in enumerate(chunks[:5], start=1):
        context_snippets.append(f"[Chunk {i}] {c.get('text', '')[:400]}")
    context = "\n\n".join(context_snippets)

    user_prompt = f"""QUESTION:
{question}

CONTEXT:
{context}

Can the QUESTION be answered using the CONTEXT? Respond with one word: "relevant" or "weak"."""

    try:
        response = _llm_client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[
                {"role": "system", "content": _GRADER_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=300,
            temperature=0.0,
        )
        raw = _extract_text(response)
        verdict = "relevant" if "relevant" in raw.lower() else "weak"
        print(f"[grade] Raw='{raw[:60]}' → Verdict: {verdict}")
        return {"grade": verdict}
    except Exception as e:
        print(f"[grade] Error: {e} — defaulting to 'weak'")
        return {"grade": "weak"}


# ══════════════════════════════════════════════════
# NODE 3 — REWRITE QUERY
# ══════════════════════════════════════════════════
_REWRITER_PROMPT = """You are a query rewriter for a RAG search system.

Rewrite the QUESTION into a better search query for a vector + keyword search engine.

Rules:
- Keep it short (5-15 words).
- Use specific keywords likely to appear in documents.
- Do NOT add new facts or change the intent.
- Output ONLY the rewritten query, nothing else.
"""


def rewrite_node(state: RAGState) -> dict:
    """Rewrite the query when retrieval grade is weak."""
    original = state.get("question", "")
    current = state.get("query", original)

    user_prompt = f"""QUESTION: {original}
Previous query: {current}

Rewrite for better retrieval:"""

    try:
        response = _llm_client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[
                {"role": "system", "content": _REWRITER_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=300,
            temperature=0.3,
        )
        new_query = _extract_text(response)
        new_query = new_query.strip().strip('"').strip("'")

        if not new_query or len(new_query) < 3:
            print(f"[rewrite] Empty rewrite → keeping original")
            return {"query": current}

        print(f"[rewrite] New query: '{new_query}'")
        return {"query": new_query}
    except Exception as e:
        print(f"[rewrite] Error: {e} — keeping original")
        return {"query": current}


# ══════════════════════════════════════════════════
# NODE 4 — GENERATE FINAL ANSWER
# ══════════════════════════════════════════════════
def generate_node(state: RAGState) -> dict:
    """Generate final grounded answer from reranked chunks."""
    question = state.get("question", "")
    chunks = state.get("reranked_chunks", [])

    print(f"[generate] Generating answer from {len(chunks)} chunks")

    result = generate_answer(question, chunks)
    return {"answer": result["answer"]}