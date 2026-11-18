from groq import Groq

import config


# Initialize Groq client once
_client = Groq(api_key=config.GROQ_API_KEY)


# System prompt — enforces grounding + citations
_SYSTEM_PROMPT = """You are a helpful enterprise knowledge assistant.

RULES:
1. Answer ONLY using the provided CONTEXT chunks.
2. Do NOT use outside knowledge or make assumptions.
3. If the context does not contain enough information, say: "I don't have enough information in the provided documents to answer this question."
4. Always cite your sources at the end using this format: [source: <filename>, page: <page or N/A>]
5. Be concise and factual. Do not invent details.
"""


def _format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a readable context block."""
    lines = []
    for i, c in enumerate(chunks, start=1):
        src = c.get("source", "unknown")
        page = c.get("page", "N/A")
        lines.append(f"--- Chunk {i} (source: {src}, page: {page}) ---")
        lines.append(c["text"])
        lines.append("")
    return "\n".join(lines)


def generate_answer(
    query: str,
    chunks: list[dict],
    max_tokens: int = 800,
    temperature: float = 0.1,
) -> dict:
    """
    Generate a grounded answer using Groq LLM + retrieved chunks.

    Args:
        query:       user question
        chunks:      top-K chunks from reranker (each with text, source, page)
        max_tokens:  LLM output limit
        temperature: low = deterministic, factual

    Returns:
        {
            "answer": str,           # the LLM's answer
            "sources": list[dict],   # unique sources cited
            "model": str,            # model used
            "used_chunks": int,      # how many chunks fed in
        }
    """
    if not chunks:
        return {
            "answer": "I don't have enough information in the provided documents to answer this question.",
            "sources": [],
            "model": config.GROQ_MODEL,
            "used_chunks": 0,
        }

    context = _format_context(chunks)
    user_prompt = f"""CONTEXT:
{context}

QUESTION:
{query}

Answer using ONLY the context above. Cite sources at the end."""

    response = _client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )

    answer_text = response.choices[0].message.content.strip()

    # Extract unique sources
    seen = set()
    sources = []
    for c in chunks:
        key = (c.get("source"), c.get("page"))
        if key not in seen:
            seen.add(key)
            sources.append({
                "source": c.get("source", "unknown"),
                "page": c.get("page"),
            })

    return {
        "answer": answer_text,
        "sources": sources,
        "model": config.GROQ_MODEL,
        "used_chunks": len(chunks),
    }