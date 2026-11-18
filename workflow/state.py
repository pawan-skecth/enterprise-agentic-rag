from typing import TypedDict, Optional


class RAGState(TypedDict, total=False):
    """State that flows through the LangGraph nodes."""
    question: str           # Original user question
    query: str              # Possibly rewritten query for retrieval
    chunks: list[dict]      # Retrieved chunks from hybrid_search
    reranked_chunks: list[dict]  # After cross-encoder rerank
    grade: str              # "relevant" | "weak"
    answer: str             # Final LLM answer
    attempts: int           # Retrieval attempt counter