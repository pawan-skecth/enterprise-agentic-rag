from sentence_transformers import CrossEncoder

import config


# Lazy singleton — model loads once per process
_reranker = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        print("[reranker] Loading model: BAAI/bge-reranker-base")
        _reranker = CrossEncoder("BAAI/bge-reranker-base")
        print("[reranker] Model loaded")
    return _reranker


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    Rerank candidate chunks using a cross-encoder.

    Args:
        query:      user question
        candidates: list of chunk dicts (from hybrid_search)
        top_k:      how many to keep after reranking

    Returns:
        List of dicts with rerank_score added, sorted descending:
            {
                "text": str,
                "source": str,
                "page": int | None,
                "rrf_score": float,      # original hybrid score
                "rerank_score": float,   # cross-encoder score
            }
    """
    if not candidates:
        return []

    model = _get_reranker()

    # Build (query, chunk_text) pairs
    pairs = [(query, c["text"]) for c in candidates]

    # Score all pairs in one batch
    scores = model.predict(pairs, show_progress_bar=False)

    # Attach scores
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)

    # Sort by rerank score descending
    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]