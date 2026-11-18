from retrieval.dense_search import dense_search
from retrieval.bm25_search import bm25_search


# RRF smoothing constant (standard value)
RRF_K = 60


def hybrid_search(
    query: str,
    top_k: int = 5,
    dense_k: int = 20,
    bm25_k: int = 20,
    source_filter: str | None = None,
) -> list[dict]:
    """
    Hybrid search using Reciprocal Rank Fusion of dense + BM25 results.

    Args:
        query:         natural-language question
        top_k:         final number of chunks to return
        dense_k:       how many candidates to fetch from dense
        bm25_k:        how many candidates to fetch from BM25
        source_filter: optional filename filter

    Returns:
        List of dicts sorted by RRF score (highest first):
            {
                "text": str,
                "source": str,
                "page": int | None,
                "rrf_score": float,
                "dense_rank": int | None,
                "bm25_rank":  int | None,
            }
    """
    # 1. Fetch from both retrievers
    dense_results = dense_search(query, top_k=dense_k, source_filter=source_filter)
    bm25_results = bm25_search(query, top_k=bm25_k, source_filter=source_filter)

    # 2. Build lookup: chunk key = (source, page, text) → unique identifier
    def key(r):
        return (r.get("source"), r.get("page"), r.get("text"))

    combined = {}

    # Add dense results with their ranks
    for rank, r in enumerate(dense_results, start=1):
        k = key(r)
        combined.setdefault(k, {
            "text": r["text"],
            "source": r["source"],
            "page": r["page"],
            "dense_rank": None,
            "bm25_rank": None,
            "rrf_score": 0.0,
        })
        combined[k]["dense_rank"] = rank
        combined[k]["rrf_score"] += 1.0 / (RRF_K + rank)

    # Add BM25 results with their ranks
    for rank, r in enumerate(bm25_results, start=1):
        k = key(r)
        combined.setdefault(k, {
            "text": r["text"],
            "source": r["source"],
            "page": r["page"],
            "dense_rank": None,
            "bm25_rank": None,
            "rrf_score": 0.0,
        })
        combined[k]["bm25_rank"] = rank
        combined[k]["rrf_score"] += 1.0 / (RRF_K + rank)

    # 3. Sort by RRF score descending, take top_k
    merged = sorted(combined.values(), key=lambda x: x["rrf_score"], reverse=True)
    return merged[:top_k]