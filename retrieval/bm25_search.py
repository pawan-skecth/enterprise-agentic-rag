import re
from functools import lru_cache

from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient

import config


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-_]", " ", text)
    return [t for t in text.split() if len(t) > 1]


@lru_cache(maxsize=1)
def _build_index():
    """
    Load all chunks from Qdrant and build a BM25 index.
    Cached so we only build once per process.
    """
    client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)

    print("[bm25] Building index from Qdrant...")
    points, _ = client.scroll(
        collection_name=config.QDRANT_COLLECTION,
        limit=10_000,
        with_payload=True,
        with_vectors=False,
    )

    docs = []
    for p in points:
        docs.append({
            "text": p.payload.get("text", ""),
            "source": p.payload.get("source", ""),
            "page": p.payload.get("page"),
        })

    tokenized = [_tokenize(d["text"]) for d in docs]
    index = BM25Okapi(tokenized)

    print(f"[bm25] Index built: {len(docs)} chunks")
    return index, docs


def bm25_search(
    query: str,
    top_k: int = 5,
    source_filter: str | None = None,
) -> list[dict]:
    """
    Keyword search using BM25 over all chunks in Qdrant.

    Args:
        query:         natural-language question (or keyword phrase)
        top_k:         number of chunks to return
        source_filter: optional filename to restrict results

    Returns:
        List of dicts (sorted by BM25 score, highest first):
            {"text": str, "source": str, "page": int|None, "score": float}
    """
    index, docs = _build_index()

    # Score all chunks
    scores = index.get_scores(_tokenize(query))

    # Pair with docs and optionally filter
    scored = [
        {**doc, "score": float(s)}
        for doc, s in zip(docs, scores)
        if s > 0  # skip zero-score chunks
        and (source_filter is None or doc["source"] == source_filter)
    ]

    # Sort descending
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]