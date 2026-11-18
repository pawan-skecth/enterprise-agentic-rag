from qdrant_client import QdrantClient

from ingestion.embedding import embed_query
import config


def _get_client() -> QdrantClient:
    return QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)


def dense_search(
    query: str,
    top_k: int = 5,
    source_filter: str | None = None,
) -> list[dict]:
    """
    Semantic search using BGE query embedding + Qdrant cosine similarity.

    Args:
        query:         natural-language question
        top_k:         number of chunks to return
        source_filter: optional filename to restrict search (e.g. "sample.pdf")

    Returns:
        List of dicts (sorted by score, highest first):
            {"text": str, "source": str, "page": int|None, "score": float}
    """
    client = _get_client()

    # 1. Embed the query (with BGE prefix)
    query_vector = embed_query(query)

    # 2. Optional metadata filter (e.g. only search resume)
    query_filter = None
    if source_filter:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        query_filter = Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=source_filter))]
        )

    # 3. Search Qdrant
    results = client.search(
        collection_name=config.QDRANT_COLLECTION,
        query_vector=query_vector,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
    )

    # 4. Format results
    return [
        {
            "text": hit.payload.get("text", ""),
            "source": hit.payload.get("source", ""),
            "page": hit.payload.get("page"),
            "score": hit.score,
        }
        for hit in results
    ]