from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, PayloadSchemaType,
)

import config


def _get_client() -> QdrantClient:
    return QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)


def ensure_collection(recreate: bool = False) -> None:
    """
    Create the Qdrant collection if it doesn't exist.
    If `recreate=True`, deletes any existing collection first (fresh start).
    """
    client = _get_client()
    name = config.QDRANT_COLLECTION

    if recreate and client.collection_exists(name):
        print(f"[qdrant] Dropping existing collection: {name}")
        client.delete_collection(name)

    if not client.collection_exists(name):
        print(f"[qdrant] Creating collection: {name} (dim={config.EMBEDDING_DIM}, cosine)")
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=config.EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )
        # Payload indexes for fast filtering
        for field in ["source", "page", "sheet"]:
            client.create_payload_index(
                collection_name=name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD
                if field in ("source", "sheet") else PayloadSchemaType.INTEGER,
            )
    else:
        print(f"[qdrant] Collection already exists: {name}")


def upsert_chunks(
    chunks: list[dict],
    vectors: list[list[float]],
    batch_size: int = 64,
    start_id: int = 0,
) -> int:
    """
    Upload chunks + vectors + metadata to Qdrant.

    Args:
        chunks:     list of chunk dicts (from chunking.py)
        vectors:    parallel list of 768-dim vectors
        batch_size: upload batch size
        start_id:   global offset — ensures IDs are unique ACROSS files

    Returns:
        Number of points uploaded
    """
    if len(chunks) != len(vectors):
        raise ValueError(
            f"Mismatch: {len(chunks)} chunks vs {len(vectors)} vectors"
        )

    client = _get_client()
    name = config.QDRANT_COLLECTION

    points = []
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        payload = {
            "text": chunk["text"],
            "source": chunk.get("source", ""),
            "page": chunk.get("page"),
            "sheet": chunk.get("sheet"),
            "chunk_id": chunk.get("chunk_id", 0),
        }
        payload = {k: v for k, v in payload.items() if v is not None}

        points.append(PointStruct(
            id=start_id + i,            # ← globally unique ID
            vector=vec,
            payload=payload,
        ))

    # Upload in batches
    uploaded = 0
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(collection_name=name, points=batch)
        uploaded += len(batch)
        print(f"[qdrant] Upserted {uploaded}/{len(points)} points")

    return uploaded