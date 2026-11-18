from qdrant_client import QdrantClient
import config


client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)

info = client.get_collection(config.QDRANT_COLLECTION)
print(f"Points in Qdrant: {info.points_count}")
print(f"Vector dim:       {info.config.params.vectors.size}")
print(f"Distance:         {info.config.params.vectors.distance}")
print()

print("Sample payloads (first 5):")
points = client.scroll(
    collection_name=config.QDRANT_COLLECTION,
    limit=5,
    with_payload=True,
)[0]

for p in points:
    src = p.payload.get("source", "?")
    page = p.payload.get("page", "—")
    text_len = len(p.payload.get("text", ""))
    preview = p.payload.get("text", "")[:80].replace("\n", " ")
    print(f"  - {src:<35} page {str(page):<5} {text_len:>4} chars | {preview}...")