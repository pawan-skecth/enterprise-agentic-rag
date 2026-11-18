from ingestion.pdf_loader import load_pdf
from ingestion.text_cleaner import clean_blocks
from ingestion.chunking import chunk_recursive
from ingestion.embedding import embed_documents
from retrieval.qdrant_upsert import ensure_collection, upsert_chunks
from qdrant_client import QdrantClient
import config


# 1. Prepare data
print("Loading sample.pdf...")
blocks = clean_blocks(load_pdf("data/raw/sample.pdf", extract_images=False))
chunks = chunk_recursive(blocks, chunk_size=500, chunk_overlap=100)
print(f"  → {len(chunks)} chunks")

# 2. Embed
print("Embedding chunks...")
vectors = embed_documents([c["text"] for c in chunks])
print(f"  → {len(vectors)} vectors (dim={len(vectors[0])})")

# 3. Create fresh collection
ensure_collection(recreate=True)

# 4. Upsert
uploaded = upsert_chunks(chunks, vectors)
print(f"\n✅ Uploaded {uploaded} points to Qdrant")

# 5. Verify — count points in the collection
client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
info = client.get_collection(config.QDRANT_COLLECTION)
print(f"\nCollection '{config.QDRANT_COLLECTION}':")
print(f"  Points count: {info.points_count}")
print(f"  Vector dim:   {info.config.params.vectors.size}")
print(f"  Distance:     {info.config.params.vectors.distance}")