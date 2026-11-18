import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
load_dotenv()

# ── Groq LLM ─────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")

# ── Embeddings ───────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
EMBEDDING_DIM   = int(os.getenv("EMBEDDING_DIM", "768"))

# ── Qdrant ───────────────────────────────
QDRANT_HOST       = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT       = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "enterprise_docs")

# ── Paths ────────────────────────────────
BASE_DIR           = Path(__file__).resolve().parent
RAW_DATA_DIR       = BASE_DIR / os.getenv("RAW_DATA_DIR", "data/raw")
PROCESSED_DATA_DIR = BASE_DIR / os.getenv("PROCESSED_DATA_DIR", "data/processed")

# ── Quick sanity check when run directly ─
if __name__ == "__main__":
    print("GROQ_API_KEY set:      ", bool(GROQ_API_KEY))
    print("GROQ_MODEL:            ", GROQ_MODEL)
    print("EMBEDDING_MODEL:       ", EMBEDDING_MODEL)
    print("EMBEDDING_DIM:         ", EMBEDDING_DIM)
    print("QDRANT:                ", f"{QDRANT_HOST}:{QDRANT_PORT}")
    print("QDRANT_COLLECTION:     ", QDRANT_COLLECTION)
    print("RAW_DATA_DIR:          ", RAW_DATA_DIR)
    print("PROCESSED_DATA_DIR:    ", PROCESSED_DATA_DIR)