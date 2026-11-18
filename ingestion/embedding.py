import threading
from sentence_transformers import SentenceTransformer
import config


# BGE recommends this prefix for QUERY embeddings only (not documents)
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Lazy singleton — model loads once per process
_model = None
_model_lock = threading.Lock()


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                print(f"[embedding] Loading model: {config.EMBEDDING_MODEL}")
                _model = SentenceTransformer(config.EMBEDDING_MODEL)
                print(f"[embedding] Model loaded (dim={config.EMBEDDING_DIM})")
    return _model


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed document chunks (no prefix — BGE expects raw text for docs)."""
    if not texts:
        return []
    model = _get_model()
    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=len(texts) > 32,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vectors.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a search query (with BGE query prefix)."""
    model = _get_model()
    vector = model.encode(
        _BGE_QUERY_PREFIX + query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vector.tolist()