"""
Option B — Auto-generate a RAGAS evaluation dataset from existing Qdrant chunks.
"""

import sys
import types

# ══════════════════════════════════════════════════
# COMPATIBILITY SHIM — must run BEFORE importing ragas
# ══════════════════════════════════════════════════
_VERTEXAI_PATH = "langchain_community.chat_models.vertexai"
if _VERTEXAI_PATH not in sys.modules:
    _stub = types.ModuleType(_VERTEXAI_PATH)
    _stub.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules[_VERTEXAI_PATH] = _stub
    try:
        import langchain_community.chat_models
        langchain_community.chat_models.vertexai = _stub
    except Exception:
        pass

# ══════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════
from langchain_core.documents import Document
from ragas.testset.synthesizers.generate import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

import config


# ══════════════════════════════════════════════════
# 1. CONNECT TO QDRANT AND FETCH CHUNKS
# ══════════════════════════════════════════════════
print("Fetching chunks from Qdrant...")
client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
points, _ = client.scroll(
    collection_name=config.QDRANT_COLLECTION,
    limit=200,
    with_payload=True,
    with_vectors=False,
)

chunks = []
for p in points:
    text = p.payload.get("text", "")
    if text:
        chunks.append(
            Document(
                page_content=text,
                metadata={
                    "source": p.payload.get("source", ""),
                    "page": p.payload.get("page"),
                },
            )
        )

print(f"Loaded {len(chunks)} chunks from Qdrant.")


# ══════════════════════════════════════════════════
# 2. SET UP THE GENERATOR (Legacy wrapper — works in your env)
# ══════════════════════════════════════════════════
# 2a. Groq LLM via LangChain
judge_llm = LangchainLLMWrapper(
    ChatGroq(
        api_key=config.GROQ_API_KEY,
        model="openai/gpt-oss-20b",
        temperature=0.0,
        timeout=180,
        max_tokens=2048,
    )
)

# 2b. Embeddings (local BGE)
judge_emb = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
)

# 2c. Build the generator
generator = TestsetGenerator(llm=judge_llm, embedding_model=judge_emb)


# ══════════════════════════════════════════════════
# 3. GENERATE THE TESTSET (using langchain docs)
# ══════════════════════════════════════════════════
print("Generating synthetic testset (this will take a few minutes)...")
testset = generator.generate_with_langchain_docs(
    documents=chunks,
    testset_size=10,
)

# ══════════════════════════════════════════════════
# 4. SAVE TO CSV
# ══════════════════════════════════════════════════
output_file = "data/synthetic_testset.csv"
testset.to_csv(output_file)
print(f"\n✅ Testset saved to {output_file}")
print("\nPreview:")
print(testset.to_pandas().head())