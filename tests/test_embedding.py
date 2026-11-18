import numpy as np
from ingestion.embedding import embed_documents, embed_query


texts = [
    "The CVE Program is the global standard for vulnerability identification.",
    "Kubernetes jobs run batch workloads to completion.",
    "BGE embeddings convert text into 768-dimensional vectors.",
]

print("Embedding 3 documents...")
vectors = embed_documents(texts)
print(f"Got {len(vectors)} vectors")
print(f"Each vector dim: {len(vectors[0])}")

print("\nEmbedding a query...")
q = embed_query("What is the CVE Program?")
print(f"Query vector dim: {len(q)}")


def cos(a, b):
    a, b = np.array(a), np.array(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


print("\nCosine similarity (query vs docs):")
for i, text in enumerate(texts):
    print(f"  doc[{i}]: {cos(q, vectors[i]):.4f}  — {text[:60]}...")