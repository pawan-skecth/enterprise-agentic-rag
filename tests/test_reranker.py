from retrieval.hybrid_search import hybrid_search
from retrieval.reranker import rerank


queries = [
    "How to monitor Kubernetes jobs with kubectl",
    "winning trades probability calculation",
    "Databricks Terraform provider for automation",
]

for q in queries:
    print("\n" + "=" * 78)
    print(f"Query: {q}")
    print("=" * 78)

    # Get top-20 from hybrid
    candidates = hybrid_search(q, top_k=20, dense_k=20, bm25_k=20)
    print(f"  Hybrid returned {len(candidates)} candidates")

    # Rerank to top-3
    top3 = rerank(q, candidates, top_k=3)

    for i, r in enumerate(top3, start=1):
        d = r["dense_rank"] if r["dense_rank"] else "—"
        b = r["bm25_rank"] if r["bm25_rank"] else "—"
        print(f"\n  [{i}] rerank={r['rerank_score']:+.4f} | RRF={r['rrf_score']:.4f} | "
              f"dense#{d} bm25#{b} | {r['source']}")
        print(f"      {r['text'][:140]}...")