from retrieval.hybrid_search import hybrid_search


queries = [
    "How to monitor Kubernetes jobs with kubectl",       # mixed: semantic + keyword
    "winning trades probability calculation",             # Excel domain
    "Databricks Terraform provider for automation",      # exact tech term
]

for q in queries:
    print("\n" + "=" * 78)
    print(f"Query: {q}")
    print("=" * 78)
    results = hybrid_search(q, top_k=3)

    for i, r in enumerate(results, start=1):
        d = f"dense#{r['dense_rank']}" if r["dense_rank"] else "dense—"
        b = f"bm25#{r['bm25_rank']}" if r["bm25_rank"] else "bm25—"
        print(f"\n  [{i}] RRF={r['rrf_score']:.4f} | {d:>8} | {b:>7} | {r['source']}")
        print(f"      {r['text'][:140]}...")