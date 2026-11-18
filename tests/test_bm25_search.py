from retrieval.bm25_search import bm25_search


queries = [
    "kubectl get jobs",                            # exact command keyword
    "winning trades probability",                  # exact phrase from Excel
    "Databricks Terraform provider",               # exact tech term
    "2R Reached Then Breakeven",                   # column header from trade journal
]

for q in queries:
    print("\n" + "=" * 70)
    print(f"Query: {q}")
    print("=" * 70)
    results = bm25_search(q, top_k=3)
    for i, r in enumerate(results, start=1):
        print(f"\n  [{i}] score={r['score']:.3f} | {r['source']} | page={r['page']}")
        print(f"      {r['text'][:150]}...")