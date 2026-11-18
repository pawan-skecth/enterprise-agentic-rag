from retrieval.dense_search import dense_search


queries = [
    "What monitoring options are available for Kubernetes jobs?",
    "Show me the trade journal probability of winning trades",
    "What tools does Databricks provide for job automation?",
]

for q in queries:
    print("\n" + "=" * 70)
    print(f"Query: {q}")
    print("=" * 70)
    results = dense_search(q, top_k=3)
    for i, r in enumerate(results, start=1):
        print(f"\n  [{i}] score={r['score']:.4f} | {r['source']} | page={r['page']}")
        print(f"      {r['text'][:150]}...")