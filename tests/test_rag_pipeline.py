from retrieval.hybrid_search import hybrid_search
from retrieval.reranker import rerank
from agents.answer_generator import generate_answer


queries = [
    "How do I monitor Kubernetes jobs?",
    "What is the probability of winning trades in the journal?",
    "What tools does Databricks offer for automation?",
]

for q in queries:
    print("\n" + "=" * 78)
    print(f"QUERY: {q}")
    print("=" * 78)

    # Full RAG pipeline
    candidates = hybrid_search(q, top_k=20, dense_k=20, bm25_k=20)
    top5 = rerank(q, candidates, top_k=5)
    result = generate_answer(q, top5)

    print(f"\n📝 ANSWER:")
    print(result["answer"])

    print(f"\n📚 SOURCES ({len(result['sources'])}):")
    for s in result["sources"]:
        page = s["page"] if s["page"] is not None else "N/A"
        print(f"   - {s['source']} (page {page})")

    print(f"\n⚙️  Model: {result['model']} | Chunks used: {result['used_chunks']}")