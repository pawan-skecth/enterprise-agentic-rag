"""
Test the full LangGraph agentic RAG pipeline.
Run: python -m tests.test_agentic_rag
"""

from workflow.rag_graph import build_rag_graph


queries = [
    "How do I monitor Kubernetes jobs?",                        # should be relevant on first try
    "What is the probability of winning trades in the journal?", # should be relevant
    "What is the capital of France?",                            # should trigger rewrite loop
]

# Build graph ONCE
print("Building RAG graph...")
graph = build_rag_graph()
print("Graph compiled.\n")

for q in queries:
    print("\n" + "=" * 78)
    print(f"QUESTION: {q}")
    print("=" * 78)

    initial_state = {
        "question": q,
        "query": q,
        "attempts": 0,
    }

    final_state = graph.invoke(initial_state)

    print("\n" + "-" * 78)
    print("📝 FINAL ANSWER:")
    print(final_state.get("answer", "<no answer>"))
    print(f"\n📊 Attempts: {final_state.get('attempts')} | Final grade: {final_state.get('grade')}")
    print(f"📚 Chunks used: {len(final_state.get('reranked_chunks', []))}")