"""
LangGraph assembly for the agentic RAG workflow.

Graph topology:
    START → guardrail ─┬─ (blocked) ──────────────────────→ END
                       └─ (safe) ──→ retrieve → grade ─┬─ (relevant) ──────────────→ generate → END
                                                       └─ (weak, attempts < MAX) ──→ rewrite → retrieve
                                                       └─ (weak, attempts >= MAX) ─→ generate → END
"""

from langgraph.graph import StateGraph, START, END

from workflow.state import RAGState
from workflow.nodes import (
    guardrail_node,
    retrieve_node,
    grade_node,
    rewrite_node,
    generate_node,
    MAX_ATTEMPTS,
)


# ══════════════════════════════════════════════════
# ROUTING FUNCTION — decides after guardrail
# ══════════════════════════════════════════════════
def route_after_guardrail(state: RAGState) -> str:
    """
    Conditional edge after guardrail:
      - blocked → END (query was unsafe)
      - safe    → retrieve (continue normal pipeline)
    """
    grade = state.get("grade", "")
    if grade == "blocked":
        print("[route] guardrail=blocked → END")
        return "end"
    print("[route] guardrail=safe → retrieve")
    return "retrieve"


# ══════════════════════════════════════════════════
# ROUTING FUNCTION — decides after grade
# ══════════════════════════════════════════════════
def route_after_grade(state: RAGState) -> str:
    """
    Conditional edge after grading:
      - relevant             → generate
      - weak + attempts left → rewrite
      - weak + no attempts   → generate (fallback)
    """
    grade = state.get("grade", "weak")
    attempts = state.get("attempts", 0)

    if grade == "relevant":
        print(f"[route] grade=relevant → generate")
        return "generate"

    if attempts >= MAX_ATTEMPTS:
        print(f"[route] grade=weak, attempts={attempts}/{MAX_ATTEMPTS} → generate (fallback)")
        return "generate"

    print(f"[route] grade=weak, attempts={attempts}/{MAX_ATTEMPTS} → rewrite")
    return "rewrite"


# ══════════════════════════════════════════════════
# BUILD THE GRAPH
# ══════════════════════════════════════════════════
def build_rag_graph():
    """Assemble and compile the LangGraph."""
    graph = StateGraph(RAGState)

    # Add nodes
    graph.add_node("guardrail", guardrail_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("generate", generate_node)

    # Entry point → guardrail first
    graph.add_edge(START, "guardrail")

    # guardrail → conditional (END if blocked, else retrieve)
    graph.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {
            "end": END,
            "retrieve": "retrieve",
        },
    )

    # retrieve → grade (always)
    graph.add_edge("retrieve", "grade")

    # grade → conditional (generate OR rewrite)
    graph.add_conditional_edges(
        "grade",
        route_after_grade,
        {
            "generate": "generate",
            "rewrite": "rewrite",
        },
    )

    # rewrite → retrieve (loop back)
    graph.add_edge("rewrite", "retrieve")

    # generate → END
    graph.add_edge("generate", END)

    return graph.compile()