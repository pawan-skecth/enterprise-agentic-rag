"""
Option A — Manual RAGAS evaluation.

Measures Faithfulness + Answer Relevancy + Context Precision + Context Recall
on 5 hand-written questions, using the actual RAG pipeline to produce responses + contexts.
"""

import sys
import types

# ══════════════════════════════════════════════════
# COMPATIBILITY SHIM — must run BEFORE importing ragas
# Fixes: ModuleNotFoundError for langchain_community.chat_models.vertexai
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
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

import config
from workflow.rag_graph import build_rag_graph


# ══════════════════════════════════════════════════
# 1. MANUAL TEST SET — you write these
# ══════════════════════════════════════════════════
TEST_CASES = [
    {
        "user_input": "How do I monitor Kubernetes jobs?",
        "reference": "Use kubectl get jobs to check status and kubectl logs to view output.",
    },
    {
        "user_input": "What is the probability of winning trades in the journal?",
        "reference": "May 2026: 0.238, June 2026: 0.286, July 2026: 0.283, September 2026: 0.333.",
    },
    {
        "user_input": "What tools does Databricks offer for automation?",
        "reference": "Databricks CLI, SDKs, REST API, and Asset Bundles.",
    },
    {
        "user_input": "How do I view the logs of a Kubernetes pod?",
        "reference": "Use kubectl logs <pod-name>, and kubectl logs -f to follow.",
    },
    {
        "user_input": "What is the probability of winning trades in September 2026?",
        "reference": "0.3333 (33.33%).",
    },
]


# ══════════════════════════════════════════════════
# 2. RUN PIPELINE — captures response + contexts
# ══════════════════════════════════════════════════
def build_eval_rows(graph):
    rows = []
    for i, tc in enumerate(TEST_CASES, start=1):
        print(f"\n[{i}/{len(TEST_CASES)}] Running: {tc['user_input']}")
        state = graph.invoke({
            "question": tc["user_input"],
            "query": tc["user_input"],
            "attempts": 0,
        })
        rows.append({
            "user_input": tc["user_input"],
            "response": state.get("answer", ""),
            "retrieved_contexts": [c["text"] for c in state.get("reranked_chunks", [])],
            "reference": tc["reference"],
        })
    return rows


# ══════════════════════════════════════════════════
# 3. RAGAS EVALUATION — all 4 metrics
# ══════════════════════════════════════════════════
def run_eval(rows):
    # Judge LLM (Groq) — fast model for high-volume judge calls
    judge_llm = LangchainLLMWrapper(
        ChatGroq(
            api_key=config.GROQ_API_KEY,
            model="openai/gpt-oss-20b",   # Fast, free-tier model
            temperature=0.0,
            timeout=120,                  # Wait longer per call
            max_retries=2,
            max_tokens=2048,
        )
    )
    # Judge embeddings (local BGE)
    judge_emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    )

    # All 4 metrics
    metrics_list = [faithfulness, answer_relevancy, context_precision, context_recall]

    # Attach judge to metrics (metric-level override)
    for m in metrics_list:
        m.llm = judge_llm
    answer_relevancy.embeddings = judge_emb

    dataset = Dataset.from_list(rows)
    result = evaluate(
        dataset=dataset,
        metrics=metrics_list,
        llm=judge_llm,
        embeddings=judge_emb,
    )
    return result


# ══════════════════════════════════════════════════
# 4. MAIN
# ══════════════════════════════════════════════════
if __name__ == "__main__":
    print("Building RAG graph...")
    graph = build_rag_graph()

    print("\nRunning 5 test questions through the pipeline...")
    rows = build_eval_rows(graph)

    print("\n" + "=" * 70)
    print("RAGAS EVALUATION — 4 METRICS")
    print("=" * 70)
    result = run_eval(rows)
    print(result)