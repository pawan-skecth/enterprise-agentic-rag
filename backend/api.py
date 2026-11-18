"""
FastAPI backend for the Enterprise RAG system.

Endpoints:
    GET  /health          → health check
    POST /ask             → run RAG pipeline, return answer + sources
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from workflow.rag_graph import build_rag_graph
import config


# ══════════════════════════════════════════════════
# APP + GRAPH (loaded once at startup)
# ══════════════════════════════════════════════════
app = FastAPI(
    title="Enterprise RAG API",
    description="Agentic RAG with hybrid search, reranking, and guardrails",
    version="1.0.0",
)

print("Loading RAG graph...")
_graph = build_rag_graph()
print("RAG graph ready.")


# ══════════════════════════════════════════════════
# REQUEST / RESPONSE MODELS
# ══════════════════════════════════════════════════
class AskRequest(BaseModel):
    question: str


class Source(BaseModel):
    source: str
    page: Optional[int] = None


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    grade: str
    attempts: int
    blocked: bool


# ══════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": config.GROQ_MODEL,
        "collection": config.QDRANT_COLLECTION,
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        state = _graph.invoke({
            "question": req.question,
            "query": req.question,
            "attempts": 0,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    # Extract unique sources
    seen = set()
    sources = []
    for c in state.get("reranked_chunks", []):
        key = (c.get("source"), c.get("page"))
        if key not in seen:
            seen.add(key)
            sources.append(Source(source=c.get("source", "?"), page=c.get("page")))

    grade = state.get("grade", "unknown")

    return AskResponse(
        answer=state.get("answer", ""),
        sources=sources,
        grade=grade,
        attempts=state.get("attempts", 0),
        blocked=(grade == "blocked"),
    )