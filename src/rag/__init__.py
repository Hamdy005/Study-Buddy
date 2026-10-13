"""
Study-Buddy RAG package with integrated Corrective RAG (CRAG).
"""

from src.rag.config import Settings, settings
from src.rag.constants import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_HIGH_THRESHOLD,
    DEFAULT_LOW_THRESHOLD,
    DEFAULT_KEEP_THRESHOLD,
)
from src.rag.schemas import (
    Chunk,
    RouterOutput,
    EvaluatorOutput,
    GraphState,
    FinalAnswer,
    Source,
    Verdict,
    Cause,
    TutorQuery,
    TutorResponse,
)
from src.rag.reranker import Reranker, JinaReranker, MockReranker, get_reranker, rerank
from src.rag.crag_llm import LLM, LangChainLLM, MockLLM
from src.rag.crag_graph import build_graph, graph
from src.rag.crag import run_crag

__all__ = [
    "Settings",
    "settings",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_HIGH_THRESHOLD",
    "DEFAULT_LOW_THRESHOLD",
    "DEFAULT_KEEP_THRESHOLD",
    "Chunk",
    "RouterOutput",
    "EvaluatorOutput",
    "GraphState",
    "FinalAnswer",
    "Source",
    "Verdict",
    "Cause",
    "TutorQuery",
    "TutorResponse",
    "Reranker",
    "JinaReranker",
    "MockReranker",
    "get_reranker",
    "rerank",
    "LLM",
    "LangChainLLM",
    "MockLLM",
    "build_graph",
    "graph",
    "run_crag",
]
