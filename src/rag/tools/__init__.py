"""
Retriever tools registry and exports for CRAG inside src/rag.
"""

from typing import Any, Awaitable, Callable
from src.rag.schemas import Source
from src.rag.tools.supabase import retrieve_supabase
from src.rag.tools.wikipedia import retrieve_wikipedia
from src.rag.tools.arxiv import retrieve_arxiv
from src.rag.tools.web import retrieve_web

# Type for retriever function
RetrieverFn = Callable[[str, int], Awaitable[list[dict[str, Any]]]]

TOOL_REGISTRY: dict[Source, RetrieverFn] = {
    "supabase": retrieve_supabase,
    "wikipedia": retrieve_wikipedia,
    "arxiv": retrieve_arxiv,
    "web": retrieve_web,
}

__all__ = [
    "retrieve_supabase",
    "retrieve_wikipedia",
    "retrieve_arxiv",
    "retrieve_web",
    "TOOL_REGISTRY",
    "RetrieverFn",
]
