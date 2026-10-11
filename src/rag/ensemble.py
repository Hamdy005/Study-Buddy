"""
LangChain Ensemble Retrieval (Vector Similarity + BM25) for Study-Buddy RAG.

Uses LangChain's built-in retrievers:
  - Vector:   SupabaseVectorRetriever (BaseRetriever querying pgvector RPC)
  - BM25:     BM25Retriever from langchain_community with per-material caching
  - Ensemble: EnsembleRetriever from langchain combining vector (60%) and BM25 (40%)
"""

from __future__ import annotations

import time
from typing import Any, Optional

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

try:
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
except ImportError:
    class Document:  # type: ignore[no-redef]
        def __init__(self, page_content: str = "", metadata: dict | None = None):
            self.page_content = page_content
            self.metadata = metadata or {}

    class BaseRetriever:  # type: ignore[no-redef]
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        def invoke(self, query: str) -> list[Document]:
            return self._get_relevant_documents(query)

        def _get_relevant_documents(self, query: str) -> list[Document]:
            return []

from src.rag.constants import (
    BM25_CACHE_TTL_SECONDS,
    BM25_SEARCH_WEIGHT,
    ENSEMBLE_OVER_RETRIEVAL_FACTOR,
    ENSEMBLE_RRF_C,
    SIMILARITY_MATCH_THRESHOLD,
    TOP_K_CHUNKS,
    VECTOR_SEARCH_WEIGHT,
)


# ── 1. Vector Search (LangChain BaseRetriever) ────────────────────────────────


class SupabaseVectorRetriever(BaseRetriever):
    """LangChain retriever for dense vector similarity search via Supabase pgvector."""

    material_id: str
    k: int = 12

    def _get_relevant_documents(self, query: str) -> list[Document]:
        try:
            from src.database import get_supabase
            from src.rag.rag import get_embedder

            embedder = get_embedder()
            query_embedding = embedder.embed_query(query)

            db = get_supabase()
            if db is None:
                return []

            result = db.rpc(
                "match_material_chunks",
                {
                    "query_embedding": query_embedding,
                    "match_material_id": self.material_id,
                    "match_threshold": SIMILARITY_MATCH_THRESHOLD,
                    "match_count": self.k,
                },
            ).execute()

            candidates = result.data or []
            return [
                Document(
                    page_content=c.get("content", ""),
                    metadata={
                        "chunk_id": c.get("chunk_id") or c.get("id"),
                        "similarity": float(c.get("similarity", 0.0)),
                        "material_id": self.material_id,
                    },
                )
                for c in candidates
            ]
        except Exception as exc:
            logger.warning(f"SupabaseVectorRetriever failed ({self.material_id}): {exc}")
            return []


def get_vector_retriever(material_id: str, k: int = 12) -> SupabaseVectorRetriever:
    """Return a LangChain vector similarity retriever for the given material."""
    return SupabaseVectorRetriever(material_id=material_id, k=max(1, int(round(k))))


# ── 2. BM25 Search (LangChain BM25Retriever with Caching) ─────────────────────

# In-memory cache for BM25 retrievers: material_id -> (timestamp, BM25Retriever)
_BM25_CACHE: dict[str, tuple[float, Any]] = {}


def get_bm25_retriever(material_id: str, k: int = 12) -> Optional[BaseRetriever]:
    """Return a cached LangChain BM25Retriever for the specified material's chunks."""
    k = max(1, int(round(k)))
    now = time.time()

    # Clean up stale cache entries
    stale_keys = [mid for mid, (ts, _) in _BM25_CACHE.items() if now - ts > BM25_CACHE_TTL_SECONDS * 2]
    for mid in stale_keys:
        _BM25_CACHE.pop(mid, None)

    # Cache hit
    if material_id in _BM25_CACHE:
        ts, cached_retriever = _BM25_CACHE[material_id]
        if now - ts <= BM25_CACHE_TTL_SECONDS:
            cached_retriever.k = k
            return cached_retriever

    # Fetch chunks from store
    try:
        from src.store import get_chunks

        chunks = get_chunks(material_id) or []
    except Exception as exc:
        logger.warning(f"Failed to fetch chunks for BM25 ({material_id}): {exc}")
        return None

    if not chunks:
        return None

    try:
        from langchain_community.retrievers import BM25Retriever

        docs = [
            Document(
                page_content=c.get("content", ""),
                metadata={
                    "chunk_id": c.get("id") or c.get("chunk_id"),
                    "material_id": material_id,
                },
            )
            for c in chunks
        ]
        retriever = BM25Retriever.from_documents(documents=docs, k=k)
        _BM25_CACHE[material_id] = (now, retriever)
        return retriever
    except Exception as exc:
        logger.warning(f"Failed to initialize LangChain BM25Retriever ({exc})")
        return None


# ── 3. Ensemble Retrieval (LangChain EnsembleRetriever) ───────────────────────


def get_ensemble_retriever(
    material_id: str,
    k: int = TOP_K_CHUNKS,
    vector_weight: float = VECTOR_SEARCH_WEIGHT,
    bm25_weight: float = BM25_SEARCH_WEIGHT,
) -> BaseRetriever:
    """Return a LangChain EnsembleRetriever combining vector (60%) and BM25 (40%).

    Retrieves top_k * ENSEMBLE_OVER_RETRIEVAL_FACTOR candidates from each sub-retriever
    and fuses them using weighted Reciprocal Rank Fusion (RRF).
    """
    k = max(1, int(round(k)))
    fetch_k = max(1, int(round(k * ENSEMBLE_OVER_RETRIEVAL_FACTOR)))

    vec_retriever = get_vector_retriever(material_id=material_id, k=fetch_k)
    bm25_retriever = get_bm25_retriever(material_id=material_id, k=fetch_k)

    if bm25_retriever is None:
        return vec_retriever

    try:
        from langchain.retrievers import EnsembleRetriever

        return EnsembleRetriever(
            retrievers=[vec_retriever, bm25_retriever],
            weights=[vector_weight, bm25_weight],
            c=ENSEMBLE_RRF_C,
        )
    except Exception as exc:
        logger.warning(f"LangChain EnsembleRetriever init failed ({exc}); falling back to vector retriever.")
        return vec_retriever