"""
Supabase vector retrieval tool for CRAG inside src/rag.
"""

from __future__ import annotations

import asyncio
from typing import Any
from loguru import logger

from src.rag.constants import TOP_K_CHUNKS
from src.rag.schemas import Source


async def retrieve_supabase(query: str, top_k: int = TOP_K_CHUNKS, material_id: str | None = None) -> list[dict[str, Any]]:
    """
    Search Supabase vector store for relevant knowledge passages using TOP_K_CHUNKS.
    Returns a list of result dictionaries with text, source, url, title.
    """
    return await asyncio.to_thread(_sync_retrieve_supabase, query, top_k, material_id)


def _sync_retrieve_supabase(query: str, top_k: int = TOP_K_CHUNKS, material_id: str | None = None) -> list[dict[str, Any]]:
    if not query.strip():
        return []

    try:
        # 1. Use Ensemble Retriever (Vector + BM25) if material_id is provided
        if material_id:
            try:
                from src.rag.ensemble import get_ensemble_retriever
                ensemble_retriever = get_ensemble_retriever(material_id=material_id, k=top_k)
                docs = ensemble_retriever.invoke(query)
                if docs:
                    return [
                        {
                            "text": doc.page_content,
                            "source": "supabase",
                            "url": doc.metadata.get("url"),
                            "title": doc.metadata.get("title", "Study Material"),
                        }
                        for doc in docs[:top_k]
                        if doc.page_content
                    ]
            except Exception as e:
                logger.debug(f"[supabase] Ensemble retrieval skipped/failed ({e}); falling back to direct RPC/table search.")

        from src.database import get_supabase
        client = get_supabase()
        if client is not None:
            # 2. Try RPC match if configured
            try:
                params: dict[str, Any] = {"query_text": query, "match_count": top_k}
                if material_id:
                    params["filter_material_id"] = material_id
                resp = client.rpc("match_material_chunks", params).execute()
                if resp.data:
                    return [
                        {
                            "text": row.get("content") or row.get("text", ""),
                            "source": "supabase",
                            "url": row.get("url"),
                            "title": row.get("title", "Study Material"),
                        }
                        for row in resp.data
                        if (row.get("content") or row.get("text"))
                    ]
            except Exception:
                pass

            # 2. Try text search / table query on material_chunks table
            try:
                table_query = client.table("material_chunks").select("*").limit(top_k)
                if material_id:
                    table_query = table_query.eq("material_id", material_id)
                words = [w for w in query.split() if len(w) > 3][:3]
                if words:
                    table_query = table_query.ilike("content", f"%{words[0]}%")
                res = table_query.execute()
                if res.data:
                    return [
                        {
                            "text": row.get("content", ""),
                            "source": "supabase",
                            "url": row.get("url"),
                            "title": row.get("title", "Study Material"),
                        }
                        for row in res.data
                        if row.get("content")
                    ]
            except Exception:
                pass
    except Exception as exc:
        logger.debug(f"Supabase retrieval skipped or failed: {exc}")

    return []
