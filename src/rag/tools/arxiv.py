"""
arXiv search tool for CRAG inside src/rag.
Uses web search configuration from src.rag.constants.
"""

from __future__ import annotations

import asyncio
from typing import Any
from loguru import logger

from src.rag.constants import (
    ARXIV_DOC_CONTENT_CHARS_MAX,
    ARXIV_DOC_CONTENT_CHARS_SUMMARY,
    ARXIV_TOP_K_RESULTS,
)


import re


def _clean_arxiv_query(query: str, max_words: int = 12) -> str:
    """Strips LaTeX formulas, punctuation, and restricts word count for arXiv."""
    cleaned = re.sub(r"\$[^$]*\$", "", query)
    cleaned = re.sub(r"\\[a-zA-Z]+", "", cleaned)
    cleaned = re.sub(r"[{}_*#@?!.,:;()]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    words = cleaned.split()
    if len(words) > max_words:
        cleaned = " ".join(words[:max_words])
    return cleaned or query[:80]


async def retrieve_arxiv(
    query: str,
    top_k: int = ARXIV_TOP_K_RESULTS,
    max_chars: int = ARXIV_DOC_CONTENT_CHARS_MAX,
    is_topic: bool = False,
) -> list[dict[str, Any]]:
    """
    Search arXiv for academic papers and preprints using project constants.
    """
    effective_max_chars = ARXIV_DOC_CONTENT_CHARS_SUMMARY if is_topic else max_chars
    return await asyncio.to_thread(_sync_retrieve_arxiv, query, top_k, effective_max_chars)


def _sync_retrieve_arxiv(
    query: str,
    top_k: int = ARXIV_TOP_K_RESULTS,
    max_chars: int = ARXIV_DOC_CONTENT_CHARS_MAX,
) -> list[dict[str, Any]]:
    if not query.strip():
        return []

    clean_q = _clean_arxiv_query(query)
    results: list[dict[str, Any]] = []

    # 1. Try direct_arxiv_search from src.rag.rag
    try:
        from src.rag.rag import direct_arxiv_search
        content = direct_arxiv_search(clean_q, max_chars=max_chars)
        if content and content.strip():
            results.append(
                {
                    "text": content.strip()[:max_chars],
                    "source": "arxiv",
                    "url": f"https://arxiv.org/abs/{clean_q.replace(' ', '_')}",
                    "title": f"arXiv: {clean_q}",
                }
            )
            return results
    except Exception as exc:
        logger.debug(f"direct_arxiv_search in src.rag.rag failed: {exc}")

    # 2. Try arxiv package
    try:
        import arxiv

        client = arxiv.Client(page_size=top_k, delay_seconds=1.0, num_retries=2)
        search = arxiv.Search(
            query=clean_q,
            max_results=top_k,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        for paper in client.results(search):
            text = f"Title: {paper.title}\nAbstract: {paper.summary}"
            url = paper.entry_id or (f"https://arxiv.org/abs/{paper.get_short_id()}" if hasattr(paper, "get_short_id") else None)
            if url and not (url.startswith("http://") or url.startswith("https://")):
                url = f"https://arxiv.org/abs/{url}"
            results.append(
                {
                    "text": text[:max_chars],
                    "source": "arxiv",
                    "url": url,
                    "title": paper.title,
                }
            )
            if len(results) >= top_k:
                break
    except Exception as exc:
        logger.debug(f"Direct arxiv package search failed: {exc}")

    if results:
        return results

    # 3. Try langchain_community ArxivAPIWrapper
    try:
        from langchain_community.utilities import ArxivAPIWrapper

        wrapper = ArxivAPIWrapper(top_k_results=top_k, doc_content_chars_max=max_chars)
        docs = wrapper.load(clean_q)
        for doc in docs:
            text = doc.page_content.strip()
            if text:
                title = doc.metadata.get("Title", "arXiv Paper")
                url = doc.metadata.get("Entry ID")
                if url and not (url.startswith("http://") or url.startswith("https://")):
                    url = f"https://arxiv.org/abs/{url}"
                results.append(
                    {
                        "text": text[:max_chars],
                        "source": "arxiv",
                        "url": url,
                        "title": title,
                    }
                )
                if len(results) >= top_k:
                    break
    except Exception as exc:
        logger.debug(f"LangChain ArxivAPIWrapper failed: {exc}")

    if results:
        return results

    # 4. Fallback to web search for arXiv research papers if direct arXiv API fails or is blocked
    try:
        from src.rag.rag import direct_ddg_search

        paper_web = direct_ddg_search(f"{clean_q} arxiv paper", max_chars=max_chars)
        if paper_web and paper_web.strip():
            results.append(
                {
                    "text": paper_web.strip()[:max_chars],
                    "source": "arxiv",
                    "url": "https://arxiv.org",
                    "title": f"arXiv Research: {clean_q}",
                }
            )
    except Exception as exc:
        logger.debug(f"DuckDuckGo arXiv fallback failed: {exc}")

    return results
