"""
Wikipedia search tool for CRAG inside src/rag.
Uses web search configuration from src.rag.constants.
"""

from __future__ import annotations

import asyncio
from typing import Any
from loguru import logger

from src.rag.constants import (
    WIKI_DOC_CONTENT_CHARS_MAX,
    WIKI_DOC_CONTENT_CHARS_SUMMARY,
    WIKI_TOP_K_RESULTS,
)


import re


def _clean_wiki_query(query: str, max_words: int = 10) -> str:
    """Strips LaTeX formulas, punctuation, and restricts word count for Wikipedia."""
    cleaned = re.sub(r"\$[^$]*\$", "", query)
    cleaned = re.sub(r"\\[a-zA-Z]+", "", cleaned)
    cleaned = re.sub(r"[{}_*#@?!.,:;()]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    words = cleaned.split()
    if len(words) > max_words:
        cleaned = " ".join(words[:max_words])
    return cleaned or query[:60]


async def retrieve_wikipedia(
    query: str,
    top_k: int = WIKI_TOP_K_RESULTS,
    max_chars: int = WIKI_DOC_CONTENT_CHARS_MAX,
    is_topic: bool = False,
) -> list[dict[str, Any]]:
    """
    Search Wikipedia for encyclopedic articles and sections using project constants.
    """
    effective_max_chars = WIKI_DOC_CONTENT_CHARS_SUMMARY if is_topic else max_chars
    return await asyncio.to_thread(_sync_retrieve_wikipedia, query, top_k, effective_max_chars)


def _sync_retrieve_wikipedia(
    query: str,
    top_k: int = WIKI_TOP_K_RESULTS,
    max_chars: int = WIKI_DOC_CONTENT_CHARS_MAX,
) -> list[dict[str, Any]]:
    if not query.strip():
        return []

    clean_q = _clean_wiki_query(query)
    results: list[dict[str, Any]] = []

    # 1. Try direct_wiki_search from src.rag.rag
    try:
        from src.rag.rag import direct_wiki_search
        content = direct_wiki_search(clean_q, max_chars=max_chars)
        if content and content.strip():
            results.append(
                {
                    "text": content.strip()[:max_chars],
                    "source": "wikipedia",
                    "url": f"https://en.wikipedia.org/wiki/{clean_q.replace(' ', '_')}",
                    "title": f"Wikipedia: {clean_q}",
                }
            )
            return results
    except Exception as exc:
        logger.debug(f"direct_wiki_search in src.rag.rag failed: {exc}")

    # 2. Try wikipedia package directly
    try:
        import wikipedia

        search_titles = wikipedia.search(clean_q, results=top_k)
        for title in search_titles:
            try:
                page = wikipedia.page(title, auto_suggest=False)
                content = page.content or page.summary or ""
                if content.strip():
                    results.append(
                        {
                            "text": content[:max_chars],
                            "source": "wikipedia",
                            "url": page.url or f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                            "title": page.title or title,
                        }
                    )
                if len(results) >= top_k:
                    break
            except Exception as e:
                logger.debug(f"Error fetching Wikipedia page '{title}': {e}")
                continue
    except Exception as exc:
        logger.debug(f"Direct wikipedia package search failed: {exc}")

    if results:
        return results

    # 3. Try langchain_community WikipediaAPIWrapper
    try:
        from langchain_community.utilities import WikipediaAPIWrapper

        api = WikipediaAPIWrapper(top_k_results=top_k, doc_content_chars_max=max_chars)
        docs = api.load(clean_q)
        for doc in docs:
            text = doc.page_content.strip()
            if text:
                title = doc.metadata.get("title", "Wikipedia Article")
                url = doc.metadata.get(
                    "source",
                    f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                )
                results.append(
                    {
                        "text": text[:max_chars],
                        "source": "wikipedia",
                        "url": url,
                        "title": title,
                    }
                )
                if len(results) >= top_k:
                    break
    except Exception as exc:
        logger.debug(f"LangChain WikipediaAPIWrapper failed: {exc}")

    if results:
        return results

    # 4. Fallback to web search for Wikipedia content if Wikipedia API is blocked by policy or fails
    try:
        from src.rag.rag import direct_ddg_search

        wiki_web = direct_ddg_search(f"{clean_q} wikipedia", max_chars=max_chars)
        if wiki_web and wiki_web.strip():
            results.append(
                {
                    "text": wiki_web.strip()[:max_chars],
                    "source": "wikipedia",
                    "url": "https://en.wikipedia.org",
                    "title": f"Wikipedia: {clean_q}",
                }
            )
    except Exception as exc:
        logger.debug(f"DuckDuckGo Wikipedia fallback failed: {exc}")

    return results
