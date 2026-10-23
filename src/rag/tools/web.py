"""
Web search tool for CRAG inside src/rag.
Uses web search configuration from src.rag.constants.
"""

from __future__ import annotations

import asyncio
from typing import Any
from loguru import logger

from src.rag.constants import (
    DUCKDUCKGO_DOC_CONTENT_CHARS_MAX,
    DUCKDUCKGO_DOC_CONTENT_CHARS_SUMMARY,
    DUCKDUCKGO_NUM_RESULTS,
)


async def retrieve_web(
    query: str,
    top_k: int = DUCKDUCKGO_NUM_RESULTS,
    max_chars: int = DUCKDUCKGO_DOC_CONTENT_CHARS_MAX,
    is_topic: bool = False,
) -> list[dict[str, Any]]:
    """
    Search the live web for recent news, events, and broad internet knowledge using project constants.
    """
    effective_max_chars = DUCKDUCKGO_DOC_CONTENT_CHARS_SUMMARY if is_topic else max_chars
    return await asyncio.to_thread(_sync_retrieve_web, query, top_k, effective_max_chars)


import re
import time


def _clean_web_query(query: str, max_words: int = 12) -> str:
    """Strips Boolean quotes, shell operators, and extra prefixes for web search."""
    try:
        from src.rag.rag import sanitize_web_query
        return sanitize_web_query(query, max_words=max_words)
    except Exception:
        cleaned = re.sub(r"\$[^$]*\$", "", query)
        cleaned = re.sub(r"\\[a-zA-Z]+", "", cleaned)
        cleaned = re.sub(r"[\"'\`]", " ", cleaned)
        cleaned = re.sub(r"[{}_*#@?!.,:;()<>]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        words = cleaned.split()
        if len(words) > max_words:
            cleaned = " ".join(words[:max_words])
        return cleaned or query[:100]


def _sync_retrieve_web(
    query: str,
    top_k: int = DUCKDUCKGO_NUM_RESULTS,
    max_chars: int = DUCKDUCKGO_DOC_CONTENT_CHARS_MAX,
) -> list[dict[str, Any]]:
    if not query.strip():
        return []

    clean_q = _clean_web_query(query)
    results: list[dict[str, Any]] = []

    # 1. Try direct_ddg_search from src.rag.rag
    try:
        from src.rag.rag import direct_ddg_search
        content = direct_ddg_search(clean_q, max_chars=max_chars)
        if content and content.strip():
            results.append(
                {
                    "text": content.strip()[:max_chars],
                    "source": "web",
                    "url": "https://duckduckgo.com",
                    "title": f"Web: {clean_q}",
                }
            )
            return results
    except Exception as exc:
        logger.debug(f"direct_ddg_search in src.rag.rag failed: {exc}")

    # 2. Try ddgs / duckduckgo_search DDGS
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*duckduckgo_search.*")
                from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            raw_results = list(ddgs.text(clean_q, max_results=top_k))
            for item in raw_results:
                text = item.get("body", "").strip()
                if text:
                    text = re.sub(r"https?://wt\.wikipedia\.org", "https://en.wikipedia.org", text)
                    url = item.get("href") or item.get("link") or "https://duckduckgo.com"
                    url = re.sub(r"https?://wt\.wikipedia\.org", "https://en.wikipedia.org", url)
                    results.append(
                        {
                            "text": text[:max_chars],
                            "source": "web",
                            "url": url,
                            "title": item.get("title", "Web Page"),
                        }
                    )
                if len(results) >= top_k:
                    break
    except Exception as exc:
        if "111" in str(exc) or "Connection refused" in str(exc) or "429" in str(exc):
            time.sleep(0.5)
        logger.debug(f"ddgs/duckduckgo_search DDGS failed: {exc}")

    if results:
        return results

    # 3. Try LangChain DuckDuckGoSearchResults
    try:
        from langchain_community.tools import DuckDuckGoSearchResults

        tool = DuckDuckGoSearchResults(max_results=top_k)
        raw_output = tool.run(clean_q)
        if isinstance(raw_output, str) and raw_output.strip() and "No good DuckDuckGo Search Result was found" not in raw_output:
            raw_output = re.sub(r"https?://wt\.wikipedia\.org", "https://en.wikipedia.org", raw_output)
            results.append(
                {
                    "text": raw_output[:max_chars],
                    "source": "web",
                    "url": "https://duckduckgo.com",
                    "title": f"Web Results for '{clean_q}'",
                }
            )
    except Exception as exc:
        logger.debug(f"LangChain DuckDuckGo tool failed: {exc}")

    return results
