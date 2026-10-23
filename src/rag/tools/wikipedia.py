"""
Wikipedia search tool for CRAG inside src/rag.
Uses web search configuration from src.rag.constants.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any
from loguru import logger

from src.rag.constants import (
    WIKI_DOC_CONTENT_CHARS_MAX,
    WIKI_DOC_CONTENT_CHARS_SUMMARY,
    WIKI_TIMEOUT_SECONDS,
    WIKI_TOP_K_RESULTS,
    WIKIPEDIA_USER_AGENT,
)


_WIKIPEDIA_PATCHED = False


WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"


def patch_wikipedia_client() -> None:
    """Configures Wikimedia User-Agent and monkeypatches wikipedia._wiki_request with a strict timeout and HTTP status verification."""
    global _WIKIPEDIA_PATCHED
    if _WIKIPEDIA_PATCHED:
        return
    try:
        import wikipedia
        import requests

        wikipedia.set_user_agent(WIKIPEDIA_USER_AGENT)
        wikipedia.wikipedia.API_URL = WIKIPEDIA_API_URL

        orig_set_lang = getattr(wikipedia, "set_lang", None)
        if orig_set_lang:
            def _safe_set_lang(prefix: str) -> None:
                orig_set_lang(prefix)
                clean_prefix = prefix.lower().replace("wt", "en")
                wikipedia.wikipedia.API_URL = f"https://{clean_prefix}.wikipedia.org/w/api.php"
            wikipedia.set_lang = _safe_set_lang

        orig_request = getattr(wikipedia.wikipedia, "_wiki_request", None)

        def _safe_wiki_request(params: dict[str, Any]) -> dict[str, Any]:
            params["format"] = "json"
            if "action" not in params:
                params["action"] = "query"
            headers = {
                "User-Agent": getattr(wikipedia.wikipedia, "USER_AGENT", WIKIPEDIA_USER_AGENT)
            }
            if (
                getattr(wikipedia.wikipedia, "RATE_LIMIT", False)
                and getattr(wikipedia.wikipedia, "RATE_LIMIT_LAST_CALL", None)
            ):
                from datetime import datetime
                import time

                last_call = wikipedia.wikipedia.RATE_LIMIT_LAST_CALL
                min_wait = getattr(wikipedia.wikipedia, "RATE_LIMIT_MIN_WAIT", None)
                if min_wait and last_call + min_wait > datetime.now():
                    wait_time = (last_call + min_wait) - datetime.now()
                    time.sleep(int(wait_time.total_seconds()))

            api_url = getattr(wikipedia.wikipedia, "API_URL", WIKIPEDIA_API_URL)
            if not api_url or "wt.wikipedia.org" in api_url or api_url.startswith("http://"):
                api_url = WIKIPEDIA_API_URL
                wikipedia.wikipedia.API_URL = WIKIPEDIA_API_URL

            r = requests.get(
                api_url,
                params=params,
                headers=headers,
                timeout=WIKI_TIMEOUT_SECONDS,
            )

            if getattr(wikipedia.wikipedia, "RATE_LIMIT", False):
                from datetime import datetime

                wikipedia.wikipedia.RATE_LIMIT_LAST_CALL = datetime.now()

            if r.status_code != 200:
                raise RuntimeError(
                    f"Wikipedia API error HTTP {r.status_code} ({r.text.strip()[:80]})"
                )
            content_type = r.headers.get("content-type", "")
            if "json" not in content_type:
                raise RuntimeError(
                    f"Wikipedia API returned non-JSON content-type '{content_type}': {r.text.strip()[:80]}"
                )
            return r.json()

        wikipedia.wikipedia._wiki_request = _safe_wiki_request
        _WIKIPEDIA_PATCHED = True
        logger.debug("Wikipedia client patched with https API_URL and safe status checks")
    except Exception as exc:
        logger.debug(f"Failed to patch wikipedia client: {exc}")


def _clean_wiki_query(query: str, max_words: int = 8) -> str:
    """Strips LaTeX formulas, question frames, and conversational filler to extract concise entity/subject search terms for Wikipedia."""
    if not query or not query.strip():
        return ""
    cleaned = re.sub(r"\$[^$]*\$", "", query)
    cleaned = re.sub(r"\\[a-zA-Z]+", "", cleaned)
    patterns = [
        r"^(what\s+is\s+(an?|the)?|what\s+are\s+(the)?|what\s+does\s+|how\s+does\s+|how\s+do\s+|how\s+to\s+|why\s+is\s+|explain\s+(the)?|describe\s+(the)?|overview\s+of\s+(the)?|definition\s+of\s+(the)?|can\s+you\s+explain\s+(the)?)\s*",
        r"\s+(and\s+what.*|does\s+it\s+provide.*|does\s+it\s+do.*|in\s+detail.*|step\s+by\s+step.*)\??$",
    ]
    for pat in patterns:
        cleaned = re.sub(pat, " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[{}_*#@?!.,:;()\"'~^/\\\[\]]", " ", cleaned)
    stop_words = {
        "what", "which", "who", "how", "why", "does", "do", "did",
        "is", "are", "was", "were", "the", "a", "an", "and", "or",
        "for", "of", "in", "to", "with", "about"
    }
    words = [w for w in cleaned.split() if w.lower() not in stop_words]
    if not words:
        words = cleaned.split()
    if len(words) > max_words:
        words = words[:max_words]
    res = " ".join(words).strip()
    return res or query.strip()[:60]


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

    patch_wikipedia_client()
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

    # 2. Try wikipedia package directly if direct_wiki_search did not yield results
    direct_api_operational = True
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
        direct_api_operational = False
        logger.debug(f"Direct wikipedia package search failed: {exc}")

    if results:
        return results

    # 3. Try langchain_community WikipediaAPIWrapper ONLY if direct API did not error out
    if direct_api_operational:
        try:
            from langchain_community.utilities import WikipediaAPIWrapper

            api = WikipediaAPIWrapper(top_k_results=top_k, doc_content_chars_max=max_chars)
            docs = api.load(clean_q)
            for doc in docs:
                text = doc.page_content.strip()
                if text:
                    title = doc.metadata.get("title", f"Wikipedia: {clean_q}")
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

        for fallback_q in [f"{clean_q} wikipedia summary", f"{clean_q} wikipedia"]:
            wiki_web = direct_ddg_search(fallback_q, max_chars=max_chars)
            if wiki_web and wiki_web.strip():
                clean_url = f"https://en.wikipedia.org/wiki/{clean_q.replace(' ', '_')}"
                clean_url = re.sub(r"https?://wt\.wikipedia\.org", "https://en.wikipedia.org", clean_url)
                results.append(
                    {
                        "text": wiki_web.strip()[:max_chars],
                        "source": "wikipedia",
                        "url": clean_url,
                        "title": f"Wikipedia: {clean_q}",
                    }
                )
                break
    except Exception as exc:
        logger.debug(f"DuckDuckGo Wikipedia fallback failed: {exc}")

    return results

