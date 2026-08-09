import re
from loguru import logger
import concurrent.futures
from langchain_community.utilities import WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper
from src.rag.rag import get_summary_llm, get_summary_fallback_llm
from .constants import (
    SUMMARIZER_PROMPT_TEMPLATE,
    WEB_SUMMARIZER_PROMPT_TEMPLATE,
    MAX_INPUT_CHARS,
    WIKI_TOP_K_RESULTS,
    WIKI_DOC_CONTENT_CHARS_MAX,
)


def summarizer_prompt():
    return SUMMARIZER_PROMPT_TEMPLATE



def clean_summary(text: str) -> str:
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Remove markdown table separator lines (e.g. |---|---|)
        if re.match(r'^[\s\|]*[-]{2,}[\s\|]*$', stripped):
            continue
        # If it's a table row, just keep it but maybe clean it up a bit
        if re.match(r'^\|.*\|$', stripped):
            parts = [p.strip() for p in stripped.split('|')]
            parts = [p for p in parts if p]
            line = ' | '.join(parts)
        # Note: We NO LONGER strip bold (**) here because the frontend uses it for styling
        cleaned.append(line)
    return '\n'.join(cleaned)


def _truncate_text(text: str, max_chars: int = MAX_INPUT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    first_len = int(max_chars * 0.6)
    last_len = max_chars - first_len
    first_part = text[:first_len]
    last_part = text[-last_len:]
    logger.info(f"Text truncated from {len(text)} to {max_chars} chars")
    return f"{first_part}\n\n...[content truncated]...\n\n{last_part}"


def summarizer(text: str) -> str:
    logger.info(f"Summarizer started for text of length {len(text)}")
    text = _truncate_text(text)
    prompt = summarizer_prompt()

    try:
        llm = get_summary_llm()
        chain = prompt | llm
        response = chain.invoke({"input": text})
        return clean_summary(response.content)
    except Exception as e:
        logger.warning(f"Primary summary model (gemini-3.5-flash-lite) failed: {e}. Falling back to gemini-3.1-flash-lite.")
        try:
            fallback_llm = get_summary_fallback_llm()
            chain = prompt | fallback_llm
            response = chain.invoke({"input": text})
            return clean_summary(response.content)
        except Exception as fallback_err:
            logger.error(f"Fallback summary generation failed: {fallback_err}", exc_info=True)
            raise fallback_err


def fetch_web_content(topic: str) -> str:
    """
    Fetch raw Wikipedia + DuckDuckGo content for *topic* and return the combined
    text string.  This is a pure data-fetching helper — no LLM is called.

    The returned text can be:
    - Chunked and stored in the DB for future reuse.
    - Passed directly to an LLM prompt as context.
    """
    logger.info(f"fetch_web_content started for topic: {topic}")

    def _fetch_wikipedia():
        try:
            wiki_api = WikipediaAPIWrapper(
                top_k_results=WIKI_TOP_K_RESULTS,
                doc_content_chars_max=WIKI_DOC_CONTENT_CHARS_MAX,
            )
            return wiki_api.run(topic)
        except Exception as e:
            logger.warning(f"Wikipedia search for '{topic}' failed: {e}")
            return ""

    def _fetch_duckduckgo():
        try:
            duck_api = DuckDuckGoSearchAPIWrapper()
            return duck_api.run(topic)
        except Exception as e:
            logger.warning(f"DuckDuckGo search for '{topic}' failed: {e}")
            return ""

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        wiki_future = executor.submit(_fetch_wikipedia)
        duck_future = executor.submit(_fetch_duckduckgo)
        wiki_content = wiki_future.result()
        duck_content = duck_future.result()

    all_content = []
    if wiki_content and wiki_content.strip():
        all_content.append(f"--- Wikipedia ---\n{wiki_content}")
    if duck_content and duck_content.strip():
        all_content.append(f"--- Web Search ---\n{duck_content}")

    if not all_content:
        logger.warning(f"No content found for topic: {topic}. Returning placeholder.")
        return f"No web content found for: {topic}"

    combined = "\n\n".join(all_content)
    logger.info(f"fetch_web_content combined length for '{topic}': {len(combined)}")
    return combined


def web_summarizer(topic: str, raw_content: str | None = None) -> str:
    """
    Generate an LLM summary for *topic*.

    If *raw_content* is provided (pre-fetched web text) it is used directly,
    skipping the Wiki/DDG network calls.  Otherwise fetch_web_content() is
    called internally so this function stays usable standalone.
    """
    logger.info(f"Web summarizer started for topic: {topic}")

    combined = raw_content if raw_content is not None else fetch_web_content(topic)
    combined = _truncate_text(combined)

    try:
        llm = get_summary_llm()
        chain = WEB_SUMMARIZER_PROMPT_TEMPLATE | llm
        response = chain.invoke({"topic": topic, "input": combined})
        return clean_summary(response.content)
    except Exception as e:
        logger.warning(f"Primary web summarizer model (gemini-3.5-flash-lite) failed: {e}. Falling back to gemini-3.1-flash-lite.")
        try:
            fallback_llm = get_summary_fallback_llm()
            chain = WEB_SUMMARIZER_PROMPT_TEMPLATE | fallback_llm
            response = chain.invoke({"topic": topic, "input": combined})
            return clean_summary(response.content)
        except Exception as fallback_err:
            logger.error(f"Fallback web summarizer failed: {fallback_err}", exc_info=True)
            raise fallback_err
