import re
import logging
import concurrent.futures
from langchain_community.utilities import WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper
from src.rag.rag import get_llm
from .constants import (
    SUMMARIZER_PROMPT_TEMPLATE,
    WEB_SUMMARIZER_PROMPT_TEMPLATE,
    MAX_INPUT_CHARS,
    WIKI_TOP_K_RESULTS,
    WIKI_DOC_CONTENT_CHARS_MAX,
)

logger = logging.getLogger(__name__)


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
    try:
        prompt = summarizer_prompt()
        llm = get_llm()
        # Modern LCEL syntax
        chain = prompt | llm
        response = chain.invoke({"input": text})
        
        raw_content = response.content
        logger.info(f"Summarizer received response of length {len(raw_content)}")
        logger.debug(f"Raw summary response: {raw_content[:500]}...")
        
        return clean_summary(raw_content)
    except Exception as e:
        logger.error(f"Summarizer failed: {str(e)}", exc_info=True)
        raise


def web_summarizer(topic: str) -> str:
    logger.info(f"Web summarizer started for topic: {topic}")

    all_content = []

    def fetch_wikipedia():
        try:
            wiki_api = WikipediaAPIWrapper(
                top_k_results=WIKI_TOP_K_RESULTS,
                doc_content_chars_max=WIKI_DOC_CONTENT_CHARS_MAX,
            )
            return wiki_api.run(topic)
        except Exception as e:
            logger.warning(f"Wikipedia search for '{topic}' failed: {e}")
            return ""

    def fetch_duckduckgo():
        try:
            duck_api = DuckDuckGoSearchAPIWrapper()
            return duck_api.run(topic)
        except Exception as e:
            logger.warning(f"DuckDuckGo search for '{topic}' failed: {e}")
            return ""

    # Execute searches in parallel to minimize latency
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        wiki_future = executor.submit(fetch_wikipedia)
        duck_future = executor.submit(fetch_duckduckgo)

        wiki_content = wiki_future.result()
        duck_content = duck_future.result()

    if wiki_content and wiki_content.strip():
        all_content.append(f"--- Wikipedia ---\n{wiki_content}")
    if duck_content and duck_content.strip():
        all_content.append(f"--- Web Search ---\n{duck_content}")

    if not all_content:
        logger.warning(f"No content found for topic: {topic}. Falling back to general knowledge.")
        all_content.append(f"No web content found for: {topic}")

    combined = "\n\n".join(all_content)
    logger.info(f"Web search combined text length for topic '{topic}': {len(combined)}")

    combined = _truncate_text(combined)
    try:
        llm = get_llm()
        chain = WEB_SUMMARIZER_PROMPT_TEMPLATE | llm
        response = chain.invoke({"topic": topic, "input": combined})
        raw_content = response.content
        logger.info(f"Web summarizer received response of length {len(raw_content)}")
        return clean_summary(raw_content)
    except Exception as e:
        logger.error(f"Web summarizer failed: {str(e)}", exc_info=True)
        raise
