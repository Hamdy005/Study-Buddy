import json
import random
import re
import logging
from typing import Optional
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import create_retriever_tool

from src.rag.rag import get_quiz_llm, web_search_tools, SupabaseRetriever
from .constants import (
    QUIZ_PROMPT_TEMPLATE,
    WEB_QUIZ_PROMPT_TEMPLATE,
    MAX_SAMPLE_CHUNKS,
    RETRIEVER_K,
    WIKI_TOP_K_RESULTS,
    WIKI_DOC_CONTENT_CHARS_MAX,
    ARXIV_TOP_K_RESULTS,
    ARXIV_DOC_CONTENT_CHARS_MAX,
)

logger = logging.getLogger(__name__)


def _quiz_prompt():
    return QUIZ_PROMPT_TEMPLATE



def smart_quiz_generator(
    difficulty,
    mcq_count,
    tf_count,
    topic_title=None,
    material_id=None,
    summary=None,
    chunks=None,
):
    random_chunks = None
    if chunks:
        if len(chunks) < 2:
            sampled = list(chunks)
        else:
            sampled = random.sample(chunks, min(MAX_SAMPLE_CHUNKS, len(chunks) - 2)) + [chunks[0], chunks[-1]]
        random.shuffle(sampled)
        random_chunks = "\n".join(sampled)

    if material_id:
        return _contextual_quiz(difficulty, mcq_count, tf_count, random_chunks, material_id)
    elif summary:
        return _summary_quiz(difficulty, mcq_count, tf_count, summary)
    elif chunks:
        return _summary_quiz(difficulty, mcq_count, tf_count, random_chunks)
    elif topic_title:
        return _web_quiz(difficulty, mcq_count, tf_count, topic_title)
    else:
        raise ValueError("No data or topic provided for quiz generation.")


def _summary_quiz(difficulty, mcq_count, tf_count, context_text):
    logger.info(f"Summary Quiz started (diff={difficulty}, mcq={mcq_count}, tf={tf_count})")
    try:
        prompt = _quiz_prompt()
        llm = get_quiz_llm()
        safe_context = context_text
        
        chain = prompt | llm
        response = chain.invoke({
            "difficulty": difficulty,
            "mcq_count": mcq_count,
            "tf_count": tf_count,
            "source_type": "summary",
            "context": safe_context,
            "agent_scratchpad": "",
        })
        
        raw_content = response.content
        logger.info(f"Summary Quiz received response of length {len(raw_content)}")
        
        # response is a message object, content is the text
        return _parse_quiz({"output": raw_content})
    except Exception as e:
        logger.error(f"Summary Quiz failed: {str(e)}", exc_info=True)
        raise


def _contextual_quiz(difficulty, mcq_count, tf_count, context, material_id):
    logger.info(f"Contextual Quiz started (material_id={material_id}, diff={difficulty})")
    try:
        prompt = _quiz_prompt()
        llm = get_quiz_llm()

        retriever = SupabaseRetriever(material_id=material_id, k=RETRIEVER_K)
        retriever_tool = create_retriever_tool(
            retriever,
            name="quiz_material_retriever",
            description="Retrieves relevant content from uploaded materials for quiz generation.",
        )

        agent = create_tool_calling_agent(llm, [retriever_tool], prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=[retriever_tool],
            verbose=False,
            return_intermediate_steps=False,
            handle_parsing_errors=True,
            max_iterations=80,
            max_execution_time=300,
        )

        safe_context = context or ""
        response = executor.invoke({
            "difficulty": difficulty,
            "source_type": "Document Embeddings",
            "mcq_count": mcq_count,
            "tf_count": tf_count,
            "agent_scratchpad": "",
            "context": safe_context,
        })
        logger.info("Contextual Quiz agent finished successfully")
        return _parse_quiz(response)
    except Exception as e:
        logger.error(f"Contextual Quiz failed: {str(e)}", exc_info=True)
        raise


def _web_quiz(difficulty, mcq_count, tf_count, topic_title):
    logger.info(f"Web Quiz started (topic={topic_title}, diff={difficulty})")
    try:
        import concurrent.futures
        from langchain_community.utilities import WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper

        def fetch_wikipedia():
            try:
                wiki_api = WikipediaAPIWrapper(
                    top_k_results=WIKI_TOP_K_RESULTS,
                    doc_content_chars_max=WIKI_DOC_CONTENT_CHARS_MAX,
                )
                return wiki_api.run(topic_title)
            except Exception as e:
                logger.warning(f"Wikipedia search for '{topic_title}' failed: {e}")
                return ""

        def fetch_duckduckgo():
            try:
                duck_api = DuckDuckGoSearchAPIWrapper()
                return duck_api.run(topic_title)
            except Exception as e:
                logger.warning(f"DuckDuckGo search for '{topic_title}' failed: {e}")
                return ""

        # Fetch in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            wiki_future = executor.submit(fetch_wikipedia)
            duck_future = executor.submit(fetch_duckduckgo)

            wiki_content = wiki_future.result()
            duck_content = duck_future.result()

        all_content = []
        if wiki_content and wiki_content.strip():
            all_content.append(f"--- Wikipedia ---\n{wiki_content}")
        if duck_content and duck_content.strip():
            all_content.append(f"--- Web Search ---\n{duck_content}")

        combined_context = "\n\n".join(all_content) if all_content else f"No web content found for: {topic_title}"

        prompt = WEB_QUIZ_PROMPT_TEMPLATE
        llm = get_quiz_llm()

        chain = prompt | llm
        response = chain.invoke({
            "topic": topic_title,
            "context": combined_context,
            "difficulty": difficulty,
            "mcq_count": mcq_count,
            "tf_count": tf_count,
            "source_type": "Web Search",
            "agent_scratchpad": "",
        })

        raw_content = response.content
        logger.info("Web Quiz chain finished successfully")
        return _parse_quiz({"output": raw_content})
    except Exception as e:
        logger.error(f"Web Quiz failed: {str(e)}", exc_info=True)
        raise


def _parse_quiz(response):
    output = response["output"] if isinstance(response, dict) else str(response)
    cleaned = re.sub(r"```(?:json)?\n?", "", output)
    cleaned = cleaned.replace("```", "").strip()
    match = re.search(r"(\{[\s\S]*\})", cleaned)
    if match:
        cleaned = match.group(1)
    return json.loads(cleaned)
