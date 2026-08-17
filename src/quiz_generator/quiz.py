import json
import random
import re
from loguru import logger
from typing import Optional
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import create_retriever_tool

from src.rag.rag import get_quiz_llm, get_quiz_fallback_llm, SupabaseRetriever, _agentic_gather_web_content
from .constants import (
    QUIZ_PROMPT_TEMPLATE,
    WEB_QUIZ_PROMPT_TEMPLATE,
    MAX_SAMPLE_CHUNKS,
    RETRIEVER_K,
)


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
    prompt = _quiz_prompt()
    safe_context = context_text

    try:
        llm = get_quiz_llm()
        chain = prompt | llm
        response = chain.invoke({
            "difficulty": difficulty,
            "mcq_count": mcq_count,
            "tf_count": tf_count,
            "source_type": "summary",
            "context": safe_context,
            "agent_scratchpad": "",
        })
        return _parse_quiz({"output": response.content})
    except Exception as e:
        logger.warning(f"Primary Quiz LLM (gemini-3.5-flash-lite) failed: {e}. Falling back to gemini-3.1-flash-lite.")
        try:
            fallback_llm = get_quiz_fallback_llm()
            chain = prompt | fallback_llm
            response = chain.invoke({
                "difficulty": difficulty,
                "mcq_count": mcq_count,
                "tf_count": tf_count,
                "source_type": "summary",
                "context": safe_context,
                "agent_scratchpad": "",
            })
            return _parse_quiz({"output": response.content})
        except Exception as fallback_err:
            logger.error(f"Fallback Quiz generation failed: {fallback_err}", exc_info=True)
            raise fallback_err


def _contextual_quiz(difficulty, mcq_count, tf_count, context, material_id):
    logger.info(f"Contextual Quiz started (material_id={material_id}, diff={difficulty})")
    prompt = _quiz_prompt()
    retriever = SupabaseRetriever(material_id=material_id, k=RETRIEVER_K)
    retriever_tool = create_retriever_tool(
        retriever,
        name="quiz_material_retriever",
        description="Retrieves relevant content from uploaded materials for quiz generation.",
    )
    safe_context = context or ""

    try:
        llm = get_quiz_llm()
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
        response = executor.invoke({
            "difficulty": difficulty,
            "source_type": "Document Embeddings",
            "mcq_count": mcq_count,
            "tf_count": tf_count,
            "agent_scratchpad": "",
            "context": safe_context,
        })
        return _parse_quiz(response)
    except Exception as e:
        logger.warning(f"Primary Contextual Quiz agent failed: {e}. Falling back to gemini-3.1-flash-lite.")
        try:
            fallback_llm = get_quiz_fallback_llm()
            agent = create_tool_calling_agent(fallback_llm, [retriever_tool], prompt)
            executor = AgentExecutor(
                agent=agent,
                tools=[retriever_tool],
                verbose=False,
                return_intermediate_steps=False,
                handle_parsing_errors=True,
                max_iterations=80,
                max_execution_time=300,
            )
            response = executor.invoke({
                "difficulty": difficulty,
                "source_type": "Document Embeddings",
                "mcq_count": mcq_count,
                "tf_count": tf_count,
                "agent_scratchpad": "",
                "context": safe_context,
            })
            return _parse_quiz(response)
        except Exception as fallback_err:
            logger.error(f"Fallback Contextual Quiz failed: {fallback_err}", exc_info=True)
            raise fallback_err


def _web_quiz(difficulty, mcq_count, tf_count, topic_title):
    logger.info(f"Web Quiz started (topic={topic_title}, diff={difficulty})")

    # Agentic tool selection: router LLM picks Wikipedia / DuckDuckGo / ArXiv
    combined_context, has_wiki, has_ddg, has_arxiv = _agentic_gather_web_content(
        query=topic_title,
        is_topic=True,
        existing_doc_context="",
        subject_title="",
    )
    if not combined_context:
        combined_context = f"No web content found for: {topic_title}"

    prompt = WEB_QUIZ_PROMPT_TEMPLATE

    try:
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
        return _parse_quiz({"output": response.content})
    except Exception as e:
        logger.warning(f"Primary Web Quiz model failed: {e}. Falling back to gemini-3.1-flash-lite.")
        try:
            fallback_llm = get_quiz_fallback_llm()
            chain = prompt | fallback_llm
            response = chain.invoke({
                "topic": topic_title,
                "context": combined_context,
                "difficulty": difficulty,
                "mcq_count": mcq_count,
                "tf_count": tf_count,
                "source_type": "Web Search",
                "agent_scratchpad": "",
            })
            return _parse_quiz({"output": response.content})
        except Exception as fallback_err:
            logger.error(f"Fallback Web Quiz failed: {fallback_err}", exc_info=True)
            raise fallback_err


def _parse_quiz(response):
    output = response["output"] if isinstance(response, dict) else str(response)
    cleaned = re.sub(r"```(?:json)?\n?", "", output)
    cleaned = cleaned.replace("```", "").strip()
    match = re.search(r"(\{[\s\S]*\})", cleaned)
    if match:
        cleaned = match.group(1)
    return json.loads(cleaned)
