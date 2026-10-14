"""
CRAG execution runner inside src/rag.
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any, AsyncGenerator

from src.rag.config import Settings, settings as default_settings
from src.rag.constants import GRAPH_RECURSION_LIMIT, TOP_K_CHUNKS
from src.rag.crag_graph import build_graph
from src.rag.crag_llm import LLM
from src.rag.reranker import Reranker
from src.rag.schemas import FinalAnswer, GraphState, Source
from src.rag.tools import RetrieverFn, TOOL_REGISTRY
from src.rag.tools.supabase import retrieve_supabase
from src.rag.tracing import ChatTracer


async def run_crag(
    question: str,
    settings: Settings | None = None,
    llm: LLM | None = None,
    reranker: Reranker | None = None,
    tools: dict[Source, RetrieverFn] | None = None,
    tracer: ChatTracer | None = None,
    material_id: str | None = None,
    token_queue: asyncio.Queue | None = None,
    on_intent: Any | None = None,
    topic: str | None = None,
    chat_history: list[dict[str, Any]] | None = None,
) -> FinalAnswer:
    """
    Run the Corrective RAG pipeline for a given user question and return a FinalAnswer.
    """
    clean_q = question.strip()
    cfg = settings or default_settings

    active_tracer = tracer
    owns_tracer = False
    if active_tracer is None and os.environ.get("LANGSMITH_API_KEY"):
        active_tracer = ChatTracer(feature="crag_pipeline")
        active_tracer.start(clean_q, source_type="crag")
        owns_tracer = True

    # Configure active tools dictionary
    if tools is not None:
        active_tools = tools
    else:
        active_tools = dict(TOOL_REGISTRY)
        if material_id:
            active_tools["supabase"] = lambda q, k=TOP_K_CHUNKS: retrieve_supabase(q, k, material_id=material_id)

    app = build_graph(
        settings=cfg,
        llm=llm,
        reranker=reranker,
        tools=active_tools,
    )

    initial_state: GraphState = {
        "original_q": clean_q,
        "tracer": active_tracer,
        "material_id": material_id,
        "token_queue": token_queue,
        "on_intent": on_intent,
        "topic": topic,
        "chat_history": chat_history or [],
    }

    config = {
        "recursion_limit": GRAPH_RECURSION_LIMIT,
    }

    try:
        final_state = await app.ainvoke(initial_state, config=config)
    except Exception as exc:
        if owns_tracer and active_tracer:
            active_tracer.finish_error(exc)
        raise exc
    finally:
        if token_queue is not None:
            await token_queue.put(None)

    answer = final_state.get("answer") or "No answer could be generated."
    citations = final_state.get("citations") or []
    low_confidence = bool(final_state.get("low_confidence", False))
    intent = final_state.get("intent")

    if owns_tracer and active_tracer:
        active_tracer.finish(answer, used_fallback=low_confidence)

    return FinalAnswer(
        answer=answer,
        citations=citations,
        low_confidence=low_confidence,
        run_id=active_tracer.run_id if active_tracer else None,
        intent=intent,
    )


async def run_crag_stream(
    question: str,
    settings: Settings | None = None,
    llm: LLM | None = None,
    reranker: Reranker | None = None,
    tools: dict[Source, RetrieverFn] | None = None,
    tracer: ChatTracer | None = None,
    material_id: str | None = None,
    on_intent: Any | None = None,
    topic: str | None = None,
    chat_history: list[dict[str, Any]] | None = None,
) -> AsyncGenerator[str, None]:
    """
    Stream CRAG response tokens in real-time as they are generated.
    Yields individual token strings.
    """
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    task = asyncio.create_task(
        run_crag(
            question=question,
            settings=settings,
            llm=llm,
            reranker=reranker,
            tools=tools,
            tracer=tracer,
            material_id=material_id,
            token_queue=queue,
            on_intent=on_intent,
            topic=topic,
            chat_history=chat_history,
        )
    )

    try:
        while True:
            get_token = asyncio.create_task(queue.get())
            done, _ = await asyncio.wait(
                {get_token, task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            if get_token in done:
                token = get_token.result()
                if token is None:
                    break
                yield token
                await asyncio.sleep(0)
            else:
                get_token.cancel()
                if task.done() and task.exception():
                    await task
                break

        await task
    except Exception as exc:
        if not task.done():
            task.cancel()
        raise exc


def cli_main():
    """Command line entrypoint."""
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        print("Usage: python -m src.rag.crag '<question>'")
        sys.exit(1)

    question = sys.argv[1]
    result = asyncio.run(run_crag(question))

    print("=" * 60)
    print("CRAG PIPELINE ANSWER")
    print("=" * 60)
    print(f"Answer: {result.answer}\n")
    print(f"Citations: {result.citations}")
    print(f"Low Confidence: {result.low_confidence}")
    print("=" * 60)


if __name__ == "__main__":
    cli_main()
