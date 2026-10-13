"""
LangGraph CRAG StateGraph definition and compilation inside src/rag.
"""

from __future__ import annotations

from typing import Any
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.rag.config import Settings, settings as default_settings
from src.rag.constants import GRAPH_RECURSION_LIMIT
from src.rag.crag_llm import LLM, LangChainLLM
from src.rag.crag_nodes import (
    create_evaluator_node,
    create_fallback_node,
    create_generate_node,
    create_init_state_node,
    create_merge_pool_node,
    create_normalize_node,
    create_reply_chitchat_node,
    create_reply_refusal_node,
    create_rerank_node,
    create_retry_check_node,
    create_router_node,
    create_retrieve_node,
    create_triage_node,
    evaluator_gate,
    retry_check_gate,
    score_gate,
    triage_gate,
)
from src.rag.reranker import Reranker, JinaReranker
from src.rag.schemas import GraphState, Source
from src.rag.tools import RetrieverFn


def build_graph(
    settings: Settings | None = None,
    llm: LLM | None = None,
    reranker: Reranker | None = None,
    tools: dict[Source, RetrieverFn] | None = None,
) -> CompiledStateGraph:
    """
    Construct, wire, and compile the CRAG StateGraph pipeline.
    """
    cfg = settings or default_settings
    llm_instance = llm or LangChainLLM()
    reranker_instance = reranker or JinaReranker(model_id=cfg.RERANKER_MODEL)

    workflow = StateGraph(GraphState)

    workflow.add_node("triage", create_triage_node(llm=llm_instance, settings=cfg))
    workflow.add_node("reply_chitchat", create_reply_chitchat_node())
    workflow.add_node("reply_refusal", create_reply_refusal_node())
    workflow.add_node("init_state", create_init_state_node())
    workflow.add_node(
        "router",
        create_router_node(
            llm=llm_instance,
            settings=cfg,
        ),
    )
    workflow.add_node("retrieve", create_retrieve_node(tools=tools, settings=cfg))
    workflow.add_node("normalize", create_normalize_node())
    workflow.add_node("merge_pool", create_merge_pool_node(settings=cfg))
    workflow.add_node("rerank", create_rerank_node(reranker=reranker_instance, settings=cfg))
    workflow.add_node("evaluator", create_evaluator_node(llm=llm_instance))
    workflow.add_node("retry_check", create_retry_check_node(settings=cfg))
    workflow.add_node("generate", create_generate_node(llm=llm_instance))
    workflow.add_node("fallback", create_fallback_node(llm=llm_instance, settings=cfg))

    workflow.add_edge(START, "triage")
    workflow.add_conditional_edges(
        "triage",
        triage_gate,
        {
            "reply_chitchat": "reply_chitchat",
            "reply_refusal": "reply_refusal",
            "init_state": "init_state",
        },
    )
    workflow.add_edge("reply_chitchat", END)
    workflow.add_edge("reply_refusal", END)
    workflow.add_edge("init_state", "router")
    workflow.add_edge("router", "retrieve")
    workflow.add_edge("retrieve", "normalize")
    workflow.add_edge("normalize", "merge_pool")
    workflow.add_edge("merge_pool", "rerank")

    workflow.add_conditional_edges(
        "rerank",
        lambda state: score_gate(state, settings=cfg),
        {
            "generate": "generate",
            "evaluator": "evaluator",
            "retry_check": "retry_check",
        },
    )

    workflow.add_conditional_edges(
        "evaluator",
        evaluator_gate,
        {
            "generate": "generate",
            "retry_check": "retry_check",
        },
    )

    workflow.add_conditional_edges(
        "retry_check",
        lambda state: retry_check_gate(state, settings=cfg),
        {
            "router": "router",
            "fallback": "fallback",
        },
    )

    workflow.add_edge("generate", END)
    workflow.add_edge("fallback", END)

    app = workflow.compile()
    return app


graph = build_graph()
