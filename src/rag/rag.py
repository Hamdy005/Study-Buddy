import os
import asyncio
import uuid
import time
import concurrent.futures
from loguru import logger
from functools import lru_cache
from typing import Optional
from pydantic import BaseModel, Field
from duckduckgo_search import DDGS
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_core.tools import StructuredTool
from src.rag.tracing import ChatTracer, record_gemini_request
from langchain_core.messages import SystemMessage, HumanMessage

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI

from src.config import settings
from src.database import get_supabase
from src.store import get_chunks, get_material
from .constants import (
    EMBEDDING_DIM,
    RAG_PROMPT_TEMPLATE_BASE,
    CHAT_TITLE_PROMPT_TEMPLATE,
    WIKI_TOP_K_RESULTS,
    WIKI_DOC_CONTENT_CHARS_MAX,
    WIKI_DOC_CONTENT_CHARS_SUMMARY,
    DUCKDUCKGO_NUM_RESULTS,
    DUCKDUCKGO_DOC_CONTENT_CHARS_MAX,
    DUCKDUCKGO_DOC_CONTENT_CHARS_SUMMARY,
    ARXIV_TOP_K_RESULTS,
    ARXIV_DOC_CONTENT_CHARS_MAX,
    ARXIV_DOC_CONTENT_CHARS_SUMMARY,
    MEMORY_WINDOW_SIZE,
    TOP_K_CHUNKS,
    RERANKER_OVER_RETRIEVAL_FACTOR,
    SIMILARITY_MATCH_THRESHOLD,
    VECTOR_SEARCH_WEIGHT,
    BM25_SEARCH_WEIGHT,
    DEFAULT_RETRIEVAL_TOOL,
)
from .ensemble import get_ensemble_retriever
from .reranker import rerank_dicts
from .batch_workers import rerank_batch, rerank_async
from .schemas import EmbeddingJob


# ── Embeddings ─────────────────────────────────────────


@lru_cache
def get_embedder():
    try:
        import sentence_transformers
    except Exception as exc:
        logger.error(f"Failed to import sentence_transformers directly: {exc}", exc_info=True)
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def store_embeddings(material_id: str, chunk_ids: list[str], chunks: list[str]):
    logger.info(f"Generating embeddings for material {material_id} ({len(chunks)} chunks)...")
    embedder = get_embedder()
    embeddings = embedder.embed_documents(chunks)

    records = [
        {"chunk_id": cid, "material_id": material_id, "embedding": emb}
        for cid, emb in zip(chunk_ids, embeddings)
    ]

    db = get_supabase()
    if db is None:
        logger.warning("Supabase not connected — embeddings computed but NOT stored (no DB).")
        return

    logger.info(f"Storing {len(records)} embeddings in Supabase...")
    for i in range(0, len(records), 50):
        db.table("material_embeddings").insert(records[i:i + 50]).execute()
    logger.info(f"Embeddings stored successfully for material {material_id}.")


def warmup_embedder():
    """Dummy forward pass to keep OpenMP/MKL thread pool alive during idle periods."""
    embedder = get_embedder()
    embedder.embed_documents(["warmup"])


async def store_embeddings_async(material_id: str, chunk_ids: list[str], chunks: list[str]):
    """
    Async variant of store_embeddings that routes embedding inference through
    the batch worker queue for batching across concurrent requests.
    Falls back to the synchronous path if the batch workers are not running.
    """
    try:
        from src.rag.batch_workers import embedding_queue, job_store
        from .schemas import EmbeddingJob

        job = EmbeddingJob(job_id=str(uuid.uuid4()), texts=chunks)
        job_store[job.job_id] = {"status": "pending", "result": None, "error": None}
        await embedding_queue.put(job)

        # Wait for the worker to process the job, but with a timeout.
        # If batch workers are not running (e.g. commented out in main.py),
        # this would hang forever — the timeout triggers a fallback to the sync path.
        try:
            await asyncio.wait_for(job.done.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            # Workers not running — clean up and fall back to sync embedding
            job_store.pop(job.job_id, None)
            logger.warning(
                f"store_embeddings_async timed out waiting for batch worker "
                f"(material={material_id}). Falling back to synchronous embedding."
            )
            await asyncio.to_thread(store_embeddings, material_id, chunk_ids, chunks)
            return

        entry = job_store.pop(job.job_id)
        if entry["status"] == "error":
            raise RuntimeError(f"Embedding failed: {entry['error']}")

        embeddings = entry["result"]

        records = [
            {"chunk_id": cid, "material_id": material_id, "embedding": emb}
            for cid, emb in zip(chunk_ids, embeddings)
        ]

        db = get_supabase()
        if db is None:
            logger.warning("Supabase not connected — embeddings computed but NOT stored (no DB).")
            return

        logger.info(f"Storing {len(records)} embeddings in Supabase for material {material_id}...")

        def _insert_records():
            for i in range(0, len(records), 50):
                db.table("material_embeddings").insert(records[i:i + 50]).execute()

        await asyncio.to_thread(_insert_records)
        logger.info(f"Embeddings stored successfully for material {material_id}.")

    except Exception as e:
        # If anything unexpected fails, fall back to sync to avoid blocking the caller.
        logger.warning(f"store_embeddings_async failed ({e}); falling back to sync.")
        await asyncio.to_thread(store_embeddings, material_id, chunk_ids, chunks)


def similarity_search(query: str, material_id: str, k: int = 5, tracer: Optional[ChatTracer] = None) -> list[dict]:
    """Retrieve top_k chunks using Ensemble Retrieval followed by Reranking.

    1. Ensemble Retrieval: invokes LangChain's EnsembleRetriever (Vector 60% + BM25 40%)
       to retrieve and fuse candidate documents.
    2. Cross-Encoder Reranking: scores candidate chunks against the query using
       the batched Jina reranker worker (or direct reranker fallback) and returns top_k chunks.
    """
    if not query or not material_id:
        return []

    fetch_k = k * RERANKER_OVER_RETRIEVAL_FACTOR

    # 1. Ensemble retrieval call in rag.py (retrieves from Vector 60% and BM25 40%)
    ensemble_retriever = get_ensemble_retriever(material_id=material_id, k=k)
    docs = ensemble_retriever.invoke(query)
    if not docs:
        return []

    # Take top (k * 3) ensemble-ranked candidate chunks to pass to reranking
    candidates = [
        {
            "content": doc.page_content,
            "chunk_id": doc.metadata.get("chunk_id"),
            "similarity": float(doc.metadata.get("similarity", 0.0)),
        }
        for doc in docs[:fetch_k]
    ]

    # 2. Reranker call with LangSmith tracing & latency measurement
    rerank_span = tracer.start_rerank_span(query=query, num_candidates=len(candidates), top_k=k) if tracer else None
    t0_rerank = time.perf_counter()
    ranked = rerank_batch(query=query, records=candidates, top_k=k)
    latency_ms = (time.perf_counter() - t0_rerank) * 1000

    if tracer:
        scores = [float(r.get("rerank_score", 0.0)) for r in ranked]
        if rerank_span:
            tracer.end_rerank_span(
                rerank_span,
                scores=scores,
                latency_ms=latency_ms,
                num_candidates=len(candidates),
                top_k=k,
            )
        else:
            tracer.log_reranker(
                query=query,
                num_candidates=len(candidates),
                top_k=k,
                latency_ms=latency_ms,
                scores=scores,
            )

    logger.info(f"⏱️ Reranker took {latency_ms:.1f}ms ({latency_ms/1000:.2f}s) for {len(candidates)} candidates.")
    return ranked


async def similarity_search_async(query: str, material_id: str, k: int = 5, tracer: Optional[ChatTracer] = None) -> list[dict]:
    """Retrieve top_k chunks asynchronously using Ensemble Retrieval and batch reranking."""
    if not query or not material_id:
        return []

    fetch_k = k * RERANKER_OVER_RETRIEVAL_FACTOR

    # 1. Ensemble retrieval call in rag.py (retrieves from Vector 60% and BM25 40%)
    ensemble_retriever = get_ensemble_retriever(material_id=material_id, k=k)
    docs = await ensemble_retriever.ainvoke(query)
    if not docs:
        return []

    # Take top (k * 3) ensemble-ranked candidate chunks to pass to reranking
    candidates = [
        {
            "content": doc.page_content,
            "chunk_id": doc.metadata.get("chunk_id"),
            "similarity": float(doc.metadata.get("similarity", 0.0)),
        }
        for doc in docs[:fetch_k]
    ]

    # 2. Async batch reranker call with LangSmith tracing & latency measurement
    rerank_span = tracer.start_rerank_span(query=query, num_candidates=len(candidates), top_k=k) if tracer else None
    t0_rerank = time.perf_counter()
    ranked = await rerank_async(query=query, records=candidates, top_k=k)
    latency_ms = (time.perf_counter() - t0_rerank) * 1000

    if tracer:
        scores = [float(r.get("rerank_score", 0.0)) for r in ranked]
        if rerank_span:
            tracer.end_rerank_span(
                rerank_span,
                scores=scores,
                latency_ms=latency_ms,
                num_candidates=len(candidates),
                top_k=k,
            )
        else:
            tracer.log_reranker(
                query=query,
                num_candidates=len(candidates),
                top_k=k,
                latency_ms=latency_ms,
                scores=scores,
            )

    logger.info(f"⏱️ Async reranker took {latency_ms:.1f}ms ({latency_ms/1000:.2f}s) for {len(candidates)} candidates.")
    return ranked


# ── LLM ────────────────────────────────────────────────

def get_llm():
    """RAG Chatbot primary LLM — Mistral AI ministral-8b-latest."""
    mistral_key = os.environ.get("MISTRAL_API_KEY")
    if not mistral_key:
        raise ValueError("MISTRAL_API_KEY is not configured in config.env. Required for Ministral 8B RAG chatbot.")
    logger.info("Initializing RAG Chatbot LLM with Mistral AI model: ministral-8b-latest")
    return ChatMistralAI(
        model="ministral-8b-latest",
        api_key=mistral_key,
        temperature=0.3,
        max_tokens=3500,
        timeout=120,
        max_retries=3,
    )


def get_rag_router_llm():
    """RAG-chatbot agentic tool-routing LLM — Mistral AI ministral-8b-latest.

    Uses Ministral 8B as the web-search router inside the RAG chatbot pipeline.
    Ministral supports .bind_tools() natively so no Gemini is needed for this step.
    Kept separate from get_llm() so temperature and token limits can be tuned independently.
    """
    mistral_key = os.environ.get("MISTRAL_API_KEY")
    if not mistral_key:
        raise ValueError("MISTRAL_API_KEY is not configured in config.env.")
    logger.info("Initializing RAG router LLM with Mistral AI model: ministral-8b-latest")
    return ChatMistralAI(
        model="ministral-8b-latest",
        api_key=mistral_key,
        temperature=0.1,   # Lower temperature for deterministic tool selection
        max_tokens=512,    # Router only needs to emit tool_calls, not long text
        timeout=30,
        max_retries=3,
    )


def get_rag_fallback_llm():
    """RAG Chatbot fallback LLM — Mistral AI ministral-3b-latest (or ministral-8b-latest).

    Used when the primary call needs a fallback. Stays 100% within the Ministral
    family — no Gemini model is involved in the chatbot path.
    """
    mistral_key = os.environ.get("MISTRAL_API_KEY")
    if not mistral_key:
        raise ValueError("MISTRAL_API_KEY is not configured in config.env.")
    logger.info("Initializing RAG Chatbot fallback LLM with Mistral AI model: ministral-3b-latest")
    return ChatMistralAI(
        model="ministral-3b-latest",
        api_key=mistral_key,
        temperature=0.3,
        max_tokens=3500,
        timeout=120,
        max_retries=3,
    )


def get_summary_llm():
    """Primary Summary Generator LLM using gemini-3.5-flash-lite."""
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        raise ValueError("GEMINI_API_KEY not found in config.env.")
    logger.info("Initializing Summary LLM with model: gemini-3.5-flash-lite")
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        api_key=gemini_key,
        temperature=0.3,
        max_output_tokens=4000,
        timeout=180,
    )


def get_summary_fallback_llm():
    """Fallback Summary Generator LLM using gemini-3.1-flash-lite."""
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        raise ValueError("GEMINI_API_KEY not found in config.env.")
    logger.info("Initializing Fallback Summary LLM with model: gemini-3.1-flash-lite")
    return ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        api_key=gemini_key,
        temperature=0.3,
        max_output_tokens=4000,
        timeout=180,
    )


def get_quiz_llm():
    """Primary Quiz Generator LLM using gemini-3.5-flash-lite."""
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        raise ValueError("GEMINI_API_KEY not found in config.env.")
    logger.info("Initializing Quiz LLM with model: gemini-3.5-flash-lite")
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        api_key=gemini_key,
        temperature=0.3,
        max_output_tokens=12000,
        timeout=300,
    )


def get_quiz_fallback_llm():
    """Fallback Quiz Generator LLM using gemini-3.1-flash-lite."""
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        raise ValueError("GEMINI_API_KEY not found in config.env.")
    logger.info("Initializing Fallback Quiz LLM with model: gemini-3.1-flash-lite")
    return ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        api_key=gemini_key,
        temperature=0.3,
        max_output_tokens=12000,
        timeout=300,
    )


def get_fallback_llm():
    """Legacy alias kept for callers outside the RAG chatbot (summarizer, quiz).
    Returns Gemini 3.1 Flash Lite — NOT used in the RAG chatbot path."""
    return get_summary_fallback_llm()


def _clean_llm_response(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, str):
                texts.append(part)
            elif isinstance(part, dict):
                if part.get("type") == "text":
                    texts.append(part.get("text", ""))
                elif "text" in part and part.get("type") != "thinking":
                    texts.append(part.get("text", ""))
        return "\n".join(t for t in texts if t)
    return str(content)


# ── Web Search Helpers ────────────────────────────────

def direct_ddg_search(query: str, max_chars: int = DUCKDUCKGO_DOC_CONTENT_CHARS_MAX) -> str:
    """
    Run a DuckDuckGo web search for *query*, trying multiple backends in order.
    Falls back from html → lite → api until a backend returns results.
    Returns an empty string if all backends fail.
    """
    for backend in ("html", "lite", "api"):
        try:
            ddgs = DDGS()
            results = list(ddgs.text(query, max_results=DUCKDUCKGO_NUM_RESULTS, backend=backend))
            if not results:
                logger.warning(f"direct_ddg_search backend='{backend}': 0 results for '{query[:60]}'")
                continue
            snippets = [
                f"{r.get('title', '')}: {r.get('body', '')}"
                for r in results
                if r.get('body')
            ]
            combined = "\n".join(snippets)
            if not combined.strip():
                continue
            logger.info(f"direct_ddg_search: {len(results)} results via backend='{backend}' chars={len(combined)}")
            return combined[:max_chars]
        except Exception as e:
            logger.warning(f"direct_ddg_search backend='{backend}' failed: {type(e).__name__}: {e}")
            continue
    logger.warning(f"direct_ddg_search: all backends failed for '{query[:60]}'")
    return ""


def direct_wiki_search(query: str, max_chars: int = WIKI_DOC_CONTENT_CHARS_MAX) -> str:
    """
    Run a targeted Wikipedia search for *query*.
    Used ONLY for topic-type materials (no PDF/URL).
    Returns an empty string if the search fails.
    """
    try:
        wiki_api = WikipediaAPIWrapper(
            top_k_results=WIKI_TOP_K_RESULTS,
            doc_content_chars_max=max_chars,
        )
        result = wiki_api.run(query)
        return result[:max_chars]
    except Exception as e:
        logger.warning(f"direct_wiki_search failed: {e}")
        return ""


def direct_arxiv_search(query: str, max_chars: int = ARXIV_DOC_CONTENT_CHARS_MAX) -> str:
    """
    Run an ArXiv paper search for *query*.
    Invoked only when the agentic router classifies the topic as scientific/technical.
    Returns an empty string if the search fails.
    """
    try:
        arxiv_api = ArxivAPIWrapper(
            top_k_results=ARXIV_TOP_K_RESULTS,
            doc_content_chars_max=max_chars,
        )
        result = arxiv_api.run(query)
        return result[:max_chars]
    except Exception as e:
        logger.warning(f"direct_arxiv_search failed: {e}")
        return ""


# ── Agentic Tool Definitions & Router ──────────────────

class _SearchInput(BaseModel):
    query: str = Field(description="The exact search query string to look up")


def _make_search_tools(is_topic: bool) -> list:
    """
    Build the LangChain StructuredTool list available to the agentic router.

    - Topics  → Wikipedia + DuckDuckGo + ArXiv (agent picks the right ones)
    - PDF/URL → DuckDuckGo + ArXiv only (vector search already covers the doc)
    """
    wiki_tool = StructuredTool(
        name="wikipedia_search",
        func=direct_wiki_search,
        args_schema=_SearchInput,
        description=(
            "Search Wikipedia for encyclopedic, well-established knowledge. "
            "This is your go-to source for foundational definitions, historical context, "
            "scientific principles, biographies, and any topic with broad public documentation. "
            "Use this when the query involves a recognized concept, person, event, field of study, "
            "or any subject that a general-purpose encyclopedia would cover authoritatively. "
            "Do NOT use for cutting-edge research not yet documented in Wikipedia, "
            "real-time events, or highly niche technical subjects where ArXiv is superior."
        ),
    )

    ddg_tool = StructuredTool(
        name="web_search",
        func=direct_ddg_search,
        args_schema=_SearchInput,
        description=(
            "Search the live web using DuckDuckGo to retrieve current, diverse, and up-to-date "
            "information from across the internet. "
            "This is your broadest and most versatile retrieval tool — use it to find recent "
            "developments, practical tutorials, software documentation, real-world examples, "
            "news, and any topic that benefits from multiple diverse perspectives. "
            "Always consider this tool — it fills the gaps left by encyclopedias and academic papers, "
            "and it excels at contemporary, applied, or rapidly evolving subjects."
        ),
    )

    arxiv_tool = StructuredTool(
        name="arxiv_search",
        func=direct_arxiv_search,
        args_schema=_SearchInput,
        description=(
            "Search ArXiv for peer-reviewed preprints and cutting-edge academic research papers. "
            "This tool delivers research-grade, technically precise content from the world's "
            "leading open-access scientific repository. "
            "Use this ONLY when the topic is clearly within an active scientific or technical domain, "
            "such as: machine learning, deep learning, LLMs, computer vision, NLP, reinforcement learning, "
            "physics, quantum computing, mathematics, statistics, biology, genomics, chemistry, "
            "neuroscience, or any subject with a strong published academic literature. "
            "Do NOT use for general knowledge, history, geography, language learning, social sciences, "
            "or everyday topics that lack a formal research literature."
        ),
    )

    if is_topic:
        return [wiki_tool, ddg_tool, arxiv_tool]
    else:
        # PDF/URL: document covers domain knowledge; only supplement with web/arxiv if needed
        return [ddg_tool, arxiv_tool]


def _agentic_gather_web_content(
    query: str,
    is_topic: bool,
    existing_doc_context: str = "",
    subject_title: str = "",
    tracer: Optional[ChatTracer] = None,
) -> tuple[str, bool, bool, bool]:
    """
    Core agentic routing engine.

    Uses a fast LLM to decide *which* search tools (if any) are worth calling
    for this specific query, then executes the selected tools in parallel.

    Args:
        query:               The user's question or study topic.
        is_topic:            True for custom topic materials; False for PDF/URL.
        existing_doc_context: For PDF/URL chatbot — the already-retrieved vector chunks
                              so the router can decide if web enrichment is needed.
        subject_title:       Material title, appended to web search queries for accuracy.
        tracer:              Optional per-turn ChatTracer instance.

    Returns:
        (combined_content, has_wiki, has_ddg, has_arxiv)
    """
    tools = _make_search_tools(is_topic)
    tool_map = {
        "wikipedia_search": (direct_wiki_search, "Wikipedia"),
        "web_search":       (direct_ddg_search,  "Web Search"),
        "arxiv_search":     (direct_arxiv_search, "ArXiv Research"),
    }

    # ── Router LLM call (Summarizer / Quiz path — uses Gemini) ──────────────────────────────────────────────────
    try:
        router_llm = get_summary_llm()
        llm_with_tools = router_llm.bind_tools(tools)

        if is_topic:
            messages = [
                SystemMessage(content=(
                    "You are an expert AI educational research router. "
                    "Your job is to select search tools to gather comprehensive background information for a study topic. "
                    "You MUST select and invoke search tools for any topic. "
                    "Call web_search for up-to-date web content. "
                    "Call wikipedia_search for foundational encyclopedic knowledge. "
                    "Call arxiv_search ONLY if the topic involves scientific, academic, machine learning, physics, math, AI, or technical research."
                )),
                HumanMessage(content=f'Study topic to research: "{query}"'),
            ]
        else:
            doc_preview = existing_doc_context[:600].strip() if existing_doc_context else ""
            messages = [
                SystemMessage(content=(
                    "You are an AI research router evaluating a user query against existing document context. "
                    "Decide whether external search tools are needed to supplement the document context. "
                    "If the document context is sufficient to answer the query, do NOT call any tools. "
                    "If the query asks for current info, recent software versions, releases, real-time facts, "
                    "or topics missing from the document context, you MUST call web_search. "
                    "If the query asks for academic paper research or cutting-edge scientific methods, call arxiv_search."
                )),
                HumanMessage(content=(
                    f'User query: "{query}"\n'
                    f'Subject: "{subject_title}"\n'
                    f'Document context retrieved:\n{doc_preview}'
                )),
            ]

        router_response = llm_with_tools.invoke(messages)
        selected_calls = getattr(router_response, "tool_calls", []) or []

        # Log Gemini router invocation (gemini-3.5-flash-lite) — Summarizer/Quiz path only
        if tracer:
            usage = getattr(router_response, "usage_metadata", None) or {}
            tracer.log_llm_call(
                model          = "gemini-3.5-flash-lite",
                prompt_text    = f"User query: {query}\nSubject: {subject_title}",
                model_settings = {"temperature": 0.3, "max_output_tokens": 4000},
                usage          = usage,
                role           = "router",
            )
        else:
            record_gemini_request("gemini-3.5-flash-lite")

    except Exception as e:
        logger.warning(f"Agentic router LLM failed ({e}); falling back to default tool selection.")
        selected_calls = []

    # Guarantee search tool execution for custom topics if LLM didn't emit calls
    if is_topic and not selected_calls:
        logger.info(f"Agentic router emitted no calls for topic '{query}'; activating intelligent fallback.")
        academic_keywords = {"llm", "lora", "quantum", "model", "neural", "transformer", "algorithm", "deep learning", "ai", "physics", "math", "genomics", "arxiv"}
        q_lower = query.lower()
        is_academic = any(kw in q_lower for kw in academic_keywords)

        selected_calls = [
            {"name": "wikipedia_search", "args": {"query": query}},
            {"name": "web_search",       "args": {"query": f"{query} {subject_title}".strip()}},
        ]
        if is_academic:
            selected_calls.append({"name": "arxiv_search", "args": {"query": query}})

    # For PDF/URL materials: if query asks for current/latest/external info and router returned 0 tools, fallback to web_search
    if not is_topic and not selected_calls:
        external_keywords = {"latest", "current", "recent", "release", "version", "today", "news", "price", "2025", "2026"}
        q_lower = query.lower()
        if any(kw in q_lower for kw in external_keywords):
            logger.info(f"External/temporal query detected for PDF material '{query}'; activating web_search fallback.")
            selected_calls = [{"name": "web_search", "args": {"query": query}}]

    if not selected_calls:
        logger.info(f"Agentic router: no tools selected for query='{query}' is_topic={is_topic}")
        return "", False, False, False

    # ── Parallel tool execution ───────────────────────────────────────────
    context_parts: list[str] = []
    has_wiki = has_ddg = has_arxiv = False

    # Determine higher character limits for Topic Creation (Gemini 3.5 Flash Lite - high context)
    wiki_max = WIKI_DOC_CONTENT_CHARS_SUMMARY if is_topic else WIKI_DOC_CONTENT_CHARS_MAX
    ddg_max = DUCKDUCKGO_DOC_CONTENT_CHARS_SUMMARY if is_topic else DUCKDUCKGO_DOC_CONTENT_CHARS_MAX
    arxiv_max = ARXIV_DOC_CONTENT_CHARS_SUMMARY if is_topic else ARXIV_DOC_CONTENT_CHARS_MAX

    def _run_tool(tool_call: dict) -> tuple[str, str, str]:
        name = tool_call.get("name", "")
        args = tool_call.get("args", {})
        raw_query = args.get("query", query)
        # Enrich DuckDuckGo query with subject title only for short queries that lack subject context
        if name == "web_search" and subject_title:
            if len(raw_query.split()) < 4 and subject_title.lower() not in raw_query.lower():
                raw_query = f"{raw_query} {subject_title}".strip()
        
        if name == "wikipedia_search":
            return name, "Wikipedia", direct_wiki_search(raw_query, max_chars=wiki_max)
        elif name == "web_search":
            return name, "Web Search", direct_ddg_search(raw_query, max_chars=ddg_max)
        elif name == "arxiv_search":
            return name, "ArXiv Research", direct_arxiv_search(raw_query, max_chars=arxiv_max)
        return name, "", ""

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(selected_calls)) as executor:
        futures = [executor.submit(_run_tool, tc) for tc in selected_calls]
        for fut in concurrent.futures.as_completed(futures):
            try:
                name, label, result = fut.result(timeout=20)
                if result.strip():
                    context_parts.append(f"--- {label} ---\n{result}")
                    if name == "wikipedia_search": has_wiki = True
                    elif name == "web_search":      has_ddg  = True
                    elif name == "arxiv_search":    has_arxiv = True
            except Exception as e:
                logger.warning(f"Tool execution error: {e}")

    combined = "\n\n".join(context_parts)
    logger.info(
        f"Agentic web gather done — wiki={has_wiki} ddg={has_ddg} arxiv={has_arxiv} "
        f"chars={len(combined)} for query='{query}'"
    )
    return combined, has_wiki, has_ddg, has_arxiv


def _agentic_gather_web_content_rag(
    query: str,
    is_topic: bool,
    existing_doc_context: str = "",
    subject_title: str = "",
    tracer: Optional[ChatTracer] = None,
) -> tuple[str, bool, bool, bool]:
    """Agentic web-search router for the RAG chatbot.

    Identical contract to _agentic_gather_web_content() but uses
    ``mistral-small-latest`` (via get_rag_router_llm()) as the tool-calling
    router so that **no Gemini model is involved in the chatbot pipeline**.

    The Summarizer and Quiz callers still use the original function which routes
    through Gemini Flash-Lite.
    """
    tools = _make_search_tools(is_topic)
    tool_map = {
        "wikipedia_search": (direct_wiki_search, "Wikipedia"),
        "web_search":       (direct_ddg_search,  "Web Search"),
        "arxiv_search":     (direct_arxiv_search, "ArXiv Research"),
    }

    # ── Check if router LLM call is even needed ────────────────────────────────
    if not is_topic and existing_doc_context:
        external_keywords = {
            "latest", "recent", "current", "release", "releases", "version", "versions",
            "today", "now", "news", "price", "prices", "2024", "2025", "2026",
            "online", "web", "google", "search", "browse", "arxiv", "paper", "papers",
            "scientific", "research paper", "state of the art", "sota", "update", "updates"
        }
        q_lower = query.lower()
        has_external_intent = any(kw in q_lower for kw in external_keywords)
        if not has_external_intent:
            logger.info(
                f"RAG: Document context present ({len(existing_doc_context)} chars) and no external "
                f"keywords in query='{query}'. Bypassing router LLM to conserve Mistral rate limit."
            )
            return "", False, False, False

    # ── Router LLM call (RAG chatbot path — uses Mistral) ──────────────────────
    selected_calls: list = []
    try:
        router_llm = get_rag_router_llm()
        llm_with_tools = router_llm.bind_tools(tools)

        if is_topic:
            messages = [
                SystemMessage(content=(
                    "You are an expert AI educational research router. "
                    "Your job is to select search tools to gather comprehensive background information for a study topic. "
                    "You MUST select and invoke search tools for any topic. "
                    "Call web_search for up-to-date web content. "
                    "Call wikipedia_search for foundational encyclopedic knowledge. "
                    "Call arxiv_search ONLY if the topic involves scientific, academic, machine learning, physics, math, AI, or technical research."
                )),
                HumanMessage(content=f'Study topic to research: "{query}"'),
            ]
        else:
            doc_preview = existing_doc_context[:600].strip() if existing_doc_context else ""
            messages = [
                SystemMessage(content=(
                    "You are an AI research router. "
                    "Decide whether external search tools are needed to supplement the document context. "
                    "If the document context is sufficient to answer the query, do NOT call any tools. "
                    "If the query asks for current info, recent software versions, releases, real-time facts, "
                    "or topics missing from the document context, call web_search. "
                    "If the query asks for academic paper research or cutting-edge scientific methods, call arxiv_search."
                )),
                HumanMessage(content=(
                    f'User query: "{query}"\n'
                    f'Subject: "{subject_title}"\n'
                    f'Document context retrieved:\n{doc_preview}'
                )),
            ]

        router_response = llm_with_tools.invoke(messages)
        selected_calls = getattr(router_response, "tool_calls", []) or []

        # Pause briefly to prevent back-to-back request collision on Mistral API (1 RPS limit)
        time.sleep(0.8)

        # Log Mistral router span in LangSmith
        if tracer:
            usage = getattr(router_response, "usage_metadata", None) or {}
            tracer.log_llm_call(
                model          = "ministral-8b-latest",
                prompt_text    = f"User query: {query}\nSubject: {subject_title}",
                model_settings = {"temperature": 0.1, "max_tokens": 512},
                usage          = usage,
                role           = "router",
            )

    except Exception as e:
        logger.warning(f"RAG agentic router (Mistral) failed ({e}); falling back to default tool selection.")
        selected_calls = []

    # ── Same fallback guarantees as the original function ──────────────────────

    if is_topic and not selected_calls:
        logger.info(f"RAG Mistral router emitted no calls for topic '{query}'; activating intelligent fallback.")
        academic_keywords = {"llm", "lora", "quantum", "model", "neural", "transformer", "algorithm", "deep learning", "ai", "physics", "math", "genomics", "arxiv"}
        q_lower = query.lower()
        is_academic = any(kw in q_lower for kw in academic_keywords)

        selected_calls = [
            {"name": "wikipedia_search", "args": {"query": query}},
            {"name": "web_search",       "args": {"query": f"{query} {subject_title}".strip()}},
        ]
        if is_academic:
            selected_calls.append({"name": "arxiv_search", "args": {"query": query}})

    if not is_topic and not selected_calls:
        external_keywords = {"latest", "current", "recent", "release", "version", "today", "news", "price", "2025", "2026"}
        q_lower = query.lower()
        if any(kw in q_lower for kw in external_keywords):
            logger.info(f"External/temporal query detected (RAG path) for '{query}'; activating web_search fallback.")
            selected_calls = [{"name": "web_search", "args": {"query": query}}]

    if not selected_calls:
        logger.info(f"RAG agentic router: no tools selected for query='{query}' is_topic={is_topic}")
        return "", False, False, False

    # ── Parallel tool execution (same as original) ─────────────────────────────
    context_parts: list[str] = []
    has_wiki = has_ddg = has_arxiv = False

    wiki_max  = WIKI_DOC_CONTENT_CHARS_SUMMARY  if is_topic else WIKI_DOC_CONTENT_CHARS_MAX
    ddg_max   = DUCKDUCKGO_DOC_CONTENT_CHARS_SUMMARY if is_topic else DUCKDUCKGO_DOC_CONTENT_CHARS_MAX
    arxiv_max = ARXIV_DOC_CONTENT_CHARS_SUMMARY  if is_topic else ARXIV_DOC_CONTENT_CHARS_MAX

    def _run_tool(tool_call: dict) -> tuple[str, str, str]:
        name = tool_call.get("name", "")
        args = tool_call.get("args", {})
        raw_query = args.get("query", query)
        if name == "web_search" and subject_title:
            if len(raw_query.split()) < 4 and subject_title.lower() not in raw_query.lower():
                raw_query = f"{raw_query} {subject_title}".strip()

        if name == "wikipedia_search":
            return name, "Wikipedia", direct_wiki_search(raw_query, max_chars=wiki_max)
        elif name == "web_search":
            return name, "Web Search", direct_ddg_search(raw_query, max_chars=ddg_max)
        elif name == "arxiv_search":
            return name, "ArXiv Research", direct_arxiv_search(raw_query, max_chars=arxiv_max)
        return name, "", ""

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(selected_calls)) as executor:
        futures = [executor.submit(_run_tool, tc) for tc in selected_calls]
        for fut in concurrent.futures.as_completed(futures):
            try:
                name, label, result = fut.result(timeout=20)
                if result.strip():
                    context_parts.append(f"--- {label} ---\n{result}")
                    if name == "wikipedia_search": has_wiki = True
                    elif name == "web_search":     has_ddg  = True
                    elif name == "arxiv_search":   has_arxiv = True
            except Exception as e:
                logger.warning(f"RAG tool execution error: {e}")

    combined = "\n\n".join(context_parts)
    logger.info(
        f"RAG agentic web gather done — wiki={has_wiki} ddg={has_ddg} arxiv={has_arxiv} "
        f"chars={len(combined)} for query='{query}'"
    )
    return combined, has_wiki, has_ddg, has_arxiv




class SupabaseRetriever(BaseRetriever):
    material_id: str
    k: int = TOP_K_CHUNKS
    tracer: Optional[ChatTracer] = None

    def _get_relevant_documents(self, query: str) -> list[Document]:
        results = similarity_search(query, self.material_id, self.k, tracer=self.tracer)
        return [
            Document(page_content=r["content"], metadata={
                "similarity": r.get("similarity"),
                "ensemble_score": r.get("ensemble_score"),
                "rerank_score": r.get("rerank_score"),
                "vector_rank": r.get("vector_rank"),
                "bm25_rank": r.get("bm25_rank"),
                "chunk_id": r.get("chunk_id"),
            })
            for r in results
        ]


# ── RAG Prompt ─────────────────────────────────────────

def _rag_prompt(
    has_ddg: bool = False,
    has_wiki: bool = False,
    has_arxiv: bool = False,
    has_knowledge_retriever: bool = False,
    subject: str = "",
):
    sources = []
    if has_wiki:
        sources.append("Wikipedia snippets")
    if has_ddg:
        sources.append("DuckDuckGo web snippets")
    if has_arxiv:
        sources.append("ArXiv research papers")

    tools_section = ""
    if sources:
        tools_section = (
            f"\n<web_search_results>\n"
            f"The following search results ({', '.join(sources)}) were retrieved specifically "
            f"for this query and are included in <context>.\n"
            f"</web_search_results>"
        )
    if has_knowledge_retriever:
        tools_section += (
            "\n<knowledge>\n"
            "Relevant excerpts from the user's learning material are also included in <context>.\n"
            "</knowledge>"
        )

    subject_line = f"\nYour current study topic is: **{subject}**." if subject else ""

    formatted_template = RAG_PROMPT_TEMPLATE_BASE.format(
        subject_line=subject_line,
        tools_section=tools_section
    )

    return PromptTemplate(
        input_variables=["chat_history", "input", "agent_scratchpad", "context"],
        template=formatted_template,
    )


# ── RAG Answer ─────────────────────────────────────────

def rag_answer(
    query: str,
    material_id: Optional[str] = None,
    chunks: Optional[list[str]] = None,
    summaries: str = "",
    memory = None,
    tracer: Optional[ChatTracer] = None,
):
    if memory is None:
        memory = ConversationBufferWindowMemory(
            input_key="input", memory_key="chat_history", return_messages=True, k=MEMORY_WINDOW_SIZE
        )

    # Fetch material info if material_id is provided
    mat = None
    if material_id:
        mat = get_material(material_id)

    is_topic = not (material_id and mat and mat.get("source_type") != "topic")

    context_parts = []
    has_chunks = False

    # Inject Subject/Topic
    if mat and mat.get("title"):
        context_parts.append(f"Subject / Topic: {mat.get('title')}")

    retrieval_chunks: list[str] = []
    retrieval_scores: list[float] = []

    if not is_topic:
        # --- Material-based query (PDF/URL): hybrid ensemble search (60% vector + 40% BM25) ---
        results = similarity_search(query, material_id, k=TOP_K_CHUNKS, tracer=tracer)
        if results:
            has_chunks = True
            retrieval_chunks = [r["content"] for r in results]
            retrieval_scores = [float(r.get("ensemble_score", r.get("rerank_score", r.get("similarity", 0.0)))) for r in results]
            chunks = retrieval_chunks
            context_parts.append("Relevant Excerpts:\n" + "\n---\n".join(chunks))

        # Fallback: summary
        if not has_chunks and summaries:
            context_parts.append(f"Material Summary (No specific excerpts found for your query):\n{summaries}")

        # Fallback: sample head + tail chunks
        if not has_chunks and not summaries:
            all_chunks = get_chunks(material_id)
            if all_chunks:
                head = all_chunks[:3]
                tail = all_chunks[-2:] if len(all_chunks) > 3 else []
                sampled = head + [c for c in tail if c not in head]
                sampled_text = "\n---\n".join(c["content"] for c in sampled)
                context_parts.append(f"Material Sample (No summary found; showing start and end of material):\n{sampled_text}")

    # Log retrieval span (only for PDF/URL turns that hit vector search)
    if tracer and retrieval_chunks:
        tracer.log_retrieval(
            query       = query,
            chunks      = retrieval_chunks,
            scores      = retrieval_scores,
            tool_chosen = DEFAULT_RETRIEVAL_TOOL,
        )

    subject_title = mat.get("title") if mat and mat.get("title") else ""

    # Build existing doc context string for the router (PDF/URL only)
    existing_doc_context = "\n\n".join(context_parts) if (not is_topic and context_parts) else ""

    # --- Agentic web context gathering (RAG chatbot — Mistral router, no Gemini) ---
    web_content, has_wiki, has_ddg, has_arxiv = _agentic_gather_web_content_rag(
        query=query,
        is_topic=is_topic,
        existing_doc_context=existing_doc_context,
        subject_title=subject_title,
        tracer=tracer,
    )
    if web_content:
        context_parts.append(web_content)

    # Log agentic web gather span
    if tracer:
        tools_called = (
            (["wikipedia_search"] if has_wiki else []) +
            (["web_search"] if has_ddg else []) +
            (["arxiv_search"] if has_arxiv else [])
        )
        tracer.log_web_gather(
            query           = query,
            tools_called    = tools_called,
            chars_retrieved = len(web_content),
        )

    context_str = "\n\n".join(context_parts) if context_parts else "No specific context provided."

    has_knowledge = not is_topic
    prompt = _rag_prompt(
        has_ddg=has_ddg,
        has_wiki=has_wiki,
        has_arxiv=has_arxiv,
        has_knowledge_retriever=has_knowledge,
        subject=subject_title,
    )

    # Safety/refusal responses must NOT be saved to memory, otherwise the
    _REFUSAL_PREFIXES = (
        "I can't respond on a gibberish",
        "I can't respond on a NSFW",
        "I can't respond on a political",
        "I can't respond on a religious",
    )

    def _is_refusal(text: str) -> bool:
        t = text.strip()
        return any(t.startswith(p) for p in _REFUSAL_PREFIXES)

    try:
        primary_llm = get_llm()
        chain = prompt | primary_llm

        memory_vars = memory.load_memory_variables({"input": query})
        chat_history = memory_vars.get("chat_history", [])

        llm_start = time.perf_counter()
        response = chain.invoke({
            "input": query,
            "context": context_str,
            "chat_history": chat_history,
            "agent_scratchpad": "",
        })
        llm_ms = (time.perf_counter() - llm_start) * 1000

        # Log LLM span with token usage and cost
        if tracer:
            usage = getattr(response, "usage_metadata", None) or {}
            tracer.log_llm_call(
                model           = "ministral-8b-latest",
                prompt_text     = context_str,
                model_settings  = {"temperature": 0.3, "max_tokens": 3500},
                usage           = usage,
                ttft_ms         = None,   # non-streaming: no TTFT
                is_fallback     = False,
                role            = "primary",
            )

        answer = _clean_llm_response(response.content)
        # Only persist non-refusal answers to memory
        if not _is_refusal(answer):
            memory.save_context({"input": query}, {"output": answer})
        return answer, memory
    except Exception as e:
        is_rate_limit = "429" in str(e) or "rate limit" in str(e).lower()
        logger.warning(f"Primary LLM call failed or rate-limited (rate_limit={is_rate_limit}): {e}. Falling back to secondary LLM.")
        if tracer:
            tracer.log_error("primary_llm", e)
        if is_rate_limit:
            logger.info("Mistral rate limit hit in rag_answer; waiting 2.0s before fallback attempt...")
            time.sleep(2.0)
        try:
            fallback_llm = get_rag_fallback_llm()
            chain = prompt | fallback_llm
            memory_vars = memory.load_memory_variables({"input": query})
            chat_history = memory_vars.get("chat_history", [])

            llm_start = time.perf_counter()
            response = chain.invoke({
                "input": query,
                "context": context_str,
                "chat_history": chat_history,
                "agent_scratchpad": "",
            })

            answer = _clean_llm_response(response.content)

            if tracer:
                usage = getattr(response, "usage_metadata", None) or {}
                if not usage.get("output_tokens"):
                    input_est = max(1, len(context_str + query) // 4)
                    output_est = max(1, len(answer) // 4)
                    usage = {
                        "input_tokens": input_est,
                        "output_tokens": output_est,
                        "total_tokens": input_est + output_est,
                    }
                tracer.log_llm_call(
                    model           = "ministral-3b-latest",
                    prompt_text     = context_str,
                    model_settings  = {"temperature": 0.3, "max_tokens": 3500},
                    usage           = usage,
                    ttft_ms         = None,
                    is_fallback     = True,
                    role            = "fallback",
                )

            # Only persist non-refusal answers to memory
            if not _is_refusal(answer):
                memory.save_context({"input": query}, {"output": answer})
            return answer, memory
        except Exception as fallback_err:
            logger.error(f"Fallback LLM call also failed: {fallback_err}")
            if tracer:
                tracer.log_error("fallback_llm", fallback_err)
            raise fallback_err


async def rag_answer_stream(
    query: str,
    material_id: Optional[str] = None,
    chunks: Optional[list[str]] = None,
    summaries: str = "",
    memory = None,
    tracer: Optional[ChatTracer] = None,
):
    """
    Async generator that streams LLM response tokens using Mistral AI only.
    Primary model: ministral-8b-latest. Fallback: mistral-small-latest.
    Gemini is NOT used anywhere in this function.
    Yields individual token strings as they arrive.
    Updates conversation memory upon completion if not a refusal.
    """
    if memory is None:
        memory = ConversationBufferWindowMemory(
            input_key="input", memory_key="chat_history", return_messages=True, k=MEMORY_WINDOW_SIZE
        )

    # Fetch material info if material_id is provided
    mat = None
    if material_id:
        mat = await asyncio.to_thread(get_material, material_id)

    is_topic = not (material_id and mat and mat.get("source_type") != "topic")

    context_parts = []
    has_chunks = False

    # Inject Subject/Topic
    if mat and mat.get("title"):
        context_parts.append(f"Subject / Topic: {mat.get('title')}")

    retrieval_chunks: list[str] = []
    retrieval_scores: list[float] = []

    if not is_topic:
        # --- Material-based query (PDF/URL): hybrid ensemble search (60% vector + 40% BM25) ---
        results = await similarity_search_async(query, material_id, k=TOP_K_CHUNKS, tracer=tracer)
        if results:
            has_chunks = True
            retrieval_chunks = [r["content"] for r in results]
            retrieval_scores = [float(r.get("ensemble_score", r.get("rerank_score", r.get("similarity", 0.0)))) for r in results]
            chunks = retrieval_chunks
            context_parts.append("Relevant Excerpts:\n" + "\n---\n".join(chunks))

        # Fallback: summary
        if not has_chunks and summaries:
            context_parts.append(f"Material Summary (No specific excerpts found for your query):\n{summaries}")

        # Fallback: sample head + tail chunks
        if not has_chunks and not summaries:
            all_chunks = await asyncio.to_thread(get_chunks, material_id)
            if all_chunks:
                head = all_chunks[:3]
                tail = all_chunks[-2:] if len(all_chunks) > 3 else []
                sampled = head + [c for c in tail if c not in head]
                sampled_text = "\n---\n".join(c["content"] for c in sampled)
                context_parts.append(f"Material Sample (No summary found; showing start and end of material):\n{sampled_text}")

    # Log retrieval span (only for PDF/URL turns that hit vector search)
    if tracer and retrieval_chunks:
        tracer.log_retrieval(
            query       = query,
            chunks      = retrieval_chunks,
            scores      = retrieval_scores,
            tool_chosen = DEFAULT_RETRIEVAL_TOOL,
        )

    subject_title = mat.get("title") if mat and mat.get("title") else ""

    # Build existing doc context string for the router (PDF/URL only)
    existing_doc_context = "\n\n".join(context_parts) if (not is_topic and context_parts) else ""

    # --- Agentic web context gathering (RAG chatbot — Mistral router, no Gemini) ---
    web_content, has_wiki, has_ddg, has_arxiv = await asyncio.to_thread(
        _agentic_gather_web_content_rag,
        query,
        is_topic,
        existing_doc_context,
        subject_title,
        tracer,
    )
    if web_content:
        context_parts.append(web_content)

    # Log agentic web gather span
    if tracer:
        tools_called = (
            (["wikipedia_search"] if has_wiki else []) +
            (["web_search"] if has_ddg else []) +
            (["arxiv_search"] if has_arxiv else [])
        )
        tracer.log_web_gather(
            query           = query,
            tools_called    = tools_called,
            chars_retrieved = len(web_content),
        )

    context_str = "\n\n".join(context_parts) if context_parts else "No specific context provided."

    has_knowledge = not is_topic
    prompt = _rag_prompt(
        has_ddg=has_ddg,
        has_wiki=has_wiki,
        has_arxiv=has_arxiv,
        has_knowledge_retriever=has_knowledge,
        subject=subject_title,
    )

    _REFUSAL_PREFIXES = (
        "I can't respond on a gibberish",
        "I can't respond on a NSFW",
        "I can't respond on a political",
        "I can't respond on a religious",
    )

    def _is_refusal(text: str) -> bool:
        t = text.strip()
        return any(t.startswith(p) for p in _REFUSAL_PREFIXES)

    memory_vars = memory.load_memory_variables({"input": query})
    chat_history = memory_vars.get("chat_history", [])

    full_answer_parts = []
    first_token_ts: Optional[float] = None
    ttft_ms: Optional[float] = None
    stream_start = time.perf_counter()
    used_fallback = False
    stream_usage: dict = {}

    try:
        primary_llm = get_llm()
        chain = prompt | primary_llm

        async for chunk in chain.astream({
            "input": query,
            "context": context_str,
            "chat_history": chat_history,
            "agent_scratchpad": "",
        }):
            u = getattr(chunk, "usage_metadata", None)
            if u:
                stream_usage = u
            token = _clean_llm_response(chunk.content)
            if token:
                if first_token_ts is None:
                    first_token_ts = time.perf_counter()
                    ttft_ms = (first_token_ts - stream_start) * 1000
                full_answer_parts.append(token)
                yield token

    except Exception as e:
        is_rate_limit = "429" in str(e) or "rate limit" in str(e).lower()
        logger.warning(f"Primary LLM streaming failed or rate-limited (rate_limit={is_rate_limit}): {e}. Falling back to secondary LLM.")
        if tracer:
            tracer.log_error("primary_llm_stream", e)
        used_fallback = True
        first_token_ts = None
        ttft_ms = None
        stream_start = time.perf_counter()
        if is_rate_limit:
            logger.info("Mistral rate limit hit on primary LLM; waiting 2.0s before fallback attempt...")
            await asyncio.sleep(2.0)
        try:
            fallback_llm = get_rag_fallback_llm()
            chain = prompt | fallback_llm

            async for chunk in chain.astream({
                "input": query,
                "context": context_str,
                "chat_history": chat_history,
                "agent_scratchpad": "",
            }):
                u = getattr(chunk, "usage_metadata", None)
                if u:
                    stream_usage = u
                token = _clean_llm_response(chunk.content)
                if token:
                    if first_token_ts is None:
                        first_token_ts = time.perf_counter()
                        ttft_ms = (first_token_ts - stream_start) * 1000
                    full_answer_parts.append(token)
                    yield token
        except Exception as fallback_err:
            logger.error(f"Fallback LLM streaming failed: {fallback_err}")
            if tracer:
                tracer.log_error("fallback_llm_stream", fallback_err)
            raise fallback_err

    full_answer = "".join(full_answer_parts)

    # Log LLM span with token usage, TTFT, and dynamic pricing
    if tracer:
        model_name = "ministral-3b-latest" if used_fallback else "ministral-8b-latest"
        settings_map = (
            {"temperature": 0.3, "max_tokens": 3500}
            if used_fallback
            else {"temperature": 0.3, "max_tokens": 3500}
        )
        # If streaming provider didn't return usage_metadata, estimate from text
        if not stream_usage or not stream_usage.get("output_tokens"):
            input_est = max(1, len(context_str + query) // 4)
            output_est = max(1, len(full_answer) // 4)
            stream_usage = {
                "input_tokens": input_est,
                "output_tokens": output_est,
                "total_tokens": input_est + output_est,
            }

        tracer.log_llm_call(
            model           = model_name,
            prompt_text     = context_str,
            model_settings  = settings_map,
            usage           = stream_usage,
            ttft_ms         = ttft_ms,
            is_fallback     = used_fallback,
            role            = "fallback" if used_fallback else "primary",
        )
        # Wire eval stubs — will be activated when Corrective RAG is integrated
        tracer.schedule_evals(query=query, answer=full_answer, context=context_str)

    if not _is_refusal(full_answer):
        memory.save_context({"input": query}, {"output": full_answer})


def extract_chat_title(query: str, material_title: Optional[str] = None) -> str:
    topic_context = ""
    if material_title:
        topic_context = f"\nNote: The user is discussing the topic '{material_title}'. If their query uses pronouns like 'its' or 'this', assume it refers to this topic. If the topic name '{material_title}' appears to be a random string or dummy name, do not use it directly; instead, create a general title related to their query, such as 'Types of the topic' or 'Elements of the topic'."

    formatted_template = CHAT_TITLE_PROMPT_TEMPLATE.format(topic_context=topic_context)

    prompt = PromptTemplate(
        input_variables=["query"],
        template=formatted_template
    )
    
    try:
        primary_llm = get_llm()
        chain = prompt | primary_llm
        response = chain.invoke({"query": query})
    except Exception as e:
        logger.warning(f"Primary LLM call failed in extract_chat_title: {e}. Falling back to secondary LLM.")
        try:
            fallback_llm = get_rag_fallback_llm()
            chain = prompt | fallback_llm
            response = chain.invoke({"query": query})
        except Exception as fallback_err:
            logger.error(f"Fallback LLM call also failed in extract_chat_title: {fallback_err}")
            clean_q = query.replace('\n', ' ').strip()
            return (clean_q[:40] + '...') if len(clean_q) > 40 else clean_q

    raw_title = _clean_llm_response(response.content)
    # Strip markdown symbols (*, #, _, `, quotes)
    title = raw_title.replace('*', '').replace('#', '').replace('_', '').replace('`', '').strip().strip('"').strip("'")
    if len(title) > 50:
        title = title[:50].rsplit(' ', 1)[0] + '...'
    return title

