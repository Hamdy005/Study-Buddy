import os
import asyncio
import uuid
import concurrent.futures
from loguru import logger
from functools import lru_cache
from typing import Optional
from pydantic import BaseModel, Field
from duckduckgo_search import DDGS
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_core.tools import StructuredTool
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
)
from .schemas import EmbeddingJob


# ── Embeddings ─────────────────────────────────────────


@lru_cache
def get_embedder():
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


def similarity_search(query: str, material_id: str, k: int = 5) -> list[dict]:
    embedder = get_embedder()
    query_embedding = embedder.embed_query(query)

    db = get_supabase()
    if db is None:
        return []
    
    result = db.rpc(
        "match_material_chunks",
        {
            "query_embedding": query_embedding,
            "match_material_id": material_id,
            "match_threshold": 0.35,
            "match_count": k,
        },
    ).execute()

    return result.data


# ── LLM ────────────────────────────────────────────────

def get_llm():
    """RAG Chatbot LLM — strictly uses Mistral AI ministral-8b-latest."""
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
    """Fallback LLM using Gemini 3.1 Flash Lite."""
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

    Returns:
        (combined_content, has_wiki, has_ddg, has_arxiv)
    """
    tools = _make_search_tools(is_topic)
    tool_map = {
        "wikipedia_search": (direct_wiki_search, "Wikipedia"),
        "web_search":       (direct_ddg_search,  "Web Search"),
        "arxiv_search":     (direct_arxiv_search, "ArXiv Research"),
    }

    # ── Router LLM call ──────────────────────────────────────────────────
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


# ── Supabase Retriever  ────────────────

class SupabaseRetriever(BaseRetriever):
    material_id: str
    k: int = TOP_K_CHUNKS

    def _get_relevant_documents(self, query: str) -> list[Document]:
        results = similarity_search(query, self.material_id, self.k)
        return [
            Document(page_content=r["content"], metadata={
                "similarity": r.get("similarity"),
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

    if not is_topic:
        # --- Material-based query (PDF/URL): vector similarity search ---
        results = similarity_search(query, material_id, k=TOP_K_CHUNKS)
        if results:
            has_chunks = True
            chunks = [r["content"] for r in results]
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

    subject_title = mat.get("title") if mat and mat.get("title") else ""

    # Build existing doc context string for the router (PDF/URL only)
    existing_doc_context = "\n\n".join(context_parts) if (not is_topic and context_parts) else ""

    # --- Agentic web context gathering ---
    web_content, has_wiki, has_ddg, has_arxiv = _agentic_gather_web_content(
        query=query,
        is_topic=is_topic,
        existing_doc_context=existing_doc_context,
        subject_title=subject_title,
    )
    if web_content:
        context_parts.append(web_content)

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

        response = chain.invoke({
            "input": query,
            "context": context_str,
            "chat_history": chat_history,
            "agent_scratchpad": "",
        })

        answer = _clean_llm_response(response.content)
        # Only persist non-refusal answers to memory
        if not _is_refusal(answer):
            memory.save_context({"input": query}, {"output": answer})
        return answer, memory
    except Exception as e:
        logger.warning(f"Primary LLM call failed or rate-limited: {e}. Falling back to secondary LLM.")
        try:
            fallback_llm = get_fallback_llm()
            chain = prompt | fallback_llm
            memory_vars = memory.load_memory_variables({"input": query})
            chat_history = memory_vars.get("chat_history", [])

            response = chain.invoke({
                "input": query,
                "context": context_str,
                "chat_history": chat_history,
                "agent_scratchpad": "",
            })
            answer = _clean_llm_response(response.content)
            # Only persist non-refusal answers to memory
            if not _is_refusal(answer):
                memory.save_context({"input": query}, {"output": answer})
            return answer, memory
        except Exception as fallback_err:
            logger.error(f"Fallback LLM call also failed: {fallback_err}")
            raise fallback_err


async def rag_answer_stream(
    query: str,
    material_id: Optional[str] = None,
    chunks: Optional[list[str]] = None,
    summaries: str = "",
    memory = None,
):
    """
    Async generator that streams LLM response tokens using Mistral AI (with Gemini fallback).
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

    if not is_topic:
        # --- Material-based query (PDF/URL): vector similarity search ---
        results = await asyncio.to_thread(similarity_search, query, material_id, k=TOP_K_CHUNKS)
        if results:
            has_chunks = True
            chunks = [r["content"] for r in results]
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

    subject_title = mat.get("title") if mat and mat.get("title") else ""

    # Build existing doc context string for the router (PDF/URL only)
    existing_doc_context = "\n\n".join(context_parts) if (not is_topic and context_parts) else ""

    # --- Agentic web context gathering (runs in thread to avoid blocking event loop) ---
    web_content, has_wiki, has_ddg, has_arxiv = await asyncio.to_thread(
        _agentic_gather_web_content,
        query,
        is_topic,
        existing_doc_context,
        subject_title,
    )
    if web_content:
        context_parts.append(web_content)

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

    try:
        primary_llm = get_llm()
        chain = prompt | primary_llm

        async for chunk in chain.astream({
            "input": query,
            "context": context_str,
            "chat_history": chat_history,
            "agent_scratchpad": "",
        }):
            token = _clean_llm_response(chunk.content)
            if token:
                full_answer_parts.append(token)
                yield token

    except Exception as e:
        logger.warning(f"Primary LLM streaming failed or rate-limited: {e}. Falling back to secondary LLM.")
        try:
            fallback_llm = get_fallback_llm()
            chain = prompt | fallback_llm

            async for chunk in chain.astream({
                "input": query,
                "context": context_str,
                "chat_history": chat_history,
                "agent_scratchpad": "",
            }):
                token = _clean_llm_response(chunk.content)
                if token:
                    full_answer_parts.append(token)
                    yield token
        except Exception as fallback_err:
            logger.error(f"Fallback LLM streaming failed: {fallback_err}")
            raise fallback_err

    full_answer = "".join(full_answer_parts)
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
            fallback_llm = get_fallback_llm()
            chain = prompt | fallback_llm
            response = chain.invoke({"query": query})
        except Exception as fallback_err:
            logger.error(f"Fallback LLM call also failed in extract_chat_title: {fallback_err}")
            raise fallback_err

    raw_title = _clean_llm_response(response.content)
    # Strip markdown symbols (*, #, _, `, quotes)
    title = raw_title.replace('*', '').replace('#', '').replace('_', '').replace('`', '').strip().strip('"').strip("'")
    if len(title) > 50:
        title = title[:50].rsplit(' ', 1)[0] + '...'
    return title

