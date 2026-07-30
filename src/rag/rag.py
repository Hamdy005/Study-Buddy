import os
import asyncio
import uuid
import logging
from functools import lru_cache
from typing import Optional
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import WikipediaAPIWrapper

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import settings
from src.database import get_supabase
from src.store import get_chunks, get_material
from .constants import (
    EMBEDDING_DIM,
    RAG_PROMPT_TEMPLATE_BASE,
    CHAT_TITLE_PROMPT_TEMPLATE,
    WIKI_TOP_K_RESULTS,
    WIKI_DOC_CONTENT_CHARS_MAX,
    DUCKDUCKGO_NUM_RESULTS,
    DUCKDUCKGO_DOC_CONTENT_CHARS_MAX,
)
from .schemas import EmbeddingJob

logger = logging.getLogger(__name__)


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
    """
    from src.rag.batch_workers import embedding_queue, job_store
    from .schemas import EmbeddingJob

    job = EmbeddingJob(job_id=str(uuid.uuid4()), texts=chunks)
    job_store[job.job_id] = {"status": "pending", "result": None, "error": None}
    await embedding_queue.put(job)
    await job.done.wait()

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
            
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _insert_records)
    logger.info(f"Embeddings stored successfully for material {material_id}.")


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
    if not os.environ.get("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY not found. Please set it in config.env.")
    logger.info(f"Initializing LLM with model: {settings.model_name}")
    return ChatGoogleGenerativeAI(
        model=settings.model_name,
        api_key=settings.gemini_api_key,
        temperature=0.3,
        max_output_tokens=2500,
        timeout=120,
    )


def get_quiz_llm():
    """Dedicated LLM instance for quiz generation with a high output token budget.
    Large quizzes (40 MCQs + 20 T/F) can produce 8000-12000 tokens of JSON,
    so we cannot reuse the chat LLM which is capped at 2000 tokens.
    """
    if not os.environ.get("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY not found. Please set it in config.env.")
    logger.info(f"Initializing Quiz LLM with model: {settings.model_name}")
    return ChatGoogleGenerativeAI(
        model=settings.model_name,
        api_key=settings.gemini_api_key,
        temperature=0.3,
        max_output_tokens=12000,
        timeout=300,
    )


def get_gemma_31b_llm():
    if not os.environ.get("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY not found. Please set it in config.env.")
    logger.info("Initializing primary LLM with model: google/gemma-4-31b-it")
    return ChatGoogleGenerativeAI(
        model="google/gemma-4-31b-it",
        api_key=settings.gemini_api_key,
        temperature=0.3,
        max_output_tokens=2500,
        timeout=120,
    )


def get_gemma_26b_llm():
    if not os.environ.get("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY not found. Please set it in config.env.")
    logger.info("Initializing fallback LLM with model: google/gemma-4-26b-it")
    return ChatGoogleGenerativeAI(
        model="google/gemma-4-26b-it",
        api_key=settings.gemini_api_key,
        temperature=0.3,
        max_output_tokens=2500,
        timeout=120,
    )


# ── Web Search Helpers ────────────────────────────────

def direct_ddg_search(query: str) -> str:
    """
    Run a targeted DuckDuckGo search for *query*.
    Used for ALL material types (topics, PDFs, URLs) to supplement context.
    Returns an empty string if the search fails.
    """
    try:
        duck_api = DuckDuckGoSearchResults(num_results=DUCKDUCKGO_NUM_RESULTS)
        raw = duck_api.run(query)
        return raw[:DUCKDUCKGO_DOC_CONTENT_CHARS_MAX]
    except Exception as e:
        logger.warning(f"direct_ddg_search failed: {e}")
        return ""


def direct_wiki_search(query: str) -> str:
    """
    Run a targeted Wikipedia search for *query*.
    Used ONLY for topic-type materials (no PDF/URL).
    Returns an empty string if the search fails.
    """
    try:
        wiki_api = WikipediaAPIWrapper(
            top_k_results=WIKI_TOP_K_RESULTS,
            doc_content_chars_max=WIKI_DOC_CONTENT_CHARS_MAX,
        )
        result = wiki_api.run(query)
        return result[:WIKI_DOC_CONTENT_CHARS_MAX]
    except Exception as e:
        logger.warning(f"direct_wiki_search failed: {e}")
        return ""


# ── Supabase Retriever  ────────────────

class SupabaseRetriever(BaseRetriever):
    material_id: str
    k: int = 4

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

def _rag_prompt(has_ddg: bool = False, has_wiki: bool = False, has_knowledge_retriever: bool = False, subject: str = ""):
    sources = []
    if has_wiki:
        sources.append("Wikipedia snippets")
    if has_ddg:
        sources.append("DuckDuckGo web snippets")

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
            input_key="input", memory_key="chat_history", return_messages=True, k=5
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
        results = similarity_search(query, material_id, k=4)
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

    # --- Wikipedia search: topics only ---
    wiki_snippets = ""
    if is_topic:
        wiki_snippets = direct_wiki_search(query)
        if wiki_snippets:
            context_parts.append(f"Wikipedia Results:\n{wiki_snippets}")

    # --- DuckDuckGo search: ALL material types (topics, PDFs, URLs) ---
    # Enrich the search query with the subject title so follow-up / short
    subject_title = mat.get("title") if mat and mat.get("title") else ""
    ddg_query = f"{query} {subject_title}".strip() if subject_title else query
    ddg_snippets = direct_ddg_search(ddg_query)
    if ddg_snippets:
        context_parts.append(f"Web Search Results (DuckDuckGo):\n{ddg_snippets}")

    context_str = "\n\n".join(context_parts) if context_parts else "No specific context provided."

    has_knowledge = not is_topic
    subject_title = mat.get("title") if mat and mat.get("title") else ""
    prompt = _rag_prompt(
        has_ddg=bool(ddg_snippets),
        has_wiki=bool(wiki_snippets),
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
        primary_llm = get_gemma_31b_llm()
        chain = prompt | primary_llm

        memory_vars = memory.load_memory_variables({"input": query})
        chat_history = memory_vars.get("chat_history", [])

        response = chain.invoke({
            "input": query,
            "context": context_str,
            "chat_history": chat_history,
            "agent_scratchpad": "",
        })

        answer = response.content
        # Only persist non-refusal answers to memory
        if not _is_refusal(answer):
            memory.save_context({"input": query}, {"output": answer})
        return answer, memory
    except Exception as e:
        logger.warning(f"Gemma 4 31B API call failed or rate-limited: {e}. Falling back to google/gemma-4-26b-it immediately.")
        try:
            fallback_llm = get_gemma_26b_llm()
            chain = prompt | fallback_llm
            memory_vars = memory.load_memory_variables({"input": query})
            chat_history = memory_vars.get("chat_history", [])

            response = chain.invoke({
                "input": query,
                "context": context_str,
                "chat_history": chat_history,
                "agent_scratchpad": "",
            })
            answer = response.content
            # Only persist non-refusal answers to memory
            if not _is_refusal(answer):
                memory.save_context({"input": query}, {"output": answer})
            return answer, memory
        except Exception as fallback_err:
            logger.error(f"Fallback Gemma 4 26B LLM call also failed: {fallback_err}")
            raise fallback_err

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
        primary_llm = get_gemma_31b_llm()
        chain = prompt | primary_llm
        response = chain.invoke({"query": query})
    except Exception as e:
        logger.warning(f"Gemma 4 31B API call failed or rate-limited in extract_chat_title: {e}. Falling back to google/gemma-4-26b-it immediately.")
        try:
            fallback_llm = get_gemma_26b_llm()
            chain = prompt | fallback_llm
            response = chain.invoke({"query": query})
        except Exception as fallback_err:
            logger.error(f"Fallback Gemma 4 26B LLM call also failed in extract_chat_title: {fallback_err}")
            raise fallback_err

    title = response.content.strip().strip('"').strip("'")
    if len(title) > 50:
        title = title[:50].rsplit(' ', 1)[0] + '...'
    return title

