import os
import asyncio
import uuid
import logging
from functools import lru_cache
from typing import Optional
from langchain_core.tools import Tool

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchResults
from langchain_core.tools.retriever import create_retriever_tool
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_openai import ChatOpenAI
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
    ARXIV_TOP_K_RESULTS,
    ARXIV_DOC_CONTENT_CHARS_MAX,
    DUCKDUCKGO_DOC_CONTENT_CHARS_MAX,
)
from .schemas import SearchInput, EmbeddingJob

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


def get_groq_llm():
    if not settings.groq_api_key:
        raise ValueError("GROQ_API_KEY not found. Please set it in config.env.")
    return ChatOpenAI(
        model="llama-3.1-8b-instant",
        base_url="https://api.groq.com/openai/v1",
        api_key=settings.groq_api_key,
        max_tokens=600,
    )


# ── Web Search Tools ───────────────────────────────────

def web_search_tools(
    has_material: bool = False,
    wiki_k: int = WIKI_TOP_K_RESULTS,
    wiki_chars: int = WIKI_DOC_CONTENT_CHARS_MAX,
    arxiv_k: int = ARXIV_TOP_K_RESULTS,
    arxiv_chars: int = ARXIV_DOC_CONTENT_CHARS_MAX,
    duck_chars: int = DUCKDUCKGO_DOC_CONTENT_CHARS_MAX,
):

    tools = []

    try:
        wiki_api = WikipediaAPIWrapper(top_k_results=wiki_k, doc_content_chars_max=wiki_chars)
        def safe_wiki_run(query: str) -> str:
            try: return wiki_api.run(query)[:wiki_k * wiki_chars]
            except Exception as e: return f"Wikipedia search failed: {e}. Try another tool."
        
        wikipedia = Tool(
            name="wikipedia",
            description="Search Wikipedia for factual, historical, or conceptual questions. Input should be a specific search query.",
            func=safe_wiki_run,
            args_schema=SearchInput
        )
        tools.append(wikipedia)
    except Exception as e:
        logger.warning(f"Skipping Wikipedia Search: {e}")
        arxiv_k = 4; arxiv_chars = 2500  # reallocate budget

    try:
        arxiv_api = ArxivAPIWrapper(top_k_results=arxiv_k, doc_content_chars_max=arxiv_chars)
        def safe_arxiv_run(query: str) -> str:
            try: return arxiv_api.run(query)[:arxiv_k * arxiv_chars]
            except Exception as e: return f"Arxiv search failed: {e}. Try another tool."

        arxiv = Tool(
            name="arxiv",
            description="Search scientific papers on Arxiv for technical, academic, or research questions in Physics, Math, CS, Biology, etc. Input should be a specific search query.",
            func=safe_arxiv_run,
            args_schema=SearchInput
        )
        tools.append(arxiv)
    except Exception as e:
        logger.warning(f"Skipping Arxiv Search: {e}")
        duck_chars += (arxiv_k * arxiv_chars)

    try:
        duck_api = DuckDuckGoSearchResults()
        def safe_duck_run(query: str) -> str:
            try: return duck_api.run(query)[:duck_chars]
            except Exception as e: return f"DuckDuckGo search failed: {e}."

        duck = Tool(
            name="duckduckgo",
            description="Search the web for current events, recent news, or general web content. Use when Wikipedia and Arxiv don't have the answer. Input should be a specific search query.",
            func=safe_duck_run,
            args_schema=SearchInput
        )
        tools.append(duck)
    except Exception as e:
        logger.warning(f"Skipping DuckDuckGo Search: {e}")

    return tools

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

def _rag_prompt(has_web_tools: bool = True, has_knowledge_retriever: bool = False, subject: str = ""):
    tools_list = []
    if has_web_tools:
        tools_list.append("- **Wikipedia Retriever** for general knowledge and conceptual explanations")
        tools_list.append("- **Arxiv Retriever** for academic and scientific research information")
        tools_list.append("- **DuckDuckGo Retriever** for the latest web-based insights")
    if has_knowledge_retriever:
        tools_list.append("- **Knowledge Retriever:** for local learning materials (vector embeddings, summaries, raw text chunks)")
    
    tools_section = ""
    if tools_list:
        tools_section = "\n<tools>\nYou have access to these tools:\n" + "\n".join(tools_list) + "\n</tools>"

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

    # Determine tool availability
    # Custom topics (no URL/file) should use web tools
    if material_id and mat and mat.get("source_type") != "topic":
        tools = []
    else:
        tools = web_search_tools(has_material=False)

    llm = get_groq_llm()

    context_parts = []
    has_chunks = False

    # Inject Subject/Topic
    if mat and mat.get("title"):
        context_parts.append(f"Subject / Topic: {mat.get('title')}")

    if material_id and mat and mat.get("source_type") != "topic":
        results = similarity_search(query, material_id, k=5)
        if results:
            has_chunks = True
            chunks = [r["content"] for r in results]
            context_parts.append(f"Relevant Excerpts:\n" + "\n---\n".join(chunks))

    # To save tokens, only pass the summary if no specific chunks were found
    if not has_chunks and summaries:
        context_parts.append(f"Material Summary (No specific excerpts found for your query):\n{summaries}")

    # Fallback: If NO chunks matched AND NO summary was generated, pass start and end chunks
    if not has_chunks and not summaries and material_id and mat and mat.get("source_type") != "topic":
        all_chunks = get_chunks(material_id)
        if all_chunks:
            # Take first 3 and last 2 chunks
            head = all_chunks[:3]
            tail = all_chunks[-2:] if len(all_chunks) > 3 else []
            # Combine without duplicates
            sampled = head + [c for c in tail if c not in head]
            sampled_text = "\n---\n".join(c["content"] for c in sampled)
            context_parts.append(f"Material Sample (No summary found; showing start and end of material):\n{sampled_text}")

    context_str = "\n\n".join(context_parts) if context_parts else "No specific context provided."

    has_knowledge = bool(material_id and mat and mat.get("source_type") != "topic")
    subject_title = mat.get("title") if mat and mat.get("title") else ""
    prompt = _rag_prompt(has_web_tools=len(tools) > 0, has_knowledge_retriever=has_knowledge, subject=subject_title)

    if tools:
        agent = create_tool_calling_agent(llm, tools, prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=False,
            return_intermediate_steps=False,
            handle_parsing_errors=True,
            max_iterations=3,
        )
        response = executor.invoke({"input": query, "context": context_str})
        return response["output"], memory
    else:
        # Simple LLM call without Agent loop to save tokens and avoid 429
        chain = prompt | llm
        
        # Load history from memory
        memory_vars = memory.load_memory_variables({"input": query})
        chat_history = memory_vars.get("chat_history", [])
        
        response = chain.invoke({
            "input": query,
            "context": context_str,
            "chat_history": chat_history,
            "agent_scratchpad": ""
        })
        
        answer = response.content
        # Save to memory manually
        memory.save_context({"input": query}, {"output": answer})
        return answer, memory

def extract_chat_title(query: str, material_title: Optional[str] = None) -> str:
    llm = get_groq_llm()
    
    topic_context = ""
    if material_title:
        topic_context = f"\nNote: The user is discussing the topic '{material_title}'. If their query uses pronouns like 'its' or 'this', assume it refers to this topic. If the topic name '{material_title}' appears to be a random string or dummy name, do not use it directly; instead, create a general title related to their query, such as 'Types of the topic' or 'Elements of the topic'."

    formatted_template = CHAT_TITLE_PROMPT_TEMPLATE.format(topic_context=topic_context)

    prompt = PromptTemplate(
        input_variables=["query"],
        template=formatted_template
    )
    chain = prompt | llm
    response = chain.invoke({"query": query})
    title = response.content.strip().strip('"').strip("'")
    if len(title) > 50:
        title = title[:50].rsplit(' ', 1)[0] + '...'
    return title
