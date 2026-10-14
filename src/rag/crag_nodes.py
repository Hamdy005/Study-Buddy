"""
CRAG node implementations integrated inside src/rag.
Features automatic LangChain tool selection on pass 0 with corrective CRAG escalation on retry.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
from typing import Any, Callable
from loguru import logger

from src.rag.config import Settings, settings as default_settings
from src.rag.constants import (
    ESCALATION_LEVELS,
    FALLBACK_NO_SOURCES_MSG,
    FALLBACK_NO_SOURCES_MSG_AR,
    FALLBACK_SYSTEM_PROMPT,
    GENERATE_HYBRID_PROMPT,
    GENERATE_SYSTEM_PROMPT,
    MAX_CHUNK_CHARS,
    MAX_QUESTION_LEN,
    MIN_QUESTION_LEN,
    ROUTER_SYSTEM_PROMPT,
    EVALUATOR_SYSTEM_PROMPT,
    TRIAGE_SYSTEM_PROMPT,
    UNSAFE_REPLIES,
    CHITCHAT_REPLIES,
    TOOL_TIMEOUT_SECONDS,
    DEFAULT_TOOL_TIMEOUT_SIMPLE_S,
    DEFAULT_TOOL_TIMEOUT_COMPLEX_S,
    DEFAULT_SIMPLE_THRESHOLD,
    TOPIC_HINTS,
    WIKI_TOP_K_RESULTS,
    ARXIV_TOP_K_RESULTS,
    DUCKDUCKGO_NUM_RESULTS,
    format_chat_history,
    router_user_message,
    evaluator_user_message,
    generate_user_message,
    fallback_user_message,
)
from src.rag.crag_llm import LLM, LangChainLLM
from src.rag.reranker import Reranker, JinaReranker, CascadeReranker
from src.rag.schemas import (
    Chunk,
    EvaluatorOutput,
    GraphState,
    RouterOutput,
    Source,
    TriageOutput,
)
from src.rag.tools import TOOL_REGISTRY, RetrieverFn
from src.rag.tracing import ChatTracer


def _log_node(node_name: str, state: GraphState, **extra):
    if node_name == "triage":
        logger.info(f"[{node_name}] {' '.join(f'{k}={v}' for k, v in extra.items())}")
        return
    sources_display = extra.pop("sources", None) or state.get("sources_to_use", [])
    if not sources_display:
        pool_chunks = state.get("ranked") or state.get("pool") or state.get("new_chunks") or []
        sources_display = list(dict.fromkeys(c.source for c in pool_chunks if hasattr(c, "source")))
    logger.info(
        f"[{node_name}] retries={state.get('retries', 0)} "
        f"sources={sources_display} "
        f"{' '.join(f'{k}={v}' for k, v in extra.items())}"
    )


# ── Text Cleaning & Chunking Helpers ───────────────────────────

def _clean_text(raw: str) -> str:
    """Strip HTML tags and excessive LaTeX markup."""
    text = re.sub(r"<[^>]+>", " ", raw)
    text = re.sub(r"\\\[(.*?)\\\]", r"\1", text)
    text = re.sub(r"\\\((.*?)\\\)", r"\1", text)
    text = re.sub(r"\$\$(.*?)\$\$", r"\1", text)
    text = re.sub(r"\$(.*?)\$", r"\1", text)
    text = re.sub(r"\\text\{([^}]+)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> list[str]:
    """Split long passage into smaller chunks if it exceeds chunk_size."""
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        if end < text_len:
            newline_idx = text.rfind("\n", start + chunk_size // 2, end)
            if newline_idx != -1:
                end = newline_idx
            else:
                period_idx = text.rfind(". ", start + chunk_size // 2, end)
                if period_idx != -1:
                    end = period_idx + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_len:
            break
        start = max(start + 1, end - overlap)

    return chunks or [text[:chunk_size]]


# ── Automatic LangChain Tool Selector Helper ───────────────────

async def _run_langchain_tool_selector(
    query: str,
    material_id: str | None = None,
    topic: str | None = None,
    chat_history: list[dict[str, Any]] | None = None,
) -> tuple[list[Source], dict[str, str]]:
    """
    Automatic tool selector on pass 0 using LangChain bind_tools from src.rag.rag.
    Detects whether tools like wikipedia_search, web_search, arxiv_search, or supabase should be used.
    Allows the LLM to select multiple tools at once from the very first time.
    Returns (selected_sources, tool_queries).
    """
    selected_sources: list[Source] = []
    tool_queries: dict[str, str] = {}

    try:
        from src.rag.rag import _make_search_tools, get_rag_router_llm, get_summary_llm
        from langchain_core.messages import SystemMessage, HumanMessage

        # Give LLM access to all search tools
        tools = _make_search_tools(is_topic=True)
        
        try:
            router_llm = get_rag_router_llm()
        except Exception:
            router_llm = get_summary_llm()

        topic_info = f"\nTopic is: {topic}\n" if topic else ""
        history_info = f"\nRecent conversation context:\n{format_chat_history(chat_history, limit=8)}\n" if chat_history else ""

        llm_with_tools = router_llm.bind_tools(tools)
        messages = [
            SystemMessage(content=(
                "You are an expert AI educational research router.\n"
                "Select search tools to gather background information for a user query.\n\n"
                "Tool Selection Policy:\n"
                "1. Single Tool Priority: If the query fits cleanly into one domain, stick to that ONE tool:\n"
                "   - Straight academic, AI/ML, math, physics, or research theory -> Call ONLY arxiv_search.\n"
                "   - Straight foundational, historical, encyclopedic concepts, or broad definitions -> Call ONLY wikipedia_search.\n"
                "   - Straight real-time, current updates, recent software releases, or practical web guides -> Call ONLY web_search.\n\n"
                "2. Overlapping Multi-Tool Rule: If the query spans multiple categories or overlaps between domains "
                "(e.g., cutting-edge academic theory applied in modern libraries, historical background overlapping with recent AI advances, "
                "or research papers requiring broader web context), you CAN and SHOULD select multiple tools at once (e.g. arxiv_search + web_search).\n\n"
                "3. Pronoun & Topic Resolution: If the query uses ambiguous pronouns (it, they, he, she, this, that) or asks to define/explain 'our topic', "
                "resolve the subject using the provided topic and conversation history, and formulate tool queries for that concrete subject.\n\n"
                "4. Args: For each selected tool, generate a clean, concise, keyword-rich search query in args (no LaTeX, no punctuation, max 10 words).\n"
                "5. Conversational / greetings: Emit 0 tool calls."
            )),
            HumanMessage(content=f'User query to research: "{query}"\n{topic_info}{history_info}'),
        ]
        resp = await llm_with_tools.ainvoke(messages)
        tool_calls = getattr(resp, "tool_calls", []) or []

        for call in tool_calls:
            name = call.get("name", "")
            args = call.get("args", {})
            sub_query = (args.get("query") or "").strip() or query
            if name == "wikipedia_search":
                if "wikipedia" not in selected_sources:
                    selected_sources.append("wikipedia")
                tool_queries["wikipedia"] = sub_query
            elif name == "web_search":
                if "web" not in selected_sources:
                    selected_sources.append("web")
                tool_queries["web"] = sub_query
            elif name == "arxiv_search":
                if "arxiv" not in selected_sources:
                    selected_sources.append("arxiv")
                tool_queries["arxiv"] = sub_query

    except Exception as exc:
        logger.debug(f"[router] Automatic LangChain tool selector skipped or failed: {exc}")

    # Fallback if no external tools selected for a topic
    if not selected_sources and not material_id:
        academic_keywords = {"llm", "lora", "quantum", "model", "neural", "transformer", "algorithm", "deep learning", "ai", "physics", "math", "genomics", "arxiv", "bellman", "reinforcement"}
        q_lower = query.lower()
        if any(kw in q_lower for kw in academic_keywords):
            selected_sources = ["arxiv", "web"]
            tool_queries["arxiv"] = query
            tool_queries["web"] = query
        else:
            selected_sources = ["wikipedia", "web"]
            tool_queries["wikipedia"] = query
            tool_queries["web"] = query

    # If material_id is provided, always include supabase as the primary document source
    if material_id and "supabase" not in selected_sources:
        selected_sources.insert(0, "supabase")
        tool_queries["supabase"] = query

    return selected_sources, tool_queries


# ── Node Implementations ───────────────────────────────────────

_KEYBOARD_ROWS = ("asdf", "qwer", "zxcv", "hjkl", "sdfg", "dfgh", "qazwsx")

_CHITCHAT_REGEX = re.compile(
    r"^\s*("
    r"hi(\s+there)?"
    r"|hey(\s+there)?"
    r"|hello(\s+there)?"
    r"|yo"
    r"|thanks?(\s+you)?"
    r"|thank\s+you(\s+so\s+much)?"
    r"|ok(ay)?"
    r"|bye(\s+bye)?"
    r"|good\s+(morning|afternoon|evening|night|day)"
    r"|مرحبا"
    r"|أ?اهلا(ً|\s+وسهلا)?"
    r"|السلام\s+عليكم(\s+ورحمة\s+الله(\s+وبركاته)?)?"
    r"|شكرا(ً)?(\s+جزيلا)?"
    r"|تمام"
    r"|باي"
    r"|مع\s+السلامة"
    r")[\s\.,!؟?~:;()\-]*$",
    re.IGNORECASE | re.UNICODE,
)


def tier0(q: str) -> str | None:
    """
    Tier 0: Pure Python, zero LLM calls, ~0 ms.
    Returns 'chitchat', 'unsafe_gibberish', or None (unsure -> tier 1).
    """
    raw = q.strip()
    if len(raw) < 2:
        return "chitchat"

    # Punctuation/whitespace stripped check
    cleaned = re.sub(r"^[\s\.,!؟?~:;()\-]+|[\s\.,!؟?~:;()\-]+$", "", raw)

    if _CHITCHAT_REGEX.match(raw) or _CHITCHAT_REGEX.match(cleaned):
        return "chitchat"

    # Gibberish check: keyboard row runs
    q_lower = raw.lower()
    for row in _KEYBOARD_ROWS:
        if row in q_lower:
            return "unsafe_gibberish"

    # 5+ repeated characters
    if re.search(r"(.)\1{4,}", raw):
        return "unsafe_gibberish"

    return None


def get_chitchat_reply(q: str) -> str:
    is_arabic = bool(re.search(r"[\u0600-\u06FF]", q))
    lang = "ar" if is_arabic else "en"
    q_lower = q.lower()
    if any(w in q_lower for w in ("thank", "شكرا", "شكر")):
        category = "thanks"
    elif any(w in q_lower for w in ("bye", "باي", "مع السلامة")):
        category = "bye"
    else:
        category = "greeting"
    return CHITCHAT_REPLIES.get(category, CHITCHAT_REPLIES["default"])[lang]


def create_triage_node(
    llm: LLM | None = None,
    settings: Settings | None = None,
) -> Callable[[GraphState], Any]:
    """Node 0: triage."""
    cfg = settings or default_settings
    llm_instance = llm or LangChainLLM()

    async def triage(state: GraphState) -> dict[str, Any]:
        t0 = time.perf_counter()
        original_q = state.get("original_q", "")

        async def _emit_intent(resolved_intent: str) -> None:
            cb = state.get("on_intent")
            if cb is not None:
                try:
                    res = cb(resolved_intent)
                    if asyncio.iscoroutine(res):
                        await res
                except Exception as cb_err:
                    logger.debug(f"[triage] on_intent callback error: {cb_err}")

        # Tier 0 check
        t0_res = tier0(original_q)
        if t0_res == "chitchat":
            reply = get_chitchat_reply(original_q)
            latency_ms = (time.perf_counter() - t0) * 1000
            _log_node("triage", state, tier=0, intent="chitchat", unsafe_type="none", n_sub=0, latency_ms=f"{latency_ms:.1f}")
            await _emit_intent("chitchat")
            return {
                "intent": "chitchat",
                "sub_questions": [],
                "reply": reply,
            }
        elif t0_res == "unsafe_gibberish":
            reply = UNSAFE_REPLIES["gibberish"]
            latency_ms = (time.perf_counter() - t0) * 1000
            _log_node("triage", state, tier=0, intent="unsafe", unsafe_type="gibberish", n_sub=0, latency_ms=f"{latency_ms:.1f}")
            await _emit_intent("unsafe")
            return {
                "intent": "unsafe",
                "sub_questions": [],
                "reply": reply,
            }

        # Tier 1: LLM structured call
        topic = state.get("topic")
        chat_history = state.get("chat_history", [])
        topic_context = f"<topic>Topic is: {topic}</topic>\n" if topic else ""
        history_context = f"<conversation_history>\n{format_chat_history(chat_history, limit=8)}\n</conversation_history>\n"
        clean_q = re.sub(r"</?\s*user_query\s*>", "", original_q, flags=re.IGNORECASE).strip()
        prompt = (
            f"{TRIAGE_SYSTEM_PROMPT}\n\n"
            f"{topic_context}"
            f"{history_context}"
            f"<user_query>{clean_q}</user_query>\n\n"
            f"Classify the user query. If the question uses ambiguous pronouns (it, they, he, she, this, that) or asks to define/explain our topic, resolve it using the topic name and conversation history (classify questions defining our topic as simple).\n"
            f"Respond with EXACTLY this format:\n"
            f"intent: chitchat | unsafe | simple | complex\n"
            f"unsafe_type: none | gibberish | nsfw | political | religious\n"
            f"sub_questions: <none, or list 1 to 3 queries each on a new line prefixed with ->"
        )

        output: TriageOutput | None = None
        try:
            output = await asyncio.wait_for(
                llm_instance.ainvoke_structured(prompt, TriageOutput),
                timeout=cfg.TRIAGE_TIMEOUT_S,
            )
        except Exception as first_exc:
            err1 = f"{type(first_exc).__name__}: {first_exc}" if str(first_exc) else type(first_exc).__name__
            logger.warning(f"[triage] Tier 1 attempt 1 failed ({err1}). Retrying once...")
            try:
                output = await asyncio.wait_for(
                    llm_instance.ainvoke_structured(prompt, TriageOutput),
                    timeout=cfg.TRIAGE_TIMEOUT_S,
                )
            except Exception as second_exc:
                err2 = f"{type(second_exc).__name__}: {second_exc}" if str(second_exc) else type(second_exc).__name__
                logger.warning(f"[triage] Tier 1 attempt 2 failed ({err2}). Failing toward complex pipeline.")
                output = TriageOutput.model_construct(
                    intent="complex",
                    unsafe_type="none",
                    sub_questions=[original_q],
                )

        if output.intent == "complex" and output.sub_questions == ["complex study query"]:
            output = TriageOutput.model_construct(
                intent="complex",
                unsafe_type="none",
                sub_questions=[original_q[:200]],
            )

        if output.intent == "chitchat":
            reply = get_chitchat_reply(original_q)
        elif output.intent == "unsafe":
            reply = UNSAFE_REPLIES.get(output.unsafe_type, UNSAFE_REPLIES["gibberish"])
        else:
            reply = None

        latency_ms = (time.perf_counter() - t0) * 1000
        tracer: ChatTracer | None = state.get("tracer")
        if tracer:
            tracer.log_llm_call(
                model=getattr(llm_instance, "model_name", None) or "ministral-8b-latest",
                prompt_text=prompt,
                model_settings={"temperature": 0.0},
                usage=getattr(llm_instance, "last_usage", None),
                latency_ms=latency_ms,
                role="triage",
                output_text=f"intent={output.intent}, unsafe_type={output.unsafe_type}, sub_questions={output.sub_questions}",
            )
            tracer.log_node(
                name="node/triage",
                run_type="chain",
                inputs={"original_q": original_q},
                outputs={
                    "intent": output.intent,
                    "unsafe_type": output.unsafe_type,
                    "sub_questions": output.sub_questions,
                },
                latency_ms=latency_ms,
            )

        _log_node(
            "triage",
            state,
            tier=1,
            intent=output.intent,
            unsafe_type=output.unsafe_type,
            n_sub=len(output.sub_questions),
            latency_ms=f"{latency_ms:.1f}",
        )
        await _emit_intent(output.intent)
        return {
            "intent": output.intent,
            "sub_questions": output.sub_questions,
            "reply": reply,
        }

    return triage


def create_reply_chitchat_node() -> Callable[[GraphState], Any]:
    """Canned reply node for chitchat."""
    async def reply_chitchat(state: GraphState) -> dict[str, Any]:
        reply = state.get("reply") or CHITCHAT_REPLIES["default"]["en"]
        queue: Any = state.get("token_queue")
        if queue is not None:
            await queue.put(reply)
            await queue.put(None)
        return {
            "answer": reply,
            "citations": [],
            "low_confidence": False,
            "reply": reply,
        }
    return reply_chitchat


def create_reply_refusal_node() -> Callable[[GraphState], Any]:
    """Canned reply node for unsafe refusal."""
    async def reply_refusal(state: GraphState) -> dict[str, Any]:
        reply = state.get("reply") or UNSAFE_REPLIES["gibberish"]
        queue: Any = state.get("token_queue")
        if queue is not None:
            await queue.put(reply)
            await queue.put(None)
        return {
            "answer": reply,
            "citations": [],
            "low_confidence": False,
            "reply": reply,
        }
    return reply_refusal


def triage_gate(state: GraphState) -> str:
    """Conditional routing from triage node."""
    intent = state.get("intent")
    if intent == "chitchat":
        return "reply_chitchat"
    elif intent == "unsafe":
        return "reply_refusal"
    return "init_state"


def create_init_state_node() -> Callable[[GraphState], Any]:
    """Node 1: init_state."""
    async def init_state(state: GraphState) -> dict[str, Any]:
        await asyncio.sleep(0)
        t0 = time.perf_counter()
        original_q = state.get("original_q", "").strip()
        if len(original_q) < MIN_QUESTION_LEN:
            raise ValueError(f"original_q must be at least {MIN_QUESTION_LEN} characters")
        if len(original_q) > MAX_QUESTION_LEN:
            original_q = original_q[:MAX_QUESTION_LEN]

        latency_ms = (time.perf_counter() - t0) * 1000
        tracer: ChatTracer | None = state.get("tracer")
        if tracer:
            tracer.log_node(
                name="node/init_state",
                run_type="chain",
                inputs={"original_q": original_q},
                outputs={"retries": 0, "sources_to_use": ["supabase"]},
                latency_ms=latency_ms,
            )

        _log_node("init_state", state, original_q=original_q[:50], latency_ms=f"{latency_ms:.1f}")
        return {
            "original_q": original_q,
            "intent": state.get("intent", "simple"),
            "sub_questions": state.get("sub_questions", []),
            "search_query": original_q,
            "sources_to_use": ["supabase"],
            "tried_sources": [],
            "pool": [],
            "ranked": [],
            "verdict": None,
            "missing": "",
            "likely_cause": "none",
            "retries": 0,
            "answer": None,
            "low_confidence": False,
            "citations": [],
            "raw_results": [],
            "new_chunks": [],
            "all_retrievers_failed": False,
            "tracer": state.get("tracer"),
            "material_id": state.get("material_id"),
            "token_queue": state.get("token_queue"),
            "topic": state.get("topic"),
            "chat_history": state.get("chat_history", []),
        }

    return init_state


def create_router_node(
    llm: LLM | None = None,
    settings: Settings | None = None,
) -> Callable[[GraphState], Any]:
    """
    Node 2: router.
    Pass 0: Complex intent runs all tools concurrently (0 LLM call); simple intent runs automatic tool calling.
    On retry: Requerying only with fixed sources (rewrites query targeting missing info).
    """
    cfg = settings or default_settings
    llm_instance = llm or LangChainLLM()

    async def router(state: GraphState) -> dict[str, Any]:
        t0 = time.perf_counter()
        tracer: ChatTracer | None = state.get("tracer")
        retries = state.get("retries", 0)
        original_q = state["original_q"]
        tried = list(state.get("tried_sources", []))
        missing = state.get("missing", "")
        likely_cause = state.get("likely_cause", "none")

        # Pass 0
        if retries == 0 and not tried:
            intent = state.get("intent", "simple")
            sub_questions = state.get("sub_questions", [])
            material_id = state.get("material_id")

            # 1. Complex query: use ALL tools concurrently!
            if intent == "complex":
                pass0_sources: list[Source] = ["supabase", "arxiv", "wikipedia", "web"]
                q_list = sub_questions if sub_questions else [original_q]
                chosen_query = q_list[0] if q_list else original_q
                tool_queries = {s: chosen_query for s in pass0_sources}
                latency_ms = (time.perf_counter() - t0) * 1000
                if tracer:
                    tracer.log_node(
                        name="node/router",
                        run_type="chain",
                        inputs={"original_q": original_q, "pass": 0, "mode": "complex_all_tools"},
                        outputs={"sources_to_use": pass0_sources, "search_query": chosen_query, "tool_queries": tool_queries},
                        latency_ms=latency_ms,
                    )
                _log_node("router", state, decision=f"pass_0 (complex -> all_tools) -> {pass0_sources}", latency_ms=f"{latency_ms:.1f}")
                return {
                    "sources_to_use": pass0_sources,
                    "search_query": chosen_query,
                    "tool_queries": tool_queries,
                }

            # 2. Simple query: automatic tool calling
            topic = state.get("topic")
            chat_history = state.get("chat_history", [])
            auto_sources, tool_queries = await _run_langchain_tool_selector(
                original_q, material_id=material_id, topic=topic, chat_history=chat_history
            )
            if auto_sources:
                latency_ms = (time.perf_counter() - t0) * 1000
                if tracer:
                    tracer.log_llm_call(
                        model=cfg.LLM_MODEL,
                        prompt_text=f"User query to research: '{original_q}'",
                        model_settings={"temperature": 0.0},
                        usage=getattr(llm_instance, "last_usage", None),
                        latency_ms=latency_ms,
                        role="router",
                        output_text=f"selected_sources={auto_sources}, tool_queries={tool_queries}",
                    )
                    tracer.log_node(
                        name="node/router",
                        run_type="chain",
                        inputs={"original_q": original_q, "pass": 0, "mode": "langchain_tools"},
                        outputs={"sources_to_use": auto_sources, "search_query": original_q, "tool_queries": tool_queries},
                        latency_ms=latency_ms,
                    )
                _log_node("router", state, decision=f"pass_0 (simple -> automatic) -> {auto_sources}", latency_ms=f"{latency_ms:.1f}")
                return {
                    "sources_to_use": auto_sources,
                    "search_query": original_q,
                    "tool_queries": tool_queries,
                }

            fallback_sources: list[Source] = ["supabase"] if material_id else ["wikipedia", "web"]
            latency_ms = (time.perf_counter() - t0) * 1000
            if tracer:
                tracer.log_node(
                    name="node/router",
                    run_type="chain",
                    inputs={"original_q": original_q, "pass": 0, "mode": "simple_fallback"},
                    outputs={"sources_to_use": fallback_sources, "search_query": original_q},
                    latency_ms=latency_ms,
                )
            _log_node("router", state, decision=f"pass_0 (simple -> fallback) -> {fallback_sources}", latency_ms=f"{latency_ms:.1f}")
            return {
                "sources_to_use": fallback_sources,
                "search_query": original_q,
                "tool_queries": {s: original_q for s in fallback_sources},
            }

        # On retry:
        retry_reason = "requery"
        intent = state.get("intent", "simple")
        topic = state.get("topic")
        chat_history = state.get("chat_history", [])
        topic_info = f"<topic>Topic is: {topic}</topic>\n" if topic else ""
        history_info = f"<conversation_history>\n{format_chat_history(chat_history, limit=8)}\n</conversation_history>\n"
        logger.info(f"[router] Retry pass {retries}: reason='requery' (missing='{missing}', intent='{intent}', topic='{topic}')")
        
        # For simple query retry (score was <= simple threshold):
        if intent == "simple":
            # Must retry with ALL tools instead of auto select, with a keyword-style search query
            sources_to_use: list[Source] = ["supabase", "arxiv", "wikipedia", "web"]
            simple_thresh = getattr(cfg, "SIMPLE_THRESHOLD", DEFAULT_SIMPLE_THRESHOLD)
            user_msg = (
                f"<original_question>{original_q}</original_question>\n"
                f"<previous_query>{state.get('search_query', original_q)}</previous_query>\n"
                f"<missing>{missing or f'Re-rank score was below simple threshold ({simple_thresh}); need targeted factual keywords.'}</missing>\n"
                f"<retry_number>{retries}</retry_number>"
            )
            prompt = (
                f"{ROUTER_SYSTEM_PROMPT}\n\n"
                f"{topic_info}"
                f"{history_info}"
                f"{user_msg}\n\n"
                f"Rewrite this simple question into concise, keyword-style search terms optimized for arXiv and search engines "
                f"(remove conversational phrases and question frames; extract 3 to 6 high-value search keywords). "
                f"If the question refers to 'our topic', 'the topic', or uses ambiguous pronouns (it, they, he, she, this, that, its), "
                f"resolve it using the topic name and conversation history to formulate search keywords for that concrete topic.\n"
                f"Return EXACTLY in this format:\n"
                f"SEARCH_QUERY: <keyword1 keyword2 ...>"
            )
        else:
            # Complex query: keep existing sources as before
            sources_to_use = list(state.get("sources_to_use") or ["supabase", "arxiv", "wikipedia", "web"])
            user_msg = (
                f"<original_question>{original_q}</original_question>\n"
                f"<previous_query>{state.get('search_query', original_q)}</previous_query>\n"
                f"<missing>{missing or 'additional context needed'}</missing>\n"
                f"<retry_number>{retries}</retry_number>"
            )
            prompt = (
                f"{ROUTER_SYSTEM_PROMPT}\n\n"
                f"{topic_info}"
                f"{history_info}"
                f"{user_msg}\n\n"
                f"Write a focused search query targeting the missing information. "
                f"If the question refers to 'our topic', 'the topic', or uses ambiguous pronouns, "
                f"resolve it using the topic name and conversation history.\n"
                f"Return EXACTLY in this format:\n"
                f"SEARCH_QUERY: <search query keywords>"
            )

        new_search_query = original_q
        try:
            res = await llm_instance.ainvoke_structured(prompt, RouterOutput)
            if res and res.search_query.strip():
                new_search_query = res.search_query.strip()
                if len(new_search_query) > 300:
                    new_search_query = new_search_query[:300].rstrip()
        except Exception as exc:
            logger.warning(
                f"[router] Router query-rewrite LLM failed ({exc}). Using missing/original fallback."
            )
            if intent == "simple":
                clean = re.sub(r"\b(what|is|are|the|explain|how|why|tell|me|about|can|you|define|describe|does)\b", "", original_q, flags=re.IGNORECASE)
                words = [w for w in re.findall(r"\w+", clean) if len(w) > 2]
                new_search_query = " ".join(words[:6]) if words else original_q
            elif missing and len(missing) >= 3:
                new_search_query = missing[:300].strip()

        if len(new_search_query) > 300:
            new_search_query = new_search_query[:300].rstrip()

        latency_ms = (time.perf_counter() - t0) * 1000
        tool_queries = {s: new_search_query for s in sources_to_use}
        if tracer:
            tracer.log_llm_call(
                model=getattr(llm_instance, "model_name", None) or cfg.LLM_MODEL,
                prompt_text=prompt,
                model_settings={"temperature": 0.0},
                usage=getattr(llm_instance, "last_usage", None),
                latency_ms=latency_ms,
                role="router",
                output_text=f"sources={sources_to_use}, query={new_search_query}",
            )
            tracer.log_node(
                name="node/router",
                run_type="chain",
                inputs={
                    "original_q": original_q,
                    "retries": retries,
                    "missing": missing,
                    "sources_to_use": sources_to_use,
                },
                outputs={
                    "sources_to_use": sources_to_use,
                    "search_query": new_search_query,
                    "tool_queries": tool_queries,
                },
                latency_ms=latency_ms,
            )

        _log_node(
            "router",
            state,
            reason="requery",
            sources=sources_to_use,
            search_query=new_search_query[:40],
            latency_ms=f"{latency_ms:.1f}",
        )
        return {
            "sources_to_use": sources_to_use,
            "search_query": new_search_query,
            "tool_queries": tool_queries,
        }

    return router


def create_retrieve_node(
    tools: dict[Source, RetrieverFn] | None = None,
    settings: Settings | None = None,
) -> Callable[[GraphState], Any]:
    """Node 3: retrieve."""
    cfg = settings or default_settings
    tool_map = tools or TOOL_REGISTRY
    default_fetch_count = cfg.TOP_K * cfg.RETRIEVE_MULTIPLIER

    def _get_fetch_count(src: Source) -> int:
        if src == "wikipedia":
            return WIKI_TOP_K_RESULTS
        elif src == "arxiv":
            return ARXIV_TOP_K_RESULTS
        elif src == "web":
            return DUCKDUCKGO_NUM_RESULTS
        return default_fetch_count

    def _get_tool_timeout(src: Source, intent: str | None) -> float:
        is_complex = (intent == "complex")
        simple_timeout = getattr(cfg, "TOOL_TIMEOUT_SIMPLE_S", DEFAULT_TOOL_TIMEOUT_SIMPLE_S)
        complex_timeout = getattr(cfg, "TOOL_TIMEOUT_COMPLEX_S", DEFAULT_TOOL_TIMEOUT_COMPLEX_S)
        return complex_timeout if is_complex else simple_timeout

    async def retrieve(state: GraphState) -> dict[str, Any]:
        t0 = time.perf_counter()
        tracer: ChatTracer | None = state.get("tracer")
        sources_to_use = state.get("sources_to_use", ["supabase"])
        search_query = state.get("search_query", state["original_q"])
        tool_queries = state.get("tool_queries", {}) or {}
        intent = state.get("intent")

        async def _run_tool(src: Source) -> list[dict[str, Any]]:
            retriever = tool_map.get(src)
            if not retriever:
                logger.warning(f"[retrieve] Unknown retriever tool: {src}")
                return []
            count = _get_fetch_count(src)
            q = tool_queries.get(src) or search_query
            tool_timeout = _get_tool_timeout(src, intent)
            try:
                return await asyncio.wait_for(
                    retriever(q, count),
                    timeout=tool_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"[retrieve] Tool '{src}' timed out after {tool_timeout}s (intent={intent or 'simple'})"
                )
                return []
            except Exception as e:
                logger.warning(f"[retrieve] Tool '{src}' failed: {e}")
                return []

        tasks = [_run_tool(s) for s in sources_to_use]
        results_nested = await asyncio.gather(*tasks, return_exceptions=True)

        raw_results: list[dict[str, Any]] = []
        all_failed = True
        for idx, res in enumerate(results_nested):
            if isinstance(res, list):
                if res:
                    all_failed = False
                raw_results.extend(res)
            elif isinstance(res, Exception):
                logger.warning(f"[retrieve] Error running tool {sources_to_use[idx]}: {res}")

        latency_ms = (time.perf_counter() - t0) * 1000
        if tracer:
            tracer.log_retrieval(
                query=search_query,
                chunks=[r.get("text", "") for r in raw_results if r.get("text")],
                scores=[],
                tool_chosen=",".join(sources_to_use),
            )
            tracer.log_node(
                name="node/retrieve",
                run_type="tool",
                inputs={
                    "search_query": search_query,
                    "sources_to_use": sources_to_use,
                },
                outputs={
                    "fetched_count": len(raw_results),
                    "all_retrievers_failed": all_failed and len(raw_results) == 0,
                    "sources_breakdown": {s: sum(1 for r in raw_results if r.get("source") == s) for s in sources_to_use},
                },
                latency_ms=latency_ms,
            )

        _log_node("retrieve", state, fetched_count=len(raw_results), all_failed=all_failed, latency_ms=f"{latency_ms:.1f}")
        return {
            "raw_results": raw_results,
            "all_retrievers_failed": all_failed and len(raw_results) == 0,
        }

    return retrieve


def create_normalize_node() -> Callable[[GraphState], Any]:
    """Node 4: normalize."""
    async def normalize(state: GraphState) -> dict[str, Any]:
        await asyncio.sleep(0)
        t0 = time.perf_counter()
        tracer: ChatTracer | None = state.get("tracer")
        raw_results: list[dict[str, Any]] = state.get("raw_results") or []
        sources_to_use: list[Source] = state.get("sources_to_use", [])
        existing_tried: list[Source] = list(state.get("tried_sources", []))

        new_tried = list(dict.fromkeys(existing_tried + sources_to_use))
        new_chunks: list[Chunk] = []
        seen_ids: set[str] = set()

        for item in raw_results:
            raw_text = item.get("text", "")
            cleaned = _clean_text(raw_text)
            if not cleaned:
                continue

            src: Source = item.get("source", "web")
            url = item.get("url")
            title = item.get("title")

            valid_url: str | None = None
            if url and (url.startswith("http://") or url.startswith("https://")):
                valid_url = url

            if src == "supabase":
                passages = [cleaned]
            else:
                passages = _chunk_text(cleaned, chunk_size=1500, overlap=150)
            for p in passages:
                if not p.strip():
                    continue
                p_text = p.strip()[:MAX_CHUNK_CHARS]
                chunk_id = hashlib.sha256(f"{src}:{p_text}".encode("utf-8")).hexdigest()[:16]
                if chunk_id in seen_ids:
                    continue
                seen_ids.add(chunk_id)

                try:
                    chunk = Chunk(
                        id=chunk_id,
                        text=p_text,
                        source=src,
                        url=valid_url,
                        title=title,
                        score=None,
                    )
                    new_chunks.append(chunk)
                except Exception as exc:
                    logger.debug(f"[normalize] Skipping chunk due to validation error: {exc}")

        latency_ms = (time.perf_counter() - t0) * 1000
        if tracer:
            tracer.log_node(
                name="node/normalize",
                run_type="tool",
                inputs={"raw_results_count": len(raw_results), "sources": sources_to_use},
                outputs={
                    "normalized_chunks_count": len(new_chunks),
                    "total_tried_sources": new_tried,
                    "chunk_ids_sample": [c.id for c in new_chunks[:5]],
                },
                latency_ms=latency_ms,
            )

        _log_node("normalize", state, normalized_chunks=len(new_chunks), total_tried=len(new_tried), latency_ms=f"{latency_ms:.1f}")
        return {
            "new_chunks": new_chunks,
            "tried_sources": new_tried,
        }

    return normalize


def create_merge_pool_node(settings: Settings | None = None) -> Callable[[GraphState], Any]:
    """Node 5: merge_pool (selective carry-over)."""
    cfg = settings or default_settings

    async def merge_pool(state: GraphState) -> dict[str, Any]:
        await asyncio.sleep(0)
        t0 = time.perf_counter()
        tracer: ChatTracer | None = state.get("tracer")
        retries = state.get("retries", 0)
        verdict = state.get("verdict")
        old_pool: list[Chunk] = list(state.get("pool", []))
        new_chunks: list[Chunk] = list(state.get("new_chunks") or state.get("_new_chunks") or [])

        # Append new chunks to old pool without discarding previous sources
        merged = old_pool + new_chunks
        carried_count = len(old_pool)

        deduped: dict[str, Chunk] = {}
        for c in merged:
            if c.id not in deduped:
                deduped[c.id] = c
            else:
                existing = deduped[c.id]
                if c.score is not None and (existing.score is None or c.score > existing.score):
                    deduped[c.id] = c

        final_list = list(deduped.values())
        final_list.sort(
            key=lambda c: (c.score is None, c.score or 0.0),
            reverse=True,
        )
        capped_pool = final_list[: cfg.POOL_MAX]

        latency_ms = (time.perf_counter() - t0) * 1000
        if tracer:
            tracer.log_node(
                name="node/merge_pool",
                run_type="chain",
                inputs={
                    "verdict": verdict,
                    "incoming_new_chunks": len(new_chunks),
                    "old_pool_size": len(old_pool),
                },
                outputs={
                    "carried_old_count": carried_count,
                    "deduped_total": len(deduped),
                    "capped_pool_size": len(capped_pool),
                },
                latency_ms=latency_ms,
            )

        _log_node("merge_pool", state, verdict=verdict, carried=len(capped_pool), latency_ms=f"{latency_ms:.1f}")
        return {"pool": capped_pool}

    return merge_pool


def create_rerank_node(
    reranker: Reranker | None = None,
    settings: Settings | None = None,
    llm: LLM | None = None,
) -> Callable[[GraphState], Any]:
    """Node 6: rerank."""
    cfg = settings or default_settings
    reranker_instance = reranker or CascadeReranker(settings=cfg, llm=llm)

    async def rerank(state: GraphState) -> dict[str, Any]:
        t0 = time.perf_counter()
        tracer: ChatTracer | None = state.get("tracer")
        pool = list(state.get("pool", []))
        original_q = state["original_q"]

        if not pool:
            latency_ms = (time.perf_counter() - t0) * 1000
            if tracer:
                tracer.log_reranker(
                    query=original_q,
                    num_candidates=0,
                    top_k=cfg.FINAL_CONTEXT_N,
                    latency_ms=latency_ms,
                    scores=[],
                    model=getattr(reranker_instance, "last_tier", cfg.RERANKER_MODEL),
                )
                tracer.log_node(
                    name="node/rerank",
                    run_type="retriever",
                    inputs={"original_q": original_q, "pool_size": 0},
                    outputs={"ranked": [], "top_score": 0.0},
                    latency_ms=latency_ms,
                )
            _log_node("rerank", state, top_score=0.0, ranked_count=0, latency_ms=f"{latency_ms:.1f}")
            return {"ranked": [], "pool": []}

        scored_pool: list[Chunk] = []
        unscored_chunks: list[Chunk] = []
        for c in pool:
            if c.score is not None:
                scored_pool.append(c)
            else:
                unscored_chunks.append(c)

        multiplier = getattr(cfg, "RETRIEVE_MULTIPLIER", 3.0)
        max_candidates = max(12, int(round(cfg.FINAL_CONTEXT_N * multiplier)))
        to_score = unscored_chunks[:max_candidates]
        leftover_unscored = unscored_chunks[max_candidates:]

        if to_score:
            texts_to_score = [c.text[:MAX_CHUNK_CHARS] for c in to_score]
            try:
                if hasattr(reranker_instance, "score_async"):
                    scores = await reranker_instance.score_async(
                        original_q,
                        texts_to_score,
                    )
                else:
                    scores = await asyncio.to_thread(
                        reranker_instance.score,
                        original_q,
                        texts_to_score,
                    )
            except Exception as exc:
                logger.warning(
                    f"[rerank] Reranker failed: {exc}. Falling back to retrieval order with default scores."
                )
                scores = [max(0.1, 0.5 - (i * 0.01)) for i in range(len(to_score))]

            for chunk, score in zip(to_score, scores):
                clamped = max(0.0, min(1.0, float(score)))
                scored_pool.append(
                    Chunk(
                        id=chunk.id,
                        text=chunk.text,
                        source=chunk.source,
                        url=chunk.url,
                        title=chunk.title,
                        score=clamped,
                    )
                )

        for i, chunk in enumerate(leftover_unscored):
            scored_pool.append(
                Chunk(
                    id=chunk.id,
                    text=chunk.text,
                    source=chunk.source,
                    url=chunk.url,
                    title=chunk.title,
                    score=max(0.05, 0.2 - (i * 0.01)),
                )
            )

        scored_pool.sort(key=lambda c: c.score or 0.0, reverse=True)

        # Guarantee top 2 chunks from each tool/source present are preserved in ranked context
        guaranteed_chunks: list[Chunk] = []
        guaranteed_ids: set[str] = set()

        sources_present = list(dict.fromkeys(c.source for c in scored_pool))
        for src in sources_present:
            src_chunks = [c for c in scored_pool if c.source == src]
            for c in src_chunks[:2]:
                if c.id not in guaranteed_ids:
                    guaranteed_chunks.append(c)
                    guaranteed_ids.add(c.id)

        other_chunks = [c for c in scored_pool if c.id not in guaranteed_ids]
        remaining_slots = max(0, cfg.FINAL_CONTEXT_N - len(guaranteed_chunks))
        selected_others = other_chunks[:remaining_slots]

        ranked = guaranteed_chunks + selected_others
        ranked.sort(key=lambda c: c.score or 0.0, reverse=True)

        top_score = ranked[0].score if ranked and ranked[0].score is not None else 0.0
        latency_ms = (time.perf_counter() - t0) * 1000

        used_model = getattr(reranker_instance, "last_tier", cfg.RERANKER_MODEL)

        if tracer:
            all_scores = [c.score for c in scored_pool if c.score is not None]
            tracer.log_reranker(
                query=original_q,
                num_candidates=len(pool),
                top_k=cfg.FINAL_CONTEXT_N,
                latency_ms=latency_ms,
                scores=all_scores,
                model=used_model,
            )
            tracer.log_node(
                name="node/rerank",
                run_type="retriever",
                inputs={"original_q": original_q, "candidate_chunks_count": len(pool)},
                outputs={
                    "ranked_count": len(ranked),
                    "top_score": round(top_score, 4),
                    "ranked_chunks": [
                        {"id": c.id, "score": round(c.score or 0.0, 4), "source": c.source, "preview": c.text[:120]}
                        for c in ranked
                    ],
                },
                latency_ms=latency_ms,
            )

        _log_node("rerank", state, top_score=f"{top_score:.3f}", ranked_count=len(ranked), latency_ms=f"{latency_ms:.1f}")
        return {
            "pool": scored_pool,
            "ranked": ranked,
        }

    return rerank


def score_gate(state: GraphState, settings: Settings | None = None) -> str:
    """Conditional edge function after rerank. No LLM call."""
    cfg = settings or default_settings
    ranked = state.get("ranked", [])
    top = ranked[0].score if ranked and ranked[0].score is not None else 0.0
    intent = state.get("intent")

    # Use round(top, 3) so 0.7995+ (displayed as 0.800) routes to generate
    top_rounded = round(top, 3)

    if intent == "simple":
        # Simple query path: depend on re-rank score only and bypass evaluator node.
        # - Above simple threshold (default 0.35) -> generate
        # - Below simple threshold -> retry_check (retries with all tools & keyword query)
        simple_thresh = getattr(cfg, "SIMPLE_THRESHOLD", DEFAULT_SIMPLE_THRESHOLD)
        if top_rounded > simple_thresh:
            decision = "generate"
        else:
            decision = "retry_check"
    else:
        # Complex query path (preserved exactly as before):
        if top_rounded >= cfg.HIGH_THRESHOLD:
            decision = "generate"
        elif top_rounded <= cfg.LOW_THRESHOLD:
            decision = "retry_check"
        else:
            decision = "evaluator"

    logger.info(f"[score_gate] intent={intent} top_score={top:.3f} -> {decision}")
    return decision


def create_evaluator_node(
    llm: LLM | None = None,
) -> Callable[[GraphState], Any]:
    """Node 8: evaluator."""
    llm_instance = llm or LangChainLLM()

    async def evaluator(state: GraphState) -> dict[str, Any]:
        t0 = time.perf_counter()
        tracer: ChatTracer | None = state.get("tracer")
        original_q = state["original_q"]
        ranked = state.get("ranked", [])

        user_msg = evaluator_user_message({
            "original_q": original_q,
            "ranked": ranked,
        })
        prompt = (
            f"{EVALUATOR_SYSTEM_PROMPT}\n\n"
            f"{user_msg}\n\n"
            f"Evaluate relevance and sufficiency. Respond with EXACTLY three lines:\n"
            f"VERDICT: correct | ambiguous | incorrect\n"
            f"CAUSE: none | bad_query\n"
            f"MISSING: <description of what is missing, or none if verdict is correct>"
        )

        evaluator_output: EvaluatorOutput | None = None
        try:
            evaluator_output = await llm_instance.ainvoke_structured(prompt, EvaluatorOutput)
        except Exception as exc:
            logger.warning(
                f"[evaluator] Evaluator LLM failed ({exc}). Falling back to safe default: ambiguous, bad_query."
            )

        if evaluator_output is None:
            evaluator_output = EvaluatorOutput(
                verdict="ambiguous",
                missing="unknown",
                likely_cause="bad_query",
            )

        latency_ms = (time.perf_counter() - t0) * 1000
        if tracer:
            tracer.log_llm_call(
                model=getattr(llm_instance, "model_name", None) or "ministral-8b-latest",
                prompt_text=prompt,
                model_settings={"temperature": 0.0},
                usage=getattr(llm_instance, "last_usage", None),
                latency_ms=latency_ms,
                role="evaluator",
                input_chunks=ranked,
                output_text=f"verdict={evaluator_output.verdict}, missing={evaluator_output.missing}, likely_cause={evaluator_output.likely_cause}",
            )
            tracer.log_node(
                name="node/evaluator",
                run_type="chain",
                inputs={
                    "original_q": original_q,
                    "input_chunks_count": len(ranked),
                    "retrieved_chunks": [
                        {"id": c.id, "source": c.source, "score": c.score, "preview": c.text[:120]}
                        for c in ranked
                    ],
                },
                outputs={
                    "verdict": evaluator_output.verdict,
                    "missing": evaluator_output.missing,
                    "likely_cause": evaluator_output.likely_cause,
                },
                latency_ms=latency_ms,
            )

        cause = evaluator_output.likely_cause
        retry_reason = "none" if evaluator_output.verdict == "correct" else "requery"
        logger.info(
            f"[evaluator] verdict={evaluator_output.verdict}, reason='{retry_reason}' (cause={cause}), missing='{evaluator_output.missing}'"
        )
        _log_node(
            "evaluator",
            state,
            verdict=evaluator_output.verdict,
            reason=retry_reason,
            missing=evaluator_output.missing[:40],
            cause=evaluator_output.likely_cause,
            latency_ms=f"{latency_ms:.1f}",
        )
        return {
            "verdict": evaluator_output.verdict,
            "missing": evaluator_output.missing,
            "likely_cause": evaluator_output.likely_cause,
        }

    return evaluator


def evaluator_gate(state: GraphState) -> str:
    """Conditional edge function after evaluator."""
    verdict = state.get("verdict")
    decision = "generate" if verdict == "correct" else "retry_check"
    logger.info(f"[evaluator_gate] verdict={verdict} -> {decision}")
    return decision


def create_retry_check_node(settings: Settings | None = None) -> Callable[[GraphState], Any]:
    """Node 9: retry_check."""
    cfg = settings or default_settings

    async def retry_check(state: GraphState) -> dict[str, Any]:
        await asyncio.sleep(0)
        t0 = time.perf_counter()
        tracer: ChatTracer | None = state.get("tracer")
        current_retries = state.get("retries", 0)
        new_retries = current_retries + 1

        missing = state.get("missing", "")
        cause = state.get("likely_cause", "bad_query")

        if not missing or state.get("verdict") is None:
            missing = "Context relevance score below low threshold"
            cause = "bad_query"

        if state.get("all_retrievers_failed") and not state.get("pool"):
            new_retries = cfg.MAX_RETRIES + 1
            missing = "All retrieval sources failed or returned empty results"
            cause = "bad_query"

        retry_reason = "requery"
        logger.info(
            f"[retry_check] Retry check: retries={current_retries}->{new_retries}, reason='requery' (cause={cause})"
        )

        latency_ms = (time.perf_counter() - t0) * 1000
        if tracer:
            tracer.log_node(
                name="node/retry_check",
                run_type="chain",
                inputs={"current_retries": current_retries, "max_retries": cfg.MAX_RETRIES, "missing": missing, "cause": cause, "reason": retry_reason},
                outputs={"new_retries": new_retries, "will_retry": new_retries <= cfg.MAX_RETRIES, "reason": retry_reason},
                latency_ms=latency_ms,
            )

        _log_node("retry_check", state, reason=retry_reason, old_retries=current_retries, new_retries=new_retries, latency_ms=f"{latency_ms:.1f}")
        return {
            "retries": new_retries,
            "missing": missing,
            "likely_cause": cause,
        }

    return retry_check


def retry_check_gate(state: GraphState, settings: Settings | None = None) -> str:
    """Conditional edge function after retry_check."""
    cfg = settings or default_settings
    retries = state.get("retries", 0)
    decision = "router" if retries <= cfg.MAX_RETRIES else "fallback"
    logger.info(f"[retry_check_gate] retries={retries} (MAX={cfg.MAX_RETRIES}) -> {decision}")
    return decision


def create_generate_node(llm: LLM | None = None) -> Callable[[GraphState], Any]:
    """Node 10: generate."""
    llm_instance = llm or LangChainLLM()

    async def generate(state: GraphState) -> dict[str, Any]:
        t0 = time.perf_counter()
        tracer: ChatTracer | None = state.get("tracer")
        original_q = state["original_q"]
        ranked = state.get("ranked", [])

        citations: list[str] = [c.url or c.title or c.source for c in ranked]
        user_msg = generate_user_message({
            "original_q": original_q,
            "ranked": ranked,
        })
        prompt = f"{GENERATE_SYSTEM_PROMPT}\n\n{user_msg}"

        queue: Any = state.get("token_queue")
        if queue is not None and hasattr(llm_instance, "astream"):
            answer_parts = []
            async for tok in llm_instance.astream(prompt):
                if tok:
                    answer_parts.append(tok)
                    await queue.put(tok)
                    await asyncio.sleep(0)
            answer = "".join(answer_parts)
            await queue.put(None)
        else:
            answer = await llm_instance.ainvoke(prompt)
            if queue is not None:
                await queue.put(answer)
                await queue.put(None)

        deduped_citations = list(dict.fromkeys(citations))

        latency_ms = (time.perf_counter() - t0) * 1000
        if tracer:
            tracer.log_llm_call(
                model=getattr(llm_instance, "model_name", None) or "ministral-8b-latest",
                prompt_text=prompt,
                model_settings={"temperature": 0.3},
                usage=getattr(llm_instance, "last_usage", None),
                latency_ms=latency_ms,
                role="primary",
                input_chunks=ranked,
                output_text=answer,
            )
            tracer.log_node(
                name="node/generate",
                run_type="chain",
                inputs={
                    "original_q": original_q,
                    "input_chunks_count": len(ranked),
                    "retrieved_chunks": [
                        {"id": c.id, "source": c.source, "score": c.score}
                        for c in ranked
                    ],
                },
                outputs={
                    "answer_preview": answer[:200],
                    "citations": deduped_citations,
                    "low_confidence": False,
                },
                latency_ms=latency_ms,
            )

        _log_node("generate", state, citations_count=len(deduped_citations), latency_ms=f"{latency_ms:.1f}")
        return {
            "answer": answer.strip(),
            "citations": deduped_citations,
            "low_confidence": False,
        }

    return generate


def create_fallback_node(
    llm: LLM | None = None,
    settings: Settings | None = None,
) -> Callable[[GraphState], Any]:
    """Node 11: fallback."""
    cfg = settings or default_settings
    llm_instance = llm or LangChainLLM()

    async def fallback(state: GraphState) -> dict[str, Any]:
        t0 = time.perf_counter()
        tracer: ChatTracer | None = state.get("tracer")
        original_q = state["original_q"]
        ranked = state.get("ranked", []) or state.get("pool", [])
        missing = state.get("missing", "Sufficient authoritative evidence")

        queue: Any = state.get("token_queue")
        if not ranked:
            latency_ms = (time.perf_counter() - t0) * 1000
            if queue is not None:
                await queue.put(FALLBACK_NO_SOURCES_MSG)
                await queue.put(None)
            if tracer:
                tracer.log_node(
                    name="node/fallback",
                    run_type="chain",
                    inputs={"original_q": original_q, "missing": missing, "chunks": 0},
                    outputs={"answer": FALLBACK_NO_SOURCES_MSG, "low_confidence": True},
                    latency_ms=latency_ms,
                )
            _log_node("fallback", state, chunks=0, latency_ms=f"{latency_ms:.1f}")
            return {
                "answer": FALLBACK_NO_SOURCES_MSG,
                "citations": [],
                "low_confidence": True,
            }

        citations: list[str] = [c.url or c.title or c.source for c in ranked]
        top_score = ranked[0].score if (ranked and ranked[0].score is not None) else 0.0

        if top_score >= cfg.LOW_THRESHOLD:
            logger.info(
                f"[fallback] top_score={top_score:.3f} >= LOW_THRESHOLD ({cfg.LOW_THRESHOLD}) -> "
                f"answering using GENERATE_HYBRID_PROMPT (context-primary + fills gaps with knowledge)"
            )
            user_msg = generate_user_message({
                "original_q": original_q,
                "ranked": ranked,
            })
            prompt = f"{GENERATE_HYBRID_PROMPT}\n\n{user_msg}"
            is_normal = True
        else:
            logger.info(
                f"[fallback] top_score={top_score:.3f} < LOW_THRESHOLD ({cfg.LOW_THRESHOLD}) -> "
                f"using FALLBACK_SYSTEM_PROMPT"
            )
            user_msg = fallback_user_message({
                "original_q": original_q,
                "ranked": ranked,
                "pool": state.get("pool", []),
                "missing": missing,
            })
            prompt = f"{FALLBACK_SYSTEM_PROMPT}\n\n{user_msg}"
            is_normal = False

        if queue is not None and hasattr(llm_instance, "astream"):
            answer_parts = []
            async for tok in llm_instance.astream(prompt):
                if tok:
                    answer_parts.append(tok)
                    await queue.put(tok)
                    await asyncio.sleep(0)
            answer = "".join(answer_parts)
            await queue.put(None)
        else:
            answer = await llm_instance.ainvoke(prompt)
            if queue is not None:
                await queue.put(answer)
                await queue.put(None)

        deduped_citations = list(dict.fromkeys(citations))

        latency_ms = (time.perf_counter() - t0) * 1000
        if tracer:
            tracer.log_llm_call(
                model=getattr(llm_instance, "model_name", None) or "ministral-8b-latest",
                prompt_text=prompt,
                model_settings={"temperature": 0.3 if is_normal else 0.2},
                usage=getattr(llm_instance, "last_usage", None),
                latency_ms=latency_ms,
                is_fallback=not is_normal,
                role="primary" if is_normal else "fallback",
                input_chunks=ranked,
                output_text=answer,
            )
            tracer.log_node(
                name="node/fallback",
                run_type="chain",
                inputs={
                    "original_q": original_q,
                    "missing": missing,
                    "input_chunks_count": len(ranked),
                    "mode": "hybrid_generation" if is_normal else "fallback_sys_prompt",
                    "top_score": top_score,
                    "retrieved_chunks": [
                        {"id": c.id, "source": c.source, "score": c.score}
                        for c in ranked
                    ],
                },
                outputs={
                    "answer_preview": answer[:200],
                    "citations": deduped_citations,
                    "low_confidence": True,
                },
                latency_ms=latency_ms,
            )

        _log_node(
            "fallback",
            state,
            top_score=f"{top_score:.3f}",
            mode="hybrid_generation" if is_normal else "fallback_sys_prompt",
            citations_count=len(deduped_citations),
            latency_ms=f"{latency_ms:.1f}",
        )
        return {
            "answer": answer.strip(),
            "citations": deduped_citations,
            "low_confidence": True,
        }

    return fallback
