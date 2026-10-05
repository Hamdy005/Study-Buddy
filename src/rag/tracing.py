"""
LangSmith Observability for Study-Buddy RAG pipeline
=====================================================

Each chat turn is logged as one **parent trace** containing:
  - input / output
  - total latency & time-to-first-token (streaming)
  - token usage & estimated cost    
  - retrieval step  (vector query, chunks, similarity scores, tool chosen)
  - final prompt    (rendered template) + model settings
  - errors          (primary & fallback)
  - metadata tags   (user_id, session_id, language, feature, environment, prompt_version)
  - user feedback   (thumbs-up/down posted via the /feedback endpoint)
  - eval stubs      (groundedness, relevance — wired but skipped until Corrective RAG lands)

Usage
-----
  from src.rag.tracing import ChatTracer

  tracer = ChatTracer(user_id=..., session_id=..., feature="ws_chat")
  tracer.start(query)

  tracer.log_retrieval(query, chunks, scores, tool_chosen)
  tracer.log_web_gather(query, tools_called, chars_retrieved)
  tracer.log_llm_call(model, prompt_text, settings, usage, ttft_ms)

  tracer.finish(answer)          # success
  tracer.finish_error(exc)       # failure

  run_id = tracer.run_id         # return to client so /feedback can reference it
"""

from __future__ import annotations

import os
import re
import time
import uuid
import logging
import threading
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger(__name__)

# ── LangSmith client (optional — tracing degrades gracefully if not configured) ──

_langsmith_client = None


def _get_client():
    """Lazily initialise the LangSmith client once."""
    global _langsmith_client
    if _langsmith_client is not None:
        return _langsmith_client

    api_key = os.environ.get("LANGSMITH_API_KEY", "")
    if not api_key:
        return None

    try:
        from langsmith import Client
        _langsmith_client = Client(api_key=api_key)
        return _langsmith_client
    except Exception as exc:
        log.warning(f"LangSmith client init failed — tracing disabled: {exc}")
        return None


from src.rag.constants import (
    ARABIC_CHAR_RATIO_THRESHOLD,
    COST_PER_1K,
    DEFAULT_ENVIRONMENT,
    DEFAULT_LANGSMITH_PROJECT,
    GEMINI_COUNT_KEY_TTL_SECONDS,
    GEMINI_FREE_REQUESTS_LIMIT,
    GEMINI_MODELS,
    PROMPT_VERSION,
)

# Backwards compatibility aliases
_COST_PER_1K = COST_PER_1K

_local_request_counts: dict[str, int] = {}
_lock = threading.Lock()


def _get_today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def record_gemini_request(model: str) -> dict:
    """
    Track daily requests for Gemini models under Google's 1 000 free request-tier.

    If daily requests for the model <= 1000:
        Pricing is request-based (free tier, $0.00).
    If daily requests exceed 1000:
        Pricing converts to the token-based method.
    """
    today = _get_today_str()
    key = f"gemini_daily_requests:{model}:{today}"
    total_key = f"gemini_daily_requests:total:{today}"
    count = 1

    try:
        from src.redis_client import get_redis
        r = get_redis()
        if r is not None:
            count = int(r.incr(key))
            r.expire(key, GEMINI_COUNT_KEY_TTL_SECONDS)
            r.incr(total_key)
            r.expire(total_key, GEMINI_COUNT_KEY_TTL_SECONDS)
        else:
            with _lock:
                _local_request_counts[key] = _local_request_counts.get(key, 0) + 1
                count = _local_request_counts[key]
                _local_request_counts[total_key] = _local_request_counts.get(total_key, 0) + 1
    except Exception as exc:
        log.warning(f"Failed to record Gemini request count in Redis: {exc}")
        with _lock:
            _local_request_counts[key] = _local_request_counts.get(key, 0) + 1
            count = _local_request_counts[key]

    is_exceeded = count > GEMINI_FREE_REQUESTS_LIMIT
    pricing_mode = "token_based" if is_exceeded else "request_based"

    return {
        "request_count": count,
        "limit": GEMINI_FREE_REQUESTS_LIMIT,
        "is_exceeded": is_exceeded,
        "pricing_mode": pricing_mode,
    }


def get_gemini_request_count(model: str) -> int:
    """Return the current daily request count for a given Gemini model."""
    today = _get_today_str()
    key = f"gemini_daily_requests:{model}:{today}"
    try:
        from src.redis_client import get_redis
        r = get_redis()
        if r is not None:
            val = r.get(key)
            return int(val) if val else 0
        return _local_request_counts.get(key, 0)
    except Exception:
        return _local_request_counts.get(key, 0)


def _estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    is_gemini_exceeded: bool = False,
) -> tuple[float, str]:
    """
    Estimate LLM cost in USD and return (cost_usd, pricing_mode).

    - For Gemini models (gemini-3.5-flash-lite, gemini-3.1-flash-lite):
      If daily requests <= 1000:
          Treated as request-based free tier -> cost is $0.00.
      If daily requests > 1000:
          Converts to token-based method -> cost computed via _COST_PER_1K.
    - For non-Gemini models (e.g. ministral-8b-latest):
      Always token-based.
    """
    rates = _COST_PER_1K.get(model, (0.0, 0.0))
    token_cost = rates[0] * input_tokens / 1000 + rates[1] * output_tokens / 1000

    if model in GEMINI_MODELS:
        if not is_gemini_exceeded:
            return 0.0, "request_based"
        return token_cost, "token_based"

    return token_cost, "token_based"


def _detect_language(text: str) -> str:
    """Heuristic language tag — extend as needed."""
    if not text:
        return "unknown"
    arabic_chars = len(re.findall(r"[\u0600-\u06FF]", text))
    if arabic_chars / max(len(text), 1) > ARABIC_CHAR_RATIO_THRESHOLD:
        return "ar"
    return "en"


# ── ChatTracer ────────────────────────────────────────────────────────────────


class ChatTracer:
    """
    Wraps a single chat turn as a parent LangSmith trace with named child spans.

    If LANGSMITH_API_KEY is not set the tracer is a no-op — the RAG pipeline
    continues to work normally without any tracing overhead.
    """

    def __init__(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        material_id: Optional[str] = None,
        feature: str = "rag_chat",         # "rag_chat" | "ws_chat"
    ):
        self.user_id     = user_id
        self.session_id  = session_id
        self.material_id = material_id
        self.feature     = feature
        self.run_id: Optional[str] = None
        self._client     = _get_client()
        self._start_ts   = time.perf_counter()
        self._run_obj    = None   # langsmith RunTree if available

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self, query: str, source_type: str = "topic") -> Optional[str]:
        """
        Open the parent trace.  Returns the run_id string (or a UUID even if
        tracing is disabled) so the caller can always return it to the frontend.
        """
        self.run_id = str(uuid.uuid4())
        if not self._client:
            return self.run_id

        project     = os.environ.get("LANGSMITH_PROJECT", DEFAULT_LANGSMITH_PROJECT)
        environment = os.environ.get("ENVIRONMENT", DEFAULT_ENVIRONMENT)

        metadata = {
            "user_id":        self.user_id,
            "session_id":     self.session_id,
            "material_id":    self.material_id,
            "language":       _detect_language(query),
            "feature":        self.feature,
            "source_type":    source_type,
            "environment":    environment,
            "prompt_version": PROMPT_VERSION,
        }

        try:
            from langsmith.run_trees import RunTree
            self._run_obj = RunTree(
                name         = f"chat_turn/{self.feature}",
                run_type     = "chain",
                id           = uuid.UUID(self.run_id),
                project_name = project,
                inputs       = {"query": query},
                extra        = {"metadata": metadata},
            )
            self._run_obj.post()
        except Exception as exc:
            log.debug(f"LangSmith start trace failed: {exc}")

        return self.run_id

    def finish(self, answer: str, used_fallback: bool = False) -> None:
        """Close the parent trace on success."""
        elapsed_ms = (time.perf_counter() - self._start_ts) * 1000
        if not self._run_obj:
            return
        try:
            self._run_obj.end(
                outputs = {"answer": answer, "used_fallback_llm": used_fallback},
                extra   = {"latency_ms": round(elapsed_ms, 1)},
            )
            self._run_obj.patch()
        except Exception as exc:
            log.debug(f"LangSmith finish trace failed: {exc}")

    def finish_error(self, error: Exception) -> None:
        """Close the parent trace on an unhandled error."""
        elapsed_ms = (time.perf_counter() - self._start_ts) * 1000
        if not self._run_obj:
            return
        try:
            self._run_obj.end(
                error = f"{type(error).__name__}: {error}",
                extra = {"latency_ms": round(elapsed_ms, 1)},
            )
            self._run_obj.patch()
        except Exception as exc:
            log.debug(f"LangSmith finish_error trace failed: {exc}")

    # ── Child spans ───────────────────────────────────────────────────────────

    def log_retrieval(
        self,
        query:       str,
        chunks:      list[str],
        scores:      list[float],
        tool_chosen: str = "vector_similarity",
    ) -> None:
        """Log the vector-search retrieval step as a child span."""
        if not self._run_obj:
            return
        try:
            child = self._run_obj.create_child(
                name     = "retrieval/vector_search",
                run_type = "retriever",
                inputs   = {"query": query, "tool": tool_chosen},
            )
            child.end(outputs={
                "num_chunks":    len(chunks),
                "top_score":     round(max(scores, default=0.0), 4),
                "min_score":     round(min(scores, default=0.0), 4),
                "avg_score":     round(sum(scores) / max(len(scores), 1), 4),
                "chunk_preview": [c[:120] for c in chunks[:3]],
            })
            child.post()
        except Exception as exc:
            log.debug(f"LangSmith log_retrieval failed: {exc}")

    def log_web_gather(
        self,
        query:           str,
        tools_called:    list[str],
        chars_retrieved: int,
    ) -> None:
        """Log the agentic web-search step as a child span."""
        if not self._run_obj:
            return
        try:
            child = self._run_obj.create_child(
                name     = "retrieval/agentic_web_gather",
                run_type = "tool",
                inputs   = {"query": query, "tools_selected": tools_called},
            )
            child.end(outputs={
                "chars_retrieved": chars_retrieved,
                "num_tools":       len(tools_called),
            })
            child.post()
        except Exception as exc:
            log.debug(f"LangSmith log_web_gather failed: {exc}")

    def log_llm_call(
        self,
        model:          str,
        prompt_text:    str,
        model_settings: dict,
        usage:          Optional[dict] = None,   # {"input_tokens": N, "output_tokens": M}
        ttft_ms:        Optional[float] = None,  # time-to-first-token in ms (streaming only)
        is_fallback:    bool = False,
        role:           Optional[str] = None,    # "primary" | "fallback" | "router"
    ) -> None:
        """Log the LLM invocation as a child span including dynamic cost estimate."""
        if not self._run_obj:
            # Even if LangSmith is disabled, maintain Gemini daily request count
            if model in GEMINI_MODELS:
                record_gemini_request(model)
            return

        try:
            span_role = role or ("fallback" if is_fallback else "primary")

            gemini_stats = None
            is_exceeded = False
            if model in GEMINI_MODELS:
                gemini_stats = record_gemini_request(model)
                is_exceeded = gemini_stats["is_exceeded"]

            input_tokens  = (usage or {}).get("input_tokens") or 0
            output_tokens = (usage or {}).get("output_tokens") or 0

            # Fallback estimation if token usage is 0 (common in streaming)
            if input_tokens == 0 and prompt_text:
                input_tokens = max(1, len(prompt_text) // 4)

            cost_usd, pricing_mode = _estimate_cost(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                is_gemini_exceeded=is_exceeded,
            )

            child = self._run_obj.create_child(
                name     = f"llm/{span_role}",
                run_type = "llm",
                inputs   = {
                    "prompt_preview": prompt_text[:500],
                    "model":          model,
                    "settings":       model_settings,
                    "pricing_mode":   pricing_mode,
                },
            )
            span_outputs = {
                "input_tokens":  input_tokens,
                "output_tokens": output_tokens,
                "total_tokens":  input_tokens + output_tokens,
                "cost_usd":      round(cost_usd, 6),
                "pricing_mode":  pricing_mode,
            }
            if gemini_stats:
                span_outputs["gemini_request_count_today"] = gemini_stats["request_count"]
                span_outputs["gemini_free_requests_limit"] = gemini_stats["limit"]
                span_outputs["gemini_converted_to_tokens"] = gemini_stats["is_exceeded"]

            if ttft_ms is not None:
                span_outputs["ttft_ms"] = round(ttft_ms, 1)

            child.end(outputs=span_outputs)
            child.post()
        except Exception as exc:
            log.debug(f"LangSmith log_llm_call failed: {exc}")

    def log_error(self, stage: str, error: Exception) -> None:
        """Log a recoverable error (e.g. primary LLM rate-limit) as a child span."""
        if not self._run_obj:
            return
        try:
            child = self._run_obj.create_child(
                name     = f"error/{stage}",
                run_type = "chain",
                inputs   = {"stage": stage},
            )
            child.end(error=f"{type(error).__name__}: {error}")
            child.post()
        except Exception as exc:
            log.debug(f"LangSmith log_error failed: {exc}")

    # ── Eval stubs (wired, skipped until Corrective RAG) ─────────────────────

    def schedule_evals(
        self,
        query:   str,
        answer:  str,
        context: str,
    ) -> None:
        """
        Placeholder for groundedness & relevance evaluations.

        TODO(corrective-rag): Replace stubs with real LLM-as-judge evaluators
        once Corrective RAG is implemented.  At that point wire:
          - groundedness  : does the answer stay within the retrieved context?
          - relevance     : does the retrieved context actually address the query?

        Evals should be sampled (e.g. 10 % of turns) to avoid cost explosion.

        Example skeleton — do NOT activate until corrective RAG is ready:

            from langsmith.evaluation import evaluate
            results = evaluate(
                lambda inputs: {"answer": answer},
                data=...,
                evaluators=[groundedness_evaluator, relevance_evaluator],
                experiment_prefix="study-buddy-rag",
            )
            # Then log scores back to the parent run via client.create_feedback().
        """
        # No-op until Corrective RAG integration — structure is ready.
        pass


# ── Feedback (called from POST /api/tutor/feedback) ───────────────────────────


def post_user_feedback(
    run_id:  str,
    score:   int,           # +1 = 👍  /  -1 = 👎
    comment: Optional[str] = None,
    user_id: Optional[str] = None,
) -> bool:
    """
    Attach a thumbs-up / thumbs-down signal to an existing LangSmith trace.

    Returns True on success, False if tracing is disabled or the call fails.

    Frontend integration TODO
    ─────────────────────────
    The /api/tutor/feedback endpoint receives the run_id returned in the chat
    response (TutorResponse.run_id) or in the WebSocket [DONE] JSON payload.
    The frontend should:
      1. Render a 👍 / 👎 button pair below each assistant message.
      2. On click POST to /api/tutor/feedback:
             { "run_id": "<run_id>", "score": 1 | -1, "comment": "optional text" }
      3. Disable the buttons after the first submission (no double-voting).
      4. Show a brief "Thanks for your feedback!" toast/snackbar on success.
    """
    client = _get_client()
    if not client:
        log.debug("LangSmith not configured — feedback not stored.")
        return False

    # LangSmith's thumbs widget expects a normalised float [0.0, 1.0]
    normalised = 1.0 if score > 0 else 0.0

    try:
        client.create_feedback(
            run_id     = run_id,
            key        = "user_thumbs",
            score      = normalised,
            comment    = comment or "",
            source_info= {"user_id": user_id} if user_id else {},
        )
        return True
    except Exception as exc:
        log.warning(f"LangSmith post_user_feedback failed: {exc}")
        return False
