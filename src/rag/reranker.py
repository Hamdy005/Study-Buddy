"""
Reranker module for Study-Buddy RAG pipeline.

Multi-Tier Cascade Architecture:
  1. Primary:    Jina AI Reranker API v2 (jina-reranker-v2-base-multilingual)
  2. Fallback 1: Cohere Rerank 3.5 (rerank-v3.5)
  3. Fallback 2: Mistral Reranker Node (ministral-8b-latest + RERANKER_LLM_SYSTEM_PROMPT)
  4. Safety Net: Heuristic Token Overlap (failsafe)

Local quantized model has been removed in favor of the resilient cloud cascade.
"""

from __future__ import annotations

import json
import os
import re
import time
from functools import lru_cache
from typing import Any, Optional, Protocol, Union, runtime_checkable

import httpx

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from src.rag.constants import (
    COHERE_RERANK_API_URL,
    COHERE_RERANK_DEFAULT_MODEL,
    COHERE_RERANK_MULTILINGUAL_MODEL,
    JINA_RERANK_API_URL,
    JINA_RERANK_DEFAULT_MODEL,
    MISTRAL_RERANK_DEFAULT_MODEL,
    RERANKER_LLM_SYSTEM_PROMPT,
    RERANKER_MAX_LENGTH,
    RERANKER_MODEL,
    RERANKER_OVER_RETRIEVAL_FACTOR,
    reranker_user_message,
)

_RERANKER_MODEL = RERANKER_MODEL


# ── Reranker Protocol ───────────────────────────────────────────
@runtime_checkable
class Reranker(Protocol):
    """Protocol defining the reranker scoring interface."""

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Score candidate texts against query. Returns float scores in [0, 1]."""
        ...


# ── Tier 1: Jina AI API v2 Reranker ─────────────────────────────
class JinaAPIReranker:
    """Primary Tier: Jina AI Reranker API v2 (https://api.jina.ai/v1/rerank).
    Model: jina-reranker-v2-base-multilingual
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = JINA_RERANK_DEFAULT_MODEL,
        endpoint: str = JINA_RERANK_API_URL,
        timeout: float = 10.0,
    ):
        self.api_key = api_key or os.getenv("JINA_API_KEY")
        self.model_name = model_name
        self.endpoint = endpoint
        self.timeout = timeout

    def _prepare_payload(self, query: str, texts: list[str]) -> tuple[dict[str, str], dict[str, Any]]:
        if not self.api_key or not self.api_key.strip():
            raise ValueError("JINA_API_KEY is not configured")
        headers = {
            "Authorization": f"Bearer {self.api_key.strip()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": texts,
            "top_n": len(texts),
        }
        return headers, payload

    def _parse_results(self, data: dict[str, Any], count: int) -> list[float]:
        results = data.get("results", [])
        scores = [0.0] * count
        for item in results:
            idx = item.get("index")
            if idx is not None and 0 <= idx < count:
                scores[idx] = max(0.0, min(1.0, float(item.get("relevance_score", 0.0))))
        return scores

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        if not query.strip():
            return [0.0] * len(texts)

        headers, payload = self._prepare_payload(query, texts)
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(self.endpoint, json=payload, headers=headers)
            resp.raise_for_status()
            return self._parse_results(resp.json(), len(texts))

    async def score_async(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        if not query.strip():
            return [0.0] * len(texts)

        headers, payload = self._prepare_payload(query, texts)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(self.endpoint, json=payload, headers=headers)
            resp.raise_for_status()
            return self._parse_results(resp.json(), len(texts))


# ── Tier 2: Cohere Rerank 3.5 ────────────────────────────────────
class CohereReranker:
    """Secondary Tier Fallback: Cohere Rerank 3.5 API (https://api.cohere.com/v2/rerank).
    Model: rerank-v3.5 (unified multilingual for 100+ languages).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = COHERE_RERANK_DEFAULT_MODEL,
        endpoint: str = COHERE_RERANK_API_URL,
        timeout: float = 10.0,
    ):
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        self.model_name = model_name
        self.endpoint = endpoint
        self.timeout = timeout

    def _prepare_payload(self, query: str, texts: list[str]) -> tuple[dict[str, str], dict[str, Any]]:
        if not self.api_key or not self.api_key.strip():
            raise ValueError("COHERE_API_KEY is not configured")
        headers = {
            "Authorization": f"Bearer {self.api_key.strip()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Client-Name": "Study-Buddy",
        }
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": texts,
            "top_n": len(texts),
        }
        return headers, payload

    def _parse_results(self, data: dict[str, Any], count: int) -> list[float]:
        results = data.get("results", [])
        scores = [0.0] * count
        for item in results:
            idx = item.get("index")
            if idx is not None and 0 <= idx < count:
                scores[idx] = max(0.0, min(1.0, float(item.get("relevance_score", 0.0))))
        return scores

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        if not query.strip():
            return [0.0] * len(texts)

        headers, payload = self._prepare_payload(query, texts)
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(self.endpoint, json=payload, headers=headers)
            resp.raise_for_status()
            return self._parse_results(resp.json(), len(texts))

    async def score_async(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        if not query.strip():
            return [0.0] * len(texts)

        headers, payload = self._prepare_payload(query, texts)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(self.endpoint, json=payload, headers=headers)
            resp.raise_for_status()
            return self._parse_results(resp.json(), len(texts))


# ── Tier 3: Mistral Reranker Node ────────────────────────────────
class MistralReranker:
    """Tertiary Tier Fallback: Mistral scoring node using RERANKER_LLM_SYSTEM_PROMPT.
    Default model: ministral-8b-latest (temperature=0.0 for calibrated scoring).
    """

    def __init__(
        self,
        llm: Any = None,
        model_name: str = MISTRAL_RERANK_DEFAULT_MODEL,
        system_prompt: str = RERANKER_LLM_SYSTEM_PROMPT,
        temperature: float = 0.0,
    ):
        self.llm = llm
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature

    def _get_llm(self) -> Any:
        if self.llm is not None:
            return self.llm
        try:
            from langchain_mistralai import ChatMistralAI
            api_key = os.getenv("MISTRAL_API_KEY")
            if not api_key:
                from src.config import Settings as GlobalSettings
                api_key = getattr(GlobalSettings, "mistral_api_key", None)
            if api_key:
                self.llm = ChatMistralAI(
                    model=self.model_name,
                    mistral_api_key=api_key,
                    temperature=self.temperature,
                )
                return self.llm
        except Exception as exc:
            logger.debug(f"Could not initialize ChatMistralAI directly ({exc}); falling back to LangChainLLM")

        from src.rag.crag_llm import LangChainLLM

        self.llm = LangChainLLM()
        return self.llm

    def _parse_llm_scores(self, raw_text: str, expected_len: int) -> list[float]:
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            cleaned = cleaned.strip()

        scores: list[float] | None = None
        # 1. Direct JSON parse
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict) and "scores" in data and isinstance(data["scores"], list):
                scores = [float(s) for s in data["scores"]]
            elif isinstance(data, list):
                scores = [float(s) for s in data]
        except Exception:
            pass

        # 2. Regex search for {"scores": [...]}
        if scores is None:
            match = re.search(r'\{\s*"scores"\s*:\s*\[([^\]]+)\]\s*\}', cleaned)
            if match:
                try:
                    num_strs = match.group(1).split(",")
                    scores = [float(s.strip()) for s in num_strs if s.strip()]
                except Exception:
                    pass

        # 3. Fallback: extract floating point numbers
        if scores is None:
            numbers = re.findall(r"\b0\.\d+\b|\b1\.0\b|\b[01]\b", cleaned)
            if numbers:
                scores = [float(n) for n in numbers]

        if not scores:
            raise ValueError(f"Could not parse scores from LLM output: {raw_text[:200]}")

        clamped = [max(0.0, min(1.0, float(s))) for s in scores]
        if len(clamped) < expected_len:
            clamped.extend([0.0] * (expected_len - len(clamped)))
        elif len(clamped) > expected_len:
            clamped = clamped[:expected_len]

        return clamped

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        if not query.strip():
            return [0.0] * len(texts)

        llm = self._get_llm()
        user_msg = reranker_user_message(query, texts)
        prompt = f"{self.system_prompt}\n\n{user_msg}"
        raw_text: str = ""

        chat_model_getter = getattr(llm, "_get_chat_model", None)
        if callable(chat_model_getter):
            try:
                from langchain_core.messages import HumanMessage, SystemMessage

                model = chat_model_getter()
                res = model.invoke([SystemMessage(content=self.system_prompt), HumanMessage(content=user_msg)])
                raw_text = getattr(res, "content", str(res))
            except Exception:
                raw_text = ""
        elif hasattr(llm, "invoke"):
            try:
                from langchain_core.messages import HumanMessage, SystemMessage

                res = llm.invoke([SystemMessage(content=self.system_prompt), HumanMessage(content=user_msg)])
                raw_text = getattr(res, "content", str(res))
            except Exception:
                raw_text = ""

        if not raw_text:
            import asyncio

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    raw_text = asyncio.run(llm.ainvoke(prompt))
                else:
                    raw_text = loop.run_until_complete(llm.ainvoke(prompt))
            except RuntimeError:
                import asyncio

                raw_text = asyncio.run(llm.ainvoke(prompt))

        return self._parse_llm_scores(raw_text, len(texts))

    async def score_async(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        if not query.strip():
            return [0.0] * len(texts)

        llm = self._get_llm()
        user_msg = reranker_user_message(query, texts)
        prompt = f"{self.system_prompt}\n\n{user_msg}"

        if hasattr(llm, "ainvoke"):
            chat_model_getter = getattr(llm, "_get_chat_model", None)
            if callable(chat_model_getter):
                try:
                    from langchain_core.messages import HumanMessage, SystemMessage

                    model = chat_model_getter()
                    if hasattr(model, "ainvoke"):
                        res = await model.ainvoke([SystemMessage(content=self.system_prompt), HumanMessage(content=user_msg)])
                        raw_text = getattr(res, "content", str(res))
                    else:
                        res = model.invoke([SystemMessage(content=self.system_prompt), HumanMessage(content=user_msg)])
                        raw_text = getattr(res, "content", str(res))
                except Exception:
                    raw_text = await llm.ainvoke(prompt)
            else:
                try:
                    from langchain_core.messages import HumanMessage, SystemMessage
                    res = await llm.ainvoke([SystemMessage(content=self.system_prompt), HumanMessage(content=user_msg)])
                    raw_text = getattr(res, "content", str(res))
                except Exception:
                    raw_text = await llm.ainvoke(prompt)
        else:
            import asyncio

            raw_text = await asyncio.to_thread(self.score, query, texts)

        return self._parse_llm_scores(raw_text, len(texts))


LLMReranker = MistralReranker


# ── Multi-Tier Cascade Reranker Coordinator ───────────────────────
class CascadeReranker:
    """Production Multi-Tier Reranker Cascade:
    1. Primary:    Jina AI Reranker API v2 (jina-reranker-v2-base-multilingual)
    2. Fallback 1: Cohere Rerank 3.5 (rerank-v3.5)
    3. Fallback 2: Mistral Reranker Node (ministral-8b-latest + RERANKER_LLM_SYSTEM_PROMPT)
    4. Safety Net: Heuristic Token Overlap (failsafe)
    """

    def __init__(
        self,
        settings: Any = None,
        llm: Any = None,
        jina_reranker: Reranker | None = None,
        cohere_reranker: Reranker | None = None,
        mistral_reranker: Reranker | None = None,
        llm_reranker: Reranker | None = None,
    ):
        from src.rag.config import default_settings, settings as global_settings

        cfg = settings or global_settings or default_settings

        jina_key = getattr(cfg, "JINA_API_KEY", None) or os.getenv("JINA_API_KEY")
        cohere_key = getattr(cfg, "COHERE_API_KEY", None) or os.getenv("COHERE_API_KEY")
        jina_model = getattr(cfg, "JINA_RERANK_MODEL", JINA_RERANK_DEFAULT_MODEL)
        cohere_model = getattr(cfg, "COHERE_RERANK_MODEL", COHERE_RERANK_DEFAULT_MODEL)
        mistral_model = getattr(cfg, "MISTRAL_RERANK_MODEL", MISTRAL_RERANK_DEFAULT_MODEL)

        self.jina_reranker = jina_reranker or JinaAPIReranker(api_key=jina_key, model_name=jina_model)
        self.cohere_reranker = cohere_reranker or CohereReranker(api_key=cohere_key, model_name=cohere_model)
        self.mistral_reranker = mistral_reranker or llm_reranker or MistralReranker(llm=llm, model_name=mistral_model)
        self.llm_reranker = self.mistral_reranker
        self.last_tier: str = "none"

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        if not query.strip():
            return [0.0] * len(texts)

        # Tier 1: Jina AI API v2
        try:
            scores = self.jina_reranker.score(query, texts)
            if scores and len(scores) == len(texts):
                self.last_tier = "jina_api"
                return scores
        except Exception as exc:
            logger.warning(f"[CascadeReranker] Tier 1 (Jina AI v2) failed: {exc}")

        # Tier 2: Cohere Rerank 3.5
        try:
            scores = self.cohere_reranker.score(query, texts)
            if scores and len(scores) == len(texts):
                self.last_tier = "cohere_3.5"
                return scores
        except Exception as exc:
            logger.warning(f"[CascadeReranker] Tier 2 (Cohere 3.5) failed: {exc}")

        # Tier 3: Mistral Reranker Node
        try:
            scores = self.mistral_reranker.score(query, texts)
            if scores and len(scores) == len(texts):
                self.last_tier = "mistral"
                return scores
        except Exception as exc:
            logger.warning(f"[CascadeReranker] Tier 3 (Mistral) failed: {exc}")

        # Tier 4: Failsafe Token Overlap
        logger.warning("[CascadeReranker] All 3 tiers failed; falling back to token overlap heuristic.")
        self.last_tier = "heuristic"
        return [self._token_overlap_score(query, t) for t in texts]

    async def score_async(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        if not query.strip():
            return [0.0] * len(texts)

        # Tier 1: Jina AI API v2
        try:
            if hasattr(self.jina_reranker, "score_async"):
                scores = await self.jina_reranker.score_async(query, texts)
            else:
                import asyncio
                scores = await asyncio.to_thread(self.jina_reranker.score, query, texts)
            if scores and len(scores) == len(texts):
                self.last_tier = "jina_api"
                return scores
        except Exception as exc:
            logger.warning(f"[CascadeReranker] Tier 1 (Jina AI v2) async failed: {exc}")

        # Tier 2: Cohere Rerank 3.5
        try:
            if hasattr(self.cohere_reranker, "score_async"):
                scores = await self.cohere_reranker.score_async(query, texts)
            else:
                import asyncio
                scores = await asyncio.to_thread(self.cohere_reranker.score, query, texts)
            if scores and len(scores) == len(texts):
                self.last_tier = "cohere_3.5"
                return scores
        except Exception as exc:
            logger.warning(f"[CascadeReranker] Tier 2 (Cohere 3.5) async failed: {exc}")

        # Tier 3: Mistral Reranker Node
        try:
            if hasattr(self.mistral_reranker, "score_async"):
                scores = await self.mistral_reranker.score_async(query, texts)
            else:
                import asyncio
                scores = await asyncio.to_thread(self.mistral_reranker.score, query, texts)
            if scores and len(scores) == len(texts):
                self.last_tier = "mistral"
                return scores
        except Exception as exc:
            logger.warning(f"[CascadeReranker] Tier 3 (Mistral) async failed: {exc}")

        # Tier 4: Failsafe Token Overlap
        logger.warning("[CascadeReranker] All 3 tiers failed; falling back to token overlap heuristic.")
        self.last_tier = "heuristic"
        return [self._token_overlap_score(query, t) for t in texts]

    @staticmethod
    def _token_overlap_score(query: str, text: str) -> float:
        q_words = set(query.lower().split())
        if not q_words:
            return 0.0
        t_words = set(text.lower().split())
        overlap = len(q_words & t_words)
        return min(1.0, max(0.0, overlap / len(q_words)))


# Alias for backward compatibility
JinaReranker = JinaAPIReranker


# ── Testing Mock Reranker ───────────────────────────────────────
class MockReranker:
    """Mock reranker for deterministic unit and integration tests."""

    def __init__(
        self,
        scores: list[float] | None = None,
        score_map: dict[str, float] | None = None,
        fail: bool = False,
    ):
        self.scores = scores
        self.score_map = score_map or {}
        self.fail = fail
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, texts: list[str]) -> list[float]:
        self.calls.append((query, list(texts)))
        if self.fail:
            raise RuntimeError("Simulated reranker failure")
        if self.scores is not None:
            if len(self.scores) >= len(texts):
                return self.scores[: len(texts)]
            return self.scores + [0.5] * (len(texts) - len(self.scores))
        if self.score_map:
            return [self.score_map.get(t, 0.5) for t in texts]
        return [0.75] * len(texts)


# ── Startup & Inference Helpers ─────────────────────────────────
@lru_cache(maxsize=1)
def get_reranker() -> CascadeReranker:
    """Return singleton instance of CascadeReranker."""
    return CascadeReranker()


_get_reranker = get_reranker


def warmup_reranker():
    """No-op warmup: local models removed in favor of remote API cascade (Jina -> Cohere -> Mistral)."""
    logger.debug("Reranker warmup: using API cascade (Jina v2 -> Cohere 3.5 -> Mistral). Ready.")


def _score_pairs_batch(pairs: list[list[str]]) -> list[float]:
    """Score pairs grouped by query using CascadeReranker."""
    if not pairs:
        return []

    query_to_indices: dict[str, list[int]] = {}
    query_to_texts: dict[str, list[str]] = {}
    for idx, pair in enumerate(pairs):
        q = pair[0]
        t = pair[1] if len(pair) > 1 else ""
        if q not in query_to_indices:
            query_to_indices[q] = []
            query_to_texts[q] = []
        query_to_indices[q].append(idx)
        query_to_texts[q].append(t)

    cascade = get_reranker()
    scores = [0.0] * len(pairs)
    for q, idxs in query_to_indices.items():
        texts = query_to_texts[q]
        q_scores = cascade.score(q, texts)
        for original_idx, s in zip(idxs, q_scores):
            scores[original_idx] = float(s)

    return scores


score_pairs_batch = _score_pairs_batch


def _compute_scores(query: str, texts: list[str]) -> Optional[list[float]]:
    """Compute relevance scores for (query, text) pairs using CascadeReranker."""
    if not texts:
        return []

    try:
        cascade = get_reranker()
        return cascade.score(query, texts)
    except Exception as exc:
        logger.warning(
            f"Cascade reranker inference failed ({type(exc).__name__}: {exc}). "
            f"Falling back to original vector-similarity ordering."
        )
        return None


def rerank(
    query: str,
    chunks: list[Union[str, dict[str, Any]]],
    top_k: int,
    tracer: Optional[Any] = None,
) -> list[Union[str, dict[str, Any]]]:
    """Score *chunks* against *query* and return the *top_k* highest-scoring ones."""
    if not chunks:
        return []

    if isinstance(chunks[0], dict):
        return rerank_dicts(query, chunks, top_k=top_k, tracer=tracer)  # type: ignore[arg-type]

    top_k = min(max(1, int(round(top_k))), len(chunks))
    if len(chunks) <= 1:
        return chunks[:top_k]

    t0 = time.perf_counter()
    str_chunks: list[str] = [str(c) for c in chunks]
    scores = _compute_scores(query, str_chunks)
    latency_ms = (time.perf_counter() - t0) * 1000

    if scores is None or len(scores) != len(str_chunks):
        logger.warning(
            f"Reranker failed after {latency_ms:.1f}ms; using original ordering."
        )
        return chunks[:top_k]

    scored = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    top_chunks = [chunk for _, chunk in scored[:top_k]]

    logger.info(
        f"⏱️ Reranker took {latency_ms:.1f}ms ({latency_ms/1000:.2f}s) for {len(chunks)} passages "
        f"| top score={scored[0][0]:.4f} | bottom kept score={scored[top_k - 1][0]:.4f}"
    )

    if tracer and hasattr(tracer, "log_reranker"):
        tracer.log_reranker(
            query=query,
            num_candidates=len(chunks),
            top_k=top_k,
            latency_ms=latency_ms,
            scores=[float(s) for s, _ in scored[:top_k]],
        )

    return top_chunks


def rerank_dicts(
    query: str,
    records: list[dict[str, Any]],
    top_k: int,
    content_key: str = "content",
    tracer: Optional[Any] = None,
) -> list[dict[str, Any]]:
    """Score a list of chunk dictionaries and return the *top_k* highest-scoring ones."""
    if not records:
        return []

    top_k = min(max(1, int(round(top_k))), len(records))
    if len(records) <= 1:
        for r in records[:top_k]:
            if "rerank_score" not in r:
                r["rerank_score"] = float(r.get("similarity", 0.0))
        return records[:top_k]

    t0 = time.perf_counter()
    texts = [str(r.get(content_key, "")) for r in records]
    scores = _compute_scores(query, texts)
    latency_ms = (time.perf_counter() - t0) * 1000

    if scores is None or len(scores) != len(records):
        logger.warning(
            f"Reranker failed after {latency_ms:.1f}ms; using fallback similarity scores."
        )
        for r in records[:top_k]:
            if "rerank_score" not in r:
                r["rerank_score"] = float(r.get("similarity", 0.0))
        return records[:top_k]

    scored_records: list[tuple[float, dict[str, Any]]] = []
    for score, rec in zip(scores, records):
        r_copy = dict(rec)
        r_copy["rerank_score"] = float(score)
        scored_records.append((float(score), r_copy))

    scored_records.sort(key=lambda x: x[0], reverse=True)
    top_records = [rec for _, rec in scored_records[:top_k]]

    logger.info(
        f"⏱️ Reranker took {latency_ms:.1f}ms ({latency_ms/1000:.2f}s) for {len(records)} candidates "
        f"| top score={scored_records[0][0]:.4f} | bottom kept score={scored_records[top_k - 1][0]:.4f}"
    )

    if tracer and hasattr(tracer, "log_reranker"):
        tracer.log_reranker(
            query=query,
            num_candidates=len(records),
            top_k=top_k,
            latency_ms=latency_ms,
            scores=[float(score) for score, _ in scored_records[:top_k]],
        )

    return top_records
