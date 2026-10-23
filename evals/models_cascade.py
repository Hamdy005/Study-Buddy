"""
LLM & Embedding Fallback Cascade for RAGAS Evaluation.

Constructs an LLM fallback cascade:
    1. Gemini 3.5 Flash Lite (gemini-3.5-flash-lite)
    2. Gemini 3.1 Flash Lite (gemini-3.1-flash-lite)

Embeddings:
    gemini-embedding-2 (Google Generative AI)
"""

import os
import asyncio
import logging
import time
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

load_dotenv("config.env")

logger = logging.getLogger("evals.cascade")

from pydantic import PrivateAttr
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings



class FallbackChatModel(BaseChatModel):
    """
    RAGAS-compatible Chat Model wrapper that cascades through models:
        Tier 1: Gemini 3.5 Flash Lite
        Tier 2: Gemini 3.1 Flash Lite

    Guarantees stateful, seamless handover: if a model fails on any evaluation question or sub-metric
    (e.g., rate limit 429, token budget, timeout), the next tier immediately continues the job
    wherever it was using modulus wrapping, preventing metric evaluation failures or NaN scores.
    """
    models: list[BaseChatModel]
    temperature: float = 0.0
    _active_index: int = PrivateAttr(default=0)
    _cooldowns: dict[int, float] = PrivateAttr(default_factory=dict)
    _cooldown_seconds: float = PrivateAttr(default=60.0)

    @property
    def _llm_type(self) -> str:
        return "fallback-chat-model"

    @property
    def model_name(self) -> str:
        if self.models:
            active = self._active_index % len(self.models)
            return getattr(self.models[active], "model_name", None) or getattr(self.models[active], "model", "fallback-chat-model")
        return "fallback-chat-model"

    @property
    def model(self) -> str:
        return self.model_name

    def _get_ordered_models(self) -> list[tuple[int, BaseChatModel]]:
        """Returns the models ordered starting from the currently healthy active tier using modulus wrapping."""
        now = time.time()
        n = len(self.models)
        if n == 0:
            return []

        active = self._active_index % n
        if active in self._cooldowns:
            if now - self._cooldowns[active] > self._cooldown_seconds:
                # Cooldown expired, clear it
                del self._cooldowns[active]
            else:
                # Active tier is still in cooldown; pick the next healthy tier via modulus
                for alt_offset in range(n):
                    cand_idx = (self._active_index + alt_offset) % n
                    if cand_idx not in self._cooldowns or (now - self._cooldowns[cand_idx] > self._cooldown_seconds):
                        self._active_index = cand_idx
                        break

        ordered = []
        for offset in range(n):
            idx = (self._active_index + offset) % n
            ordered.append((idx, self.models[idx]))
        return ordered

    def _generate(
        self,
        messages: list[Any],
        stop: Optional[list[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        last_err = None
        max_attempts = 2

        for attempt in range(max_attempts):
            ordered = self._get_ordered_models()
            for idx, model in ordered:
                model_name = getattr(model, "model_name", None) or getattr(model, "model", f"tier-{idx+1}")
                try:
                    logger.info(f"[Cascade Tier {idx+1}] Invoking sync model '{model_name}'...")
                    call_kwargs: dict[str, Any] = {}
                    if stop:
                        call_kwargs["stop"] = stop
                    if "temperature" in kwargs:
                        call_kwargs["temperature"] = kwargs["temperature"]
                    elif hasattr(model, "temperature"):
                        call_kwargs["temperature"] = self.temperature

                    result = model._generate(messages, **call_kwargs)
                    if result and getattr(result, "generations", None):
                        # Successful call: lock this model in as the healthy active tier
                        self._active_index = idx
                        if idx in self._cooldowns:
                            del self._cooldowns[idx]
                        return result
                    raise RuntimeError(f"Model '{model_name}' returned empty generations.")
                except Exception as e:
                    logger.warning(f"[Cascade Tier {idx+1}] Model '{model_name}' failed ({type(e).__name__}: {e}).")
                    last_err = e
                    self._cooldowns[idx] = time.time()
                    next_idx = (idx + 1) % len(self.models)
                    self._active_index = next_idx
                    next_name = getattr(self.models[next_idx], "model_name", None) or getattr(self.models[next_idx], "model", f"tier-{next_idx+1}")
                    logger.info(
                        f"[Cascade Handover] Seamlessly handing over to '{next_name}' (Tier {next_idx+1}) "
                        f"to continue the evaluation question without interruption..."
                    )

            if attempt < max_attempts - 1:
                time.sleep(2.0)

        raise last_err or RuntimeError("All model cascade tiers failed.")

    async def _agenerate(
        self,
        messages: list[Any],
        stop: Optional[list[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        last_err = None
        max_attempts = 2

        for attempt in range(max_attempts):
            ordered = self._get_ordered_models()
            for idx, model in ordered:
                model_name = getattr(model, "model_name", None) or getattr(model, "model", f"tier-{idx+1}")
                tier_timeout = getattr(model, "request_timeout", None) or getattr(model, "timeout", None) or 30.0
                try:
                    logger.info(f"[Cascade Tier {idx+1}] Invoking async model '{model_name}' (timeout={tier_timeout}s)...")
                    call_kwargs: dict[str, Any] = {}
                    if stop:
                        call_kwargs["stop"] = stop
                    if "temperature" in kwargs:
                        call_kwargs["temperature"] = kwargs["temperature"]
                    elif hasattr(model, "temperature"):
                        call_kwargs["temperature"] = self.temperature

                    # Use asyncio.wait_for with tier_timeout to guarantee provider switching never hangs
                    coro = model._agenerate(messages, **call_kwargs)
                    result = await asyncio.wait_for(coro, timeout=float(tier_timeout))

                    if result and getattr(result, "generations", None):
                        # Successful call: lock this model in as the healthy active tier
                        self._active_index = idx
                        if idx in self._cooldowns:
                            del self._cooldowns[idx]
                        return result
                    raise RuntimeError(f"Model '{model_name}' returned empty generations.")
                except (asyncio.TimeoutError, TimeoutError) as e:
                    logger.warning(f"[Cascade Tier {idx+1}] Model '{model_name}' timed out after {tier_timeout}s.")
                    last_err = TimeoutError(f"Tier {idx+1} ({model_name}) timed out after {tier_timeout}s")
                except Exception as e:
                    logger.warning(f"[Cascade Tier {idx+1}] Model '{model_name}' failed ({type(e).__name__}: {e}).")
                    last_err = e

                # Record failure & cooldown for this model, and seamlessly hand over to the next model via modulus
                self._cooldowns[idx] = time.time()
                next_idx = (idx + 1) % len(self.models)
                self._active_index = next_idx
                next_name = getattr(self.models[next_idx], "model_name", None) or getattr(self.models[next_idx], "model", f"tier-{next_idx+1}")
                logger.info(
                    f"[Cascade Handover] Seamlessly handing over to '{next_name}' (Tier {next_idx+1}) "
                    f"to continue the evaluation question without interruption..."
                )

            if attempt < max_attempts - 1:
                logger.info("[Cascade] All tiers failed in current pass; waiting 2s for transient rate limit window before retry...")
                await asyncio.sleep(2.0)

        raise last_err or RuntimeError("All model cascade tiers failed.")


def build_judge_llm_cascade() -> BaseChatModel:
    """
    Constructs the 2-tier LLM fallback cascade for RAGAS evaluation:
        1. Gemini 3.5 Flash Lite (gemini-3.5-flash-lite)
        2. Gemini 3.1 Flash Lite (gemini-3.1-flash-lite)
    """
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_key:
        raise RuntimeError("GEMINI_API_KEY is required in config.env for evaluation LLM cascade tiers.")

    cascade_tiers: list[BaseChatModel] = [
        # ── Tier 1: Gemini 3.5 Flash Lite ───────────────────────────
        ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite",
            google_api_key=gemini_key,
            temperature=0.0,
            request_timeout=25.0,
            max_retries=0,
        ),
        # ── Tier 2: Gemini 3.1 Flash Lite ───────────────────────────
        ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite",
            google_api_key=gemini_key,
            temperature=0.0,
            request_timeout=25.0,
            max_retries=0,
        ),
    ]

    logger.info(f"Initialized Judge LLM Cascade with {len(cascade_tiers)} tiers: Gemini 3.5 Flash Lite -> Gemini 3.1 Flash Lite.")
    return FallbackChatModel(models=cascade_tiers)


def get_judge_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Returns the Gemini Embedding 2 model."""
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_key:
        raise RuntimeError("GEMINI_API_KEY is required in config.env for Gemini Embedding 2.")

    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2",
        google_api_key=gemini_key,
    )
