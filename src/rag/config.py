"""
CRAG Settings configuration inside src/rag.
"""

from __future__ import annotations

from typing_extensions import Self
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.rag.constants import (
    DEFAULT_CARRY_MAX,
    DEFAULT_FINAL_CONTEXT_N,
    DEFAULT_HIGH_THRESHOLD,
    DEFAULT_KEEP_THRESHOLD,
    DEFAULT_LOW_THRESHOLD,
    DEFAULT_MAX_RETRIES,
    DEFAULT_POOL_MAX,
    DEFAULT_RERANK_WINDOW,
    DEFAULT_RETRIEVE_MULTIPLIER,
    DEFAULT_TOP_K,
    DEFAULT_LLM_MODEL,
    DEFAULT_TRIAGE_TIMEOUT_S,
    DEFAULT_MAX_SUBQUERIES,
    DEFAULT_TOOL_TIMEOUT_SIMPLE_S,
    DEFAULT_TOOL_TIMEOUT_COMPLEX_S,
    RERANKER_QUANTIZED_MODEL,
)


class Settings(BaseSettings):
    """
    CRAG Settings class with strict runtime validation.
    """
    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    MAX_RETRIES: int = Field(default=DEFAULT_MAX_RETRIES, ge=0, le=2)
    TOP_K: int = Field(default=DEFAULT_TOP_K, ge=1, le=50)
    RETRIEVE_MULTIPLIER: int = Field(default=DEFAULT_RETRIEVE_MULTIPLIER, ge=1, le=6)
    RERANK_WINDOW: int = Field(default=DEFAULT_RERANK_WINDOW)
    FINAL_CONTEXT_N: int = Field(default=DEFAULT_FINAL_CONTEXT_N, ge=1, le=20)
    POOL_MAX: int = Field(default=DEFAULT_POOL_MAX, ge=10, le=100)
    CARRY_MAX: int = Field(default=DEFAULT_CARRY_MAX, ge=0, le=10)

    HIGH_THRESHOLD: float = Field(default=DEFAULT_HIGH_THRESHOLD, ge=0.0, le=1.0)
    LOW_THRESHOLD: float = Field(default=DEFAULT_LOW_THRESHOLD, ge=0.0, le=1.0)
    KEEP_THRESHOLD: float = Field(default=DEFAULT_KEEP_THRESHOLD, ge=0.0, le=1.0)

    TRIAGE_TIMEOUT_S: float = Field(default=DEFAULT_TRIAGE_TIMEOUT_S, ge=1.0, le=30.0)
    MAX_SUBQUERIES: int = Field(default=DEFAULT_MAX_SUBQUERIES, ge=1, le=3)
    TOOL_TIMEOUT_SIMPLE_S: float = Field(default=DEFAULT_TOOL_TIMEOUT_SIMPLE_S, ge=1.0, le=60.0)
    TOOL_TIMEOUT_COMPLEX_S: float = Field(default=DEFAULT_TOOL_TIMEOUT_COMPLEX_S, ge=1.0, le=60.0)

    # Integrations
    SUPABASE_URL: str | None = None
    SUPABASE_KEY: str | None = None
    OPENAI_API_KEY: str | None = None
    MISTRAL_API_KEY: str | None = None
    TAVILY_API_KEY: str | None = None
    LLM_MODEL: str = DEFAULT_LLM_MODEL
    RERANKER_MODEL: str = RERANKER_QUANTIZED_MODEL

    @field_validator("RERANK_WINDOW")
    @classmethod
    def validate_rerank_window(cls, v: int) -> int:
        if v not in (384, 512, 1024):
            raise ValueError(f"RERANK_WINDOW must be one of {{384, 512, 1024}}, got {v}")
        return v

    @model_validator(mode="after")
    def validate_constraints(self) -> Self:
        if not (self.LOW_THRESHOLD < self.HIGH_THRESHOLD):
            raise ValueError(
                f"LOW_THRESHOLD ({self.LOW_THRESHOLD}) must be strictly less than "
                f"HIGH_THRESHOLD ({self.HIGH_THRESHOLD})"
            )

        if not (self.LOW_THRESHOLD <= self.KEEP_THRESHOLD <= self.HIGH_THRESHOLD):
            raise ValueError(
                f"KEEP_THRESHOLD ({self.KEEP_THRESHOLD}) must satisfy "
                f"LOW_THRESHOLD ({self.LOW_THRESHOLD}) <= KEEP_THRESHOLD <= HIGH_THRESHOLD ({self.HIGH_THRESHOLD})"
            )

        if not (self.CARRY_MAX <= self.FINAL_CONTEXT_N <= self.POOL_MAX):
            raise ValueError(
                f"Constraint violated: CARRY_MAX ({self.CARRY_MAX}) <= "
                f"FINAL_CONTEXT_N ({self.FINAL_CONTEXT_N}) <= POOL_MAX ({self.POOL_MAX})"
            )

        return self


settings = Settings()
