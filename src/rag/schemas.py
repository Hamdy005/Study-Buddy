from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from typing import Optional, Any, Literal

# Search input schema for tools
class SearchInput(BaseModel):
    query: str = Field(description="The search query or topic to look up")


# Batch worker job dataclasses
@dataclass
class EmbeddingJob:
    """Batch embedding of multiple texts (for store_embeddings)."""
    job_id: str
    texts: list[str]
    done: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class RerankJob:
    """Batch reranking of query-candidate pairs across concurrent requests."""
    job_id: str
    query: str
    records: list[dict[str, Any]]
    top_k: int
    content_key: str = "content"
    done: asyncio.Event = field(default_factory=asyncio.Event)




# Tutor query and response schemas
class TutorQuery(BaseModel):
    query: str = Field(..., min_length=2, description="User question (at least 2 characters)")
    source_type: str = "web"
    material_id: Optional[str] = None
    session_id: Optional[str] = None   # preferred
    memory_id: Optional[str] = None    # legacy fallback


class TutorResponse(BaseModel):
    answer: str
    source: str
    time_taken: float
    memory_id: str
    run_id: Optional[str] = None   # LangSmith trace ID — used by the feedback endpoint
    intent: Optional[str] = None   # Classified intent (simple, complex, chitchat, unsafe)


class FeedbackRequest(BaseModel):
    """
    Payload for POST /api/tutor/feedback.

    Frontend TODO
    ─────────────
    After receiving a chat response, render 👍 / 👎 buttons next to the
    assistant message. On click, POST this payload to /api/tutor/feedback.
    Disable both buttons after the first submission to prevent double-voting.
    """
    run_id:  str
    score:   int            # +1 = thumbs-up  /  -1 = thumbs-down
    comment: Optional[str] = None   # optional free-text reason


# Chat session models
class SessionRequest(BaseModel):
    material_id: str
    title: Optional[str] = "Chat Session"


class RenameSessionRequest(BaseModel):
    title: str


class ExtractTitleRequest(BaseModel):
    query: str


class SaveChatRequest(BaseModel):
    material_id: str
    messages: list[dict[str, Any]]


# ── CRAG Pipeline Schemas ─────────────────────────────────────────
import hashlib
from typing_extensions import Self, TypedDict
from pydantic import ConfigDict, field_validator, model_validator
from src.rag.constants import (
    MAX_CHUNK_CHARS,
    MAX_MISSING_LEN,
    MAX_SEARCH_QUERY_LEN,
    MIN_SEARCH_QUERY_LEN,
    SourceType,
    VerdictType,
    CauseType,
)

Source = SourceType
Verdict = VerdictType
Cause = CauseType


class Chunk(BaseModel):
    """
    Normalized document passage retrieved from a retrieval source.
    """
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1, description="Stable hash of source + text")
    text: str = Field(..., min_length=1, max_length=MAX_CHUNK_CHARS, description="Passage text")
    source: Source
    url: str | None = Field(default=None, description="Source URL; must be http(s) if provided")
    title: str | None = Field(default=None, description="Source document or page title")
    score: float | None = Field(default=None, ge=0.0, le=1.0, description="Rerank score in [0, 1]")

    @field_validator("text", mode="before")
    @classmethod
    def strip_and_validate_text(cls, v: Any) -> str:
        if isinstance(v, str):
            v = v.strip()
            if not v:
                raise ValueError("Chunk text cannot be empty or whitespace only")
        return v

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str | None) -> str | None:
        if v is not None:
            v_stripped = v.strip()
            if not (v_stripped.startswith("http://") or v_stripped.startswith("https://")):
                raise ValueError("url must begin with http:// or https://")
            return v_stripped
        return v

    @classmethod
    def create(
        cls,
        source: Source,
        text: str,
        url: str | None = None,
        title: str | None = None,
        score: float | None = None,
        chunk_id: str | None = None,
    ) -> Chunk:
        """Create a Chunk with deterministic SHA-256 ID if chunk_id is not supplied."""
        clean_text = text.strip()
        computed_id = (
            chunk_id
            if chunk_id
            else hashlib.sha256(f"{source}:{clean_text}".encode("utf-8")).hexdigest()[:16]
        )
        return cls(
            id=computed_id,
            text=clean_text,
            source=source,
            url=url,
            title=title,
            score=score,
        )


class RouterOutput(BaseModel):
    """
    Structured output from the Router node specifying sources to query and the search query.
    """
    model_config = ConfigDict(extra="forbid")

    sources: list[Source] = Field(
        default_factory=lambda: ["supabase"],
        min_length=1,
        description="List of unique sources to query",
    )
    search_query: str = Field(
        ...,
        min_length=MIN_SEARCH_QUERY_LEN,
        max_length=MAX_SEARCH_QUERY_LEN,
        description="Search query to execute against selected sources",
    )
    tool_queries: dict[str, str] = Field(
        default_factory=dict,
        description="Optional mapping of source name to specialized search query",
    )

    @model_validator(mode="before")
    @classmethod
    def populate_search_query_if_missing(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "search_queries" in data and not data.get("search_query"):
                sqs = data.get("search_queries")
                if isinstance(sqs, list) and sqs:
                    data["search_query"] = str(sqs[0]).strip()
            search_q = data.get("search_query")
            tool_q = data.get("tool_queries")
            if (search_q is None or not str(search_q).strip()) and isinstance(tool_q, dict) and tool_q:
                first_q = next(iter(tool_q.values()), None)
                if first_q:
                    data["search_query"] = str(first_q).strip()
        return data

    @field_validator("search_query", mode="before")
    @classmethod
    def strip_search_query(cls, v: Any) -> str:
        if isinstance(v, str):
            v = v.strip()
        return v

    @field_validator("sources")
    @classmethod
    def validate_sources(cls, v: list[Source]) -> list[Source]:
        if not v:
            raise ValueError("sources list cannot be empty")
        if len(v) != len(set(v)):
            raise ValueError("sources list cannot contain duplicate sources")
        return v


class EvaluatorOutput(BaseModel):
    """
    Structured output from the Evaluator node judging context quality.
    """
    model_config = ConfigDict(extra="forbid")

    verdict: Verdict = Field(..., description="Relevance and sufficiency verdict")
    missing: str = Field(
        ...,
        max_length=MAX_MISSING_LEN,
        description="Description of what is missing; non-empty unless verdict is 'correct'",
    )
    likely_cause: Cause = Field(
        ...,
        description="Root cause of retrieval gap; 'none' if and only if verdict is 'correct'",
    )

    @field_validator("missing", mode="before")
    @classmethod
    def strip_missing(cls, v: Any) -> str:
        if isinstance(v, str):
            v = v.strip()
            if len(v) > MAX_MISSING_LEN:
                v = v[:MAX_MISSING_LEN].rstrip()
            return v
        return v

    @model_validator(mode="after")
    def validate_verdict_consistency(self) -> Self:
        if self.verdict == "correct":
            if self.likely_cause != "none":
                raise ValueError("likely_cause must be 'none' when verdict is 'correct'")
        else:
            if not self.missing or not self.missing.strip():
                raise ValueError(
                    f"missing must be non-empty when verdict is '{self.verdict}'"
                )
            if self.likely_cause == "none":
                raise ValueError(
                    f"likely_cause cannot be 'none' when verdict is '{self.verdict}'"
                )
        return self


Intent = Literal["chitchat", "unsafe", "simple", "complex"]
UnsafeType = Literal["none", "gibberish", "nsfw", "political", "religious"]


class TriageOutput(BaseModel):
    """
    Structured output from the triage node classifying user intent.
    """
    model_config = ConfigDict(extra="forbid")

    intent: Literal["chitchat", "unsafe", "simple", "complex"]
    unsafe_type: Literal["none", "gibberish", "nsfw", "political", "religious"]
    sub_questions: list[str] = Field(default_factory=list)

    @field_validator("sub_questions")
    @classmethod
    def validate_sub_questions(cls, v: list[str]) -> list[str]:
        if not isinstance(v, list):
            raise ValueError("sub_questions must be a list")
        if len(v) > 3:
            raise ValueError(f"At most 3 sub_questions allowed, got {len(v)}")
        cleaned: list[str] = []
        for item in v:
            if not isinstance(item, str):
                raise ValueError("Each sub_question must be a string")
            s = item.strip()
            if not (3 <= len(s) <= 200):
                raise ValueError(f"Each sub_question must be 3..200 chars, got {len(s)}")
            cleaned.append(s)
        return cleaned

    @model_validator(mode="after")
    def validate_triage(self) -> Self:
        if self.intent == "unsafe":
            if self.unsafe_type == "none":
                raise ValueError("unsafe_type cannot be 'none' when intent is 'unsafe'")
        else:
            if self.unsafe_type != "none":
                raise ValueError("unsafe_type must be 'none' when intent is not 'unsafe'")

        if self.intent == "complex":
            if not self.sub_questions:
                raise ValueError("sub_questions must be non-empty (1..3) when intent is 'complex'")
        else:
            if self.sub_questions:
                raise ValueError(f"sub_questions must be empty list for intent '{self.intent}'")

        return self


class GraphState(TypedDict, total=False):
    """
    Full state maintained across the LangGraph CRAG pipeline.
    """
    original_q: str                    # min_length=2, max_length=2000, never mutated
    intent: Literal["chitchat", "unsafe", "simple", "complex"] | None
    sub_questions: list[str]           # only for complex, max MAX_SUBQUERIES
    reply: str | None                  # canned text for chitchat/unsafe
    search_query: str                  # current (possibly rewritten) query
    sources_to_use: list[Source]
    tried_sources: list[Source]        # unique, only grows
    pool: list[Chunk]                  # replaced by merge_pool, never by a plain overwrite elsewhere
    ranked: list[Chunk]                # top FINAL_CONTEXT_N after rerank, score set
    verdict: Verdict | None
    missing: str
    likely_cause: Cause
    retries: int                       # 0 <= retries <= MAX_RETRIES + 1
    answer: str | None
    low_confidence: bool
    citations: list[str]
    raw_results: list[dict[str, Any]]
    new_chunks: list[Chunk]
    all_retrievers_failed: bool
    tracer: Any                        # Optional ChatTracer instance for LangSmith observability
    material_id: str | None            # Optional material ID for targeted document retrieval
    token_queue: Any                   # Optional asyncio.Queue for streaming response tokens
    tool_queries: dict[str, str]       # Optional per-source specialized search queries
    on_intent: Any                     # Optional callback invoked when intent is classified


class FinalAnswer(BaseModel):
    """
    Final output returned to caller.
    """
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(..., min_length=1, description="Synthesized answer text")
    citations: list[str] = Field(default_factory=list, description="Citations used in answer")
    low_confidence: bool = Field(default=False, description="Flag indicating low-confidence fallback")
    run_id: str | None = Field(default=None, description="LangSmith trace run ID")
    intent: str | None = Field(default=None, description="Classified intent from triage")
