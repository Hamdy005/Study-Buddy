import asyncio
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from typing import Optional, Any

# Search input schema for tools
class SearchInput(BaseModel):
    query: str = Field(description="The search query or topic to look up")


# Batch worker job dataclass
@dataclass
class EmbeddingJob:
    """Batch embedding of multiple texts (for store_embeddings)."""
    job_id: str
    texts: list[str]
    done: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class ValidationJob:
    """Batch validation of a topic text."""
    job_id: str
    text: str
    done: asyncio.Event = field(default_factory=asyncio.Event)



# Tutor query and response schemas
class TutorQuery(BaseModel):
    query: str
    source_type: str = "web"
    material_id: Optional[str] = None
    session_id: Optional[str] = None   # preferred
    memory_id: Optional[str] = None    # legacy fallback


class TutorResponse(BaseModel):
    answer: str
    source: str
    time_taken: float
    memory_id: str


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
