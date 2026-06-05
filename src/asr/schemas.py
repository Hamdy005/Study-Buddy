import asyncio
from dataclasses import dataclass, field


@dataclass
class AudioJob:
    """Single audio transcription job queued for batch inference."""
    job_id: str
    audio_path: str          # path to the temp audio file on disk
    language: str            # "en" | "ar"
    done: asyncio.Event = field(default_factory=asyncio.Event)
