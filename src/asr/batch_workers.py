"""
ASR Batch Workers — Async batching infrastructure for voice transcription.

Architecture (mirrors Raij/src/smart_search/batch_workers.py, audio workers only):
  - Two asyncio.Queues: audio_en_queue and audio_ar_queue.
  - Two worker coroutines per queue drain jobs in micro-batches.
  - Workers offload heavy inference to a thread via run_in_executor.
  - A shared in-memory job_store tracks job status + results.
  - A warmup loop periodically keeps OpenMP threads alive between requests.
    Parakeet warmup runs every PARAKEET_WARMUP_EVERY cycles (full transcribe,
    must use the full pipeline to avoid corrupting TDT decoder state).
    wav2vec2 warmup runs every cycle (cheap raw forward pass).
"""

import asyncio
import time
import uuid
import logging
from typing import Any

from .constants import ASR_BATCH_MAX, ASR_BATCH_WINDOW_S, WARMUP_INTERVAL_S, PARAKEET_WARMUP_EVERY
from .schemas import AudioJob

logger = logging.getLogger(__name__)


# ═══════════════════════ Job Store ════════════════════════

job_store: dict[str, dict[str, Any]] = {}
"""
{
    "<job_id>": {
        "status": "pending" | "processing" | "done" | "error",
        "result": <str transcript> | None,
        "error": <str> | None,
    }
}
"""


def create_job() -> str:
    """Create a new pending job and return its ID."""
    job_id = str(uuid.uuid4())
    job_store[job_id] = {"status": "pending", "result": None, "error": None}
    return job_id


# ═══════════════════════ Request-in-Flight Gate ════════════════════════

_request_in_flight_count = 0


def set_request_in_flight(active: bool):
    """Increment/decrement in-flight counter used to gate warmup cycles."""
    global _request_in_flight_count
    if active:
        _request_in_flight_count += 1
    else:
        _request_in_flight_count = max(0, _request_in_flight_count - 1)


def is_request_in_flight() -> bool:
    return _request_in_flight_count > 0


# ═══════════════════════ Queues ════════════════════════

audio_en_queue: asyncio.Queue[AudioJob] = asyncio.Queue()
audio_ar_queue: asyncio.Queue[AudioJob] = asyncio.Queue()


# ═══════════════════════ Workers ════════════════════════

async def audio_en_worker():
    """
    Drains up to ASR_BATCH_MAX English audio jobs every ASR_BATCH_WINDOW_S seconds.
    Runs one batched Parakeet transcription via run_in_executor.
    Writes transcript into job_store and sets job.done.
    Cleans up temp audio files after processing.
    """
    import os
    from .models import transcribe_en_batch

    loop = asyncio.get_event_loop()

    while True:
        first_job: AudioJob = await audio_en_queue.get()
        batch: list[AudioJob] = [first_job]

        # Collect up to (ASR_BATCH_MAX - 1) more within the time window
        deadline = loop.time() + ASR_BATCH_WINDOW_S
        while len(batch) < ASR_BATCH_MAX:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                job = await asyncio.wait_for(audio_en_queue.get(), timeout=remaining)
                batch.append(job)
            except asyncio.TimeoutError:
                break

        for job in batch:
            job_store[job.job_id]["status"] = "processing"

        try:
            set_request_in_flight(True)
            audio_paths = [job.audio_path for job in batch]
            transcripts = await loop.run_in_executor(None, transcribe_en_batch, audio_paths)

            for job, transcript in zip(batch, transcripts):
                if not transcript.strip():
                    job_store[job.job_id]["status"] = "error"
                    job_store[job.job_id]["error"] = "Could not transcribe audio. Please try again and speak clearly."
                else:
                    job_store[job.job_id]["status"] = "done"
                    job_store[job.job_id]["result"] = transcript
                job.done.set()

        except Exception as e:
            logger.error(f"English ASR batch failed: {e}", exc_info=True)
            for job in batch:
                job_store[job.job_id]["status"] = "error"
                job_store[job.job_id]["error"] = str(e)
                if not job.done.is_set():
                    job.done.set()
        finally:
            set_request_in_flight(False)
            for job in batch:
                try:
                    os.unlink(job.audio_path)
                except Exception:
                    pass


async def audio_ar_worker():
    """
    Drains up to ASR_BATCH_MAX Arabic audio jobs every ASR_BATCH_WINDOW_S seconds.
    Runs one batched wav2vec2 transcription via run_in_executor.
    Writes transcript into job_store and sets job.done.
    Cleans up temp audio files after processing.
    """
    import os
    from .models import transcribe_ar_batch

    loop = asyncio.get_event_loop()

    while True:
        first_job: AudioJob = await audio_ar_queue.get()
        batch: list[AudioJob] = [first_job]

        deadline = loop.time() + ASR_BATCH_WINDOW_S
        while len(batch) < ASR_BATCH_MAX:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                job = await asyncio.wait_for(audio_ar_queue.get(), timeout=remaining)
                batch.append(job)
            except asyncio.TimeoutError:
                break

        for job in batch:
            job_store[job.job_id]["status"] = "processing"

        try:
            set_request_in_flight(True)
            audio_paths = [job.audio_path for job in batch]
            transcripts = await loop.run_in_executor(None, transcribe_ar_batch, audio_paths)

            for job, transcript in zip(batch, transcripts):
                if not transcript.strip():
                    job_store[job.job_id]["status"] = "error"
                    job_store[job.job_id]["error"] = "لم يتم التعرف على الصوت. الرجاء المحاولة مرة أخرى والتحدث بوضوح."
                else:
                    job_store[job.job_id]["status"] = "done"
                    job_store[job.job_id]["result"] = transcript
                job.done.set()

        except Exception as e:
            logger.error(f"Arabic ASR batch failed: {e}", exc_info=True)
            for job in batch:
                job_store[job.job_id]["status"] = "error"
                job_store[job.job_id]["error"] = str(e)
                if not job.done.is_set():
                    job.done.set()
        finally:
            set_request_in_flight(False)
            for job in batch:
                try:
                    os.unlink(job.audio_path)
                except Exception:
                    pass


# ═══════════════════════ Warmup Loop ════════════════════════

async def _asr_warmup_loop():
    """
    Periodically poke both ASR models to prevent OpenMP/MKL thread pool
    spin-down during idle periods.

    - wav2vec2: every WARMUP_INTERVAL_S seconds (raw forward pass, ~5-15ms)
    - Parakeet: every PARAKEET_WARMUP_EVERY cycles (~6 min at 45s/cycle)
      Uses full model.transcribe() to avoid corrupting TDT decoder cache.

    Skipped entirely if a real request is in flight.
    """
    from .models import warmup_parakeet, warmup_wav2vec2

    loop = asyncio.get_event_loop()
    parakeet_cycle = 0

    while True:
        await asyncio.sleep(WARMUP_INTERVAL_S)
        if is_request_in_flight():
            continue

        t0 = time.monotonic()
        try:
            await loop.run_in_executor(None, warmup_wav2vec2)
            parakeet_cycle += 1
            if parakeet_cycle >= PARAKEET_WARMUP_EVERY:
                parakeet_cycle = 0
                await loop.run_in_executor(None, warmup_parakeet)
        except Exception as e:
            logger.warning(f"⚠️ ASR warmup cycle error (non-fatal): {e}")
            continue

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(f"🔥 ASR warmup cycle done in {elapsed_ms:.0f}ms")


# ═══════════════════════ Startup ════════════════════════

_asr_workers_started = False


def start_asr_workers():
    """
    Launch all ASR async worker coroutines. Call once during app startup.
    - 2 English audio workers (Parakeet)
    - 2 Arabic audio workers (wav2vec2)
    - 1 warmup loop
    """
    global _asr_workers_started
    if _asr_workers_started:
        return
    _asr_workers_started = True

    for i in range(2):
        asyncio.create_task(audio_en_worker(), name=f"asr_en_worker_{i}")
    for i in range(2):
        asyncio.create_task(audio_ar_worker(), name=f"asr_ar_worker_{i}")
    asyncio.create_task(_asr_warmup_loop(), name="asr_warmup_loop")

    logger.info("✅ ASR batch workers started (2 EN + 2 AR + warmup loop)")
