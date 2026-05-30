"""
Embedding Batch Workers — Async batching infrastructure for embedding inference.

Architecture:
  - Single asyncio.Queue for all embedding jobs.
  - Dedicated async worker coroutines drain the queue in micro-batches.
  - Workers offload heavy inference to a thread via run_in_executor.
  - A shared in-memory job_store dict tracks job status + results.
  - Warmup loop periodically does a dummy forward pass to keep OpenMP threads alive.
"""

import asyncio
import time
import uuid
import logging
from typing import Any

from .constants import BATCH_MAX_SIZE, BATCH_WINDOW_S, WARMUP_INTERVAL_S
from .schemas import EmbeddingJob

logger = logging.getLogger(__name__)


# ═══════════════════════ Job Store ════════════════════════

job_store: dict[str, dict[str, Any]] = {}
"""
{
    "<job_id>": {
        "status": "pending" | "processing" | "done" | "error",
        "result": <list[list[float]] for EmbeddingJob> | None,
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
    """Increment/decrement in-flight counter. Thread-safe enough for a gate."""
    global _request_in_flight_count
    if active:
        _request_in_flight_count += 1
    else:
        _request_in_flight_count = max(0, _request_in_flight_count - 1)


def is_request_in_flight() -> bool:
    return _request_in_flight_count > 0


# ═══════════════════════ Queue ════════════════════════

embedding_queue: asyncio.Queue[EmbeddingJob] = asyncio.Queue()


# ═══════════════════════ Workers ════════════════════════


async def embedding_worker():
    """
    Drains up to {BATCH_MAX_SIZE} embedding jobs every {BATCH_WINDOW_S * 1000:.0f}ms.

    One SentenceTransformer forward pass per batch:
      1. Collect texts from all jobs in the batch
      2. get_embedder().embed_documents(all_texts) → raw [B, D] embeddings
      3. Distribute results back to individual jobs

    Results are written into job_store and each job's done Event is set.
    """
    from src.rag.rag import get_embedder

    loop = asyncio.get_event_loop()

    while True:
        # Wait for at least one job
        first_job: EmbeddingJob = await embedding_queue.get()
        batch: list[EmbeddingJob] = [first_job]

        # Collect up to 7 more within the time window
        deadline = loop.time() + BATCH_WINDOW_S
        while len(batch) < BATCH_MAX_SIZE:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                job = await asyncio.wait_for(embedding_queue.get(), timeout=remaining)
                batch.append(job)
            except asyncio.TimeoutError:
                break

        try:
            set_request_in_flight(True)

            # Gather all texts from all jobs in the batch
            all_texts: list[str] = []
            text_counts: list[int] = []
            for job in batch:
                all_texts.extend(job.texts)
                text_counts.append(len(job.texts))

            # Single forward pass for the entire batch
            embedder = get_embedder()
            all_embeddings = await loop.run_in_executor(
                None, embedder.embed_documents, all_texts
            )

            # Distribute results back to individual jobs
            idx = 0
            for i, job in enumerate(batch):
                n = text_counts[i]
                job_result = all_embeddings[idx: idx + n]
                idx += n

                # Use .get() or setdefault to avoid KeyError if initialization was missed
                if job.job_id not in job_store:
                    job_store[job.job_id] = {"status": "pending", "result": None, "error": None}
                
                job_store[job.job_id].update({
                    "status": "done",
                    "result": job_result
                })
                job.done.set()

        except Exception as e:
            logger.error(f"Embedding batch failed: {e}", exc_info=True)
            for job in batch:
                if job.job_id not in job_store:
                    job_store[job.job_id] = {"status": "error", "result": None, "error": str(e)}
                else:
                    job_store[job.job_id].update({
                        "status": "error",
                        "error": str(e)
                    })
                # Critical: always set the event so the request doesn't hang
                if not job.done.is_set():
                    job.done.set()
        finally:
            set_request_in_flight(False)



# ═══════════════════════ Warmup Loop ════════════════════════


async def _warmup_loop():
    """
    Periodically does a dummy forward pass to prevent OpenMP/MKL thread pool
    spin-down during idle periods.

    Skipped entirely if a real request is in flight.
    """
    from src.rag.rag import warmup_embedder

    loop = asyncio.get_event_loop()

    while True:
        await asyncio.sleep(WARMUP_INTERVAL_S)
        if is_request_in_flight():
            continue
        t0 = time.monotonic()
        try:
            await loop.run_in_executor(None, warmup_embedder)
        except Exception as e:
            logger.warning(f"Warmup cycle error (non-fatal): {e}")
            continue
        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(f"Warmup cycle done ({elapsed_ms:.0f}ms)")


# ═══════════════════════ Startup ════════════════════════

_workers_started = False


def start_workers():
    """
    Launch all async worker coroutines. Call once during app startup.

    - 1 embedding worker (batched SentenceTransformer inference)
    - 1 warmup loop (keeps OpenMP threads alive)
    """
    global _workers_started
    if _workers_started:
        return
    _workers_started = True

    # Use only 1 worker to save RAM on this environment
    asyncio.create_task(embedding_worker(), name="embedding_worker_0")
    asyncio.create_task(_warmup_loop(), name="warmup_loop")

    logger.info("Batch workers started (embedding worker + warmup loop)")
