"""
Batch Workers — Async batching infrastructure for embedding and reranking inference.

Architecture:
  - Single asyncio.Queue for embedding jobs and reranking jobs.
  - Dedicated async worker coroutines drain queues in micro-batches.
  - Workers offload heavy inference to threads via run_in_executor.
  - A shared in-memory job_store dict tracks job status + results.
  - Warmup loop periodically does a dummy forward pass for both embedder
    and reranker to keep PyTorch/OpenMP threads alive during idle periods.
"""

import asyncio
import time
import uuid
from loguru import logger
from typing import Any, Optional

from .constants import (
    BATCH_MAX_SIZE,
    BATCH_WINDOW_S,
    WARMUP_INTERVAL_S,
    RERANK_BATCH_MAX_SIZE,
    RERANK_BATCH_WINDOW_S,
)
from .schemas import EmbeddingJob, RerankJob


# ═══════════════════════ Job Store ════════════════════════

job_store: dict[str, dict[str, Any]] = {}
"""
{
    "<job_id>": {
        "status": "pending" | "processing" | "done" | "error",
        "result": <list[list[float]] for EmbeddingJob | list[dict] for RerankJob> | None,
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


# ═══════════════════════ Queues & Loop ════════════════════════

embedding_queue: asyncio.Queue[EmbeddingJob] = asyncio.Queue()
rerank_queue: asyncio.Queue[RerankJob] = asyncio.Queue()
_main_loop: Optional[asyncio.AbstractEventLoop] = None


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


async def rerank_worker():
    """
    Drains up to {RERANK_BATCH_MAX_SIZE} rerank jobs every {RERANK_BATCH_WINDOW_S * 1000:.0f}ms.

    Batch reranking forward pass:
      1. Collect (query, doc_text) pairs across all jobs in the micro-batch.
      2. Call _score_pairs_batch(all_pairs) in thread pool.
      3. Distribute scores back to jobs, add 'rerank_score', sort descending, slice top_k.
      4. Results are written into job_store and each job's done Event is set.
    """
    from src.rag.reranker import _score_pairs_batch

    loop = asyncio.get_event_loop()

    while True:
        first_job: RerankJob = await rerank_queue.get()
        batch: list[RerankJob] = [first_job]

        deadline = loop.time() + RERANK_BATCH_WINDOW_S
        while len(batch) < RERANK_BATCH_MAX_SIZE:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                job = await asyncio.wait_for(rerank_queue.get(), timeout=remaining)
                batch.append(job)
            except asyncio.TimeoutError:
                break

        try:
            set_request_in_flight(True)

            all_pairs: list[list[str]] = []
            job_pair_counts: list[int] = []

            for job in batch:
                cnt = 0
                for rec in job.records:
                    text = str(rec.get(job.content_key, ""))
                    all_pairs.append([job.query, text])
                    cnt += 1
                job_pair_counts.append(cnt)

            if all_pairs:
                all_scores = await loop.run_in_executor(
                    None, _score_pairs_batch, all_pairs
                )
            else:
                all_scores = []

            idx = 0
            for i, job in enumerate(batch):
                n = job_pair_counts[i]
                job_scores = all_scores[idx : idx + n]
                idx += n

                scored_records: list[tuple[float, dict[str, Any]]] = []
                for s, rec in zip(job_scores, job.records):
                    r_copy = dict(rec)
                    r_copy["rerank_score"] = float(s)
                    scored_records.append((float(s), r_copy))

                scored_records.sort(key=lambda x: x[0], reverse=True)
                top_records = [r for _, r in scored_records[: job.top_k]]

                if job.job_id not in job_store:
                    job_store[job.job_id] = {"status": "pending", "result": None, "error": None}

                job_store[job.job_id].update({
                    "status": "done",
                    "result": top_records,
                })
                job.done.set()

        except Exception as e:
            logger.error(f"Reranking batch failed: {e}", exc_info=True)
            for job in batch:
                fallback_records = []
                for r in job.records[: job.top_k]:
                    r_copy = dict(r)
                    if "rerank_score" not in r_copy:
                        r_copy["rerank_score"] = float(r_copy.get("similarity", 0.0))
                    fallback_records.append(r_copy)

                if job.job_id not in job_store:
                    job_store[job.job_id] = {"status": "error", "result": fallback_records, "error": str(e)}
                else:
                    job_store[job.job_id].update({
                        "status": "error",
                        "result": fallback_records,
                        "error": str(e),
                    })
                if not job.done.is_set():
                    job.done.set()
        finally:
            set_request_in_flight(False)


# ═══════════════════════ Warmup Loop ════════════════════════


async def _warmup_loop():
    """
    Periodically does a dummy forward pass for both embedder and reranker models
    to prevent OpenMP/MKL thread pool spin-down during idle periods.

    Skipped entirely if a real request is in flight.
    """
    from src.rag.rag import warmup_embedder
    from src.rag.reranker import warmup_reranker

    loop = asyncio.get_event_loop()

    while True:
        await asyncio.sleep(WARMUP_INTERVAL_S)
        if is_request_in_flight():
            continue
        t0 = time.monotonic()
        try:
            await loop.run_in_executor(None, warmup_embedder)
            await loop.run_in_executor(None, warmup_reranker)
        except Exception as e:
            logger.warning(f"Warmup cycle error (non-fatal): {e}")
            continue
        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(f"Warmup cycle done (embedder + reranker) ({elapsed_ms:.0f}ms)")


# ═══════════════════════ Startup ════════════════════════

_workers_started = False


def start_workers():
    """
    Launch all async worker coroutines. Call once during app startup.

    - 1 embedding worker (batched SentenceTransformer inference)
    - 1 reranker worker (batched Jina cross-encoder inference)
    - 1 warmup loop (keeps OpenMP threads alive for embedder and reranker)
    """
    global _workers_started, _main_loop
    if _workers_started:
        return
    _workers_started = True

    try:
        _main_loop = asyncio.get_running_loop()
    except RuntimeError:
        _main_loop = asyncio.get_event_loop()

    # Launch worker tasks on the running event loop
    asyncio.create_task(embedding_worker(), name="embedding_worker_0")
    asyncio.create_task(rerank_worker(), name="rerank_worker_0")
    asyncio.create_task(_warmup_loop(), name="warmup_loop")

    logger.info("Batch workers started (embedding worker + rerank worker + warmup loop)")


# ═══════════════════════ Submission Helpers ════════════════════════


async def rerank_async(
    query: str,
    records: list[dict[str, Any]],
    top_k: int,
    content_key: str = "content",
    timeout: float = 30.0,
) -> list[dict[str, Any]]:
    """
    Async interface to route reranking through the batch worker queue.
    Falls back to synchronous rerank_dicts if workers are not started or on timeout/failure.
    """
    from src.rag.reranker import rerank_dicts

    if not records:
        return []
    if len(records) <= 1 or not _workers_started:
        return rerank_dicts(query, records, top_k=top_k, content_key=content_key)

    job_id = str(uuid.uuid4())
    job = RerankJob(
        job_id=job_id,
        query=query,
        records=records,
        top_k=top_k,
        content_key=content_key,
    )
    job_store[job_id] = {"status": "pending", "result": None, "error": None}
    await rerank_queue.put(job)

    try:
        await asyncio.wait_for(job.done.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        job_store.pop(job_id, None)
        logger.warning("rerank_async timed out waiting for batch worker. Falling back to direct rerank_dicts.")
        return rerank_dicts(query, records, top_k=top_k, content_key=content_key)

    entry = job_store.pop(job_id, None)
    if entry and entry.get("result") is not None:
        return entry["result"]
    return rerank_dicts(query, records, top_k=top_k, content_key=content_key)


def rerank_batch(
    query: str,
    records: list[dict[str, Any]],
    top_k: int,
    content_key: str = "content",
    timeout: float = 30.0,
) -> list[dict[str, Any]]:
    """
    Synchronous interface for batched reranking.
    - If called from a worker thread while the main event loop is running, dispatches
      to rerank_async via asyncio.run_coroutine_threadsafe.
    - If called from the main thread event loop or if workers are not started,
      safely calls rerank_dicts directly without blocking the event loop or hanging.
    """
    from src.rag.reranker import rerank_dicts

    if not records:
        return []
    if len(records) <= 1 or not _workers_started or _main_loop is None or not _main_loop.is_running():
        return rerank_dicts(query, records, top_k=top_k, content_key=content_key)

    try:
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if current_loop is _main_loop:
            # Running on main thread event loop: avoid blocking wait, use direct rerank_dicts
            return rerank_dicts(query, records, top_k=top_k, content_key=content_key)

        # Running in a worker thread: dispatch to main event loop queue
        future = asyncio.run_coroutine_threadsafe(
            rerank_async(query, records, top_k=top_k, content_key=content_key, timeout=timeout),
            _main_loop,
        )
        return future.result(timeout=timeout)
    except Exception as e:
        logger.warning(f"rerank_batch dispatch failed ({e}); falling back to direct rerank_dicts.")
        return rerank_dicts(query, records, top_k=top_k, content_key=content_key)

