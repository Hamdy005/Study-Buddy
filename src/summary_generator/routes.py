import asyncio
import time
from loguru import logger
from fastapi import APIRouter, HTTPException, Depends

from src.summary_generator.summary import summarizer, web_summarizer, fetch_web_content
from src.materials.text_utils import chunk_text
from src.rag.rag import store_embeddings_async
from src.store import (
    get_material, get_chunks, save_chunks, save_summary,
    get_summary as get_stored_summary, update_material_status,
    atomic_check_and_increment_daily_limit, decrement_daily_usage, ADMIN_EMAILS,
)
from src.dependencies import get_current_user_id, get_current_user
from src.config import settings
from .schemas import SummarizeRequest, SummarizeResponse
from .constants import MAX_COMBINED_TEXT_LEN

router = APIRouter(prefix="/api/materials", tags=["Summarizer"])


@router.post("/summarize", response_model=SummarizeResponse)
async def generate_summary(
    body: SummarizeRequest,
    user_id: str = Depends(get_current_user_id),
    current_user=Depends(get_current_user),
):
    # Atomically reserve a daily request slot before starting expensive generation.
    user_email = current_user.get("email") if isinstance(current_user, dict) else getattr(current_user, "email", None)
    is_admin = bool(user_email and user_email in ADMIN_EMAILS)
    if not atomic_check_and_increment_daily_limit(user_id, email=user_email, limit=20):
        raise HTTPException(429, "Daily limit of 20 requests reached. Come back tomorrow!")

    mat = get_material(body.material_id)
    if not mat:
        if not is_admin:
            decrement_daily_usage(user_id)
        raise HTTPException(404, "Material not found")
    if mat.get("user_id") != user_id:
        if not is_admin:
            decrement_daily_usage(user_id)
        raise HTTPException(403, "Access denied")

    try:
        start = time.time()
        loop = asyncio.get_event_loop()

        if mat.get("source_type") == "topic":
            topic_title = mat.get("title", "topic")

            # ── Lazy cache check ───────────────────────────────────────────────
            # Check whether we already have stored chunks for this topic.
            # If yes: use them directly (no web fetching).
            # If no: fetch web content, chunk + embed it, then summarize.
            existing_chunks = await loop.run_in_executor(None, get_chunks, body.material_id)

            if existing_chunks:
                # Fast path — use cached chunks
                logger.info(
                    f"Topic '{topic_title}' has {len(existing_chunks)} cached chunks — "
                    "skipping web fetch for summarization."
                )
                combined = "\n".join(c["content"] for c in existing_chunks)
                if len(combined) > MAX_COMBINED_TEXT_LEN:
                    half = MAX_COMBINED_TEXT_LEN // 2
                    combined = combined[:half] + combined[-half:]
                summary = await loop.run_in_executor(None, summarizer, combined)

            else:
                # First-time path — fetch, cache, then summarize
                logger.info(
                    f"Topic '{topic_title}' has no cached chunks — fetching web content."
                )
                raw_content = await loop.run_in_executor(None, fetch_web_content, topic_title)

                # Chunk + store + embed (mirrors the URL pipeline)
                chunks_texts = await loop.run_in_executor(
                    None, lambda: chunk_text(raw_content, chunk_size=600, chunk_overlap=100)
                )
                if chunks_texts:
                    chunk_ids = await loop.run_in_executor(
                        None, save_chunks, body.material_id, chunks_texts
                    )
                    await store_embeddings_async(body.material_id, chunk_ids, chunks_texts)
                    logger.info(
                        f"Cached {len(chunks_texts)} chunks for topic '{topic_title}'."
                    )

                # Generate summary from the raw fetched content
                summary = await loop.run_in_executor(
                    None, web_summarizer, topic_title, raw_content
                )

        else:
            chunks_list = get_chunks(body.material_id)
            if not chunks_list:
                raise HTTPException(400, "No text chunks found in this material")
            combined = "\n".join(c["content"] for c in chunks_list)
            if len(combined) > MAX_COMBINED_TEXT_LEN:
                half_len = MAX_COMBINED_TEXT_LEN // 2
                combined = combined[:half_len] + combined[-half_len:]
            summary = await loop.run_in_executor(None, summarizer, combined)

        elapsed = time.time() - start

        save_summary(
            material_id=body.material_id,
            user_id=user_id,
            summary=summary,
            time_taken=elapsed,
            model_name=settings.model_name,
        )

        return SummarizeResponse(summary=summary, time_taken=elapsed)
    except HTTPException:
        # Refund on any HTTP exception raised inside the try block (e.g. 400).
        if not is_admin:
            decrement_daily_usage(user_id)
        raise
    except Exception as e:
        if not is_admin:
            decrement_daily_usage(user_id)
        raise HTTPException(500, f"Summarization failed: {e}")


@router.get("/{material_id}/summary")
async def get_material_summary(
    material_id: str,
    user_id: str = Depends(get_current_user_id),
    current_user=Depends(get_current_user),
):
    mat = get_material(material_id)
    if not mat or mat.get("user_id") != user_id:
        raise HTTPException(403, "Access denied")

    summary = get_stored_summary(material_id)
    if not summary:
        logger.info("no summary found")
        return {"summary": None, "time_taken": 0}
    return {"summary": summary["summary"], "time_taken": summary.get("time_taken", 0)}
