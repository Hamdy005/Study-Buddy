import asyncio
import time
import logging 
from fastapi import APIRouter, HTTPException, Depends

from src.summary_generator.summary import summarizer, web_summarizer
from src.store import get_material, get_chunks, save_summary, get_summary as get_stored_summary, check_daily_limit, increment_daily_usage
from src.dependencies import get_current_user_id, get_current_user
from src.config import settings
from .schemas import SummarizeRequest, SummarizeResponse
from .constants import MAX_COMBINED_TEXT_LEN

router = APIRouter(prefix="/api/materials", tags=["Summarizer"])
logger = logging.getLogger(__name__)


@router.post("/summarize", response_model=SummarizeResponse)
async def generate_summary(
    body: SummarizeRequest,
    user_id: str = Depends(get_current_user_id),
    current_user=Depends(get_current_user),
):
    # Rate limit check
    user_email = current_user.get("email") if isinstance(current_user, dict) else getattr(current_user, "email", None)
    if not check_daily_limit(user_id, email=user_email, limit=20):
        raise HTTPException(429, "Daily limit of 20 requests reached. Come back tomorrow!")

    mat = get_material(body.material_id)
    if not mat:
        raise HTTPException(404, "Material not found")
    if mat.get("user_id") != user_id:
        raise HTTPException(403, "Access denied")

    try:
        start = time.time()
        loop = asyncio.get_event_loop()

        if mat.get("source_type") == "topic":
            topic_title = mat.get("title", "topic")
            summary = await loop.run_in_executor(None, web_summarizer, topic_title)
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

        # Only increment limit if the summary was generated successfully and saved without error
        if not (user_email and user_email in settings.admin_emails if hasattr(settings, "admin_emails") else False):
            increment_daily_usage(user_id)

        return SummarizeResponse(summary=summary, time_taken=elapsed)
    except Exception as e:
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
