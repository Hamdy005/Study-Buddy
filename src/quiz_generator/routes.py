import asyncio
import logging
from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
from src.quiz_generator.quiz import smart_quiz_generator
from src.store import get_material, get_chunks, save_chunks, get_summary, save_quiz, get_quizzes, save_quiz_result, get_quiz_results, check_daily_limit, increment_daily_usage, ADMIN_EMAILS
from src.dependencies import get_current_user_id, get_current_user
from src.config import settings
from .schemas import QuizRequest, QuizResponse, SaveQuizResultRequest
from .constants import (
    MIN_MCQ_COUNT,
    MAX_MCQ_COUNT,
    MIN_TF_COUNT,
    MAX_TF_COUNT,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/quiz", tags=["Quiz"])


@router.get("/list")
async def get_quiz_list(
    material_id: Optional[str] = None,
    user_id: str = Depends(get_current_user_id),
):
    return get_quizzes(material_id=material_id, user_id=user_id)


@router.post("/generate", response_model=QuizResponse)
async def generate_quiz(
    body: QuizRequest,
    user_id: str = Depends(get_current_user_id),
    current_user=Depends(get_current_user),
):
    # Rate limit check
    user_email = current_user.get("email") if isinstance(current_user, dict) else getattr(current_user, "email", None)
    if not check_daily_limit(user_id, email=user_email, limit=20):
        raise HTTPException(429, "Daily limit of 20 requests reached. Come back tomorrow!")

    body.difficulty = body.difficulty.capitalize()
    if body.mcq_count < MIN_MCQ_COUNT or body.mcq_count > MAX_MCQ_COUNT:
        raise HTTPException(400, f"MCQ count must be between {MIN_MCQ_COUNT} and {MAX_MCQ_COUNT}")
    if body.tf_count < MIN_TF_COUNT or body.tf_count > MAX_TF_COUNT:
        raise HTTPException(400, f"True/False count must be between {MIN_TF_COUNT} and {MAX_TF_COUNT}")

    try:
        quiz = None
        material_id = body.material_id

        if body.source_type in ("web", "topic"):
            topic_title = body.topic
            if not topic_title and body.material_id:
                mat = get_material(body.material_id)
                if mat:
                    if mat.get("user_id") != user_id:
                        raise HTTPException(403, "Access denied")
                    topic_title = mat.get("title")

            if not topic_title:
                raise HTTPException(400, "Topic title or valid material_id is required for web-based quiz")

            loop = asyncio.get_event_loop()

            # ── Lazy cache check ──────────────────────────────────────────────
            # If this topic material already has cached chunks (from a prior
            # summary/quiz), use the contextual path (vector retriever).
            # Otherwise fetch web content now, cache it, then generate.
            existing_chunks = []
            if body.material_id:
                existing_chunks = await loop.run_in_executor(None, get_chunks, body.material_id)

            if existing_chunks:
                # Fast path — use cached embeddings via SupabaseRetriever
                logger.info(
                    f"Topic '{topic_title}' has {len(existing_chunks)} cached chunks — "
                    "using contextual quiz path."
                )
                chunks_texts = [c["content"] for c in existing_chunks]
                quiz = await loop.run_in_executor(
                    None,
                    lambda: smart_quiz_generator(
                        difficulty=body.difficulty,
                        mcq_count=body.mcq_count,
                        tf_count=body.tf_count,
                        material_id=body.material_id,  # triggers SupabaseRetriever
                        chunks=chunks_texts,
                    )
                )

            else:
                # First-time path — fetch web content, cache it, then quiz
                logger.info(
                    f"Topic '{topic_title}' has no cached chunks — fetching web content for quiz."
                )
                from src.summary_generator.summary import fetch_web_content
                from src.materials.text_utils import chunk_text
                from src.rag.rag import store_embeddings_async

                raw_content = await loop.run_in_executor(None, fetch_web_content, topic_title)

                chunks_texts = await loop.run_in_executor(
                    None, lambda: chunk_text(raw_content, chunk_size=600, chunk_overlap=100)
                )
                if chunks_texts and body.material_id:
                    chunk_ids = await loop.run_in_executor(
                        None, save_chunks, body.material_id, chunks_texts
                    )
                    await store_embeddings_async(body.material_id, chunk_ids, chunks_texts)
                    logger.info(
                        f"Cached {len(chunks_texts)} chunks for topic '{topic_title}'."
                    )

                quiz = await loop.run_in_executor(
                    None,
                    lambda: smart_quiz_generator(
                        difficulty=body.difficulty,
                        mcq_count=body.mcq_count,
                        tf_count=body.tf_count,
                        topic_title=topic_title,
                        chunks=chunks_texts if chunks_texts else None,
                    )
                )

        elif body.source_type in ("pdf", "url"):
            mat = get_material(body.material_id) if body.material_id else None
            if not mat:
                raise HTTPException(400, f"No {body.source_type} material found")
            if mat.get("user_id") != user_id:
                raise HTTPException(403, "Access denied")

            chunks_list = get_chunks(body.material_id)
            chunks_texts = [c["content"] for c in chunks_list] if chunks_list else []
            summary_record = get_summary(body.material_id)
            summary_text = summary_record["summary"] if summary_record else None

            loop = asyncio.get_event_loop()
            quiz = await loop.run_in_executor(
                None,
                lambda: smart_quiz_generator(
                    difficulty=body.difficulty,
                    mcq_count=body.mcq_count,
                    tf_count=body.tf_count,
                    material_id=body.material_id if mat.get("status") == "ready" else None,
                    summary=summary_text,
                    chunks=chunks_texts,
                )
            )
        else:
            raise HTTPException(400, f"Unknown source_type: {body.source_type}")

        saved = save_quiz(
            user_id=user_id,
            material_id=material_id,
            source_type="web" if body.source_type == "topic" else body.source_type,
            difficulty=body.difficulty,
            mcq_count=body.mcq_count,
            tf_count=body.tf_count,
            quiz_data=quiz,
            model_name=settings.model_name,
        )

        # Only increment limit if the quiz was generated successfully and saved without error
        if not (user_email and user_email in ADMIN_EMAILS):
            increment_daily_usage(user_id)

        return QuizResponse(quiz=quiz, quiz_id=saved["id"])
    except ValueError as e:
        logger.warning(f"Validation error in generate_quiz: {str(e)}")
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Quiz generation failed: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Quiz generation failed: {e}")


@router.post("/save-result")
async def save_result(
    body: SaveQuizResultRequest,
    user_id: str = Depends(get_current_user_id),
):
    save_quiz_result(body.quiz_id, user_id, body.result_data)
    return {"status": "ok"}


@router.get("/results/{quiz_id}")
async def load_results(
    quiz_id: str,
    user_id: str = Depends(get_current_user_id),
):
    # Verify the quiz belongs to the requesting user before returning results
    quizzes = get_quizzes(user_id=user_id)
    if not any(q.get("id") == quiz_id for q in quizzes):
        raise HTTPException(403, "Access denied")
    return get_quiz_results(quiz_id)
