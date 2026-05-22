import asyncio
import logging
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional

from src.quiz_generator.quiz import smart_quiz_generator
from src.store import get_material, get_chunks, get_summary, save_quiz, get_quizzes, save_quiz_result, get_quiz_results, check_and_increment_daily_limit
from src.dependencies import get_current_user_id, get_current_user
from src.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/quiz", tags=["Quiz"])


class QuizRequest(BaseModel):
    difficulty: str = "Medium"
    mcq_count: int = 4
    tf_count: int = 3
    source_type: str = "web"
    material_id: Optional[str] = None
    topic: Optional[str] = None


class QuizResponse(BaseModel):
    quiz: dict
    quiz_id: str


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
    if not check_and_increment_daily_limit(user_id, email=user_email, limit=10):
        raise HTTPException(429, "Daily limit of 10 requests reached. Come back tomorrow!")

    body.difficulty = body.difficulty.capitalize()
    if body.mcq_count < 1 or body.mcq_count > 20:
        raise HTTPException(400, "MCQ count must be between 1 and 20")
    if body.tf_count < 1 or body.tf_count > 20:
        raise HTTPException(400, "True/False count must be between 1 and 20")

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
            quiz = await loop.run_in_executor(
                None,
                lambda: smart_quiz_generator(
                    difficulty=body.difficulty,
                    mcq_count=body.mcq_count,
                    tf_count=body.tf_count,
                    topic_title=topic_title,
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

        return QuizResponse(quiz=quiz, quiz_id=saved["id"])
    except ValueError as e:
        logger.warning(f"Validation error in generate_quiz: {str(e)}")
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Quiz generation failed: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Quiz generation failed: {e}")


class SaveQuizResultRequest(BaseModel):
    quiz_id: str
    result_data: dict


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
