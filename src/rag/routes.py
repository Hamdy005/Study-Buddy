import time
import asyncio
import logging
from fastapi import APIRouter, HTTPException, Depends
from typing import Optional, Any

from src.rag.rag import rag_answer, extract_chat_title
from src.dependencies import get_current_user, get_current_user_id
from src.store import (
    get_material, get_chunks, get_summary, get_or_create_memory,
    # Session-based chat
    create_chat_session, list_chat_sessions, get_chat_session,
    rename_chat_session, delete_chat_session,
    append_session_message, get_session_messages,
    check_and_increment_daily_limit,
    # Legacy
    save_chat_messages, get_chat_messages,
)
from src.summary_generator.summary import clean_summary
from .schemas import TutorQuery, TutorResponse, SessionRequest, RenameSessionRequest, ExtractTitleRequest, SaveChatRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tutor", tags=["Tutor"])


@router.post("/ask", response_model=TutorResponse)
async def ask_tutor(
    body: TutorQuery,
    user_id: str = Depends(get_current_user_id),
    current_user=Depends(get_current_user),
):
    if not body.query.strip():
        raise HTTPException(400, "Query cannot be empty")

    # Determine memory key (prefer session_id for persistence)
    mem_key = body.session_id or body.memory_id

    # If session_id is provided, seed memory from DB history so context survives restarts
    seed_msgs: list[dict] | None = None
    if body.session_id:
        try:
            seed_msgs = get_session_messages(body.session_id)
        except Exception:
            seed_msgs = None

    memory, memory_id = get_or_create_memory(mem_key, seed_messages=seed_msgs)
    # Persist user message immediately so it's not lost if generation takes time
    if body.session_id:
        try:
            append_session_message(body.session_id, "user", body.query.strip())
        except Exception:
            pass

    start = time.time()
    try:
        loop = asyncio.get_event_loop()
        if body.source_type in ("pdf", "url"):
            mat = get_material(body.material_id) if body.material_id else None
            if not mat:
                raise HTTPException(400, f"No {body.source_type} material found. Upload one first.")
            if mat.get("user_id") != user_id:
                raise HTTPException(403, "Access denied")

            material_id = body.material_id
            
            # Fetch summary to be used as fallback in rag_answer
            mat_summary = get_summary(material_id)
            summary_text = mat_summary.get("summary", "") if mat_summary else ""

            # Use rag_answer for both "ready" and other statuses (if chunks exist)
            answer, memory = await loop.run_in_executor(
                None, 
                lambda: rag_answer(
                    query=body.query, 
                    material_id=material_id, 
                    memory=memory,
                    summaries=summary_text
                )
            )
            source = f"{body.source_type.upper()} (embeddings vector)"

        else:
            answer, memory = await loop.run_in_executor(
                None,
                lambda: rag_answer(query=body.query, material_id=body.material_id, memory=memory)
            )
            source = "Web Search"

    except Exception as e:
        logger.error(f"Error in ask_tutor: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Error generating answer: {e}")

    cleaned_answer = clean_summary(answer)
    elapsed = time.time() - start

    # Safety/refusal responses must not be saved to DB history either.
    _REFUSAL_PREFIXES = (
        "I can't respond on a gibberish",
        "I can't respond on a NSFW",
        "I can't respond on a political",
        "I can't respond on a religious",
    )
    is_refusal = any(cleaned_answer.strip().startswith(p) for p in _REFUSAL_PREFIXES)

    # Persist assistant response (only if not a refusal)
    if body.session_id and not is_refusal:
        try:
            append_session_message(body.session_id, "assistant", cleaned_answer)
        except Exception:
            pass  # don't fail the response if saving fails

    return TutorResponse(answer=cleaned_answer, source=source, time_taken=elapsed, memory_id=memory_id)


# ── Chat Session Routes ──────────────────────────────────


@router.get("/sessions")
async def list_sessions(
    material_id: str,
    user_id: str = Depends(get_current_user_id)
):
    import uuid
    try:
        uuid.UUID(material_id)
    except ValueError:
        return []
    return list_chat_sessions(material_id, user_id)


@router.post("/sessions")
async def create_session(
    body: SessionRequest,
    user_id: str = Depends(get_current_user_id)
):
    import uuid
    try:
        uuid.UUID(body.material_id)
    except ValueError:
        raise HTTPException(400, "Legacy topic format detected. Please delete this topic from your dashboard and recreate it to enable persistent chat.")
    session = create_chat_session(user_id, body.material_id, body.title or "Chat Session")
    return session


@router.get("/sessions/{session_id}/messages")
async def get_messages(
    session_id: str,
    user_id: str = Depends(get_current_user_id),
):
    session = get_chat_session(session_id)
    if not session or session.get("user_id") != user_id:
        raise HTTPException(403, "Access denied")
    messages = get_session_messages(session_id)
    return messages


@router.patch("/sessions/{session_id}")
async def rename_session(
    session_id: str,
    body: RenameSessionRequest,
    user_id: str = Depends(get_current_user_id),
):
    session = get_chat_session(session_id)
    if not session or session.get("user_id") != user_id:
        raise HTTPException(403, "Access denied")
    rename_chat_session(session_id, body.title)
    return {"status": "ok"}


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    user_id: str = Depends(get_current_user_id),
):
    session = get_chat_session(session_id)
    if not session or session.get("user_id") != user_id:
        raise HTTPException(403, "Access denied")
    delete_chat_session(session_id)
    return {"status": "ok"}

@router.post("/sessions/{session_id}/extract-title")
async def extract_title(
    session_id: str,
    body: ExtractTitleRequest,
    user_id: str = Depends(get_current_user_id),
):
    try:
        session = get_chat_session(session_id)
        if not session or session.get("user_id") != user_id:
            raise HTTPException(403, "Access denied")
        
        material_title = None
        if session and session.get("material_id"):
            mat = get_material(session["material_id"])
            if mat:
                material_title = mat.get("title")

        loop = asyncio.get_event_loop()
        title = await loop.run_in_executor(None, lambda: extract_chat_title(body.query, material_title))
        rename_chat_session(session_id, title)
        return {"status": "ok", "title": title}
    except Exception as e:
        logger.error(f"Failed to extract title for session {session_id}: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Failed to extract title: {e}")


# ── Legacy save/load chat (kept for backward compat) ────


@router.post("/chat/save")
async def save_chat(
    body: SaveChatRequest,
    user_id: str = Depends(get_current_user_id),
    current_user=Depends(get_current_user),
):
    save_chat_messages(body.material_id, user_id, body.messages)
    return {"status": "ok"}


@router.get("/chat/{material_id}")
async def load_chat(
    material_id: str,
    current_user=Depends(get_current_user),
):
    messages = get_chat_messages(material_id)
    return {"messages": messages}
