import time
import json
import asyncio
import jwt as pyjwt
from loguru import logger
from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect, status
from typing import Optional, Any

from src.config import settings
from src.database import get_supabase, get_auth_supabase
from src.rag.rag import rag_answer, extract_chat_title, rag_answer_stream
from src.rag.constants import REFUSAL_PREFIXES
from src.dependencies import get_current_user, get_current_user_id, DEV_USER_ID, _verify_token_cached
from src.store import (
    get_material, get_chunks, get_summary, get_or_create_memory, append_memory_message,
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

router = APIRouter(prefix="/api/tutor", tags=["Tutor"])
ws_router = APIRouter(tags=["WebSocket Chat"])


def _authenticate_ws_token(token: Optional[str]) -> Optional[str]:
    """
    Authenticate a JWT token passed during WebSocket auth handshake.
    Returns user_id string if valid, None otherwise.
    """
    client = get_auth_supabase() or get_supabase()

    # Dev mode — no Supabase configured
    if client is None:
        return DEV_USER_ID

    if not token:
        return None

    # 1. Stateless custom JWT
    try:
        from src.auth.jwt_utils import decode_access_token
        payload = decode_access_token(token)
        user_id = payload.get("sub")
        if user_id:
            return str(user_id)
    except Exception:
        pass

    # 2. Supabase JWT secret verification
    if settings.supabase_jwt_secret:
        try:
            payload = pyjwt.decode(
                token,
                settings.supabase_jwt_secret,
                algorithms=["HS256", "HS384", "HS512"],
                options={"verify_aud": False},
            )
            user_id = payload.get("sub")
            if user_id:
                return str(user_id)
        except Exception:
            pass

    # 3. Cached Supabase user lookup
    try:
        user = _verify_token_cached(client, token)
        if user and getattr(user, "id", None):
            return str(user.id)
    except Exception as e:
        logger.debug(f"WS token validation error: {e}")

    return None


@ws_router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for real-time streaming LLM chat responses.
    
    Handshake Protocol:
    1. Client connects to ws(s)://<host>/ws/chat (No tokens in URL).
    2. Client MUST send an initial JSON message: {"type": "auth", "token": "<JWT>"}
    3. Server verifies token. Sends {"type": "auth_ok"} on success or closes with 1008 on failure.
    4. Client sends chat JSON messages: {"query": "...", "material_id": "...", "session_id": "...", "source_type": "..."}
    5. Server streams token strings one by one, followed by "[DONE]".
    """
    await websocket.accept()

    # Step 1: Two-step Auth Handshake
    try:
        raw_msg = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
        auth_data = json.loads(raw_msg)
    except Exception as e:
        logger.warning(f"WS auth handshake timeout or invalid message: {e}")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Auth timeout or invalid payload")
        return

    if not isinstance(auth_data, dict) or auth_data.get("type") != "auth":
        logger.warning("WS auth handshake failed: first message was not of type 'auth'")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="First message must be type 'auth'")
        return

    token = auth_data.get("token")
    user_id = _authenticate_ws_token(token)

    if not user_id:
        logger.warning("WS auth handshake failed: invalid or expired token")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid authentication token")
        return

    # Send auth confirmation
    await websocket.send_json({"type": "auth_ok"})
    logger.info(f"WebSocket client authenticated successfully for user_id={user_id}")

    # Step 2: Message Loop
    while True:
        try:
            msg_text = await websocket.receive_text()
        except WebSocketDisconnect:
            logger.info(f"WebSocket client disconnected gracefully (user: {user_id}).")
            break
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            break

        try:
            data = json.loads(msg_text)
        except json.JSONDecodeError:
            await websocket.send_json({"type": "error", "message": "Invalid JSON payload"})
            continue

        query = data.get("query", "").strip()
        if not query:
            await websocket.send_json({"type": "error", "message": "Query cannot be empty"})
            continue

        material_id = data.get("material_id")
        session_id = data.get("session_id")
        source_type = data.get("source_type", "topic")

        # Material ownership check for pdf/url sources
        if source_type in ("pdf", "url") and material_id:
            mat = get_material(material_id)
            if not mat:
                await websocket.send_json({"type": "error", "message": f"No {source_type} material found."})
                continue
            if mat.get("user_id") != user_id:
                await websocket.send_json({"type": "error", "message": "Access denied"})
                continue

        # Load / seed memory
        mem_key = session_id or material_id
        seed_msgs = None
        if session_id:
            try:
                seed_msgs = get_session_messages(session_id)
            except Exception:
                seed_msgs = None

        memory, memory_id = get_or_create_memory(mem_key, seed_messages=seed_msgs)

        # Persist user message to session DB immediately
        if session_id:
            try:
                append_session_message(session_id, "user", query)
            except Exception as e:
                logger.warning(f"Failed to append user session message: {e}")

        # Summary fallback
        summary_text = ""
        if material_id:
            mat_summary = get_summary(material_id)
            summary_text = mat_summary.get("summary", "") if mat_summary else ""

        # Stream response
        full_answer_parts = []
        try:
            async for token_chunk in rag_answer_stream(
                query=query,
                material_id=material_id,
                summaries=summary_text,
                memory=memory,
            ):
                full_answer_parts.append(token_chunk)
                await websocket.send_text(token_chunk)

            # Send special completion signal
            await websocket.send_text("[DONE]")

            full_answer = "".join(full_answer_parts)
            cleaned_answer = clean_summary(full_answer)

            is_refusal = any(cleaned_answer.strip().startswith(p) for p in REFUSAL_PREFIXES)

            # Persist assistant response if not a refusal
            if session_id and not is_refusal:
                try:
                    append_session_message(session_id, "assistant", cleaned_answer)
                except Exception as e:
                    logger.warning(f"Failed to append assistant session message: {e}")

            if session_id and not is_refusal:
                append_memory_message(memory_id, "user", query)
                append_memory_message(memory_id, "assistant", cleaned_answer)

        except Exception as e:
            logger.error(f"Error during WS chat streaming for user {user_id}: {e}", exc_info=True)
            await websocket.send_json({"type": "error", "message": f"Error generating answer: {str(e)}"})



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
    is_refusal = any(cleaned_answer.strip().startswith(p) for p in REFUSAL_PREFIXES)

    # Persist assistant response (only if not a refusal)
    if body.session_id and not is_refusal:
        try:
            append_session_message(body.session_id, "assistant", cleaned_answer)
        except Exception:
            pass  # don't fail the response if saving fails

    # Keep Redis memory cache in sync (append user + assistant messages)
    if body.session_id and not is_refusal:
        append_memory_message(memory_id, "user", body.query.strip())
        append_memory_message(memory_id, "assistant", cleaned_answer)

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
