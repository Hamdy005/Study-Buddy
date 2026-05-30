import time
import asyncio
import logging
import validators
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks, Header, Request
from postgrest.exceptions import APIError

from src.materials.text_utils import text_from_pdf, chunk_text, scrap_website
from src.rag.rag import store_embeddings, store_embeddings_async
from src.store import create_material, get_material, update_material_status, save_chunks, list_materials, delete_material, rename_material, is_title_taken
from src.dependencies import get_current_user_id, get_current_user
from src.database import get_supabase, get_auth_supabase
from .constants import ALLOWED_TYPES, MAX_SIZE_MB, MAX_SIZE_BYTES
from .schemas import URLInput, RenameMaterialRequest, BulkDeleteRequest, TopicRequest, SearchRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/materials", tags=["Materials"])


def _validate_pdf_upload(file: UploadFile) -> None:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, "Only PDFs allowed")
    size = getattr(file, "size", None)
    if size is None:
        try:
            file.file.seek(0, 2)
            size = file.file.tell()
            file.file.seek(0)
        except Exception:
            size = None
    if size is not None and size > MAX_SIZE_BYTES:
        raise HTTPException(400, "File too large")
    if size is None:
        raise HTTPException(400, "Could not determine file size")


@router.get("")
def get_materials(
    user_id: str = Depends(get_current_user_id),
    current_user=Depends(get_current_user),
):
    return list_materials(user_id)

@router.get("/{material_id}")
def get_material_by_id(
    material_id: str,
    user_id: str = Depends(get_current_user_id),
    current_user=Depends(get_current_user),
):
    mat = get_material(material_id)
    if not mat:
        raise HTTPException(404, "Material not found")
    if mat.get("user_id") != user_id:
        raise HTTPException(403, "Access denied")
    return mat

async def _process_pdf_background(material_id: str, file_content: bytes):
    try:
        loop = asyncio.get_event_loop()
        # Skip processing if this user already has a material with this title
        mat = await loop.run_in_executor(None, get_material, material_id)
        if mat and await loop.run_in_executor(
            None,
            lambda: is_title_taken(mat.get("title", ""), exclude_id=material_id, user_id=mat.get("user_id"))
        ):
            logger.info(f"Skipping processing for {material_id}: duplicate title")
            await loop.run_in_executor(None, update_material_status, material_id, "failed", "Duplicate title. Please rename to retry.")
            return

        from io import BytesIO
        raw = await loop.run_in_executor(None, text_from_pdf, BytesIO(file_content))
        chunks = await loop.run_in_executor(None, chunk_text, raw)
        chunk_ids = await loop.run_in_executor(None, save_chunks, material_id, chunks)

        await loop.run_in_executor(None, update_material_status, material_id, "processing")
        await store_embeddings_async(material_id, chunk_ids, chunks)
        await loop.run_in_executor(None, update_material_status, material_id, "ready")
        logger.info(f"Background processing complete for material {material_id}")
    except Exception as e:
        logger.error(f"Background processing failed for material {material_id}: {e}", exc_info=True)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, update_material_status, material_id, "failed", str(e))

async def _process_url_background(material_id: str, url: str):
    try:
        loop = asyncio.get_event_loop()
        # Skip if this user already has a material with this title
        mat = await loop.run_in_executor(None, get_material, material_id)
        if mat and await loop.run_in_executor(
            None,
            lambda: is_title_taken(mat.get("title", ""), exclude_id=material_id, user_id=mat.get("user_id"))
        ):
            logger.info(f"Skipping URL processing for {material_id}: duplicate title, waiting for rename")
            return

        raw = await loop.run_in_executor(None, scrap_website, url)
        chunks = await loop.run_in_executor(None, lambda: chunk_text(raw, chunk_size=600, chunk_overlap=100))
        chunk_ids = await loop.run_in_executor(None, save_chunks, material_id, chunks)

        await loop.run_in_executor(None, update_material_status, material_id, "processing")
        await store_embeddings_async(material_id, chunk_ids, chunks)
        await loop.run_in_executor(None, update_material_status, material_id, "ready")
        logger.info(f"Background processing complete for URL material {material_id}")
    except Exception as e:
        logger.error(f"Background processing failed for URL material {material_id}: {e}", exc_info=True)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, update_material_status, material_id, "failed", str(e))

@router.post("/upload-pdf")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id),
    current_user=Depends(get_current_user),
):
    _validate_pdf_upload(file)

    try:
        loop = asyncio.get_event_loop()
        content = await file.read()
        material = await loop.run_in_executor(None, lambda: create_material(
            user_id=user_id,
            source_type="pdf",
            title=file.filename,
        ))
        material_id = material["id"]

        background_tasks.add_task(_process_pdf_background, material_id, content)

        return {
            "status": "processing_started",
            "material_id": material_id,
            "title": file.filename,
        }
    except Exception as e:
        logger.error(f"upload_pdf failed: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to start PDF processing: {e}")

@router.post("/scrape-url")
async def scrape_url(
    input: URLInput,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id),
    current_user=Depends(get_current_user),
):
    if not validators.url(input.url):
        raise HTTPException(400, "Invalid URL provided")

    try:
        loop = asyncio.get_event_loop()
        material = await loop.run_in_executor(None, lambda: create_material(
            user_id=user_id,
            source_type="url",
            title=input.url,
            url=input.url,
        ))
        material_id = material["id"]

        # Skip processing if title conflicts within user scope — user must rename first
        is_taken = await loop.run_in_executor(
            None,
            lambda: is_title_taken(input.url, exclude_id=material_id, user_id=user_id)
        )
        if not is_taken:
            background_tasks.add_task(_process_url_background, material_id, input.url)

        return {
            "status": "processing_started",
            "material_id": material_id,
            "title": input.url,
        }
    except Exception as e:
        logger.error(f"scrape_url failed: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to start scraping: {e}")


@router.post("/bulk-delete")
async def bulk_delete_materials(
    body: BulkDeleteRequest,
    user_id: str = Depends(get_current_user_id),
    current_user=Depends(get_current_user),
):
    from concurrent.futures import ThreadPoolExecutor

    def _delete_one(mid: str):
        mat = get_material(mid)
        if mat and mat.get("user_id") == user_id:
            delete_material(mid)

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=min(len(body.material_ids), 8)) as pool:
        await asyncio.gather(
            *[loop.run_in_executor(pool, _delete_one, mid) for mid in body.material_ids]
        )
    return {"status": "ok"}

@router.patch("/{material_id}")
async def rename_material_endpoint(
    material_id: str,
    body: RenameMaterialRequest,
    user_id: str = Depends(get_current_user_id),
    current_user=Depends(get_current_user),
):
    loop = asyncio.get_event_loop()
    mat = await loop.run_in_executor(None, lambda: get_material(material_id))
    if not mat:
        raise HTTPException(404, "Material not found")
    if mat.get("user_id") != user_id:
        raise HTTPException(403, "Not authorized to rename this material")
    # Topics have no URL — renaming is disabled for them
    if mat.get("source_type") == "url" and not mat.get("url"):
        raise HTTPException(403, "Custom topic names cannot be changed")
    new_title = body.title.strip()
    if not new_title:
        raise HTTPException(400, "Title cannot be empty")
    is_taken = await loop.run_in_executor(
        None,
        lambda: is_title_taken(new_title, exclude_id=material_id, user_id=user_id)
    )
    if is_taken:
        raise HTTPException(409, "You already have a material with this title")
    await loop.run_in_executor(None, lambda: rename_material(material_id, new_title))

    # If the material was pending due to title conflict, try processing now
    if mat.get("source_type") == "url" and mat.get("status") == "pending":
        url = mat.get("url")
        if url and not is_title_taken(new_title, exclude_id=material_id, user_id=user_id):
            asyncio.ensure_future(_process_url_background(material_id, url))

    return {"status": "ok"}


from src.materials.validator import validate_topic_input_async

@router.post("/topic")
async def create_topic(
    body: TopicRequest,
    user_id: str = Depends(get_current_user_id)
):
    topic_str = body.topic.strip()
    if not topic_str:
        raise HTTPException(400, "Topic title cannot be empty")
        
    # NSFW and gibberish validation using batch worker
    validation_res = await validate_topic_input_async(topic_str)
    if validation_res != "ALLOWED":
        raise HTTPException(400, validation_res)

    # Rely on the DB-level UNIQUE constraint on (user_id, title)
    try:
        mat = create_material(
            user_id=user_id,
            title=topic_str,
            source_type="topic"
        )
    except APIError as e:
        # Supabase raises APIError with code "23505" on unique-constraint violations
        if "23505" in str(e) or "duplicate" in str(e).lower() or "unique" in str(e).lower():
            raise HTTPException(409, "You already have a material with this title")
        raise HTTPException(500, f"Failed to create topic: {e}")
    update_material_status(mat["id"], "ready", "Topic ready")
    return {"material_id": mat["id"], "title": mat["title"]}



@router.post("/search")
def search_materials(
    request: Request,
    body: SearchRequest,
    user_id: str = Depends(get_current_user_id),
):
    supabase = get_supabase()
    if not supabase:
        return {"results": []}

    result = supabase.rpc(
        "search_materials_by_title",
        {"p_query": body.q, "p_user_id": user_id}
    ).execute()

    return {"results": result.data}
    
@router.delete("/{material_id}")
def delete_material_endpoint(
    material_id: str,
    user_id: str = Depends(get_current_user_id),
    current_user=Depends(get_current_user),
):
    mat = get_material(material_id)
    if not mat:
        raise HTTPException(404, "Material not found")
    if mat.get("user_id") != user_id:
        raise HTTPException(403, "Not authorized to delete this material")
    delete_material(material_id)
    return {"status": "ok"}
