import asyncio
import os
import tempfile
import uuid
import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse

from src.dependencies import get_current_user_id
from src.asr.schemas import AudioJob
from src.asr.batch_workers import audio_en_queue, audio_ar_queue, job_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/asr", tags=["ASR"])

# Allowed audio MIME types from browsers (MediaRecorder output)
_ALLOWED_MIME_PREFIXES = ("audio/", "video/webm")  # webm is video/* but contains audio

_ASR_TIMEOUT_S = 60  # max seconds to wait for a transcription result


def convert_to_wav_16k(input_path: str) -> str:
    """
    Convert an input audio file (e.g. .webm, .ogg, .mp3) to a standard 16kHz mono WAV file
    using ffmpeg. If ffmpeg is not available or fails, returns the original path.
    """
    import subprocess
    import tempfile
    import os

    fd, output_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        "-f", "wav",
        output_path
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        # Delete original temporary file if conversion succeeded
        try:
            os.unlink(input_path)
        except Exception:
            pass
        return output_path
    except Exception as e:
        logger.warning(f"ffmpeg conversion failed: {e}. Falling back to original file.")
        try:
            os.unlink(output_path)
        except Exception:
            pass
        return input_path


@router.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(..., description="Audio recording from the browser (.webm, .wav, .ogg)"),
    language: str = Form(..., description="Language code: 'en' for English, 'ar' for Arabic"),
    user_id: str = Depends(get_current_user_id),
):
    """
    Transcribe an audio file using the selected language model:
    - 'en': nvidia/parakeet-tdt-0.6b-v2 (NeMo)
    - 'ar': IbrahimAmin/egyptian-arabic-wav2vec2-xlsr-53 (HuggingFace)

    The audio is queued for batch inference and the response waits
    for the transcript to be ready (up to 60 seconds).
    """
    if language not in ("en", "ar"):
        raise HTTPException(400, "Invalid language. Must be 'en' or 'ar'.")

    # Validate MIME type loosely (browsers vary on exact content-type for webm)
    content_type = audio.content_type or ""
    if not any(content_type.startswith(prefix) for prefix in _ALLOWED_MIME_PREFIXES):
        logger.warning(f"Unexpected audio content-type: {content_type} — allowing anyway")

    # Save upload to a temp file (workers clean up after processing)
    suffix = ".webm"
    if audio.filename:
        _, ext = os.path.splitext(audio.filename)
        if ext:
            suffix = ext

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            content = await audio.read()
            tmp.write(content)
    except Exception as e:
        logger.error(f"Failed to save audio upload: {e}", exc_info=True)
        raise HTTPException(500, "Failed to save audio file.")

    # Convert to standard 16kHz mono WAV format using ffmpeg
    tmp_path = convert_to_wav_16k(tmp_path)

    # Create job and queue it
    job_id = str(uuid.uuid4())
    job = AudioJob(job_id=job_id, audio_path=tmp_path, language=language)
    job_store[job_id] = {"status": "pending", "result": None, "error": None}

    if language == "en":
        await audio_en_queue.put(job)
    else:
        await audio_ar_queue.put(job)

    # Wait for the worker to finish (timeout = _ASR_TIMEOUT_S)
    try:
        await asyncio.wait_for(job.done.wait(), timeout=_ASR_TIMEOUT_S)
    except asyncio.TimeoutError:
        # Clean up the temp file if the worker hasn't done so
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass
        job_store.pop(job_id, None)
        raise HTTPException(504, "Transcription timed out. Please try a shorter recording.")

    entry = job_store.pop(job_id, {})
    if entry.get("status") == "error":
        raise HTTPException(500, entry.get("error", "Transcription failed."))

    transcript = entry.get("result", "")
    return JSONResponse({"transcript": transcript})
