"""
ASR Models — English (Parakeet) and Arabic (wav2vec2) singletons.

Strategy (ported from Raij/src/models.py, hotword biasing removed):
  - Lazy-loaded singletons: models load on first call, not at import time.
  - Warmup on load: a silent audio pass pre-JITs the computation graph so
    the first real request has the same latency as subsequent ones.
  - Batch inference: both transcribe_*_batch functions accept a list of
    audio file paths and run a single forward pass.
  - Thread lock on Parakeet: model.transcribe() is stateful (TDT decoder),
    so we serialize all calls behind _en_model_lock.
  - Warmup functions (warmup_parakeet / warmup_wav2vec2) are called
    periodically by the batch worker warmup loop to prevent OpenMP
    thread pool spin-down during idle periods.
"""

import os
import threading
from loguru import logger

# Force PyTorch path, no TensorFlow
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

# Cap OpenMP threads — adjust if running on a GPU server with more cores
os.environ.setdefault("OMP_NUM_THREADS", "2")

import torch
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
torch.set_num_interop_threads(1)

_audio_model_en = None
_audio_model_ar = None
_en_model_lock = threading.Lock()
_ar_model_lock = threading.Lock()


# ═══════════════════════ English ASR (Parakeet) ════════════════════════

def get_audio_model_en():
    """
    Loads nvidia/parakeet-tdt-0.6b-v2 via NeMo.
    Runs a 1-second silence warmup to pre-JIT internal computation graphs.
    Uses greedy_batch decoding strategy for best throughput.
    No hotword biasing — general-purpose decoding.
    """
    global _audio_model_en
    if _audio_model_en is not None:
        return _audio_model_en

    import wave
    import tempfile
    import nemo.collections.asr as nemo_asr

    logger.info("Loading English ASR model (nvidia/parakeet-tdt-0.6b-v2)...")
    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")

    # ── Warmup: transcribe 1s of silence to pre-JIT computation graphs ──
    warmup_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            warmup_path = f.name
        with wave.open(warmup_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00" * 32000)  # 1s of silence at 16kHz
        model.freeze()
        with torch.no_grad():
            model.transcribe([warmup_path])
        model.unfreeze()
        logger.info("✅ Parakeet warmup complete")
    except Exception as e:
        logger.warning(f"⚠️ Parakeet warmup failed (non-fatal): {e}")
    finally:
        if warmup_path:
            try:
                os.unlink(warmup_path)
            except Exception:
                pass

    # ── Switch to greedy_batch for speed (no hotword biasing) ──
    try:
        from omegaconf import OmegaConf
        decoding_cfg = OmegaConf.structured(model.cfg.decoding)
        OmegaConf.update(decoding_cfg, "strategy", "greedy_batch")
        if hasattr(decoding_cfg, "greedy"):
            OmegaConf.update(decoding_cfg, "greedy.max_symbols", 5)
        try:
            model.change_decoding_strategy(decoding_cfg, verbose=False)
            logger.info("✅ Parakeet decoding strategy: greedy_batch")
        except Exception as strat_e:
            logger.warning(f"⚠️ greedy_batch strategy failed ({strat_e}), using default")
    except Exception as e:
        logger.warning(f"⚠️ Parakeet decoding strategy setup failed (non-fatal): {e}")

    _audio_model_en = model
    logger.info("✅ English ASR model (Parakeet) loaded and ready")
    return _audio_model_en


def transcribe_en_batch(audio_paths: list[str]) -> list[str]:
    """
    Batch transcription for English audio using Parakeet.
    NeMo's model.transcribe() natively handles batching internally —
    it pads to the same length and runs a single forward pass.
    Serialized behind _en_model_lock (Parakeet TDT decoder is stateful).
    Returns a list of transcription strings (one per input path).
    """
    model = get_audio_model_en()
    with _en_model_lock:
        with torch.no_grad():
            transcriptions = model.transcribe(audio_paths)
            if isinstance(transcriptions, tuple):
                transcriptions = transcriptions[0]
            return [
                (t.text if hasattr(t, "text") else str(t)).strip().rstrip(".")
                for t in transcriptions
            ]


def warmup_parakeet():
    """
    Keeps Parakeet's OpenMP threads alive via a 0.5s silence transcription.
    Must use model.transcribe() (not raw encoder) to avoid corrupting the
    TDT decoder cache. No-op if model is not yet loaded.
    """
    if _audio_model_en is None:
        return
    import wave
    import tempfile
    model = _audio_model_en
    warmup_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            warmup_path = f.name
        with wave.open(warmup_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00" * 16000)  # 0.5s of silence
        with _en_model_lock:
            model.eval()
            with torch.no_grad():
                model.transcribe([warmup_path])
    except Exception as e:
        logger.warning(f"⚠️ Parakeet warmup error (non-fatal): {e}")
    finally:
        if warmup_path:
            try:
                os.unlink(warmup_path)
            except Exception:
                pass


# ═══════════════════════ Arabic ASR (wav2vec2) ════════════════════════

def get_audio_model_ar():
    """
    Loads IbrahimAmin/egyptian-arabic-wav2vec2-xlsr-53 via HuggingFace Transformers.
    Runs a 0.5s dummy forward pass to warm up OpenMP threads.
    No hotword biasing — pure greedy argmax decoding.
    """
    global _audio_model_ar
    if _audio_model_ar is not None:
        return _audio_model_ar

    import numpy as np
    from transformers import Wav2Vec2ForCTC, AutoProcessor

    model_name = "IbrahimAmin/egyptian-arabic-wav2vec2-xlsr-53"
    logger.info(f"Loading Arabic ASR model ({model_name})...")

    processor = AutoProcessor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    model.eval()

    # ── Warmup: single forward pass on dummy audio ──
    try:
        dummy = np.zeros(8000, dtype=np.float32)
        warmup_inputs = processor(
            [dummy], sampling_rate=16000, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            model(**warmup_inputs)
        logger.info("✅ Arabic ASR warmup complete")
    except Exception as e:
        logger.warning(f"⚠️ Arabic ASR warmup failed (non-fatal): {e}")

    _audio_model_ar = {"model": model, "processor": processor, "model_name": model_name}
    logger.info("✅ Arabic ASR model (wav2vec2) loaded and ready")
    return _audio_model_ar


def _load_audio_file(path: str):
    """
    Load an audio file to a 16kHz mono float32 numpy array.
    Tries soundfile first (fast, no subprocess), then falls back to
    librosa (handles more formats including webm via ffmpeg backend).
    """
    import numpy as np
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != 16000:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=16000)
        return data.astype(np.float32)
    except Exception:
        import librosa
        data, _ = librosa.load(path, sr=16000, mono=True)
        return data.astype(np.float32)


def transcribe_ar_batch(audio_paths: list[str]) -> list[str]:
    """
    Batch transcription for Arabic audio using wav2vec2.
    Loads all audio concurrently, pads to same length,
    runs one forward pass, and decodes via greedy argmax.
    Returns a list of transcription strings (one per input path).
    """
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor

    ar = get_audio_model_ar()
    model = ar["model"]
    processor = ar["processor"]

    # Load all waveforms concurrently
    with ThreadPoolExecutor(max_workers=min(len(audio_paths), 8)) as executor:
        waveforms_raw = list(executor.map(_load_audio_file, audio_paths))

    # Guard against empty waveforms from failed decodes
    final_texts = [""] * len(audio_paths)
    valid_indices: list[int] = []
    valid_waveforms: list[np.ndarray] = []

    for idx, wav in enumerate(waveforms_raw):
        arr = np.asarray(wav, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            logger.warning(f"⚠️ Arabic ASR: empty waveform at index {idx}, skipping")
            continue
        valid_indices.append(idx)
        valid_waveforms.append(arr)

    if not valid_waveforms:
        return final_texts

    inputs = processor(
        valid_waveforms,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )

    with _ar_model_lock:
        with torch.no_grad():
            outputs = model(**inputs)

    predicted_ids = torch.argmax(outputs.logits, dim=-1)
    transcriptions = processor.batch_decode(predicted_ids)

    for local_i, text in enumerate(transcriptions):
        final_texts[valid_indices[local_i]] = text.strip().rstrip(".")

    return final_texts


def warmup_wav2vec2():
    """
    Lightweight raw forward pass to keep wav2vec2's OpenMP threads alive.
    No-op if the Arabic model is not yet loaded.
    """
    if _audio_model_ar is None:
        return
    import numpy as np
    ar = _audio_model_ar
    dummy = np.zeros(8000, dtype=np.float32)
    inputs = ar["processor"](
        [dummy], sampling_rate=16000, return_tensors="pt", padding=True
    )
    with torch.no_grad():
        ar["model"](**inputs)
