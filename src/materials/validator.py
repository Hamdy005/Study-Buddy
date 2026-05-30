import re
import logging
from src.rag.rag import get_llm
from src.config import settings

logger = logging.getLogger(__name__)

# Load English NSFW word list from config settings (kept out of committed code)
LOCAL_NSFW_WORDS = set(settings.local_nsfw_words)

# Load Arabic NSFW word list from config settings (kept out of committed code)
LOCAL_ARABIC_NSFW_WORDS = set(settings.local_arabic_nsfw_words)

def is_local_gibberish(text: str) -> bool:
    text_clean = text.strip().lower()
    
    # Empty or whitespace only
    if not text_clean:
        return True
        
    # Entirely digits
    if re.match(r"^\d+$", text_clean):
        return True
        
    # Repeated characters (e.g., aaaa, ssssss)
    if re.search(r"(.)\1{4,}", text_clean):
        return True
        
    # Consecutive repeating words/patterns (e.g., asd asd asd, hello hello)
    words = text_clean.split()
    if len(words) >= 3 and len(set(words)) == 1:
        return True

    # Consonant-only gibberish (e.g., sdfghjkl, qwrtypsdfg)
    # Allow short names/abbreviations, but flag longer purely consonant strings
    if len(text_clean) > 5 and re.match(r"^[bcdfghjklmnpqrstvwxyz]+$", text_clean):
        return True

    return False

def is_local_nsfw(text: str) -> bool:
    text_clean = text.strip().lower()
    
    # Check Arabic NSFW words
    words = text_clean.split()
    for w in words:
        if w in LOCAL_ARABIC_NSFW_WORDS:
            return True
            
    # Substring checks for high-signal Arabic NSFW roots
    arabic_substrings = {"سكس", "بورن", "شرموط", "منيوك", "قحبة", "قحبه", "متناك"}
    for root in arabic_substrings:
        if root in text_clean:
            return True
            
    # Substring checks for high-signal English NSFW roots
    high_signal_substrings = {"porn", "nude", "sex", "vagina", "penis", "clitoris"}
    for root in high_signal_substrings:
        if root in text_clean:
            return True
            
    # Check for direct matches or common word boundary matches for English
    for word in LOCAL_NSFW_WORDS:
        # If it was already checked as a high-signal substring, skip
        if word in high_signal_substrings:
            continue
        pattern = rf"\b{re.escape(word)}\b"
        if re.search(pattern, text_clean):
            return True
        # Also check obfuscated variations like b**tch, f**k
        obfuscated = word[0] + r"\*+" + word[-1] if len(word) > 2 else ""
        if obfuscated and re.search(rf"\b{obfuscated}\b", text_clean):
            return True
            
    # Check common obfuscated patterns (e.g., f*ck, b*tch, f**k)
    # Match any word that contains asterisks inside it
    if "*" in text_clean:
        # Additional safety check for common swear structures
        for word in ["fuck", "bitch", "shit", "cunt", "asshole", "bastard"]:
            parts = list(word)
            pattern_parts = [parts[0]]
            for char in parts[1:-1]:
                # Allow the character itself, or one or more asterisks
                pattern_parts.append(rf"({re.escape(char)}|\*+)")
            pattern_parts.append(parts[-1])
            pattern = rf"\b{''.join(pattern_parts)}\b"
            if re.search(pattern, text_clean):
                return True
                
    return False

from transformers import pipeline

_translator = None
_gibberish_detector = None
_nsfw_classifier = None

def get_translator():
    global _translator
    if _translator is None:
        logger.info("Loading translation model Helsinki-NLP/opus-mt-ar-en...")
        _translator = pipeline("translation", model="Helsinki-NLP/opus-mt-ar-en", device="cpu")
    return _translator

def get_gibberish_detector():
    global _gibberish_detector
    if _gibberish_detector is None:
        logger.info("Loading gibberish detector model madhurjindal/autonlp-Gibberish-Detector-492513457...")
        _gibberish_detector = pipeline("text-classification", model="madhurjindal/autonlp-Gibberish-Detector-492513457", device="cpu")
    return _gibberish_detector

def get_nsfw_classifier():
    global _nsfw_classifier
    if _nsfw_classifier is None:
        logger.info("Loading NSFW text classifier model michelleli99/NSFW_text_classifier...")
        _nsfw_classifier = pipeline("text-classification", model="michelleli99/NSFW_text_classifier", device="cpu")
    return _nsfw_classifier

def is_arabic(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text))

def validate_topics_batch(texts: list[str]) -> list[str]:
    """
    Validates a batch of topics.
    For each topic text:
      1. Detect if Arabic. If Arabic, translate it to English.
      2. Run the gibberish detector model.
      3. Run the NSFW text classifier model.
    Returns:
      A list of results: "ALLOWED", "gibberish words", or "not safe for work words".
    """
    logger.info(f"Validating batch of {len(texts)} topics...")
    results = ["ALLOWED"] * len(texts)
    
    # 1. Quick local pre-checks
    for i, text in enumerate(texts):
        if is_local_nsfw(text):
            results[i] = "not safe for work words"
        elif is_local_gibberish(text):
            results[i] = "gibberish words"
            
    # Collect indices of texts that passed local checks and need ML model check
    pending_indices = [i for i, res in enumerate(results) if res == "ALLOWED"]
    if not pending_indices:
        return results
        
    processed_texts = [texts[i] for i in pending_indices]
    
    # Translate Arabic inputs to English
    arabic_indices = [i for i, idx in enumerate(pending_indices) if is_arabic(processed_texts[i])]
    if arabic_indices:
        arabic_texts = [processed_texts[i] for i in arabic_indices]
        try:
            translator = get_translator()
            translations = translator(arabic_texts)
            for idx, translation in zip(arabic_indices, translations):
                translated_text = translation.get("translation_text", "").strip()
                processed_texts[idx] = translated_text
                logger.info(f"Translated Arabic topic '{texts[pending_indices[idx]]}' to '{translated_text}'")
                
                # Check translated text against local English NSFW list
                if is_local_nsfw(translated_text):
                    results[pending_indices[idx]] = "not safe for work words"
        except Exception as e:
            logger.error(f"Batch translation failed: {e}", exc_info=True)

    # Run Gibberish detector on all processed texts
    try:
        gibberish_detector = get_gibberish_detector()
        gibberish_preds = gibberish_detector(processed_texts)
        for i, pred in enumerate(gibberish_preds):
            actual_idx = pending_indices[i]
            # Only set if it hasn't already been flagged by translation local NSFW check
            if results[actual_idx] == "ALLOWED":
                label = pred.get("label", "").lower()
                if label in ("noise", "word salad"):
                    results[actual_idx] = "gibberish words"
    except Exception as e:
        logger.error(f"Batch gibberish detection failed: {e}", exc_info=True)

    # Run NSFW text classifier on all processed texts
    try:
        nsfw_classifier = get_nsfw_classifier()
        nsfw_preds = nsfw_classifier(processed_texts)
        for i, pred in enumerate(nsfw_preds):
            actual_idx = pending_indices[i]
            label = pred.get("label", "").lower()
            if "nsfw" in label:
                # NSFW always takes precedence over gibberish detection
                results[actual_idx] = "not safe for work words"
    except Exception as e:
        logger.error(f"Batch NSFW classification failed: {e}", exc_info=True)

    return results

async def validate_topic_input_async(topic: str) -> str:
    """
    Asynchronously validates a topic string by routing it through the validation batch queue.
    """
    import uuid
    from src.rag.batch_workers import validation_queue, validation_job_store
    from src.rag.schemas import ValidationJob

    job = ValidationJob(job_id=str(uuid.uuid4()), text=topic)
    validation_job_store[job.job_id] = {"status": "pending", "result": None, "error": None}
    await validation_queue.put(job)
    await job.done.wait()

    entry = validation_job_store.pop(job.job_id)
    if entry["status"] == "error":
        raise RuntimeError(f"Validation failed: {entry['error']}")

    return entry["result"]

def validate_topic_input(topic: str) -> str:
    """
    Synchronous validation fallback (e.g. for non-async contexts).
    """
    if is_local_nsfw(topic):
        return "not safe for work words"
    if is_local_gibberish(topic):
        return "gibberish words"
        
    res = validate_topics_batch([topic])
    return res[0]

def warmup_validation_models():
    """
    Dummy forward passes to warm up translation, gibberish detection, and NSFW classification models.
    """
    logger.info("Warming up translation and text classification models...")
    try:
        translator = get_translator()
        translator("مرحبا")
    except Exception as e:
        logger.warning(f"Translation warmup failed: {e}")
        
    try:
        gibberish = get_gibberish_detector()
        gibberish("hello")
    except Exception as e:
        logger.warning(f"Gibberish detector warmup failed: {e}")
        
    try:
        nsfw = get_nsfw_classifier()
        nsfw("hello")
    except Exception as e:
        logger.warning(f"NSFW classifier warmup failed: {e}")
    logger.info("Validation models warmup complete.")

