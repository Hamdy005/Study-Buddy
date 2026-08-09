import re
from loguru import logger
from src.rag.rag import get_llm
from src.config import settings

# Load English NSFW word list from config settings (kept out of committed code)
LOCAL_NSFW_WORDS = set(settings.local_nsfw_words)

# Load Arabic NSFW word list from config settings (kept out of committed code)
LOCAL_ARABIC_NSFW_WORDS = set(settings.local_arabic_nsfw_words)

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

def validate_topics_batch(texts: list[str]) -> list[str]:
    """
    Validates a batch of topics.
    For each topic text:
      Checks against the local English and Arabic NSFW lists.
    Returns:
      A list of results: "ALLOWED" or "not safe for work words".
    """
    logger.info(f"Validating batch of {len(texts)} topics...")
    results = ["ALLOWED"] * len(texts)
    
    for i, text in enumerate(texts):
        if is_local_nsfw(text):
            results[i] = "not safe for work words"
            
    return results


def validate_topic_input(topic: str) -> str:
    """
    Synchronous validation fallback (e.g. for non-async contexts).
    """
    if is_local_nsfw(topic):
        return "not safe for work words"
        
    return "ALLOWED"

