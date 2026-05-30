import re
import logging
from src.rag.rag import get_llm
from src.config import settings

logger = logging.getLogger(__name__)

# Basic English NSFW word list to catch obvious cases instantly
LOCAL_NSFW_WORDS = {
    "porn", "sex", "nude", "bitch", "fuck", "asshole", "cunt", "dick",
    "pussy", "nigger", "faggot", "bastard", "slut", "whore", "cock",
    "boob", "tit", "vagina", "penis", "clitoris"
}

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
    
    # Substring checks for high-signal NSFW roots
    high_signal_substrings = {"porn", "nude", "sex", "vagina", "penis", "clitoris"}
    for root in high_signal_substrings:
        if root in text_clean:
            return True
            
    # Check for direct matches or common word boundary matches
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

def validate_topic_input(topic: str) -> str:
    """
    Validates a topic string.
    Returns:
        str: "ALLOWED" if the topic is valid, or a error message string if blocked.
    """
    # 1. Quick Local Checks
    if is_local_nsfw(topic):
        return "NSFW words, profanity, or slang are not allowed."
        
    if is_local_gibberish(topic):
        return "Topic appears to be gibberish or meaningless text."

    # 2. LLM Check for multi-lingual and advanced cases
    try:
        llm = get_llm()
        prompt = (
            "You are a content filter for an educational application.\n"
            f"Analyze the topic: \"{topic}\"\n\n"
            "Determine if it contains:\n"
            "1. NSFW content, profanity, swearing, slang insults, sexual references, or pornographic terms in ANY language.\n"
            "2. Gibberish, random sequences of characters/numbers (e.g., \"12321321\", \"asdsaba\", \"aaaabbbb\", \"esaejsaioejasoi\").\n"
            "3. Completely meaningless or troll input.\n\n"
            "Respond in one of these two formats:\n"
            "- If allowed: ALLOWED\n"
            "- If blocked: BLOCKED: <reason in English>\n"
            "Do not output any markdown, tags, or extra words. Just the raw text."
        )
        
        response = llm.invoke(prompt)
        result = response.content.strip()
        
        if result == "ALLOWED":
            return "ALLOWED"
        elif result.startswith("BLOCKED:"):
            reason = result.replace("BLOCKED:", "").strip()
            return reason or "Topic is not allowed."
        else:
            # Fallback if the LLM output structure was unexpected
            if "blocked" in result.lower():
                return "Topic contains content that is not allowed."
            return "ALLOWED"
            
    except Exception as e:
        logger.error(f"LLM topic validation failed: {e}", exc_info=True)
        # In case of API failure, fall back to allowing if it passed local checks
        return "ALLOWED"
