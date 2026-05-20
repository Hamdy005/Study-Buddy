import re
import logging
from langchain.prompts import PromptTemplate
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from src.rag.rag import get_llm

logger = logging.getLogger(__name__)


def summarizer_prompt():
    return PromptTemplate(
        input_variables=["input"],
        template="""
You are an expert academic assistant tasked with creating a comprehensive and well-structured summary of educational material.

**INSTRUCTIONS:**
1. Analyze the provided text and identify the main topics, key concepts, and important details.
2. Create a coherent summary that flows logically from introduction to conclusion.
3. Focus on educational value - highlight definitions, theories, examples, and practical applications.
4. Maintain academic tone while ensuring clarity and accessibility.
5. Organize the summary with clear sections and logical progression.
6. If the content contains a list of facts, make sure the final summary presents them as a numbered list.

**STRUCTURE YOUR SUMMARY AS FOLLOWS:**
- **Overview**: Begin with a 2-3 sentence high-level overview of the entire content
- **Key Topics**: List the main topics covered in the material (Use a NUMBERED LIST: 1. , 2. , 3. ...)
- **Detailed Summary**: Provide a comprehensive section-by-section summary covering all important concepts
- **Key Takeaways**: Highlight the most important points, definitions, and conclusions (Use a NUMBERED LIST: 1. , 2. , 3. ...)
- **Educational Value**: Explain how this material helps in understanding the subject

**FORMATTING RULES:**
- Do NOT use markdown tables or pipe characters (|)
- Do NOT use separator lines (---, ===)
- Use [[[[### HEADER ###]]]] for Main Section Headings (e.g. [[[[### Detailed Summary ###]]]])
- Use [[[[>>> HEADER <<<]]]] for Sub-topics or Sub-headings inside a section (e.g. [[[[>>> Introduction <<<]]]])
- Do NOT put punctuation (like colons or periods) at the end of the text inside the markers.
- Use **Text** for important keywords, topics, or terms you want to highlight within paragraphs.
- Use numbered lists or bullet points instead of tables

**CONTENT TO SUMMARIZE:**
{input}

**REMEMBER:**
- Be thorough but concise
- Maintain academic accuracy
- Use clear, educational language
- Focus on what would be most helpful for a student studying this material
- **STRICT RULE:** Ensure that the opening and closing brackets match EXACTLY in number. If you start with [[[[###, you MUST end with ###]]]]. Do not omit any brackets.
""",
    )


def clean_summary(text: str) -> str:
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Remove markdown table separator lines (e.g. |---|---|)
        if re.match(r'^[\s\|]*[-]{2,}[\s\|]*$', stripped):
            continue
        # If it's a table row, just keep it but maybe clean it up a bit
        if re.match(r'^\|.*\|$', stripped):
            parts = [p.strip() for p in stripped.split('|')]
            parts = [p for p in parts if p]
            line = ' | '.join(parts)
        # Note: We NO LONGER strip bold (**) here because the frontend uses it for styling
        cleaned.append(line)
    return '\n'.join(cleaned)


MAX_INPUT_CHARS = 15000


def _truncate_text(text: str, max_chars: int = MAX_INPUT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    first_len = int(max_chars * 0.6)
    last_len = max_chars - first_len
    first_part = text[:first_len]
    last_part = text[-last_len:]
    logger.info(f"Text truncated from {len(text)} to {max_chars} chars")
    return f"{first_part}\n\n...[content truncated]...\n\n{last_part}"


def summarizer(text: str) -> str:
    logger.info(f"Summarizer started for text of length {len(text)}")
    text = _truncate_text(text)
    try:
        prompt = summarizer_prompt()
        llm = get_llm()
        # Modern LCEL syntax
        chain = prompt | llm
        response = chain.invoke({"input": text})
        
        raw_content = response.content
        logger.info(f"Summarizer received response of length {len(raw_content)}")
        logger.debug(f"Raw summary response: {raw_content[:500]}...")
        
        return clean_summary(raw_content)
    except Exception as e:
        logger.error(f"Summarizer failed: {str(e)}", exc_info=True)
        raise


def web_summarizer(topic: str) -> str:
    logger.info(f"Web summarizer started for topic: {topic}")

    all_content = []

    try:
        wiki_api = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=6000)
        wiki_content = wiki_api.run(topic)
        if wiki_content and wiki_content.strip():
            all_content.append(f"--- Wikipedia ---\n{wiki_content}")
    except Exception as e:
        logger.warning(f"Wikipedia search for '{topic}' failed: {e}")

    try:
        arxiv_api = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=7000)
        arxiv_content = arxiv_api.run(topic)
        if arxiv_content and arxiv_content.strip():
            all_content.append(f"--- Arxiv ---\n{arxiv_content}")
    except Exception as e:
        logger.warning(f"Arxiv search for '{topic}' failed: {e}")

    if not all_content:
        logger.warning(f"No content found for topic: {topic}. Falling back to general knowledge.")
        all_content.append(f"Topic: {topic}\n\nPlease provide a comprehensive educational summary of this topic based on your general knowledge.")

    combined = "\n\n".join(all_content)
    logger.info(f"Web search combined text length for topic '{topic}': {len(combined)}")

    return summarizer(combined)
