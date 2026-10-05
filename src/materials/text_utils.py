import re
# pyrefly: ignore [missing-import]
import pdfplumber
from loguru import logger
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader


def _clean_text(text: str) -> str:
    if not text:
        return ""
    # Remove NULL bytes (\x00 / \u0000) which PostgreSQL text format cannot accept
    return text.replace("\x00", "").replace("\u0000", "")


def _normalize_pdf_text(text: str) -> str:
    """Normalize raw page text extracted by pdfplumber.

    Steps:
    1. Remove soft hyphens at line-ends (e.g. "fundamen-\\ntal" -> "fundamental").
    2. Convert single newlines (not blank-line separators) to a space.
    3. Collapse 3+ consecutive newlines into exactly "\\n\\n".
    4. Collapse repeated spaces / tabs into one space.
    """
    # 1. Dehyphenation
    text = re.sub(r"-\n(?=\w)", "", text)
    # 2. Inline line-wraps -> space
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # 3. Excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 4. Repeated whitespace
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def text_from_pdf(pdf_file) -> str:
    """Extract text from a PDF file using pdfplumber.

    Accepts any file-like object or bytes-like object (e.g. io.BytesIO).
    Pages are joined with "\\n\\n" so LangChain's RecursiveCharacterTextSplitter
    can split cleanly at page boundaries.

    Raises a logged warning (but does not crash) on corrupted / encrypted PDFs
    and emits a warning if the file appears to be scanned (needs OCR).
    """
    pages_text: list[str] = []
    total_pages: int = 0

    try:
        with pdfplumber.open(pdf_file) as pdf:
            total_pages = len(pdf.pages)
            for page in pdf.pages:
                raw = page.extract_text()
                if raw and raw.strip():
                    pages_text.append(_normalize_pdf_text(raw))
    except Exception as exc:
        logger.error(f"Failed to open or read PDF: {exc}", exc_info=True)
        return ""

    if not pages_text:
        return ""

    full_text = "\n\n".join(pages_text)

    # Scanned-PDF heuristic: < 50 non-whitespace chars per page on average
    char_count = len(full_text.replace("\n", "").replace(" ", ""))
    if total_pages > 0 and (char_count / total_pages) < 50:
        logger.warning(
            f"PDF may be scanned (avg {char_count / total_pages:.1f} non-whitespace "
            f"chars/page across {total_pages} pages). OCR is required for accurate "
            "text extraction — current output will be empty or incomplete."
        )

    return _clean_text(full_text)


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 150):
    cleaned = _clean_text(text)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Prefer paragraph breaks, then sentence endings, then word boundaries.
        # This avoids the default fallback that cuts on raw PDF line-wraps (\n).
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(cleaned)
    return [_clean_text(c) for c in chunks if _clean_text(c).strip()]


def scrap_website(url: str) -> str:
    loader = UnstructuredURLLoader(urls=[url], ssl_verify=True)
    data = loader.load()
    raw = data[0].page_content if data else ""
    return _clean_text(raw)
