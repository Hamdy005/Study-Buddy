import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader


def _clean_text(text: str) -> str:
    if not text:
        return ""
    # Remove NULL bytes (\x00 / \u0000) which PostgreSQL text format cannot accept
    return text.replace("\x00", "").replace("\u0000", "")


def text_from_pdf(pdf_file) -> str:
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted
    return _clean_text(text)


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 150):
    cleaned = _clean_text(text)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(cleaned)
    return [_clean_text(c) for c in chunks if _clean_text(c).strip()]


def scrap_website(url: str) -> str:
    loader = UnstructuredURLLoader(urls=[url], ssl_verify=True)
    data = loader.load()
    raw = data[0].page_content if data else ""
    return _clean_text(raw)
