from pathlib import Path
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

from src.config import settings

def get_vector_store() -> Chroma:
    """Instantiates a persistent Chroma vector database Client."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        google_api_key=settings.GOOGLE_API_KEY
    )
    return Chroma(
        persist_directory=str(settings.CHROMA_DB_DIR),
        embedding_function=embeddings
    )

def ingest_pdf(file_path: Path) -> Dict[str, Any]:
    """Loads a PDF, generates text chunks, and commits them to Chroma."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not file_path.is_file() or file_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file: {file_path}")
    loader = PyPDFLoader(str(file_path))
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(pages)

    if not chunks:
        raise ValueError(f"No extractable text found in PDF: {file_path.name}")

    for chunk in chunks:
        chunk.metadata["filename"] = file_path.name

    vector_store = get_vector_store()
    vector_store.delete(where={"filename": file_path.name})
    vector_store.add_documents(chunks)

    return {
        "filename": file_path.name,
        "total_pages": len(pages),
        "total_chunks": len(chunks)
    }
