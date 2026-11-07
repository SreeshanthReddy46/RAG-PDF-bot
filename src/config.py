import os
from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

class Settings(BaseSettings):
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    CHROMA_DB_DIR: Path = Path(os.getenv("CHROMA_DB_DIR", "./data/chroma_db"))
    UPLOAD_DIR: Path = Path(os.getenv("UPLOAD_DIR", "./data/uploads"))

    EMBEDDING_MODEL: str = "models/text-embeddings-004"
    LLM_MODEL: str = "gemini-1.5-pro"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150
    TOP_K_RETRIEVAL: int = 4

settings = Settings()

if not settings.CHROMA_DB_DIR.is_absolute():
    settings.CHROMA_DB_DIR = (_PROJECT_ROOT / settings.CHROMA_DB_DIR).resolve()
if not settings.UPLOAD_DIR.is_absolute():
    settings.UPLOAD_DIR = (_PROJECT_ROOT / settings.UPLOAD_DIR).resolve()

def _ensure_directory(path: Path) -> None:
    """Create a configured data directory and reject conflicting files clearly."""
    if path.exists() and not path.is_dir():
        # Empty files can be left behind by a failed initialization; remove only
        # those harmless placeholders so the configured directory can be used.
        if path.stat().st_size == 0:
            path.unlink()
        else:
            raise NotADirectoryError(
                f"Configured data directory is a file: {path}. "
                "Move or rename it, then restart the application."
            )
    path.mkdir(parents=True, exist_ok=True)


_ensure_directory(settings.CHROMA_DB_DIR)
_ensure_directory(settings.UPLOAD_DIR)
