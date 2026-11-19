from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.getenv("RAG_DATA_DIR", str(ROOT_DIR / "data"))).resolve()
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "qwen3.7-text-embedding-flash")
    generation_model: str = os.getenv("GENERATION_MODEL", "qwen3.7-plus")
    rewrite_model: str = os.getenv("REWRITE_MODEL", "qwen3.8-flash")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "80"))
    top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.32"))
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "30"))

    @property
    def db_path(self) -> Path:
        return self.data_dir / "knowledge.db"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "documents"

    @property
    def index_path(self) -> Path:
        return self.data_dir / "vectors.faiss"


settings = Settings()

