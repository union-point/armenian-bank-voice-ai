import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class RagConfig:
    data_directory: str = os.getenv("RAG_DATA_DIR", "data")

    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "64"))

    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    embed_batch_size: int = int(os.getenv("RAG_EMBED_BATCH_SIZE", "16"))

    top_k: int = int(os.getenv("RAG_TOP_K", "5"))

    def __post_init__(self) -> None:
        Path(self.data_directory).mkdir(parents=True, exist_ok=True)


rag_config = RagConfig()
