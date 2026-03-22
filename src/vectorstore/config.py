import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class VectorStoreConfig:
    chromadb_persist_directory: str = os.getenv("CHROMADB_PERSIST_DIR", ".chromadb")
    collection_name: str = os.getenv("CHROMADB_COLLECTION", "knowledge_base")

    def __post_init__(self):
        Path(self.chromadb_persist_directory).mkdir(parents=True, exist_ok=True)


vectorstore_config = VectorStoreConfig()
