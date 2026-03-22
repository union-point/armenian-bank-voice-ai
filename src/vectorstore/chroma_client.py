from typing import Optional

import chromadb
from chromadb.config import Settings

from .config import vectorstore_config


class ChromaClient:
    _instance: Optional["ChromaClient"] = None
    _client: Optional[chromadb.PersistentClient] = None
    _collection = None

    def __new__(cls) -> "ChromaClient":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        persist_directory = (
            persist_directory or vectorstore_config.chromadb_persist_directory
        )
        collection_name = collection_name or vectorstore_config.collection_name

        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )

        if self._collection is None:
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Bank knowledge base for RAG"},
            )

    def get_collection(self):
        if self._collection is None:
            self.initialize()
        return self._collection

    def delete_collection(self) -> None:
        if self._client is not None:
            self._client.delete_collection(vectorstore_config.collection_name)
            self._collection = None

    def reset(self) -> None:
        if self._client is not None:
            self._client.reset()
            self._collection = None

    def count(self) -> int:
        collection = self.get_collection()
        return collection.count()


chroma_client = ChromaClient()


def initialize_vectorstore() -> ChromaClient:
    chroma_client.initialize()
    return chroma_client
