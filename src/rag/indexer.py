import logging
from typing import Optional

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.vectorstore import chroma_client, initialize_vectorstore

from .config import RagConfig, rag_config

logger = logging.getLogger(__name__)

# Module-level cache so the model is only instantiated once per process.
_embed_model_cache: Optional[HuggingFaceEmbedding] = None


def get_embed_model(config: RagConfig = rag_config) -> HuggingFaceEmbedding:
    """
    Return a cached HuggingFaceEmbedding
    """
    global _embed_model_cache

    if _embed_model_cache is not None:
        return _embed_model_cache

    logger.info("Loading embedding model: %s", config.embedding_model)
    _embed_model_cache = HuggingFaceEmbedding(
        model_name=config.embedding_model,
        embed_batch_size=config.embed_batch_size,
        # trust_remote_code=False
    )
    return _embed_model_cache


def _get_vector_store(config: RagConfig = rag_config) -> ChromaVectorStore:
    """
    Return a ChromaVectorStore backed by the existing persistent collection.
    The ChromaClient singleton is initialised on first call if needed.
    """
    initialize_vectorstore()
    collection = chroma_client.get_collection()

    return ChromaVectorStore(chroma_collection=collection)


def build_index(config: RagConfig = rag_config) -> VectorStoreIndex:
    """
    Create (or reconnect to) the VectorStoreIndex backed by ChromaDB.
    """
    embed_model = get_embed_model(config)
    vector_store = _get_vector_store(config)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(
        nodes=[],
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=False,
    )
    logger.info("VectorStoreIndex connected to ChromaDB collection.")
    return index


def add_nodes_to_index(
    nodes: list[TextNode],
    index: VectorStoreIndex,
) -> None:
    """
    Embed *nodes* and add them to *index* (which writes through to ChromaDB).

    Args:
        nodes:         Chunked TextNode objects.
        index:         VectorStoreIndex returned by build_index().
    """
    if not nodes:
        logger.warning("add_nodes_to_index: received empty node list — nothing to do.")
        return

    logger.info("Indexing %d node(s) into ChromaDB …", len(nodes))
    index.insert_nodes(nodes)
    logger.info(
        "Done. ChromaDB collection now has %d vector(s).", chroma_client.count()
    )
