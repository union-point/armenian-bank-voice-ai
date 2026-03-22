import logging
from typing import Optional

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, TextNode

from .config import RagConfig, rag_config

logger = logging.getLogger(__name__)


def chunk_documents(
    documents: list[Document],
    config: RagConfig = rag_config,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> list[TextNode]:
    """
    Split *documents* into TextNode chunks.

    Args:
        documents:     List of llama_index Documents (output of loader.py).
        config:        RagConfig instance (defaults to module singleton).
        chunk_size:    Override config.chunk_size for this call.
        chunk_overlap: Override config.chunk_overlap for this call.

    Returns:
        Flat list of TextNode objects with parent-document metadata merged in.
    """
    if not documents:
        logger.warning("chunk_documents received an empty document list.")
        return []

    effective_chunk_size = chunk_size or config.chunk_size
    effective_overlap = chunk_overlap or config.chunk_overlap

    splitter = SentenceSplitter(
        chunk_size=effective_chunk_size,
        chunk_overlap=effective_overlap,
    )

    nodes: list[TextNode] = splitter.get_nodes_from_documents(documents)

    logger.info(
        "Chunked %d document(s) into %d node(s) (chunk_size=%d, chunk_overlap=%d)",
        len(documents),
        len(nodes),
        effective_chunk_size,
        effective_overlap,
    )
    return nodes
