import logging
from dataclasses import dataclass, field
from typing import Optional

from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters

from .config import RagConfig, rag_config
from .indexer import build_index

logging.getLogger("llama_index").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieved chunk plus its similarity score and source metadata."""

    text: str
    score: float
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.text[:120].replace("\n", " ")
        return f"RetrievalResult(score={self.score:.4f}, text={preview!r})"


def retrieve(
    query: str,
    category: Optional[str] = None,
    bank_name: Optional[str] = None,
    top_k: Optional[int] = None,
    config: RagConfig = rag_config,
) -> list[RetrievalResult]:
    """
    Query ChromaDB for the *top_k* most relevant chunks for *query*.

    Args:
        query:  The natural-language question or keyword string.
        category: Optional category filter (e.g. 'credit', 'deposit', 'branch').
        bank_name: Optional bank name filter.
        top_k:  Number of results to return (defaults to config.top_k).
        config: RagConfig instance.

    Returns:
        List of RetrievalResult objects sorted by descending similarity.
    """
    if not query.strip():
        logger.warning("retrieve() called with an empty query.")
        return []

    k = top_k if top_k is not None else config.top_k

    index = build_index(config)

    filters = []
    if category:
        filters.append(ExactMatchFilter(key="category", value=category))
    if bank_name:
        filters.append(ExactMatchFilter(key="bank_name", value=bank_name))

    metadata_filters = MetadataFilters(filters=filters) if filters else None

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=k,
        filters=metadata_filters,
    )

    node_with_scores = retriever.retrieve(query)

    results: list[RetrievalResult] = []
    for nws in node_with_scores:
        results.append(
            RetrievalResult(
                text=nws.node.get_content(),
                score=nws.score or 0.0,
                metadata=nws.node.metadata or {},
            )
        )

    logger.info("Query: %r → %d result(s) (top_k=%d)", query[:60], len(results), k)
    return results
