import json
import logging
from pathlib import Path
from typing import Optional

from llama_index.core.schema import Document, TextNode

from src.rag.chunker import chunk_documents
from src.rag.config import RagConfig, rag_config
from src.rag.indexer import add_nodes_to_index, build_index

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_documents(
    data_dir: Optional[str],
    config: RagConfig = rag_config,
) -> list[Document]:
    """
    Load all supported documents from *data_dir* (defaults to config.data_directory).

    Returns a list of llama_index Document objects, one per file.
    """
    directory = Path(data_dir or config.data_directory)

    if not directory.exists():
        logger.warning(
            "Data directory %s does not exist — returning empty list.", directory
        )
        return []

    documents: list[Document] = []

    for bank_dir in directory.iterdir():
        if not bank_dir.is_dir():
            continue

        bank_id = bank_dir.name
        logger.info(f"Processing bank: {bank_id}")

        for category_file in bank_dir.glob("*.json"):
            category = category_file.stem

            try:
                with open(category_file, "r", encoding="utf-8") as f:
                    items = json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse {category_file}: {e}")
                continue

            for item in items:
                text = item.get("text", "")
                if not text.strip():
                    continue

                metadata = item.get("metadata", {})
                metadata["bank_name"] = bank_id
                metadata["category"] = category.rstrip("s")
                metadata["source_file"] = str(category_file.relative_to(directory))

                doc = Document(
                    text=text,
                    metadata=metadata,
                    # Exclude metadata keys from the embedded text to save tokens.
                    excluded_embed_metadata_keys=["source_file"],
                    excluded_llm_metadata_keys=["source_file"],
                )
                documents.append(doc)

            logger.info(f"  {category}: {len(items)} items")

    return documents


def run_pipeline(
    data_dir: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    config: RagConfig = rag_config,
) -> int:
    """
    Full ingestion pipeline: Load → Chunk → Embed → Index.

    Args:
        data_dir:      Directory to load documents from (overrides config).
        chunk_size:    Token size per chunk (overrides config).
        chunk_overlap: Token overlap between chunks (overrides config).
        config:        RagConfig instance.

    Returns:
        Number of nodes (chunks) successfully indexed.
    """
    logger.info("=== RAG Pipeline: starting ingestion ===")

    # ── Load ──

    documents = load_documents(data_dir=data_dir)
    if not documents:
        logger.warning("No documents found — pipeline finished with 0 nodes.")
        return 0
    logger.info("Loaded %d document(s).", len(documents))

    # ── Chunk ──
    nodes = chunk_documents(
        documents,
        config=config,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not nodes:
        logger.warning("Chunking produced 0 nodes — pipeline aborted.")
        return 0

    # ── Embed + Index ──
    index = build_index(config)
    add_nodes_to_index(nodes, index)

    logger.info("=== RAG Pipeline: ingestion complete ===")
    return len(nodes)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest scraped bank data into vectorestore"
    )
    parser.add_argument("--data-dir", type=str, help="Path to data directory")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset ChromaDB collection before ingestion",
    )
    args = parser.parse_args()

    if args.reset:
        from src.vectorstore import chroma_client

        chroma_client.reset()
        logger.info("ChromaDB collection reset")

    count = run_pipeline(args.data_dir)
    print(f"\nIngested {count} nodes into ChromaDB")
