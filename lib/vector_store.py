from __future__ import annotations
"""Vector store module using ChromaDB for document chunk indexing and retrieval."""

from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

DATA_DIR = Path(__file__).parent.parent / "data"
CHROMA_DIR = DATA_DIR / "chromadb"
COLLECTION_NAME = "bmo_mda_2025"
METADATA_SCHEMA_VERSION = 2


def _get_client() -> chromadb.ClientAPI:
    """Get a persistent ChromaDB client."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def _get_embedding_fn() -> SentenceTransformerEmbeddingFunction:
    """Get the sentence-transformer embedding function."""
    return SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )


def get_collection() -> chromadb.Collection:
    """Get or create the ChromaDB collection."""
    client = _get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_get_embedding_fn(),
        metadata={"hnsw:space": "cosine"},
    )


def is_indexed() -> bool:
    """Check if documents have already been indexed with current schema."""
    try:
        collection = get_collection()
        if collection.count() == 0:
            return False
        # Verify schema version by checking a sample document's metadata
        sample = collection.peek(limit=1)
        if sample and sample["metadatas"] and sample["metadatas"][0]:
            meta = sample["metadatas"][0]
            return meta.get("_schema_version") == METADATA_SCHEMA_VERSION
        return False
    except Exception:
        return False


def reset_collection() -> None:
    """Delete and recreate the collection (for schema migration)."""
    client = _get_client()
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass


def index_documents(chunks: list[dict], progress_callback=None) -> int:
    """Index document chunks into ChromaDB.

    Args:
        chunks: List of chunk dicts from pdf_processor.extract_chunks()
        progress_callback: Optional callable(current, total) for progress updates

    Returns:
        Number of chunks indexed
    """
    # Reset collection if schema version mismatch
    if not is_indexed():
        reset_collection()

    collection = get_collection()

    # Skip if already indexed with current schema
    if collection.count() > 0:
        return collection.count()

    # Index in batches to avoid memory issues
    batch_size = 100
    total = len(chunks)

    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]

        ids = [c["id"] for c in batch]
        documents = [c["text"] for c in batch]
        metadatas = [
            {
                "heading": c["heading"],
                "heading_top": c.get("heading_top", c["heading"].split(" > ")[0]),
                "heading_sub": c.get("heading_sub", c["heading"].split(" > ")[-1] if " > " in c["heading"] else ""),
                "chunk_type": c.get("chunk_type", c.get("content_type", "text")),
                "page_number": c["page_number"],
                "source_file": c.get("source_file", ""),
                "time_period_str": c.get("time_period_str", ""),
                "_schema_version": METADATA_SCHEMA_VERSION,
            }
            for c in batch
        ]

        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

        if progress_callback:
            progress_callback(min(i + batch_size, total), total)

    return collection.count()


def query(
    text: str,
    heading_filter: str | None = None,
    chunk_type_filter: str | None = None,
    time_period_filter: str | None = None,
    top_k: int = 5,
) -> list[dict]:
    """Query the vector store for relevant chunks.

    Args:
        text: Query text
        heading_filter: Optional heading prefix to filter results
        chunk_type_filter: Optional chunk type filter ("text", "table", "list", "image", "chart_caption")
        time_period_filter: Optional time period filter (e.g., "Q3 2023", "FY2025")
        top_k: Number of results to return

    Returns:
        List of dicts with keys: id, text, heading, heading_top, heading_sub,
        chunk_type, page_number, source_file, time_period_str, distance
    """
    collection = get_collection()

    if collection.count() == 0:
        return []

    # Build where filter
    where = None
    conditions = []

    if heading_filter:
        conditions.append({"heading": {"$contains": heading_filter}})
    if chunk_type_filter:
        conditions.append({"chunk_type": chunk_type_filter})
    if time_period_filter:
        conditions.append({"time_period_str": {"$contains": time_period_filter}})

    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}

    try:
        results = collection.query(
            query_texts=[text],
            n_results=top_k,
            where=where,
        )
    except Exception:
        # Fallback without filters if they cause issues
        results = collection.query(
            query_texts=[text],
            n_results=top_k,
        )

    # Format results
    formatted = []
    if results and results["ids"] and results["ids"][0]:
        for i, doc_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            formatted.append(
                {
                    "id": doc_id,
                    "text": results["documents"][0][i],
                    "heading": meta.get("heading", ""),
                    "heading_top": meta.get("heading_top", ""),
                    "heading_sub": meta.get("heading_sub", ""),
                    "chunk_type": meta.get("chunk_type", meta.get("content_type", "text")),
                    "page_number": meta.get("page_number", 0),
                    "source_file": meta.get("source_file", ""),
                    "time_period_str": meta.get("time_period_str", ""),
                    "distance": results["distances"][0][i] if results.get("distances") else None,
                }
            )

    return formatted
