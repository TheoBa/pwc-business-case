from __future__ import annotations
"""Vector store module using ChromaDB for document chunk indexing and retrieval."""

from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

DATA_DIR = Path(__file__).parent.parent / "data"
CHROMA_DIR = DATA_DIR / "chromadb"
COLLECTION_NAME = "bmo_mda_2025"


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
    """Check if documents have already been indexed."""
    try:
        collection = get_collection()
        return collection.count() > 0
    except Exception:
        return False


def index_documents(chunks: list[dict], progress_callback=None) -> int:
    """Index document chunks into ChromaDB.

    Args:
        chunks: List of chunk dicts from pdf_processor.extract_chunks()
        progress_callback: Optional callable(current, total) for progress updates

    Returns:
        Number of chunks indexed
    """
    collection = get_collection()

    # Skip if already indexed
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
                "content_type": c["content_type"],
                "page_number": c["page_number"],
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
    content_type_filter: str | None = None,
    top_k: int = 5,
) -> list[dict]:
    """Query the vector store for relevant chunks.

    Args:
        text: Query text
        heading_filter: Optional heading prefix to filter results
        content_type_filter: Optional content type filter ("text", "table", "image")
        top_k: Number of results to return

    Returns:
        List of dicts with keys: id, text, heading, content_type, page_number, distance
    """
    collection = get_collection()

    if collection.count() == 0:
        return []

    # Build where filter
    where = None
    conditions = []

    if heading_filter:
        conditions.append({"heading": {"$contains": heading_filter}})
    if content_type_filter:
        conditions.append({"content_type": content_type_filter})

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
            formatted.append(
                {
                    "id": doc_id,
                    "text": results["documents"][0][i],
                    "heading": results["metadatas"][0][i]["heading"],
                    "content_type": results["metadatas"][0][i]["content_type"],
                    "page_number": results["metadatas"][0][i]["page_number"],
                    "distance": results["distances"][0][i] if results.get("distances") else None,
                }
            )

    return formatted
