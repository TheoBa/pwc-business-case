# Further Consideration: Embedding Model Upgrade

## Context

The current pipeline uses `all-MiniLM-L6-v2` (384-dimensional, ~22M params) from the `sentence-transformers` library as the embedding model for ChromaDB. This model is a general-purpose English sentence encoder — fast and lightweight, but not trained on financial or domain-specific corpora.

## Problem

Financial documents contain specialized vocabulary (e.g., "risk-weighted assets", "tier 1 capital ratio", "credit loss provisions", "net interest margin") and numerical relationships that general-purpose embeddings handle poorly. Semantic similarity between "operating revenue" and "top-line income" may score low because these terms rarely co-occur in general training data. This directly impacts retrieval recall — relevant chunks rank lower than they should.

## Candidate Upgrades

### Option A: `BAAI/bge-base-en-v1.5` (768-dim, ~110M params)
- **Pros:** Top-ranked on MTEB benchmark, instruction-tuned (can prefix queries with "Represent this sentence for searching relevant passages:"), significantly better at asymmetric retrieval (short query → long passage)
- **Cons:** ~5x larger than MiniLM, slower inference, higher memory usage
- **Integration:** Drop-in replacement in `vector_store.py` — change `SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-base-en-v1.5")`
- **Requires full re-index** of ChromaDB since embedding dimensions change (384 → 768)

### Option B: `nomic-embed-text` via Ollama (768-dim)
- **Pros:** Runs through Ollama (already a dependency), no separate `sentence-transformers` needed, good benchmark scores, long context window (8192 tokens vs 256 for MiniLM)
- **Cons:** Requires Ollama to be running for indexing (not just querying), slightly slower than local sentence-transformers
- **Integration:** Replace `SentenceTransformerEmbeddingFunction` with ChromaDB's `OllamaEmbeddingFunction` — `chromadb.utils.embedding_functions.OllamaEmbeddingFunction(model_name="nomic-embed-text")`
- **Requires:** `ollama pull nomic-embed-text` before first use

### Option C: Domain fine-tuned model
- **Pros:** Best possible retrieval quality for financial text
- **Cons:** Requires labeled financial Q&A pairs for fine-tuning, significant effort, ongoing maintenance
- **Recommendation:** Only pursue if Options A/B don't meet quality bar after testing

## Recommended Approach

**Option B (`nomic-embed-text` via Ollama)** is the best balance of quality, simplicity, and architectural consistency:
- Consolidates all model serving through Ollama (already running for LLM)
- 8192 token context handles long financial paragraphs without truncation (MiniLM truncates at 256 tokens — many financial chunks exceed this)
- No additional Python dependencies

## Implementation Sketch

1. Pull model: `ollama pull nomic-embed-text`
2. Update `lib/vector_store.py`: replace embedding function with `OllamaEmbeddingFunction`
3. Delete `data/chromadb/` to force re-index with new embeddings
4. Remove `sentence-transformers` from `pyproject.toml` (if no longer needed)
5. Benchmark: compare retrieval precision@5 on 10-20 test queries before/after

## Relationship to Metadata Plan

This is **orthogonal** to the metadata tagging plan — they improve different retrieval axes (semantic quality vs. structured filtering). Can be done before, after, or in parallel. If done together, only one re-index is needed.
