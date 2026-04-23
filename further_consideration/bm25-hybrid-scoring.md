# Further Consideration: BM25 Hybrid Scoring

## Context

The current retrieval pipeline relies solely on **dense vector cosine similarity** via ChromaDB's HNSW index. The metadata tagging plan adds structured filtering (chunk_type, time_period, heading) on top of this, but the core ranking mechanism remains pure semantic search.

## Problem

Dense embeddings excel at capturing semantic meaning ("net income" ≈ "bottom line profit") but can fail on **exact term matching** — critical in financial RAG where users query specific entity names, ticker symbols, exact metric labels, or regulatory terms (e.g., "CET1 ratio", "PCL", "AIRB approach"). A user asking about "AIRB" expects chunks containing that exact acronym to rank highest, but a semantic-only search might surface chunks about "internal risk models" instead.

## What BM25 Adds

BM25 (Best Matching 25) is a sparse retrieval algorithm based on term frequency and inverse document frequency. It ranks documents by **exact keyword overlap** with the query, weighted by how rare each term is across the corpus.

**Hybrid search** combines both scores:
```
final_score = α × dense_similarity + (1 - α) × bm25_score
```
Where `α` (typically 0.5-0.7) controls the balance. This gives you:
- Semantic understanding for paraphrased queries ("What did the bank earn?" → matches "net income" chunks)
- Exact keyword matching for precise queries ("CET1 ratio Q3" → matches chunks with those exact terms)

## Implementation Options

### Option A: `rank_bm25` library (lightweight)
- **Library:** `rank_bm25` (~50 lines of integration code)
- **How it works:**
  1. At index time: build a BM25 index from all chunk texts (in-memory, alongside ChromaDB)
  2. At query time: get top-K from ChromaDB (dense) + top-K from BM25 (sparse)
  3. Merge results using Reciprocal Rank Fusion (RRF) or weighted linear combination
- **Pros:** Minimal dependency, easy to implement, no infrastructure changes
- **Cons:** BM25 index lives in memory (not persisted — must rebuild on app restart), no tokenizer tuning
- **Integration points:**
  - `lib/vector_store.py`: add `BM25Okapi` index built from chunk texts in `index_documents()`
  - `lib/vector_store.py`: add `hybrid_query()` method that queries both indexes and merges
  - Persist BM25 index to disk via `pickle` to avoid rebuild on every restart

### Option B: ChromaDB with external BM25 (e.g., SQLite FTS5)
- **How it works:** Store chunk texts in a SQLite FTS5 table alongside ChromaDB, query both
- **Pros:** Persistent, battle-tested full-text search
- **Cons:** More complex, two storage systems to maintain, ChromaDB already uses SQLite internally (potential conflicts)

### Option C: Replace ChromaDB with a hybrid-native DB
- **Alternatives:** Qdrant (supports sparse+dense natively), Weaviate (BM25+vector), Milvus
- **Pros:** Single system handles both dense and sparse retrieval, built-in fusion algorithms
- **Cons:** Major migration effort, heavier infrastructure, defeats the "local zero-config" design goal

## Recommended Approach

**Option A (`rank_bm25`)** — minimal disruption, good results:

1. Add `rank_bm25` to `pyproject.toml`
2. In `lib/vector_store.py`:
   - After `index_documents()`, build `BM25Okapi` from tokenized chunk texts
   - Pickle the BM25 index to `data/bm25_index.pkl` for persistence
   - Add `hybrid_query(text, top_k=5, alpha=0.6, **filters)`:
     - Get `top_k * 2` results from ChromaDB (dense)
     - Get `top_k * 2` results from BM25 (sparse)
     - Apply metadata filters to BM25 results manually (since BM25 has no native metadata filtering)
     - Merge using RRF: `score = 1/(rank_dense + 60) + 1/(rank_sparse + 60)`
     - Return top_k merged results
3. Update `lib/llm_client.py`: `stream_rag_response()` calls `hybrid_query()` instead of `query()`
4. Add `alpha` slider to Problem Solving sidebar for advanced users (optional)

## Reciprocal Rank Fusion (RRF) Details

RRF is preferred over weighted linear combination because it doesn't require score normalization across different systems:

```
RRF_score(doc) = Σ 1 / (k + rank_i(doc))
```

Where `k` is a constant (typically 60) and `rank_i(doc)` is the rank of the document in retrieval system `i`. Documents appearing in both result sets get higher combined scores.

## Expected Impact

- **Exact term queries** (acronyms, specific metrics): significant improvement
- **Semantic/paraphrased queries**: minimal change (dense retrieval still dominates)
- **Mixed queries** ("CET1 ratio compared to peers"): best of both worlds

## Relationship to Metadata Plan

This is **complementary** to the metadata tagging plan:
- Metadata filtering narrows the search space (e.g., only Q3 tables)
- BM25 hybrid scoring improves ranking within that filtered space
- Both can be implemented independently; combining them gives the strongest retrieval

## Relationship to Embedding Upgrade

If the embedding model is upgraded (e.g., to `nomic-embed-text`), the dense retrieval quality improves, which may reduce the marginal benefit of BM25. However, BM25 still adds value for exact-match queries regardless of embedding quality.
