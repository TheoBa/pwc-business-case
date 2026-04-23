# Plan: Structured Metadata Tags for Hybrid Retrieval

Upgrade the RAG pipeline from pure cosine-similarity search to **hybrid search with rich metadata filtering**. Replace PyMuPDF with Unstructured.io for layout-aware parsing, define a strict metadata schema, and build a lightweight LLM-powered query planner that auto-extracts metadata filters from user questions. Update the Discovery UI to expose the new metadata dimensions.

---

## Phase 1: Upgrade PDF Parser to Unstructured.io

1. Add `unstructured[pdf]` to `pyproject.toml`, remove `pymupdf`
2. Rewrite `lib/pdf_processor.py` — replace font-based `_classify_block()` with Unstructured's `partition_pdf()` which auto-detects element types (`Title`, `NarrativeText`, `Table`, `ListItem`, `Image`)
3. Map Unstructured element types → schema's `chunk_type` values (`text`, `table`, `list`, `image`, `chart_caption`), preserve heading hierarchy via `Title` element tracking

## Phase 2: Define Strict Metadata Schema

Every chunk will carry:

| Field | Example | Purpose |
|-------|---------|---------|
| `source_file` | `"BMO_AR2025_MDA.pdf"` | Provenance tracking |
| `page_number` | `42` | Citations & auditing |
| `chunk_type` | `"table"` | Content-type filtering |
| `heading` | `"EDTF > Capital Adequacy"` | Full hierarchy path |
| `heading_top` | `"EDTF"` | Top-level section filter |
| `heading_sub` | `"Capital Adequacy"` | Leaf sub-heading |
| `time_period_str` | `"FY2025,Q3"` | Temporal filtering |

4. Create `_extract_time_periods(text)` using regex (`FY\d{4}`, `Q[1-4]\s?\d{4}`, standalone years) in `lib/pdf_processor.py`
5. Split existing `heading` field on `" > "` to derive `heading_top` and `heading_sub`

## Phase 3: Update Vector Store Indexing & Filtering

6. Update `index_documents()` in `lib/vector_store.py` to upsert all new metadata fields (ChromaDB requires str/int/float — store time_period as comma-joined string)
7. Extend `query()` with `time_period_filter` param using `{"time_period_str": {"$contains": value}}`; rename `content_type` → `chunk_type`
8. Add schema version check to force re-index when metadata schema changes (depends on steps 6-7)

## Phase 4: Build Lightweight LLM Query Planner

9. Add `plan_query(question) → dict` in `lib/llm_client.py` — prompts existing Ollama LLM to extract a JSON object with `search_text` + `filters` (chunk_type, time_period, heading_top) from the user's question. Includes valid filter values in the prompt. Falls back to pure semantic search if JSON parsing fails
10. Update `stream_rag_response()` to call `plan_query()` first, pass extracted filters to `vector_store.query()`. Manual sidebar filter overrides LLM-extracted heading (depends on step 9)

## Phase 5: Update Discovery & Problem Solving UI

11. Update `app_pages/discovery.py` — add `chunk_type` and `time_period` multi-select filters, update dataframe columns (parallel with step 10)
12. Update `app_pages/problem_solving.py` — add time_period sidebar dropdown, display query plan in collapsible expander, update source expander with new metadata fields

## Phase 6: Re-index and Test

13. Delete `data/chunks.json` and `data/chromadb/` to force full re-processing
14. Run end-to-end and verify with test queries:
    - "What was operating revenue in Q3 2023?" → extracts `time_period=Q3 2023`, `chunk_type=table`
    - "Summarize risk management" → extracts `heading_top=Risk Management`
    - General question → no filters applied

---

## Relevant Files

- `lib/pdf_processor.py` — Replace PyMuPDF with Unstructured.io, add time_period extraction, expand metadata schema
- `lib/vector_store.py` — Store new metadata, add filters, schema versioning
- `lib/llm_client.py` — Add `plan_query()`, update `stream_rag_response()`
- `app_pages/discovery.py` — New filters, updated dataframe
- `app_pages/problem_solving.py` — Time period filter, query plan display
- `pyproject.toml` — Dependency changes

## Verification

1. `python -c "from unstructured.partition.pdf import partition_pdf; print('OK')"` — verify install
2. Run `streamlit run streamlit_app.py` — no import errors
3. Inspect `data/chunks.json` — every chunk has all 8 schema fields
4. Discovery page shows new filters and updated dataframe columns
5. Ask "What was net income in Q3?" in Problem Solving — verify query plan expander shows extracted filters
6. Manually inspect 5-10 chunks — `chunk_type` matches actual content

## Decisions

- **Unstructured.io** (local, no API keys) over LlamaParse/Azure DI
- **Custom LLM query planner** over LangChain SelfQueryRetriever — avoids large framework dependency
- **Regex** for time period extraction from chunk text
- **Full re-index required** — old data not backward-compatible
- **ChromaDB array workaround:** time_period stored as comma-joined string with `$contains`
