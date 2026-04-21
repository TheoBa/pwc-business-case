# Plan: PwC Business Case — Streamlit Financial Document Insight App

## TL;DR
Build a multipage Streamlit app that ingests the BMO 2025 Annual Report (MDA), creates a tagged vector database segmented by headings, and provides a conversational interface for financial document analysis. Three tabs: Discovery (explore document structure), Problem Solving (RAG chat with citations), Conclusion (summary & architecture overview). Uses `st.navigation` with `position="top"` for 3-tab layout.

## Architecture

- **PDF Processing**: PyMuPDF (fitz) or pdfplumber for text/table/image extraction
- **Chunking**: Segment by document headings; tag each chunk with metadata (heading hierarchy, content type: text/table/image, page number)
- **Vector DB**: ChromaDB (local, zero-config) with embedding via `sentence-transformers` or OpenAI embeddings
- **LLM**: OpenAI GPT-4o (or configurable) for chat + RAG responses
- **Frontend**: Streamlit multipage app with `st.navigation(position="top")`

## Steps

### Phase 1: Project Setup
1. Create `pyproject.toml` with dependencies: `streamlit`, `openai`, `chromadb`, `pymupdf`, `sentence-transformers`, `pandas`
2. Create `.streamlit/config.toml` with page config and theme
3. Create directory structure:
   ```
   streamlit_app.py          # Entry point with st.navigation
   app_pages/
     discovery.py            # Tab 1: Document discovery
     problem_solving.py      # Tab 2: RAG chat interface
     conclusion.py           # Tab 3: Summary & architecture
   lib/
     pdf_processor.py        # PDF ingestion, chunking, tagging
     vector_store.py         # ChromaDB wrapper (index + query)
     llm_client.py           # OpenAI client with RAG prompt
   data/                     # Downloaded PDF + cached embeddings
   ```

### Phase 2: PDF Processing & Vector Store (depends on Phase 1)
4. **`lib/pdf_processor.py`** — Download BMO PDF, extract text by page, detect headings via font size/bold, segment into chunks with metadata:
   - `heading`: section heading hierarchy (e.g., "Risk Management > Credit Risk")
   - `content_type`: "text" | "table" | "image_caption"
   - `page_number`: source page(s)
   - `chunk_text`: the actual content
5. **`lib/vector_store.py`** — ChromaDB collection with:
   - Embedding function (sentence-transformers `all-MiniLM-L6-v2` or OpenAI `text-embedding-3-small`)
   - `index_documents(chunks)` — upsert with metadata filters
   - `query(text, filters=None, top_k=5)` — retrieve with optional heading/type filters
   - Cache the indexed collection to disk so it persists across reruns

### Phase 3: LLM Client (parallel with Phase 2)
6. **`lib/llm_client.py`** — OpenAI wrapper with:
   - RAG prompt template: inject retrieved context chunks with page citations
   - Streaming support (return generator for `st.write_stream`)
   - Dual mode: knowledge-base queries (with citations) vs general queries (no citations)
   - Intent detection: determine if user query targets a specific document segment, and if ambiguous, return clarification request

### Phase 4: Streamlit App — Entry Point
7. **`streamlit_app.py`** — Main entry:
   - `st.set_page_config(page_title="BMO Financial Insights", layout="wide")`
   - Initialize session state: `messages`, `selected_section`, `vector_store_ready`
   - `st.navigation` with 3 pages, `position="top"`
   - Trigger vector store indexing on first load (with `st.spinner`)

### Phase 5: Tab 1 — Discovery (depends on Phase 2)
8. **`app_pages/discovery.py`** — Explore the document structure:
   - Display document heading tree (expandable `st.expander` per section)
   - Show chunk count per section, content type distribution (text vs table vs image)
   - `st.dataframe` with all chunks: heading, type, page, preview text — filterable
   - Sidebar filters: filter by heading, content type, page range
   - `st.metric` cards: total pages, total chunks, sections count, tables count

### Phase 6: Tab 2 — Problem Solving / Chat (depends on Phase 2, 3)
9. **`app_pages/problem_solving.py`** — RAG chat interface:
   - Chat UI using `st.chat_message` + `st.chat_input` pattern
   - Session state for `messages` list
   - Sidebar: optional section filter (`st.selectbox` with document headings) to scope queries
   - On user input:
     a. Detect intent (knowledge-base vs general)
     b. If KB: query vector store (with optional section filter), build RAG prompt, stream response with `st.write_stream`
     c. Append citations as expandable source cards (`st.expander` showing page number, heading, snippet)
     d. If general: direct LLM call without RAG context, no citations
     e. If ambiguous segment: ask clarifying question before answering
   - `st.pills` for suggestion chips on empty chat (e.g., "What are BMO's key risks?", "Summarize credit risk exposure")
   - `st.feedback("thumbs")` on each assistant message

### Phase 7: Tab 3 — Conclusion (parallel with Phase 5, 6)
10. **`app_pages/conclusion.py`** — Summary & architecture:
    - Display solution architecture diagram (embedded image or Mermaid via `st.graphviz_chart` or markdown image)
    - Key design decisions (chunking strategy, embedding model, vector DB choice)
    - Approach for evaluation: offline (retrieval precision/recall, answer quality) + online (user feedback, latency)
    - `st.expander` sections for each architecture component
    - Link back to business case requirements with checkmarks showing coverage

## Relevant Files
- `streamlit_app.py` — Entry point, navigation, global state init
- `app_pages/discovery.py` — Document exploration tab
- `app_pages/problem_solving.py` — RAG chat tab
- `app_pages/conclusion.py` — Architecture & evaluation tab
- `lib/pdf_processor.py` — PDF extraction, heading detection, chunking
- `lib/vector_store.py` — ChromaDB indexing and querying
- `lib/llm_client.py` — OpenAI client, RAG prompts, intent detection
- `.streamlit/config.toml` — Theme and page config
- `pyproject.toml` — Dependencies

## Verification
1. Run `streamlit run streamlit_app.py` — all 3 tabs load without errors
2. Discovery tab: heading tree renders, chunk dataframe is filterable, metrics show correct counts
3. Problem Solving tab: ask "What is BMO's net income?" — response streams with page citations; ask "What is the capital of France?" — response has no citations
4. Problem Solving tab: ask ambiguous query like "Tell me about risk" — app asks which risk section (credit, market, operational, etc.)
5. Conclusion tab: architecture diagram renders, all business case requirements are addressed
6. Test section filter: select a specific heading in sidebar, verify chat scopes to that section only

## Decisions
- **ChromaDB** over Pinecone/Weaviate — zero config, local, perfect for demo; mention cloud alternatives in conclusion tab
- **3 horizontal tabs via `st.navigation(position="top")`** — clean UX for exactly 3 pages, no sidebar clutter
- **PyMuPDF** for PDF parsing — handles text + tables + images; fallback to pdfplumber if table extraction is poor
- **Sentence-transformers local embedding** as default — no API key needed for embeddings; OpenAI embedding as optional upgrade
- **Scope**: Single document (BMO AR 2025 MDA). Multi-document support excluded but architecture supports it.

## Resolved Considerations
1. **API Key management**: Use `st.secrets` with `.streamlit/secrets.toml` (gitignored). ✅ Decided.
2. **PDF caching**: ~30s first load acceptable. Cache ChromaDB to `data/`. Show progress bar. ✅ Decided.
3. **Table extraction quality**: Start with PyMuPDF. Revisit later if needed. ✅ Deferred.
