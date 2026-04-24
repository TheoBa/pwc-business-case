# Plan: Redesign Conclusion Tab as Architecture Presentation Page

## TL;DR
Replace the current `app_pages/conclusion.py` with a presentation-ready "Solution Architecture" page. Two main sections: (1) an interactive design flow showing how the major systems (PDF ingestion, chunking, vector store, LLM, UI) interact, and (2) a structured list of key design decisions with pros/cons and reasoning for rejected alternatives. Rename the tab from "Conclusion" to "Architecture" in `streamlit_app.py`.

## Steps

### Phase 1: Update Navigation (streamlit_app.py)
1. Rename the "Conclusion" page entry to "Architecture" (title and icon stay `:material/architecture:`)

### Phase 2: Rewrite conclusion.py — Design Flow Section
2. Replace the page title/caption to reflect "Solution Architecture" purpose
3. Build an **end-to-end system flow diagram** using a Mermaid `graph LR` showing:
   - PDF source → Unstructured.io parsing → Heading-based chunking + metadata tagging → ChromaDB vector indexing
   - User query → LLM intent detection (KB / general / ambiguous) → Query planner (filter extraction) → ChromaDB retrieval → RAG prompt construction → Llama 3.1 streaming → Streamlit chat UI
   - Clarification loop when intent is ambiguous
4. Add a **component breakdown** section with bordered containers (one per system) explaining each module's role:
   - **PDF Processor** (`lib/pdf_processor.py`): Unstructured.io `partition_pdf` with `hi_res`, heading detection via `Title` element depth, chunk splitting at sentence boundaries (2000 char max), time period regex extraction, schema-versioned JSON cache
   - **Vector Store** (`lib/vector_store.py`): ChromaDB persistent client, `all-MiniLM-L6-v2` embeddings (384-dim), cosine similarity, metadata filtering (`$contains`/`$and`), schema-versioned indexing with dedup
   - **LLM Client** (`lib/llm_client.py`): Ollama (OpenAI-compatible API), intent detection (JSON classification), LLM query planner (auto-extracts `chunk_type`, `time_period`, `heading_top` filters), streaming RAG with citation enforcement, fallback to unfiltered search
   - **Streamlit Frontend** (`streamlit_app.py` + `app_pages/`): 3-tab `st.navigation`, session state for chat + chunks, sidebar filters, suggestion chips, feedback thumbs

### Phase 3: Rewrite conclusion.py — Key Decisions Section
5. Build a **Key Decisions** section using `st.expander` cards, each containing:
   - Decision title
   - What we chose and why (short rationale)
   - Pros and cons in a two-column layout
   - Alternatives considered and why they were rejected

   Decisions to document (sourced from plans/ and further_consideration/):

   a. **Unstructured.io over PyMuPDF/LlamaParse/Azure Document Intelligence**
      - Chose: Unstructured.io (`partition_pdf` with `hi_res`) — layout-aware, local, free, handles text/tables/images, element-typed output
      - Rejected PyMuPDF: no native element typing (can't distinguish Title vs NarrativeText vs Table)
      - Rejected LlamaParse: requires API key, cloud dependency
      - Rejected Azure DI: cloud-only, cost, vendor lock-in

   b. **ChromaDB over Pinecone/Weaviate/Azure AI Search**
      - Chose: ChromaDB — zero-config, local persistent storage, built-in cosine similarity, good enough for single-document demo
      - Rejected Pinecone: cloud-only, requires API key, overkill for demo
      - Rejected Weaviate: heavier setup, better for multi-tenant production
      - Rejected Azure AI Search: cloud-only, cost

   c. **Heading-based chunking over fixed-size / recursive splitting**
      - Chose: Heading-based segmentation using Unstructured's `Title` element depth — preserves document's natural structure, enables hierarchical metadata (`heading_top > heading_sub`)
      - Rejected fixed-size: breaks mid-sentence, loses section context
      - Rejected recursive (LangChain-style): better than fixed but still ignores document structure

   d. **LLM query planner over LangChain SelfQueryRetriever / manual filter UI**
      - Chose: Custom lightweight LLM query planner — extracts `chunk_type`, `time_period`, `heading_top` filters from natural language, with fallback to unfiltered search
      - Rejected LangChain SelfQueryRetriever: heavy dependency, less control over filter schema
      - Rejected manual-only filters: poor UX — users shouldn't have to know document structure upfront

   e. **Ollama + Llama 3.1 over OpenAI GPT-4o / Azure OpenAI / Claude**
      - Chose: Ollama with Llama 3.1 8B — fully offline, no API costs, OpenAI-compatible API, good enough for demo
      - Rejected GPT-4o/Claude: requires API key, ongoing cost, cloud dependency (but recommended for production)

   f. **`all-MiniLM-L6-v2` embeddings (with upgrade path noted)**
      - Chose: sentence-transformers `all-MiniLM-L6-v2` — local, no API, 384-dim, fast
      - Trade-off: 256 token context limit, general-purpose vocabulary
      - Upgrade path: `nomic-embed-text` via Ollama (768-dim, 8192 tokens) — consolidates model serving

   g. **Intent detection (KB/general/ambiguous) over single-path RAG**
      - Chose: 3-way LLM intent classification — routes queries appropriately, avoids hallucinated citations on general questions, triggers clarification when ambiguous
      - Rejected single-path RAG: would cite irrelevant chunks for "what is GDP?" type questions

   h. **BM25 hybrid scoring deferred (noted as future improvement)**
      - Current: pure dense cosine similarity
      - Considered: hybrid BM25 + dense with Reciprocal Rank Fusion
      - Why deferred: metadata filtering already narrows search space significantly, adding BM25 is orthogonal improvement for exact term matching (CET1, PCL, etc.)

### Phase 4: Optional — Production Architecture
6. Keep a condensed "Production Architecture" Mermaid diagram showing cloud deployment topology (already exists in current conclusion.py, refine for presentation clarity)

## Relevant Files
- `app_pages/conclusion.py` — full rewrite
- `streamlit_app.py` — rename tab title from "Conclusion" to "Architecture"
- `plans/plan.md` — source of decisions (reference only)
- `plans/plan-metadata-tags.md` — source of decisions (reference only)
- `further_consideration/bm25-hybrid-scoring.md` — source of deferred decision (reference only)
- `further_consideration/embedding-model-upgrade.md` — source of upgrade path (reference only)

## Verification
1. Run `streamlit run streamlit_app.py` and verify the "Architecture" tab loads without errors
2. Confirm Mermaid diagrams render correctly in the Streamlit app
3. Verify all expanders open/close properly and two-column pros/cons layout renders correctly
4. Check the page is scannable and presentation-ready (clear visual hierarchy, no walls of text)

## Decisions
- **Mermaid for diagrams**: Streamlit renders Mermaid natively via `st.markdown` — no extra dependencies
- **Expanders for decisions**: each decision is collapsible so the page stays clean for presentation
- **Two-column pros/cons**: `st.columns(2)` inside each expander for visual comparison
- **No code changes to lib/**: this is purely a UI/presentation change
- **Keep tech stack summary**: condensed version at the bottom for quick reference
