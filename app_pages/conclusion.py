from __future__ import annotations
"""Architecture tab — solution design flow, key decisions, and production overview."""

import base64
import json

import streamlit as st


def _render_mermaid(chart: str) -> None:
    """Render a Mermaid diagram as an SVG image via mermaid.ink."""
    theme_config = {
        "theme": "base",
        "themeVariables": {
            "primaryColor": "#D4E4F7",
            "primaryTextColor": "#1A1A2E",
            "primaryBorderColor": "#4A6FA5",
            "secondaryColor": "#E8F0FE",
            "tertiaryColor": "#F0F4F8",
            "lineColor": "#4A6FA5",
            "textColor": "#1A1A2E",
            "mainBkg": "#FFFFFF",
            "nodeBorder": "#4A6FA5",
            "clusterBkg": "#F5F7FA",
            "clusterBorder": "#B0C4DE",
            "edgeLabelBackground": "#FFFFFF",
            "fontSize": "14px",
        },
    }
    mermaid_def = json.dumps({"code": chart.strip(), "mermaid": theme_config})
    encoded = base64.urlsafe_b64encode(mermaid_def.encode("utf-8")).decode("ascii")
    url = f"https://mermaid.ink/svg/{encoded}"
    st.image(url, use_container_width=True)


st.title(":material/architecture: Solution Architecture")
st.caption("Design flow, key decisions, and production deployment overview — presentation support.")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: End-to-End Design Flow
# ═══════════════════════════════════════════════════════════════════════════════
st.header("Design flow")
st.markdown(
    "The system implements a **Retrieval-Augmented Generation (RAG)** pipeline for "
    "financial document analysis. The diagram below shows how each major component "
    "interacts end-to-end."
)

_render_mermaid("""
graph LR
    subgraph Ingestion["Document Ingestion"]
        A[BMO 2025 MDA<br/>PDF] -->|Unstructured.io<br/>hi_res strategy| B[Layout-Aware<br/>Parsing]
        B -->|Title depth<br/>detection| C[Heading-Based<br/>Chunking]
        C -->|heading, chunk_type<br/>page, time_period| D[Metadata<br/>Tagging]
    end

    subgraph Storage["Vector Storage"]
        D -->|sentence-transformers<br/>all-MiniLM-L6-v2| E[Embedding<br/>Generation]
        E -->|Cosine similarity<br/>384-dim vectors| F[(ChromaDB<br/>Persistent Store)]
    end

    subgraph Query["Query Pipeline"]
        G[User Question] -->|Llama 3.1| H{Intent<br/>Detection}
        H -->|knowledge_base| I[LLM Query<br/>Planner]
        H -->|general| J[Direct LLM<br/>Response]
        H -->|ambiguous| K[Clarification<br/>Request]
        I -->|Extract filters:<br/>chunk_type, time_period,<br/>heading_top| L[Metadata-Filtered<br/>Vector Search]
        F -.->|Top-K retrieval| L
        L -->|Context +<br/>citations| M[RAG Prompt<br/>Construction]
    end

    subgraph Response["Response Generation"]
        M -->|Streaming| N[Llama 3.1<br/>via Ollama]
        J --> N
        N -->|Token-by-token| O[Streamlit<br/>Chat UI]
        K -->|Section list| O
    end
""")

# ── Component Breakdown ──
st.subheader("Component breakdown")

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.markdown("##### :material/description: PDF Processor")
        st.markdown("""
        `lib/pdf_processor.py`

        - **Parser**: Unstructured.io `partition_pdf` with `hi_res` strategy
        - **Heading detection**: `Title` element depth → hierarchical `heading_top > heading_sub`
        - **Chunk splitting**: sentence-boundary splitting at ~2 000 chars max
        - **Metadata**: `heading`, `chunk_type` (text/table/image), `page_number`, `time_period_str`
        - **Caching**: schema-versioned JSON (`chunks.json`) — skip re-parsing on reload
        """)

    with st.container(border=True):
        st.markdown("##### :material/neurology: LLM Client")
        st.markdown("""
        `lib/llm_client.py`

        - **Model**: Llama 3.1 8B via Ollama (OpenAI-compatible API, fully offline)
        - **Intent detection**: 3-way JSON classification → KB / general / ambiguous
        - **Query planner**: LLM extracts `chunk_type`, `time_period`, `heading_top` filters from natural language
        - **RAG prompt**: injects retrieved context with `[Page X, Section: Y]` citation format
        - **Fallback**: if filtered search returns < 2 results, retries without LLM-extracted filters
        - **Streaming**: token-by-token delivery for responsive UX
        """)

with col2:
    with st.container(border=True):
        st.markdown("##### :material/database: Vector Store")
        st.markdown("""
        `lib/vector_store.py`

        - **Database**: ChromaDB with persistent local storage
        - **Embeddings**: `all-MiniLM-L6-v2` (sentence-transformers) — 384-dim, cosine similarity
        - **Indexing**: batched upsert with deduplication, schema-versioned metadata
        - **Filtering**: `$contains` / `$and` operators on `heading`, `chunk_type`, `time_period_str`
        - **Auto-reset**: detects schema version mismatch → re-indexes automatically
        """)

    with st.container(border=True):
        st.markdown("##### :material/web: Streamlit Frontend")
        st.markdown("""
        `streamlit_app.py` + `app_pages/`

        - **Navigation**: 3-tab `st.navigation(position="top")` — Discovery, Problem Solving, Architecture
        - **Session state**: persists `messages`, `chunks`, `vector_store_ready` across reruns
        - **Discovery tab**: document structure explorer with filters (section, type, time period, page range)
        - **Problem Solving tab**: RAG chat with sidebar scope filters, suggestion chips, citation cards, feedback thumbs
        - **Startup**: auto-downloads PDF + indexes on first load with progress indicator
        """)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Technology Stack Summary
# ═══════════════════════════════════════════════════════════════════════════════
st.header("Technology stack")

with st.container(border=True):
    cols = st.columns(4)
    with cols[0]:
        st.markdown("**:material/web: Frontend**")
        st.markdown("- Streamlit 1.45+\n- Multi-page `st.navigation`\n- Session state chat")
    with cols[1]:
        st.markdown("**:material/settings: Processing**")
        st.markdown("- Unstructured.io (PDF)\n- sentence-transformers\n- Regex time-period extraction")
    with cols[2]:
        st.markdown("**:material/database: Storage**")
        st.markdown("- ChromaDB (vectors)\n- JSON chunk cache\n- Schema versioning")
    with cols[3]:
        st.markdown("**:material/smart_toy: AI / LLM**")
        st.markdown("- Ollama + Llama 3.1 8B\n- RAG with citations\n- Intent detection + query planner")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Key Design Decisions
# ═══════════════════════════════════════════════════════════════════════════════
st.header("Key design decisions")
st.markdown(
    "Each decision below explains **what we chose**, **why**, and **what alternatives "
    "were considered**. Expand any card for the full trade-off analysis."
)

# ── Decision A: PDF Parser ──
with st.expander("**1. Unstructured.io** for PDF parsing", icon=":material/description:"):
    st.markdown(
        "**Chosen:** Unstructured.io `partition_pdf` with `hi_res` strategy — layout-aware, "
        "local, free, handles text/tables/images with element-typed output."
    )
    col_pro, col_con = st.columns(2)
    with col_pro:
        st.markdown("""
        :green[**Pros**]
        - Native element typing (`Title`, `Table`, `NarrativeText`, `Image`)
        - Heading depth detection enables hierarchical chunking
        - Runs locally — no API key, no cloud dependency
        - Supports `hi_res` strategy for table structure inference
        """)
    with col_con:
        st.markdown("""
        :red[**Cons**]
        - Heavier install footprint (`unstructured[pdf]` + Tesseract)
        - Slower than lightweight parsers (~30s for 100-page PDF)
        - Occasional misclassification of decorative elements as headings
        """)
    st.divider()
    st.markdown("""
    **Alternatives rejected:**

    | Alternative | Why not |
    |---|---|
    | **PyMuPDF** | No native element typing — can't distinguish `Title` vs `NarrativeText` vs `Table` without custom heuristics. Originally used in v1, replaced when metadata tagging was added. |
    | **LlamaParse** | Requires API key and cloud connectivity — adds external dependency for a demo-scoped project. |
    | **Azure Document Intelligence** | Cloud-only, per-page cost, vendor lock-in. Excellent for production but overkill for offline demo. |
    """)

# ── Decision B: Vector Database ──
with st.expander("**2. ChromaDB** for vector storage", icon=":material/database:"):
    st.markdown(
        "**Chosen:** ChromaDB — zero-config, local persistent storage, built-in cosine "
        "similarity, good enough for a single-document demo."
    )
    col_pro, col_con = st.columns(2)
    with col_pro:
        st.markdown("""
        :green[**Pros**]
        - Zero infrastructure — `pip install chromadb` and go
        - Persistent local storage (survives app restarts)
        - Built-in metadata filtering (`$contains`, `$and`)
        - Cosine similarity search out of the box
        """)
    with col_con:
        st.markdown("""
        :red[**Cons**]
        - Not suitable for multi-user production (single-process)
        - No managed backups or replication
        - Limited query operators compared to cloud vector DBs
        """)
    st.divider()
    st.markdown("""
    **Alternatives rejected:**

    | Alternative | Why not |
    |---|---|
    | **Pinecone** | Cloud-only, requires API key. Overkill for single-document demo. Better for multi-tenant SaaS. |
    | **Weaviate** | Heavier local setup (Docker), better suited for production multi-tenant deployments. |
    | **Azure AI Search** | Cloud-only, per-query cost. Recommended for production Azure deployments. |
    """)

# ── Decision C: Chunking Strategy ──
with st.expander("**3. Heading-based chunking** over fixed-size / recursive splitting", icon=":material/content_cut:"):
    st.markdown(
        "**Chosen:** Heading-based segmentation using Unstructured's `Title` element depth — "
        "preserves the document's natural structure and enables hierarchical metadata "
        "(`heading_top > heading_sub`)."
    )
    col_pro, col_con = st.columns(2)
    with col_pro:
        st.markdown("""
        :green[**Pros**]
        - Chunks follow the document's logical sections
        - Enables scoped retrieval (search within a specific section)
        - Hierarchical headings power the sidebar section filter
        - Each chunk carries meaningful context (not a random 512-token window)
        """)
    with col_con:
        st.markdown("""
        :red[**Cons**]
        - Depends on correct heading detection (parser-dependent)
        - Long sections still need secondary splitting (sentence-boundary at 2 000 chars)
        - Doesn't work well for documents without clear heading structure
        """)
    st.divider()
    st.markdown("""
    **Alternatives rejected:**

    | Alternative | Why not |
    |---|---|
    | **Fixed-size chunking** (e.g., 512 tokens) | Breaks mid-sentence, loses section context. Chunks have no meaningful metadata. |
    | **Recursive splitting** (LangChain-style) | Better than fixed-size but still ignores document structure. No heading hierarchy. |
    """)

# ── Decision D: LLM Query Planner ──
with st.expander("**4. Custom LLM query planner** for metadata filter extraction", icon=":material/tune:"):
    st.markdown(
        "**Chosen:** A lightweight LLM query planner that extracts `chunk_type`, `time_period`, "
        "and `heading_top` filters from the user's natural language question, with automatic "
        "fallback to unfiltered search."
    )
    col_pro, col_con = st.columns(2)
    with col_pro:
        st.markdown("""
        :green[**Pros**]
        - Users don't need to know the document structure upfront
        - Automatically narrows search to tables when user asks for numbers
        - Time-period extraction catches "Q3 2023", "FY2025", etc.
        - Graceful fallback: if filtered results are sparse, retries unfiltered
        """)
    with col_con:
        st.markdown("""
        :red[**Cons**]
        - Adds one extra LLM call per query (~1s latency)
        - Filter extraction can occasionally be wrong (mitigated by fallback)
        - Requires careful prompt engineering for consistent JSON output
        """)
    st.divider()
    st.markdown("""
    **Alternatives rejected:**

    | Alternative | Why not |
    |---|---|
    | **LangChain SelfQueryRetriever** | Heavy dependency, less control over the filter schema and fallback logic. |
    | **Manual-only sidebar filters** | Poor UX — users shouldn't need to know document structure before asking a question. Sidebar filters are kept as an *optional* override. |
    """)

# ── Decision E: LLM Choice ──
with st.expander("**5. Ollama + Llama 3.1** for inference", icon=":material/smart_toy:"):
    st.markdown(
        "**Chosen:** Ollama with Llama 3.1 8B — fully offline, no API costs, "
        "OpenAI-compatible API, sufficient quality for demo."
    )
    col_pro, col_con = st.columns(2)
    with col_pro:
        st.markdown("""
        :green[**Pros**]
        - Fully offline — no API keys, no cloud dependency, no cost
        - OpenAI-compatible API (swap to GPT-4o with one config change)
        - Good enough reasoning for financial Q&A on structured context
        - Streaming support for responsive chat UX
        """)
    with col_con:
        st.markdown("""
        :red[**Cons**]
        - Weaker reasoning than GPT-4o / Claude on complex financial analysis
        - Requires local GPU or decent CPU (8B model ~4GB RAM)
        - JSON output from intent detection / query planner can be inconsistent
        """)
    st.divider()
    st.markdown("""
    **Alternatives rejected (but recommended for production):**

    | Alternative | Why not for demo | Production fit |
    |---|---|---|
    | **GPT-4o** | Requires API key, ongoing cost | Excellent — strongest reasoning |
    | **Azure OpenAI** | Cloud-only, subscription required | Best for enterprise Azure deployments |
    | **Claude** | Requires API key | Strong alternative for long-context analysis |
    """)

# ── Decision F: Embedding Model ──
with st.expander("**6. all-MiniLM-L6-v2** embeddings (with upgrade path)", icon=":material/transform:"):
    st.markdown(
        "**Chosen:** `all-MiniLM-L6-v2` via sentence-transformers — local, no API, "
        "384-dim, fast inference."
    )
    col_pro, col_con = st.columns(2)
    with col_pro:
        st.markdown("""
        :green[**Pros**]
        - Zero-config: `pip install sentence-transformers`
        - Fast inference, small model footprint
        - No API key required
        - Well-established, widely benchmarked
        """)
    with col_con:
        st.markdown("""
        :red[**Cons**]
        - 256 token context limit (truncates long financial paragraphs)
        - General-purpose vocabulary — financial terms may embed poorly
        - 384-dim vectors (lower precision than 768+ dim models)
        """)
    st.divider()
    st.markdown("""
    **Upgrade path considered:**

    **`nomic-embed-text` via Ollama** — 768-dim, 8 192 token context. Consolidates all model
    serving through Ollama (LLM + embeddings). Handles long financial paragraphs without truncation.
    Orthogonal improvement to metadata filtering — they improve different retrieval axes.
    """)

# ── Decision G: Intent Detection ──
with st.expander("**7. Three-way intent detection** (KB / general / ambiguous)", icon=":material/route:"):
    st.markdown(
        "**Chosen:** 3-way LLM intent classification before routing queries — "
        "avoids hallucinated citations on general questions and triggers clarification "
        "when the user's intent is ambiguous."
    )
    col_pro, col_con = st.columns(2)
    with col_pro:
        st.markdown("""
        :green[**Pros**]
        - KB queries get RAG + citations; general queries get clean responses
        - Ambiguous queries trigger clarification with section suggestions
        - Prevents the system from citing irrelevant chunks for off-topic questions
        - Matches the business case requirement: "no citations for non-KB info"
        """)
    with col_con:
        st.markdown("""
        :red[**Cons**]
        - Extra LLM call adds ~1s latency per query
        - Classification can be wrong (e.g., borderline KB/general questions)
        - Requires careful prompt engineering for consistent JSON output
        """)
    st.divider()
    st.markdown("""
    **Alternative rejected:**

    | Alternative | Why not |
    |---|---|
    | **Single-path RAG** (always retrieve + cite) | Would cite irrelevant chunks for "What is GDP?" type questions. Violates the requirement to not cite for non-KB queries. |
    """)

# ── Decision H: BM25 Hybrid Scoring ──
with st.expander("**8. BM25 hybrid scoring** — deferred (future improvement)", icon=":material/schedule:"):
    st.markdown(
        "**Current approach:** Pure dense cosine similarity via ChromaDB. "
        "BM25 hybrid scoring was evaluated but deferred."
    )
    col_pro, col_con = st.columns(2)
    with col_pro:
        st.markdown("""
        :green[**Why it would help**]
        - Better exact-term matching for financial acronyms (CET1, PCL, AIRB)
        - Hybrid score: `α × dense + (1-α) × BM25`
        - Reciprocal Rank Fusion (RRF) for combining results
        """)
    with col_con:
        st.markdown("""
        :orange[**Why deferred**]
        - Metadata filtering already narrows the search space significantly
        - Adding BM25 is orthogonal — it improves ranking *within* the filtered set
        - Additional dependency (`rank_bm25`) and index maintenance
        - Current retrieval quality is sufficient for demo scope
        """)
    st.divider()
    st.markdown("""
    **Recommended implementation (when needed):**
    1. Build `BM25Okapi` index from tokenized chunk texts
    2. Get top-K from ChromaDB (dense) + top-K from BM25 (sparse)
    3. Merge using Reciprocal Rank Fusion (RRF)
    """)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Production Architecture
# ═══════════════════════════════════════════════════════════════════════════════
st.header("Production cloud architecture")
st.markdown(
    "For a production deployment, the same pipeline maps onto managed cloud services. "
    "The diagram below shows a representative Azure/AWS topology."
)

_render_mermaid("""
graph TB
    subgraph Frontend["Frontend"]
        A[Streamlit App<br/>Azure App Service / GCP Cloud Run]
    end

    subgraph API["API Layer"]
        B[FastAPI Gateway<br/>Azure API Management]
    end

    subgraph Processing["Processing"]
        C[PDF Ingestion Pipeline<br/>Azure Functions / AWS Lambda]
        D[Embedding Service<br/>Azure OpenAI / Vertex AI]
    end

    subgraph Storage["Storage"]
        E[Vector DB<br/>Azure AI Search / Pinecone]
        F[Document Store<br/>Azure Blob / S3]
        G[Metadata DB<br/>PostgreSQL / CosmosDB]
    end

    subgraph LLM["LLM"]
        H[LLM Service<br/>Azure OpenAI GPT-4o]
    end

    subgraph Monitoring["Monitoring"]
        I[Logging & Metrics<br/>Application Insights]
        J[User Feedback DB]
    end

    A --> B
    B --> C
    B --> D
    B --> H
    C --> F
    D --> E
    B --> E
    H --> A
    A --> I
    A --> J
""")
