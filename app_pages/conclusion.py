from __future__ import annotations
"""Conclusion tab — solution architecture, design decisions, and evaluation approach."""

import streamlit as st

st.title(":material/architecture: Conclusion")
st.caption("Solution architecture, key design decisions, and evaluation approach.")

# ── Solution architecture ──
st.header("Solution architecture")

st.markdown("""
This application implements a **Retrieval-Augmented Generation (RAG)** pipeline
for financial document analysis. The architecture is designed for a production-level
deployment using cloud resources.
""")

# Architecture diagram using Mermaid-style text (rendered as code block)
st.subheader("System flow")

architecture_mermaid = """
```mermaid
graph LR
    A[PDF Document] -->|PyMuPDF| B[Text Extraction]
    B -->|Heading Detection| C[Chunking & Tagging]
    C -->|Metadata: heading, type, page| D[ChromaDB Vector Store]
    D -->|Semantic Search| E[RAG Pipeline]
    F[User Query] -->|Intent Detection| G{Knowledge Base?}
    G -->|Yes| E
    G -->|No| H[Direct LLM Call]
    G -->|Ambiguous| I[Clarification Request]
    E -->|Context + Citations| J[Llama 3.1 via Ollama]
    H --> J
    J -->|Streaming| K[Streamlit Chat UI]
    I --> K
```
"""
st.markdown(architecture_mermaid)

# Production architecture
st.subheader("Production cloud architecture")

prod_architecture = """
```mermaid
graph TB
    subgraph "Frontend"
        A[Streamlit App<br/>Azure App Service / GCP Cloud Run]
    end

    subgraph "API Layer"
        B[FastAPI Gateway<br/>Azure API Management]
    end

    subgraph "Processing"
        C[PDF Ingestion Pipeline<br/>Azure Functions / AWS Lambda]
        D[Embedding Service<br/>Azure OpenAI / Vertex AI]
    end

    subgraph "Storage"
        E[Vector DB<br/>Azure AI Search / Pinecone / Weaviate]
        F[Document Store<br/>Azure Blob / S3]
        G[Metadata DB<br/>PostgreSQL / CosmosDB]
    end

    subgraph "LLM"
        H[LLM Service<br/>Azure OpenAI / Self-hosted Ollama]
    end

    subgraph "Monitoring"
        I[Logging & Metrics<br/>Application Insights / CloudWatch]
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
```
"""
st.markdown(prod_architecture)

# ── Design decisions ──
st.header("Key design decisions")

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.subheader("Chunking strategy")
        st.markdown("""
        - **Heading-based segmentation**: chunks follow the document's natural structure
        - **Metadata tagging**: each chunk carries `heading`, `content_type` (text/table/image), and `page_number`
        - **Hierarchical headings**: e.g., "Risk Management > Credit Risk" enables scoped retrieval
        - **Size limits**: chunks split at ~2000 chars at sentence boundaries to balance context vs precision
        """)

    with st.container(border=True):
        st.subheader("Embedding model")
        st.markdown("""
        - **Default**: `all-MiniLM-L6-v2` (sentence-transformers) — local, no API key, 384-dim
        - **Production upgrade**: OpenAI `text-embedding-3-small` or Ollama embeddings for higher quality
        - **Trade-off**: local model enables zero-config demo; cloud model offers better semantic matching
        """)

with col2:
    with st.container(border=True):
        st.subheader("Vector database")
        st.markdown("""
        - **Demo**: ChromaDB (local, zero-config, persistent to disk)
        - **Production**: Azure AI Search, Pinecone, or Weaviate for scalability, managed backups, filtering
        - **Why ChromaDB for demo**: no infrastructure needed, instant setup, cosine similarity built-in
        """)

    with st.container(border=True):
        st.subheader("LLM & RAG approach")
        st.markdown("""
        - **Model**: Llama 3.1 8B via Ollama (fully offline, no API costs)
        - **Production alternative**: GPT-4o or Claude for stronger reasoning on financial data
        - **Intent detection**: classifies queries as knowledge-base, general, or ambiguous
        - **Citation enforcement**: system prompt requires `[Page X, Section: Y]` format
        - **Streaming**: token-by-token delivery for responsive UX
        """)

# ── Evaluation approach ──
st.header("Evaluation approach")

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.subheader("Offline evaluation")
        st.markdown("""
        **Retrieval quality:**
        - Precision@K: fraction of retrieved chunks that are relevant
        - Recall@K: fraction of relevant chunks that are retrieved
        - MRR (Mean Reciprocal Rank): position of first relevant result

        **Answer quality:**
        - LLM-as-judge: dedicated evaluation model rates answer faithfulness, relevance, and completeness
        - Citation accuracy: verify cited pages actually contain the stated information
        - Golden dataset: 50+ curated Q&A pairs from the BMO MDA for regression testing
        """)

with col2:
    with st.container(border=True):
        st.subheader("Online evaluation")
        st.markdown("""
        **User feedback:**
        - Thumbs up/down on each response (captured in-app)
        - Session-level satisfaction surveys

        **Performance metrics:**
        - Response latency (P50, P95, P99)
        - Token usage and cost per query
        - Retrieval cache hit rates

        **Behavioral monitoring:**
        - Query intent distribution (KB vs general vs ambiguous)
        - Section coverage: which document areas are most/least queried
        - Clarification request rate (high rate = poor chunking/retrieval)
        """)

# ── Business case requirements checklist ──
st.header("Business case coverage")

requirements = [
    ("Conversations within and outside the knowledge base", True, "Intent detection routes to RAG (with citations) or direct LLM (without citations)"),
    ("Cite relevant document pages and relevance for KB queries", True, "Each RAG response includes [Page X, Section: Y] citations with expandable source cards"),
    ("No citations for information outside knowledge base", True, "General queries bypass RAG pipeline entirely"),
    ("Response based on user intended document segment", True, "Sidebar section filter + heading-scoped vector search"),
    ("Seek clarification if document segment is unclear", True, "Ambiguous intent detection triggers clarification with section list"),
    ("Financial document with text, tables, and images", True, "PyMuPDF extracts all content types; metadata tags distinguish them"),
    ("Segment information based on document headings", True, "Heading detection via font size/bold analysis; hierarchical tagging"),
    ("Vector database with appropriate tags", True, "ChromaDB with heading, content_type, page_number metadata filters"),
]

for req, met, detail in requirements:
    icon = ":white_check_mark:" if met else ":x:"
    with st.expander(f"{icon} {req}"):
        st.write(detail)

# ── Tech stack summary ──
st.header("Technology stack")

with st.container(border=True):
    cols = st.columns(4)
    with cols[0]:
        st.markdown("**Frontend**")
        st.markdown("- Streamlit\n- Multi-page navigation")
    with cols[1]:
        st.markdown("**Processing**")
        st.markdown("- PyMuPDF (PDF)\n- sentence-transformers")
    with cols[2]:
        st.markdown("**Storage**")
        st.markdown("- ChromaDB (vectors)\n- Local file cache")
    with cols[3]:
        st.markdown("**AI/LLM**")
        st.markdown("- Ollama + Llama 3.1\n- RAG pipeline")
