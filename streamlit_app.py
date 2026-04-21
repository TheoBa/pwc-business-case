from __future__ import annotations
import streamlit as st

st.set_page_config(
    page_title="BMO Financial Insights",
    page_icon=":material/account_balance:",
    layout="wide",
)

# ── Global session state ──
st.session_state.setdefault("messages", [])
st.session_state.setdefault("chunks", [])
st.session_state.setdefault("vector_store_ready", False)

# ── Navigation ──
page = st.navigation(
    [
        st.Page("app_pages/discovery.py", title="Discovery", icon=":material/search:"),
        st.Page(
            "app_pages/problem_solving.py",
            title="Problem solving",
            icon=":material/chat:",
        ),
        st.Page(
            "app_pages/conclusion.py",
            title="Conclusion",
            icon=":material/architecture:",
        ),
    ],
    position="top",
)

# ── Index documents on first load ──
if not st.session_state.vector_store_ready:
    from lib import pdf_processor, vector_store

    with st.status("Preparing knowledge base...", expanded=True) as status:
        st.write("Downloading BMO 2025 MDA report...")
        pdf_processor.download_pdf()

        st.write("Extracting document chunks...")
        chunks = pdf_processor.extract_chunks()
        st.session_state.chunks = chunks

        st.write(f"Indexing {len(chunks)} chunks into vector store...")
        count = vector_store.index_documents(
            chunks,
            progress_callback=lambda current, total: st.write(
                f"Indexed {current}/{total} chunks"
            ),
        )

        st.session_state.vector_store_ready = True
        status.update(label=f"Knowledge base ready — {count} chunks indexed", state="complete")

page.run()
