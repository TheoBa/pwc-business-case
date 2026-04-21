from __future__ import annotations
"""Discovery tab — explore the document structure, headings, and chunk metadata."""

import pandas as pd
import streamlit as st

from lib.pdf_processor import get_all_headings, get_headings

st.title(":material/search: Discovery")
st.caption("Explore the structure and contents of BMO's 2025 Management Discussion & Analysis report.")

chunks = st.session_state.get("chunks", [])

if not chunks:
    st.warning("No document chunks loaded. Return to the main page to index the document.")
    st.stop()

# ── Sidebar filters ──
with st.sidebar:
    st.header("Filters")

    all_headings = get_all_headings(chunks)
    top_headings = get_headings(chunks)
    content_types = sorted({c["content_type"] for c in chunks})
    page_numbers = sorted({c["page_number"] for c in chunks})

    selected_heading = st.selectbox(
        "Section",
        ["All sections"] + top_headings,
    )

    selected_type = st.selectbox(
        "Content type",
        ["All types"] + content_types,
    )

    page_range = st.slider(
        "Page range",
        min_value=min(page_numbers),
        max_value=max(page_numbers),
        value=(min(page_numbers), max(page_numbers)),
    )

# ── Apply filters ──
filtered = chunks
if selected_heading != "All sections":
    filtered = [c for c in filtered if c["heading"].startswith(selected_heading)]
if selected_type != "All types":
    filtered = [c for c in filtered if c["content_type"] == selected_type]
filtered = [c for c in filtered if page_range[0] <= c["page_number"] <= page_range[1]]

# ── KPI metrics ──
with st.container(horizontal=True):
    st.metric("Total pages", max(page_numbers), border=True)
    st.metric("Total chunks", len(chunks), border=True)
    st.metric("Sections", len(top_headings), border=True)
    st.metric(
        "Tables detected",
        sum(1 for c in chunks if c["content_type"] == "table"),
        border=True,
    )

with st.container(horizontal=True):
    st.metric("Filtered chunks", len(filtered), border=True)
    st.metric(
        "Avg chunk length",
        f"{sum(len(c['text']) for c in filtered) // max(len(filtered), 1)} chars",
        border=True,
    )

# ── Content type distribution ──
col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.subheader("Content type distribution")
        type_counts = pd.DataFrame(
            [{"Type": c["content_type"]} for c in filtered]
        )
        if not type_counts.empty:
            st.bar_chart(type_counts["Type"].value_counts())

with col2:
    with st.container(border=True):
        st.subheader("Chunks per section (top 15)")
        heading_counts = {}
        for c in filtered:
            top = c["heading"].split(" > ")[0]
            heading_counts[top] = heading_counts.get(top, 0) + 1
        if heading_counts:
            hc_df = (
                pd.DataFrame(
                    [{"Section": k, "Chunks": v} for k, v in heading_counts.items()]
                )
                .sort_values("Chunks", ascending=False)
                .head(15)
            )
            st.bar_chart(hc_df, x="Section", y="Chunks")

# ── Document heading tree ──
st.subheader("Document sections")

for heading in top_headings:
    section_chunks = [c for c in filtered if c["heading"].startswith(heading)]
    if not section_chunks:
        continue

    with st.expander(f"**{heading}** ({len(section_chunks)} chunks)"):
        # Show sub-headings
        sub_headings = sorted(
            {c["heading"] for c in section_chunks if " > " in c["heading"]}
        )
        if sub_headings:
            for sub in sub_headings:
                sub_chunks = [c for c in section_chunks if c["heading"] == sub]
                sub_name = sub.split(" > ", 1)[1]
                st.markdown(
                    f"  - **{sub_name}** — {len(sub_chunks)} chunks, "
                    f"pages {min(c['page_number'] for c in sub_chunks)}-"
                    f"{max(c['page_number'] for c in sub_chunks)}"
                )

        # Preview first chunk
        preview = section_chunks[0]["text"][:300]
        st.caption(f"Preview: {preview}...")

# ── Filterable chunk table ──
st.subheader("All chunks")

df = pd.DataFrame(
    [
        {
            "Heading": c["heading"],
            "Type": c["content_type"],
            "Page": c["page_number"],
            "Preview": c["text"][:150] + "..." if len(c["text"]) > 150 else c["text"],
            "Length": len(c["text"]),
        }
        for c in filtered
    ]
)

if not df.empty:
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Preview": st.column_config.TextColumn("Preview", width="large"),
            "Length": st.column_config.ProgressColumn(
                "Length", min_value=0, max_value=int(df["Length"].max())
            ),
        },
    )
else:
    st.info("No chunks match the current filters.")
