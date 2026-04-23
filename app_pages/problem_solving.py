from __future__ import annotations
"""Problem solving tab — RAG chat interface with citations and section filtering."""

import streamlit as st

from lib.llm_client import (
    detect_intent,
    get_clarification_message,
    stream_general_response,
    stream_rag_response,
)
from lib.pdf_processor import get_headings, get_all_time_periods

st.title(":material/chat: Problem solving")
st.caption(
    "Ask questions about BMO's 2025 Annual Report. "
    "Knowledge-base answers include page citations. General questions are also welcome."
)

chunks = st.session_state.get("chunks", [])
if not chunks:
    st.warning("No document chunks loaded. Return to the main page to index the document.")
    st.stop()

# ── Sidebar: section filter ──
top_headings = get_headings(chunks)
time_periods = get_all_time_periods(chunks)

with st.sidebar:
    st.header("Query scope")
    heading_filter = st.selectbox(
        "Filter by section",
        ["All sections"] + top_headings,
        help="Scope the search to a specific document section",
    )

    time_period_filter = st.selectbox(
        "Filter by time period",
        ["All periods"] + time_periods,
        help="Scope the search to a specific time period",
    )

    st.divider()

    def clear_chat():
        st.session_state.messages = []

    st.button("Clear chat", on_click=clear_chat, icon=":material/delete:")

active_filter = None if heading_filter == "All sections" else heading_filter
active_time_period = None if time_period_filter == "All periods" else time_period_filter

# ── Suggestion chips (before first message) ──
SUGGESTIONS = {
    ":blue[:material/trending_up:] What is BMO's net income?": "What is BMO's net income for 2025?",
    ":green[:material/warning:] Key risk factors": "What are BMO's key risk factors discussed in the MDA?",
    ":orange[:material/account_balance:] Capital adequacy": "Summarize BMO's capital adequacy and CET1 ratio.",
    ":red[:material/bar_chart:] Revenue breakdown": "Provide a breakdown of BMO's revenue by operating segment.",
    ":violet[:material/credit_card:] Credit risk exposure": "What is BMO's credit risk exposure and how is it managed?",
}

if not st.session_state.messages:
    selected = st.pills(
        "Try asking:",
        list(SUGGESTIONS.keys()),
        label_visibility="collapsed",
    )
    if selected:
        st.session_state.messages.append(
            {"role": "user", "content": SUGGESTIONS[selected]}
        )
        st.rerun()

# ── Display chat history ──
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        # Show sources for assistant messages
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f":material/source: {len(msg['sources'])} sources cited"):
                for src in msg["sources"]:
                    chunk_type = src.get('chunk_type', src.get('content_type', 'text'))
                    time_info = f" | Period: {src['time_period_str']}" if src.get('time_period_str') else ""
                    st.markdown(
                        f"**Page {src['page_number']}** — {src['heading']} "
                        f"({chunk_type}{time_info})"
                    )
                    st.caption(src["text"][:200] + "..." if len(src["text"]) > 200 else src["text"])
                    st.divider()

        # Feedback on assistant messages
        if msg["role"] == "assistant":
            st.feedback("thumbs", key=f"feedback_{i}")

# ── Handle new input ──
if prompt := st.chat_input("Ask a question about BMO's 2025 Annual Report..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Detect intent
    with st.spinner("Analyzing question..."):
        intent_result = detect_intent(prompt, top_headings)

    intent = intent_result.get("intent", "knowledge_base")

    with st.chat_message("assistant"):
        if intent == "ambiguous":
            # Ask for clarification
            relevant_sections = intent_result.get("sections", top_headings[:5])
            response = get_clarification_message(prompt, relevant_sections)
            st.write(response)
            st.session_state.messages.append(
                {"role": "assistant", "content": response, "sources": []}
            )

        elif intent == "general":
            # General response without RAG
            response = st.write_stream(
                stream_general_response(prompt, st.session_state.messages[:-1])
            )
            st.session_state.messages.append(
                {"role": "assistant", "content": response, "sources": []}
            )

        else:
            # Knowledge base RAG response
            response = st.write_stream(
                stream_rag_response(
                    prompt,
                    st.session_state.messages[:-1],
                    heading_filter=active_filter,
                )
            )

            # Get sources and query plan from the last RAG call
            sources = getattr(stream_rag_response, "_last_sources", [])
            query_plan = getattr(stream_rag_response, "_last_query_plan", None)
            st.session_state.messages.append(
                {"role": "assistant", "content": response, "sources": sources, "query_plan": query_plan}
            )

            # Show query plan
            if query_plan and query_plan.get("filters"):
                active_filters = {k: v for k, v in query_plan["filters"].items() if v}
                if active_filters:
                    with st.expander(":material/tune: Query plan"):
                        st.markdown(f"**Search text:** {query_plan.get('search_text', prompt)}")
                        for key, val in active_filters.items():
                            st.markdown(f"**{key}:** {val}")

            # Show sources
            if sources:
                with st.expander(f":material/source: {len(sources)} sources cited"):
                    for src in sources:
                        chunk_type = src.get('chunk_type', src.get('content_type', 'text'))
                        time_info = f" | Period: {src['time_period_str']}" if src.get('time_period_str') else ""
                        st.markdown(
                            f"**Page {src['page_number']}** — {src['heading']} "
                            f"({chunk_type}{time_info})"
                        )
                        st.caption(
                            src["text"][:200] + "..."
                            if len(src["text"]) > 200
                            else src["text"]
                        )
                        st.divider()

    st.rerun()
