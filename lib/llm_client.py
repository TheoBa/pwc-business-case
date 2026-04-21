from __future__ import annotations
"""LLM client module with RAG prompt construction, streaming, and intent detection."""

from openai import OpenAI
import streamlit as st

from lib import vector_store

SYSTEM_PROMPT = """You are a financial analyst assistant specializing in BMO's 2025 Annual Report \
(Management's Discussion and Analysis). You help users extract insights from the document.

Rules:
1. When answering from the knowledge base, ALWAYS cite the source with page numbers and section headings.
   Format citations as: [Page X, Section: Y]
2. When answering general questions outside the knowledge base, respond normally without citations.
3. If the user's query is ambiguous about which document section they mean, ask a clarifying question \
   listing the available sections before answering.
4. For table data, present numbers clearly and note the page/section source.
5. Be precise with financial figures — do not round unless the user asks for a summary.
6. If you don't know something or the context doesn't contain the answer, say so clearly."""

RAG_PROMPT_TEMPLATE = """Answer the user's question using the following context from BMO's 2025 Annual Report (MDA).

CONTEXT:
{context}

IMPORTANT:
- Cite every fact with [Page X, Section: Y] using the metadata provided.
- If the context doesn't fully answer the question, say what you can and note the gap.
- For tables, preserve the numerical precision from the source.

USER QUESTION: {question}"""

INTENT_DETECTION_PROMPT = """Analyze the user's question and determine:
1. Is this a question about the BMO 2025 Annual Report / MDA document? (knowledge_base)
2. Is this a general question not related to the document? (general)
3. Is the question ambiguous about which section of the document they're referring to? (ambiguous)

If ambiguous, list which sections might be relevant.

Respond in exactly this JSON format (no markdown):
{{"intent": "knowledge_base|general|ambiguous", "sections": ["section1", "section2"], "reasoning": "brief explanation"}}

User question: {question}"""


def _get_client() -> OpenAI:
    """Get OpenAI client using st.secrets."""
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


def detect_intent(question: str, available_sections: list[str]) -> dict:
    """Detect whether a question targets the knowledge base, is general, or is ambiguous.

    Returns:
        dict with keys: intent ("knowledge_base", "general", "ambiguous"), sections, reasoning
    """
    client = _get_client()

    sections_str = ", ".join(available_sections[:20])  # Limit to avoid token overflow
    prompt = INTENT_DETECTION_PROMPT.format(question=question)
    prompt += f"\n\nAvailable document sections: {sections_str}"

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=300,
    )

    content = response.choices[0].message.content.strip()

    # Parse JSON response
    import json

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Fallback: assume knowledge base if we can't parse
        return {"intent": "knowledge_base", "sections": [], "reasoning": "Parse error, defaulting to KB"}


def _format_context(results: list[dict]) -> str:
    """Format vector search results into context string for RAG prompt."""
    context_parts = []
    for i, r in enumerate(results, 1):
        context_parts.append(
            f"[Source {i}] Page {r['page_number']}, Section: {r['heading']}, "
            f"Type: {r['content_type']}\n{r['text']}"
        )
    return "\n\n---\n\n".join(context_parts)


def stream_rag_response(
    question: str,
    chat_history: list[dict],
    heading_filter: str | None = None,
    top_k: int = 5,
):
    """Stream a RAG response with document context.

    Args:
        question: User's question
        chat_history: Previous messages for context
        heading_filter: Optional section filter
        top_k: Number of chunks to retrieve

    Yields:
        Token strings for st.write_stream

    Returns via generator .value after exhaustion:
        Tuple of (full_response, sources) where sources is list of dicts
    """
    # Retrieve relevant chunks
    results = vector_store.query(
        text=question,
        heading_filter=heading_filter,
        top_k=top_k,
    )

    context = _format_context(results) if results else "No relevant context found in the document."

    # Build messages
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add recent chat history (last 10 messages to stay within context window)
    for msg in chat_history[-10:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Add the RAG-augmented question
    rag_question = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
    messages.append({"role": "user", "content": rag_question})

    client = _get_client()
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        stream=True,
        temperature=0.1,
    )

    # Yield chunks for st.write_stream
    full_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            full_response += token
            yield token

    # Store sources for citation display (accessed after streaming)
    stream_rag_response._last_sources = results
    stream_rag_response._last_response = full_response


def stream_general_response(question: str, chat_history: list[dict]):
    """Stream a general (non-RAG) response.

    Yields token strings for st.write_stream.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for msg in chat_history[-10:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": question})

    client = _get_client()
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        stream=True,
        temperature=0.3,
    )

    full_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            full_response += token
            yield token

    stream_general_response._last_response = full_response


def get_clarification_message(question: str, sections: list[str]) -> str:
    """Generate a clarification message when the query is ambiguous."""
    section_list = "\n".join(f"- **{s}**" for s in sections[:10])
    return (
        f"Your question could relate to multiple sections of the document. "
        f"Could you clarify which area you're interested in?\n\n"
        f"Relevant sections:\n{section_list}\n\n"
        f"You can also select a specific section from the sidebar filter."
    )
